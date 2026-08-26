/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "layers/npu_torch/deepseek_v4_cp_ownership.h"

#include <glog/logging.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <map>
#include <tuple>
#include <utility>

namespace xllm::layer {
namespace {

constexpr uint64_t kFnvOffsetBasis = 1469598103934665603ULL;
constexpr uint64_t kFnvPrime = 1099511628211ULL;

struct RouteEvent {
  int64_t source_row_index = -1;
  int64_t absolute_position = -1;
  int64_t window_id = -1;
  int32_t window_offset = -1;
  int32_t sequence_id = -1;
  int32_t source_rank = -1;
  int32_t destination_rank = -1;
  bool cache_write = false;
  int64_t destination_local_slot = -1;
  int64_t output_rope_index = -1;
};

struct WindowInfo {
  int32_t sequence_id = -1;
  int32_t state_owner_rank = -1;
  int64_t window_id = -1;
  int64_t latest_event_index = -1;
  int64_t latest_absolute_position = -1;
};

void hash_bytes(uint64_t* hash, const void* data, size_t size) {
  CHECK(hash != nullptr);
  const uint8_t* bytes = static_cast<const uint8_t*>(data);
  for (size_t index = 0; index < size; ++index) {
    *hash ^= bytes[index];
    *hash *= kFnvPrime;
  }
}

template <typename T>
void hash_value(uint64_t* hash, const T& value) {
  hash_bytes(hash, &value, sizeof(T));
}

torch::Tensor make_int64_tensor(const std::vector<int64_t>& values,
                                const torch::Device& device) {
  return torch::tensor(
             values,
             torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU))
      .to(device, /*non_blocking=*/true);
}

torch::Tensor make_int32_tensor(const std::vector<int32_t>& values,
                                const torch::Device& device) {
  return torch::tensor(
             values,
             torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU))
      .to(device, /*non_blocking=*/true);
}

torch::Tensor make_uint8_tensor(const std::vector<uint8_t>& values,
                                const torch::Device& device) {
  return torch::tensor(
             values,
             torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU))
      .to(device, /*non_blocking=*/true);
}

torch::Tensor normalize_block_table(const torch::Tensor& block_table,
                                    int64_t sequence_count,
                                    const char* name) {
  CHECK(block_table.defined()) << name << " must be defined";
  CHECK(block_table.device().is_cpu())
      << name << " must use host metadata; device-to-host sync is forbidden";
  CHECK_EQ(block_table.dim(), 2) << name << " must be two-dimensional";
  CHECK_EQ(block_table.size(0), sequence_count)
      << name << " sequence count mismatch";
  CHECK(block_table.scalar_type() == torch::kInt ||
        block_table.scalar_type() == torch::kLong)
      << name << " must use int32 or int64";
  return block_table.to(torch::kInt64).contiguous();
}

int64_t block_id_at(const torch::TensorAccessor<int64_t, 2>& table,
                    int32_t sequence_id,
                    int64_t block_index,
                    const char* name) {
  CHECK_GE(block_index, 0);
  CHECK_LT(block_index, table.size(1))
      << name << " does not cover block index " << block_index
      << " for sequence " << sequence_id;
  const int64_t global_block_id = table[sequence_id][block_index];
  CHECK_GE(global_block_id, 0)
      << name << " contains an invalid block id at sequence " << sequence_id
      << ", block index " << block_index;
  return global_block_id;
}

int32_t source_rank_for_offset(int32_t query_offset,
                               int32_t chunk_length,
                               int32_t cp_size) {
  CHECK_GT(chunk_length, 0);
  const int32_t chunk_id = query_offset / chunk_length;
  return chunk_id < cp_size ? chunk_id : 2 * cp_size - 1 - chunk_id;
}

int64_t source_local_row_index(int32_t query_offset,
                               int32_t chunk_length,
                               int64_t local_sequence_offset,
                               int32_t cp_size) {
  const int32_t chunk_id = query_offset / chunk_length;
  const int32_t offset_in_chunk = query_offset % chunk_length;
  const int64_t half_offset = chunk_id < cp_size ? 0 : chunk_length;
  return local_sequence_offset + half_offset + offset_in_chunk;
}

bool canonical_event_less(const RouteEvent& lhs, const RouteEvent& rhs) {
  return std::tie(lhs.sequence_id,
                  lhs.window_id,
                  lhs.window_offset,
                  lhs.absolute_position,
                  lhs.source_rank) <
         std::tie(rhs.sequence_id,
                  rhs.window_id,
                  rhs.window_offset,
                  rhs.absolute_position,
                  rhs.source_rank);
}

Dsv4CpRouteDescriptor build_descriptor(const std::vector<RouteEvent>& events,
                                        int32_t cp_size,
                                        int32_t cp_rank,
                                        const torch::Device& device) {
  std::vector<std::vector<std::vector<RouteEvent>>> routes(
      static_cast<size_t>(cp_size),
      std::vector<std::vector<RouteEvent>>(static_cast<size_t>(cp_size)));
  for (const RouteEvent& event : events) {
    CHECK_GE(event.source_rank, 0);
    CHECK_LT(event.source_rank, cp_size);
    CHECK_GE(event.destination_rank, 0);
    CHECK_LT(event.destination_rank, cp_size);
    routes[event.source_rank][event.destination_rank].emplace_back(event);
  }

  int64_t padded_rows_per_peer = 0;
  for (std::vector<std::vector<RouteEvent>>& source_routes : routes) {
    for (std::vector<RouteEvent>& owner_routes : source_routes) {
      std::sort(owner_routes.begin(), owner_routes.end(), canonical_event_less);
      padded_rows_per_peer =
          std::max(padded_rows_per_peer,
                   static_cast<int64_t>(owner_routes.size()));
    }
  }

  const int64_t fixed_row_count = cp_size * padded_rows_per_peer;
  std::vector<int64_t> send_row_indices(fixed_row_count, -1);
  std::vector<int64_t> send_valid_indices;
  std::vector<int32_t> send_owner_ranks(fixed_row_count, -1);
  std::vector<int32_t> send_sequence_ids(fixed_row_count, -1);
  std::vector<int64_t> send_window_ids(fixed_row_count, -1);
  std::vector<int32_t> send_window_offsets(fixed_row_count, -1);
  std::vector<uint8_t> send_cache_write_flags(fixed_row_count, 0);
  std::vector<int32_t> send_real_counts(cp_size, 0);

  std::vector<int32_t> recv_source_ranks(fixed_row_count, -1);
  std::vector<int32_t> recv_sequence_ids(fixed_row_count, -1);
  std::vector<int64_t> recv_window_ids(fixed_row_count, -1);
  std::vector<int32_t> recv_window_offsets(fixed_row_count, -1);
  std::vector<uint8_t> recv_cache_write_flags(fixed_row_count, 0);
  std::vector<int32_t> recv_real_counts(cp_size, 0);
  std::vector<std::pair<RouteEvent, int64_t>> canonical_recv_slots;
  send_valid_indices.reserve(static_cast<size_t>(std::count_if(
      events.begin(), events.end(), [cp_rank](const RouteEvent& event) {
        return event.source_rank == cp_rank;
      })));

  for (int32_t peer = 0; peer < cp_size; ++peer) {
    const std::vector<RouteEvent>& sends = routes[cp_rank][peer];
    send_real_counts[peer] = static_cast<int32_t>(sends.size());
    for (size_t index = 0; index < sends.size(); ++index) {
      const int64_t slot = peer * padded_rows_per_peer +
                           static_cast<int64_t>(index);
      const RouteEvent& event = sends[index];
      send_valid_indices.emplace_back(slot);
      send_row_indices[slot] = event.source_row_index;
      send_owner_ranks[slot] = event.destination_rank;
      send_sequence_ids[slot] = event.sequence_id;
      send_window_ids[slot] = event.window_id;
      send_window_offsets[slot] = event.window_offset;
      send_cache_write_flags[slot] = event.cache_write ? 1 : 0;
    }

    const std::vector<RouteEvent>& receives = routes[peer][cp_rank];
    recv_real_counts[peer] = static_cast<int32_t>(receives.size());
    for (size_t index = 0; index < receives.size(); ++index) {
      const int64_t slot = peer * padded_rows_per_peer +
                           static_cast<int64_t>(index);
      const RouteEvent& event = receives[index];
      recv_source_ranks[slot] = event.source_rank;
      recv_sequence_ids[slot] = event.sequence_id;
      recv_window_ids[slot] = event.window_id;
      recv_window_offsets[slot] = event.window_offset;
      recv_cache_write_flags[slot] = event.cache_write ? 1 : 0;
      canonical_recv_slots.emplace_back(event, slot);
    }
  }

  std::sort(canonical_recv_slots.begin(),
            canonical_recv_slots.end(),
            [](const auto& lhs, const auto& rhs) {
              return canonical_event_less(lhs.first, rhs.first);
            });
  std::vector<int64_t> recv_canonical_indices(fixed_row_count, -1);
  std::vector<int64_t> recv_canonical_local_slots;
  recv_canonical_local_slots.reserve(canonical_recv_slots.size());
  for (size_t index = 0; index < canonical_recv_slots.size(); ++index) {
    recv_canonical_indices[index] = canonical_recv_slots[index].second;
    recv_canonical_local_slots.emplace_back(
        canonical_recv_slots[index].first.destination_local_slot);
  }

  Dsv4CpRouteDescriptor descriptor;
  descriptor.send_row_indices = make_int64_tensor(send_row_indices, device);
  descriptor.send_valid_indices =
      make_int64_tensor(send_valid_indices, device);
  descriptor.send_owner_ranks = make_int32_tensor(send_owner_ranks, device);
  descriptor.send_sequence_ids = make_int32_tensor(send_sequence_ids, device);
  descriptor.send_window_ids = make_int64_tensor(send_window_ids, device);
  descriptor.send_window_offsets =
      make_int32_tensor(send_window_offsets, device);
  descriptor.send_cache_write_flags =
      make_uint8_tensor(send_cache_write_flags, device);
  descriptor.send_real_counts = make_int32_tensor(send_real_counts, device);
  descriptor.recv_source_ranks = make_int32_tensor(recv_source_ranks, device);
  descriptor.recv_sequence_ids = make_int32_tensor(recv_sequence_ids, device);
  descriptor.recv_window_ids = make_int64_tensor(recv_window_ids, device);
  descriptor.recv_window_offsets =
      make_int32_tensor(recv_window_offsets, device);
  descriptor.recv_cache_write_flags =
      make_uint8_tensor(recv_cache_write_flags, device);
  descriptor.recv_real_counts = make_int32_tensor(recv_real_counts, device);
  descriptor.recv_canonical_indices =
      make_int64_tensor(recv_canonical_indices, device);
  descriptor.recv_canonical_local_slots =
      make_int64_tensor(recv_canonical_local_slots, device);
  descriptor.padded_rows_per_peer = padded_rows_per_peer;
  descriptor.send_real_row_count = events.empty()
                                       ? 0
                                       : std::count_if(
                                             events.begin(),
                                             events.end(),
                                             [cp_rank](const RouteEvent& event) {
                                               return event.source_rank ==
                                                      cp_rank;
                                             });
  descriptor.recv_real_row_count =
      static_cast<int64_t>(canonical_recv_slots.size());
  return descriptor;
}

Dsv4CpOwnerMetadata build_owner_metadata(
    const std::vector<RouteEvent>& events,
    const Dsv4CpRouteDescriptor& route,
    const torch::TensorAccessor<int64_t, 2>& state_block_table,
    int64_t state_block_size,
    int32_t cp_size,
    int32_t cp_rank,
    const torch::Device& device,
    Dsv4CpCacheAddressing cache_addressing) {
  std::vector<RouteEvent> owner_events;
  owner_events.reserve(static_cast<size_t>(route.recv_real_row_count));
  for (const RouteEvent& event : events) {
    if (event.destination_rank == cp_rank) {
      owner_events.emplace_back(event);
    }
  }
  std::sort(owner_events.begin(), owner_events.end(), canonical_event_less);
  CHECK_EQ(owner_events.size(), route.recv_real_row_count);

  std::vector<int32_t> row_sequence_ids;
  std::vector<int64_t> row_window_ids;
  std::vector<int32_t> row_window_offsets;
  std::vector<int32_t> row_absolute_positions;
  row_sequence_ids.reserve(owner_events.size());
  row_window_ids.reserve(owner_events.size());
  row_window_offsets.reserve(owner_events.size());
  row_absolute_positions.reserve(owner_events.size());

  std::vector<int32_t> window_sequence_ids;
  std::vector<int64_t> window_ids;
  std::vector<int32_t> q_cu_seq_lens = {0};
  std::vector<int32_t> start_positions;
  std::vector<uint8_t> output_cache_write_flags;
  std::vector<int64_t> output_cache_write_row_indices;
  std::vector<int64_t> output_rope_indices;
  std::vector<int32_t> local_state_block_table;
  const int64_t state_block_columns = state_block_table.size(1);
  const Dsv4CpCacheLayout cache_layout(cp_size, cp_rank);

  int64_t owner_row_index = 0;
  size_t event_index = 0;
  while (event_index < owner_events.size()) {
    const RouteEvent& first = owner_events[event_index];
    const int32_t sequence_id = first.sequence_id;
    const int64_t state_block_index =
        first.absolute_position / state_block_size;
    const int32_t start_position =
        static_cast<int32_t>(first.absolute_position);
    CHECK_EQ(static_cast<int64_t>(start_position), first.absolute_position)
        << "DSV4 owner Core start position exceeds int32 metadata range";

    int64_t current_window_id = first.window_id;
    int64_t current_output_rope_index = first.output_rope_index;
    bool current_window_cache_write = false;
    int64_t current_window_cache_write_row = -1;
    int64_t previous_absolute_position = first.absolute_position - 1;
    int64_t segment_row_count = 0;
    while (event_index < owner_events.size() &&
           owner_events[event_index].sequence_id == sequence_id &&
           owner_events[event_index].absolute_position / state_block_size ==
               state_block_index) {
      const RouteEvent& event = owner_events[event_index++];
      CHECK_EQ(event.absolute_position, previous_absolute_position + 1)
          << "DSV4 owner Core absolute positions must be contiguous";
      if (event.window_id != current_window_id) {
        window_sequence_ids.emplace_back(sequence_id);
        window_ids.emplace_back(current_window_id);
        output_cache_write_flags.emplace_back(
            current_window_cache_write ? 1 : 0);
        output_cache_write_row_indices.emplace_back(
            current_window_cache_write_row);
        output_rope_indices.emplace_back(current_output_rope_index);
        current_window_id = event.window_id;
        current_output_rope_index = event.output_rope_index;
        current_window_cache_write = false;
        current_window_cache_write_row = -1;
      }
      CHECK_EQ(event.output_rope_index, current_output_rope_index)
          << "DSV4 owner Core window must use one compressed RoPE row";
      CHECK(!current_window_cache_write || !event.cache_write)
          << "DSV4 owner Core window has duplicate cache-write rows";
      row_sequence_ids.emplace_back(event.sequence_id);
      row_window_ids.emplace_back(event.window_id);
      row_window_offsets.emplace_back(event.window_offset);
      row_absolute_positions.emplace_back(
          static_cast<int32_t>(event.absolute_position));
      if (event.cache_write) {
        current_window_cache_write = true;
        current_window_cache_write_row = owner_row_index;
      }
      previous_absolute_position = event.absolute_position;
      ++owner_row_index;
      ++segment_row_count;
    }

    CHECK_GT(segment_row_count, 0);
    window_sequence_ids.emplace_back(sequence_id);
    window_ids.emplace_back(current_window_id);
    output_cache_write_flags.emplace_back(current_window_cache_write ? 1 : 0);
    output_cache_write_row_indices.emplace_back(
        current_window_cache_write_row);
    output_rope_indices.emplace_back(current_output_rope_index);
    CHECK_LE(owner_row_index,
             static_cast<int64_t>(std::numeric_limits<int32_t>::max()));
    q_cu_seq_lens.emplace_back(static_cast<int32_t>(owner_row_index));
    start_positions.emplace_back(start_position);

    for (int64_t block_index = 0; block_index < state_block_columns;
         ++block_index) {
      const int64_t global_block_id =
          state_block_table[sequence_id][block_index];
      const int64_t local_block_id =
          global_block_id % cp_size == cp_rank
              ? cache_layout.physical_block_id(global_block_id,
                                               cache_addressing)
              : -1;
      CHECK_LE(local_block_id,
               static_cast<int64_t>(std::numeric_limits<int32_t>::max()))
          << "DSV4 owner-local block id exceeds int32 metadata range";
      local_state_block_table.emplace_back(
          static_cast<int32_t>(local_block_id));
    }
  }

  const int64_t segment_count =
      static_cast<int64_t>(start_positions.size());
  const int64_t output_row_count =
      static_cast<int64_t>(window_sequence_ids.size());
  torch::Tensor local_state_table =
      torch::tensor(local_state_block_table,
                    torch::TensorOptions().dtype(torch::kInt32))
          .view({segment_count, state_block_columns})
          .to(device, /*non_blocking=*/true);

  Dsv4CpOwnerMetadata metadata;
  metadata.recv_canonical_indices = route.recv_canonical_indices
                                        .slice(/*dim=*/0,
                                               /*start=*/0,
                                               route.recv_real_row_count)
                                        .clone();
  metadata.row_sequence_ids = make_int32_tensor(row_sequence_ids, device);
  metadata.row_window_ids = make_int64_tensor(row_window_ids, device);
  metadata.row_window_offsets =
      make_int32_tensor(row_window_offsets, device);
  metadata.row_absolute_positions =
      make_int32_tensor(row_absolute_positions, device);
  metadata.window_sequence_ids =
      make_int32_tensor(window_sequence_ids, device);
  metadata.window_ids = make_int64_tensor(window_ids, device);
  metadata.q_cu_seq_lens = make_int32_tensor(q_cu_seq_lens, device);
  metadata.start_positions = make_int32_tensor(start_positions, device);
  metadata.local_state_block_table = std::move(local_state_table);
  metadata.output_cache_write_flags =
      make_uint8_tensor(output_cache_write_flags, device);
  metadata.output_cache_write_row_indices =
      make_int64_tensor(output_cache_write_row_indices, device);
  metadata.output_rope_indices = make_int64_tensor(output_rope_indices, device);
  metadata.row_capacity = route.recv_source_ranks.numel();
  metadata.real_row_count = route.recv_real_row_count;
  metadata.segment_count = segment_count;
  metadata.output_row_count = output_row_count;
  metadata.committed_output_row_count = static_cast<int64_t>(std::count(
      output_cache_write_flags.begin(), output_cache_write_flags.end(), 1));
  return metadata;
}

std::vector<RouteEvent> build_swa_events(
    const CpRowLayout& row_layout,
    const std::vector<int32_t>& global_q_seq_lens,
    const std::vector<int32_t>& global_kv_seq_lens,
    const torch::TensorAccessor<int64_t, 2>& swa_block_table,
    int64_t swa_block_size,
    Dsv4CpCacheAddressing cache_addressing) {
  CHECK_EQ(global_q_seq_lens.size(), global_kv_seq_lens.size());
  CHECK_GT(swa_block_size, 0);
  const int32_t cp_size = row_layout.cp_size();
  const int32_t chunk_count = 2 * cp_size;
  std::vector<RouteEvent> events;
  events.reserve(static_cast<size_t>(row_layout.global_real_token_count()));
  int64_t local_sequence_offset = 0;
  for (int32_t sequence_id = 0;
       sequence_id < static_cast<int32_t>(global_q_seq_lens.size());
       ++sequence_id) {
    const int32_t query_length = global_q_seq_lens[sequence_id];
    const int32_t kv_length = global_kv_seq_lens[sequence_id];
    CHECK_GE(query_length, 0);
    CHECK_GE(kv_length, query_length);
    const int32_t start_position = kv_length - query_length;
    const int32_t padded_length =
        ((query_length + chunk_count - 1) / chunk_count) * chunk_count;
    const int32_t chunk_length = padded_length / chunk_count;

    for (int32_t query_offset = 0; query_offset < query_length;
         ++query_offset) {
      const int64_t absolute_position = start_position + query_offset;
      const int64_t block_index = absolute_position / swa_block_size;
      const int64_t global_block_id =
          block_id_at(swa_block_table,
                      sequence_id,
                      block_index,
                      "DSV4 SWA block table");
      const int32_t owner_rank =
          static_cast<int32_t>(global_block_id % cp_size);
      const int32_t source_rank =
          source_rank_for_offset(query_offset, chunk_length, cp_size);
      const int64_t source_row = source_local_row_index(query_offset,
                                                        chunk_length,
                                                        local_sequence_offset,
                                                        cp_size);
      CHECK_LT(source_row, row_layout.local_padded_token_count());
      const Dsv4CpCacheLayout cache_layout(cp_size, owner_rank);
      const int64_t local_slot = cache_layout.physical_slot(
          global_block_id,
          absolute_position % swa_block_size,
          swa_block_size,
          cache_addressing);
      events.emplace_back(RouteEvent{source_row,
                                     absolute_position,
                                     absolute_position,
                                     /*window_offset=*/0,
                                     sequence_id,
                                     source_rank,
                                     owner_rank,
                                     /*cache_write=*/true,
                                     local_slot,
                                     /*output_rope_index=*/-1});
    }
    local_sequence_offset += 2 * chunk_length;
  }
  CHECK_EQ(local_sequence_offset, row_layout.local_padded_token_count());
  return events;
}

uint64_t compute_signature(const CpRowLayout& row_layout,
                           const std::vector<int32_t>& q_seq_lens,
                           const std::vector<int32_t>& kv_seq_lens,
                           const torch::Tensor& state_block_table,
                           const torch::Tensor& compressed_block_table,
                           int64_t compress_ratio,
                           int64_t state_block_size,
                           int64_t compressed_block_size,
                           Dsv4CpCacheAddressing cache_addressing) {
  uint64_t signature = kFnvOffsetBasis;
  hash_value(&signature, row_layout.signature());
  hash_value(&signature, compress_ratio);
  hash_value(&signature, state_block_size);
  hash_value(&signature, compressed_block_size);
  hash_value(&signature, cache_addressing);
  if (!q_seq_lens.empty()) {
    hash_bytes(&signature,
               q_seq_lens.data(),
               q_seq_lens.size() * sizeof(int32_t));
    hash_bytes(&signature,
               kv_seq_lens.data(),
               kv_seq_lens.size() * sizeof(int32_t));
  }
  hash_bytes(&signature,
             state_block_table.data_ptr<int64_t>(),
             state_block_table.numel() * sizeof(int64_t));
  hash_bytes(&signature,
             compressed_block_table.data_ptr<int64_t>(),
             compressed_block_table.numel() * sizeof(int64_t));
  return signature;
}

}  // namespace

const Dsv4CpRouteDescriptor& Dsv4CpOwnershipPlan::main_route() const {
  return main_route_;
}

const Dsv4CpRouteDescriptor& Dsv4CpOwnershipPlan::index_route() const {
  return index_route_;
}

const Dsv4CpRouteDescriptor& Dsv4CpOwnershipPlan::swa_route() const {
  return swa_route_;
}

const Dsv4CpRouteDescriptor&
Dsv4CpOwnershipPlan::compressed_cache_route() const {
  return compressed_cache_route_;
}

const Dsv4CpOwnerMetadata& Dsv4CpOwnershipPlan::owner_metadata() const {
  return owner_metadata_;
}

uint64_t Dsv4CpOwnershipPlan::signature() const { return signature_; }

Dsv4CpRouteDescriptor Dsv4CpOwnershipPlanner::build_swa_route(
    const CpRowLayout& row_layout,
    const std::vector<int32_t>& global_q_seq_lens,
    const std::vector<int32_t>& global_kv_seq_lens,
    const torch::Tensor& swa_global_block_table,
    int64_t swa_block_size,
    const torch::Device& device,
    Dsv4CpCacheAddressing cache_addressing) const {
  CHECK_GT(row_layout.cp_size(), 1);
  const int64_t sequence_count =
      static_cast<int64_t>(global_q_seq_lens.size());
  const torch::Tensor swa_table = normalize_block_table(
      swa_global_block_table, sequence_count, "DSV4 SWA block table");
  const std::vector<RouteEvent> events = build_swa_events(
      row_layout,
      global_q_seq_lens,
      global_kv_seq_lens,
      swa_table.accessor<int64_t, 2>(),
      swa_block_size,
      cache_addressing);
  return build_descriptor(
      events, row_layout.cp_size(), row_layout.cp_rank(), device);
}

Dsv4CpOwnershipPlan Dsv4CpOwnershipPlanner::build(
    const CpRowLayout& row_layout,
    const std::vector<int32_t>& global_q_seq_lens,
    const std::vector<int32_t>& global_kv_seq_lens,
    const torch::Tensor& state_global_block_table,
    const torch::Tensor& compressed_global_block_table,
    int64_t compress_ratio,
    int64_t state_block_size,
    int64_t compressed_block_size,
    const torch::Device& device,
    Dsv4CpCacheAddressing cache_addressing) const {
  CHECK_GT(row_layout.cp_size(), 1);
  CHECK_EQ(global_q_seq_lens.size(), global_kv_seq_lens.size());
  CHECK(compress_ratio == 4 || compress_ratio == 128)
      << "DSV4 compressor ownership supports C4 or C128";
  CHECK_GT(state_block_size, 0);
  CHECK_GT(compressed_block_size, 0);
  CHECK_EQ(state_block_size % compress_ratio, 0)
      << "compression windows must not cross state blocks";

  const int64_t sequence_count =
      static_cast<int64_t>(global_q_seq_lens.size());
  const torch::Tensor state_table = normalize_block_table(
      state_global_block_table, sequence_count, "DSV4 state block table");
  const torch::Tensor compressed_table = normalize_block_table(
      compressed_global_block_table,
      sequence_count,
      "DSV4 compressed block table");
  const auto state_accessor = state_table.accessor<int64_t, 2>();
  const auto compressed_accessor = compressed_table.accessor<int64_t, 2>();

  const int32_t cp_size = row_layout.cp_size();
  const int32_t cp_rank = row_layout.cp_rank();
  const int32_t chunk_count = 2 * cp_size;
  std::vector<RouteEvent> state_events;
  std::map<std::pair<int32_t, int64_t>, WindowInfo> windows;
  int64_t local_sequence_offset = 0;
  int64_t output_rope_offset = 0;

  for (int32_t sequence_id = 0;
       sequence_id < static_cast<int32_t>(global_q_seq_lens.size());
       ++sequence_id) {
    const int32_t query_length = global_q_seq_lens[sequence_id];
    const int32_t kv_length = global_kv_seq_lens[sequence_id];
    CHECK_GE(query_length, 0);
    CHECK_GE(kv_length, query_length);
    const int32_t start_position = kv_length - query_length;
    const int64_t first_window_id = start_position / compress_ratio;
    const int32_t padded_length =
        ((query_length + chunk_count - 1) / chunk_count) * chunk_count;
    const int32_t chunk_length = padded_length / chunk_count;

    for (int32_t query_offset = 0; query_offset < query_length;
         ++query_offset) {
      const int64_t absolute_position = start_position + query_offset;
      const int64_t state_block_index =
          absolute_position / state_block_size;
      const int64_t state_global_block_id = block_id_at(state_accessor,
                                                        sequence_id,
                                                        state_block_index,
                                                        "DSV4 state block table");
      const int32_t state_owner_rank =
          static_cast<int32_t>(state_global_block_id % cp_size);
      const int32_t source_rank =
          source_rank_for_offset(query_offset, chunk_length, cp_size);
      const int64_t source_row = source_local_row_index(query_offset,
                                                        chunk_length,
                                                        local_sequence_offset,
                                                        cp_size);
      CHECK_LT(source_row, row_layout.local_padded_token_count());
      const int64_t window_id = absolute_position / compress_ratio;
      const int32_t window_offset =
          static_cast<int32_t>(absolute_position % compress_ratio);
      const int64_t output_rope_index =
          output_rope_offset + window_id - first_window_id;

      state_events.emplace_back(RouteEvent{source_row,
                                           absolute_position,
                                           window_id,
                                           window_offset,
                                           sequence_id,
                                           source_rank,
                                           state_owner_rank,
                                           /*cache_write=*/false,
                                           /*destination_local_slot=*/-1,
                                           output_rope_index});
      const std::pair<int32_t, int64_t> window_key{sequence_id, window_id};
      auto [window_it, inserted] = windows.emplace(
          window_key,
          WindowInfo{sequence_id,
                     state_owner_rank,
                     window_id,
                     static_cast<int64_t>(state_events.size() - 1),
                     absolute_position});
      if (!inserted) {
        CHECK_EQ(window_it->second.state_owner_rank, state_owner_rank)
            << "one compression window cannot span state owners";
        window_it->second.latest_event_index =
            static_cast<int64_t>(state_events.size() - 1);
        window_it->second.latest_absolute_position = absolute_position;
      }
    }
    if (query_length > 0) {
      const int64_t last_window_id =
          (static_cast<int64_t>(kv_length) - 1) / compress_ratio;
      output_rope_offset += last_window_id - first_window_id + 1;
    }
    local_sequence_offset += 2 * chunk_length;
  }

  CHECK_EQ(local_sequence_offset, row_layout.local_padded_token_count());
  std::vector<int64_t> owner_output_rows(cp_size, 0);
  std::vector<RouteEvent> compressed_cache_events;
  compressed_cache_events.reserve(windows.size());
  for (const auto& [window_key, window] : windows) {
    (void)window_key;
    CHECK_GE(window.latest_event_index, 0);
    const bool cache_write =
        (window.latest_absolute_position + 1) % compress_ratio == 0;
    state_events[window.latest_event_index].cache_write = cache_write;

    const int64_t source_output_row =
        owner_output_rows[window.state_owner_rank]++;
    if (!cache_write) {
      continue;
    }

    const int64_t compressed_block_index =
        window.window_id / compressed_block_size;
    const int64_t compressed_global_block_id = block_id_at(
        compressed_accessor,
        window.sequence_id,
        compressed_block_index,
        "DSV4 compressed block table");
    const int32_t compressed_owner_rank =
        static_cast<int32_t>(compressed_global_block_id % cp_size);
    const Dsv4CpCacheLayout compressed_layout(cp_size,
                                              compressed_owner_rank);
    const int64_t compressed_local_slot = compressed_layout.physical_slot(
        compressed_global_block_id,
        window.window_id % compressed_block_size,
        compressed_block_size,
        cache_addressing);
    compressed_cache_events.emplace_back(RouteEvent{
        source_output_row,
        window.latest_absolute_position,
        window.window_id,
        static_cast<int32_t>(window.latest_absolute_position % compress_ratio),
        window.sequence_id,
        window.state_owner_rank,
        compressed_owner_rank,
        /*cache_write=*/cache_write,
        compressed_local_slot,
        /*output_rope_index=*/-1});
  }

  Dsv4CpOwnershipPlan plan;
  plan.main_route_ =
      build_descriptor(state_events, cp_size, cp_rank, device);
  plan.index_route_ = plan.main_route_;
  plan.owner_metadata_ = build_owner_metadata(state_events,
                                              plan.main_route_,
                                              state_accessor,
                                              state_block_size,
                                              cp_size,
                                              cp_rank,
                                              device,
                                              cache_addressing);
  plan.swa_route_ = build_swa_route(row_layout,
                                    global_q_seq_lens,
                                    global_kv_seq_lens,
                                    state_table,
                                    state_block_size,
                                    device,
                                    cache_addressing);
  plan.compressed_cache_route_ =
      build_descriptor(compressed_cache_events, cp_size, cp_rank, device);
  plan.signature_ = compute_signature(row_layout,
                                      global_q_seq_lens,
                                      global_kv_seq_lens,
                                      state_table,
                                      compressed_table,
                                      compress_ratio,
                                      state_block_size,
                                      compressed_block_size,
                                      cache_addressing);
  return plan;
}

}  // namespace xllm::layer
