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

#include "layers/npu_torch/deepseek_v4_cp_attention_exchange.h"

#include <glog/logging.h>

#include <limits>
#include <utility>
#include <vector>

#include "framework/parallel_state/process_group.h"

namespace xllm::layer {

Dsv4CpAttentionExchange::Dsv4CpAttentionExchange(ProcessGroup* cp_group)
    : cp_group_(cp_group) {
  CHECK(cp_group_ != nullptr)
      << "DSV4 CP owner exchange requires a process group";
}

torch::Tensor Dsv4CpAttentionExchange::replicate_query_rows(
    const torch::Tensor& local_padded_rows) const {
  CHECK(local_padded_rows.defined());
  CHECK_GE(local_padded_rows.dim(), 1);
  CHECK(local_padded_rows.is_contiguous());
  CHECK_EQ(local_padded_rows.device(), cp_group_->device());
  return cp_group_->allgather_base_sync(local_padded_rows);
}

torch::Tensor Dsv4CpAttentionExchange::pack_send_rows(
    const torch::Tensor& local_padded_rows,
    const Dsv4CpRouteDescriptor& route) {
  CHECK(local_padded_rows.defined());
  CHECK_EQ(local_padded_rows.dim(), 2)
      << "DSV4 CP owner exchange accepts two-dimensional rows";
  CHECK(local_padded_rows.is_contiguous());
  CHECK(route.send_row_indices.defined());
  CHECK_EQ(route.send_row_indices.dim(), 1);
  CHECK_EQ(route.send_row_indices.scalar_type(), torch::kInt64);
  CHECK_EQ(route.send_row_indices.device(), local_padded_rows.device());

  torch::Tensor send_rows =
      torch::zeros({route.send_row_indices.numel(), local_padded_rows.size(1)},
                   local_padded_rows.options());
  CHECK(route.send_valid_indices.defined());
  CHECK_EQ(route.send_valid_indices.scalar_type(), torch::kInt64);
  CHECK_EQ(route.send_valid_indices.device(), local_padded_rows.device());
  torch::Tensor valid_slots = route.send_valid_indices;
  CHECK_EQ(valid_slots.numel(), route.send_real_row_count);
  if (valid_slots.numel() == 0) {
    return send_rows;
  }

  torch::Tensor source_rows =
      route.send_row_indices.index_select(/*dim=*/0, valid_slots);
  if (source_rows.device().is_cpu()) {
    CHECK(source_rows.lt(local_padded_rows.size(0)).all().item<bool>())
        << "DSV4 CP route source row exceeds local-padded input";
  }
  torch::Tensor values = local_padded_rows.index_select(/*dim=*/0, source_rows);
  send_rows.index_copy_(/*dim=*/0, valid_slots, values);
  return send_rows;
}

torch::Tensor Dsv4CpAttentionExchange::unpack_received_rows(
    const torch::Tensor& received_rows,
    const Dsv4CpRouteDescriptor& route) {
  CHECK(received_rows.defined());
  CHECK_EQ(received_rows.dim(), 2);
  CHECK(received_rows.is_contiguous());
  CHECK_EQ(received_rows.size(0), route.recv_source_ranks.numel());
  CHECK(route.recv_canonical_indices.defined());
  CHECK_EQ(route.recv_canonical_indices.scalar_type(), torch::kInt64);
  CHECK_EQ(route.recv_canonical_indices.device(), received_rows.device());
  if (route.recv_real_row_count == 0) {
    return received_rows.slice(/*dim=*/0, /*start=*/0, /*end=*/0);
  }

  torch::Tensor canonical_indices = route.recv_canonical_indices.slice(
      /*dim=*/0, /*start=*/0, route.recv_real_row_count);
  if (canonical_indices.device().is_cpu()) {
    CHECK(canonical_indices.ge(0).all().item<bool>());
    CHECK(canonical_indices.lt(received_rows.size(0)).all().item<bool>());
  }
  return received_rows.index_select(/*dim=*/0, canonical_indices).contiguous();
}

torch::Tensor Dsv4CpAttentionExchange::select_owner_output_rows(
    const torch::Tensor& global_rows,
    const Dsv4CpOwnerMetadata& owner_metadata) {
  CHECK(global_rows.defined());
  CHECK_GE(global_rows.dim(), 1);
  CHECK(owner_metadata.output_rope_indices.defined());
  CHECK_EQ(owner_metadata.output_rope_indices.dim(), 1);
  CHECK_EQ(owner_metadata.output_rope_indices.scalar_type(), torch::kInt64);
  CHECK_EQ(owner_metadata.output_rope_indices.device(), global_rows.device());
  CHECK_EQ(owner_metadata.output_rope_indices.numel(),
           owner_metadata.output_row_count);
  if (owner_metadata.output_row_count == 0) {
    return global_rows.slice(/*dim=*/0, /*start=*/0, /*end=*/0).contiguous();
  }
  if (global_rows.device().is_cpu()) {
    CHECK(owner_metadata.output_rope_indices.ge(0).all().item<bool>());
    CHECK(owner_metadata.output_rope_indices.lt(global_rows.size(0))
              .all()
              .item<bool>())
        << "DSV4 owner output row exceeds the global row count";
  }
  return global_rows
      .index_select(/*dim=*/0, owner_metadata.output_rope_indices)
      .contiguous();
}

void Dsv4CpAttentionExchange::write_received_cache_rows(
    torch::Tensor& cache,
    const torch::Tensor& received_rows,
    const Dsv4CpRouteDescriptor& route) {
  CHECK(cache.defined());
  CHECK(received_rows.defined());
  CHECK_GE(cache.dim(), 2);
  CHECK_GE(received_rows.dim(), 2);
  CHECK(cache.is_contiguous());
  CHECK(received_rows.is_contiguous());
  CHECK_EQ(cache.scalar_type(), received_rows.scalar_type());
  CHECK_EQ(cache.device(), received_rows.device());
  CHECK(route.recv_canonical_local_slots.defined());
  CHECK_EQ(route.recv_canonical_local_slots.dim(), 1);
  CHECK_EQ(route.recv_canonical_local_slots.scalar_type(), torch::kInt64);
  CHECK_EQ(route.recv_canonical_local_slots.device(), cache.device());
  CHECK_EQ(route.recv_canonical_local_slots.numel(), route.recv_real_row_count);

  const int64_t row_width = received_rows.size(received_rows.dim() - 1);
  CHECK_GT(row_width, 0);
  CHECK_EQ(cache.size(cache.dim() - 1), row_width);
  torch::Tensor values = received_rows.reshape({-1, row_width});
  CHECK_EQ(values.size(0), route.recv_real_row_count);
  if (values.size(0) == 0) {
    return;
  }

  torch::Tensor cache_rows = cache.view({-1, row_width});
  if (cache.device().is_cpu()) {
    CHECK(route.recv_canonical_local_slots.ge(0).all().item<bool>());
    CHECK(route.recv_canonical_local_slots.lt(cache_rows.size(0))
              .all()
              .item<bool>())
        << "DSV4 owner cache slot exceeds the physical cache capacity";
  }
  cache_rows.index_copy_(/*dim=*/0,
                         route.recv_canonical_local_slots,
                         values);
}

torch::Tensor Dsv4CpAttentionExchange::select_query_rank_rows(
    const torch::Tensor& replicated_rows,
    int32_t query_rank) {
  CHECK(replicated_rows.defined());
  CHECK_GE(replicated_rows.dim(), 1);
  CHECK_GE(query_rank, 0);
  CHECK_LT(query_rank, replicated_rows.size(0));
  return replicated_rows.select(/*dim=*/0, query_rank).contiguous();
}

torch::Tensor Dsv4CpAttentionExchange::build_query_sequence_indices(
    const std::vector<int32_t>& local_padded_seq_lens,
    const torch::Device& device) {
  int64_t padded_row_count = 0;
  for (int32_t sequence_length : local_padded_seq_lens) {
    CHECK_GE(sequence_length, 0);
    padded_row_count += sequence_length;
  }
  std::vector<int64_t> sequence_indices;
  sequence_indices.reserve(static_cast<size_t>(padded_row_count));
  for (int64_t sequence = 0;
       sequence < static_cast<int64_t>(local_padded_seq_lens.size());
       ++sequence) {
    sequence_indices.insert(
        sequence_indices.end(),
        static_cast<size_t>(local_padded_seq_lens[sequence]),
        sequence);
  }
  return torch::tensor(
             sequence_indices,
             torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU))
      .to(device, /*non_blocking=*/true);
}

Dsv4CpOwnerIndexMap Dsv4CpAttentionExchange::build_owner_index_map(
    const torch::Tensor& global_block_table,
    const std::vector<int32_t>& global_seq_lens,
    int64_t block_size,
    int32_t cp_size,
    int32_t cp_rank,
    const torch::Device& device,
    Dsv4CpCacheAddressing cache_addressing) {
  CHECK(global_block_table.defined());
  CHECK(global_block_table.device().is_cpu())
      << "DSV4 owner index map requires host block-table metadata";
  CHECK_EQ(global_block_table.dim(), 2);
  CHECK_EQ(global_block_table.size(0),
           static_cast<int64_t>(global_seq_lens.size()));
  CHECK(global_block_table.scalar_type() == torch::kInt ||
        global_block_table.scalar_type() == torch::kLong);
  CHECK_GT(block_size, 0);
  CHECK_GT(cp_size, 1);
  CHECK_GE(cp_rank, 0);
  CHECK_LT(cp_rank, cp_size);

  torch::Tensor block_table = global_block_table.to(torch::kInt64).contiguous();
  const auto table = block_table.accessor<int64_t, 2>();
  const Dsv4CpCacheLayout cache_layout(cp_size, cp_rank);
  const int64_t sequence_count = block_table.size(0);
  int64_t max_global_seq_len = 0;
  int64_t max_local_key_len = 0;
  int64_t max_local_block_count = 0;
  std::vector<int32_t> local_key_seq_lens;
  std::vector<int32_t> local_block_counts;
  local_key_seq_lens.reserve(global_seq_lens.size());
  local_block_counts.reserve(global_seq_lens.size());
  for (int64_t sequence = 0; sequence < sequence_count; ++sequence) {
    const int32_t seq_len = global_seq_lens[sequence];
    CHECK_GE(seq_len, 0);
    const int64_t block_count =
        (static_cast<int64_t>(seq_len) + block_size - 1) / block_size;
    CHECK_LE(block_count, block_table.size(1))
        << "DSV4 owner block table does not cover sequence " << sequence;
    int64_t local_tokens = 0;
    int64_t local_blocks = 0;
    for (int64_t block = 0; block < block_count; ++block) {
      const int64_t global_block_id = table[sequence][block];
      CHECK_GE(global_block_id, 0)
          << "DSV4 owner block table contains padding in a live block";
      if (global_block_id % cp_size == cp_rank) {
        ++local_blocks;
        const int64_t block_tokens = std::min<int64_t>(
            block_size, static_cast<int64_t>(seq_len) - block * block_size);
        local_tokens += block_tokens;
      }
    }
    CHECK_LE(local_tokens, std::numeric_limits<int32_t>::max());
    CHECK_LE(local_blocks, std::numeric_limits<int32_t>::max());
    local_key_seq_lens.emplace_back(static_cast<int32_t>(local_tokens));
    local_block_counts.emplace_back(static_cast<int32_t>(local_blocks));
    max_global_seq_len =
        std::max(max_global_seq_len, static_cast<int64_t>(seq_len));
    max_local_key_len = std::max(max_local_key_len, local_tokens);
    max_local_block_count = std::max(max_local_block_count, local_blocks);
  }

  std::vector<int32_t> local_block_table_values(
      static_cast<size_t>(sequence_count * max_local_block_count), 0);
  std::vector<int64_t> local_to_global_values(
      static_cast<size_t>(sequence_count * max_local_key_len), -1);
  std::vector<int64_t> global_to_local_values(
      static_cast<size_t>(sequence_count * max_global_seq_len), -1);
  for (int64_t sequence = 0; sequence < sequence_count; ++sequence) {
    const int32_t seq_len = global_seq_lens[sequence];
    const int64_t block_count =
        (static_cast<int64_t>(seq_len) + block_size - 1) / block_size;
    int64_t local_block_index = 0;
    int64_t local_token_index = 0;
    for (int64_t block = 0; block < block_count; ++block) {
      const int64_t global_block_id = table[sequence][block];
      if (global_block_id % cp_size != cp_rank) {
        continue;
      }
      CHECK_LT(local_block_index, max_local_block_count);
      local_block_table_values[sequence * max_local_block_count +
                               local_block_index] = static_cast<int32_t>(
          cache_layout.physical_block_id(global_block_id, cache_addressing));
      ++local_block_index;
      const int64_t block_begin = block * block_size;
      const int64_t block_end =
          std::min<int64_t>(block_begin + block_size, seq_len);
      for (int64_t global_index = block_begin; global_index < block_end;
           ++global_index) {
        CHECK_LT(local_token_index, max_local_key_len);
        local_to_global_values[sequence * max_local_key_len +
                               local_token_index] = global_index;
        global_to_local_values[sequence * max_global_seq_len + global_index] =
            local_token_index;
        ++local_token_index;
      }
    }
    CHECK_EQ(local_token_index, local_key_seq_lens[sequence]);
    CHECK_EQ(local_block_index, local_block_counts[sequence]);
  }

  Dsv4CpOwnerIndexMap index_map;
  const torch::TensorOptions device_int32 =
      torch::TensorOptions().dtype(torch::kInt32).device(device);
  const torch::TensorOptions device_int64 =
      torch::TensorOptions().dtype(torch::kInt64).device(device);
  index_map.local_block_table =
      torch::tensor(local_block_table_values,
                    torch::TensorOptions().dtype(torch::kInt32))
          .view({sequence_count, max_local_block_count})
          .to(device, /*non_blocking=*/true);
  index_map.local_key_seq_lens =
      torch::tensor(local_key_seq_lens, device_int32);
  index_map.local_to_global_indices =
      torch::tensor(local_to_global_values,
                    torch::TensorOptions().dtype(torch::kInt64))
          .view({sequence_count, max_local_key_len})
          .to(device, /*non_blocking=*/true);
  index_map.global_to_local_indices =
      torch::tensor(global_to_local_values,
                    torch::TensorOptions().dtype(torch::kInt64))
          .view({sequence_count, max_global_seq_len})
          .to(device, /*non_blocking=*/true);
  return index_map;
}

Dsv4CpOwnerQliMetadata Dsv4CpAttentionExchange::build_owner_qli_metadata(
    const torch::Tensor& global_block_table,
    const std::vector<int32_t>& global_q_seq_lens,
    const std::vector<int32_t>& global_kv_seq_lens,
    int64_t compressed_block_size,
    int64_t compress_ratio,
    int32_t cp_size,
    int32_t cp_rank,
    const torch::Device& device,
    Dsv4CpCacheAddressing cache_addressing) {
  CHECK_EQ(global_q_seq_lens.size(), global_kv_seq_lens.size());
  CHECK_GT(compressed_block_size, 0);
  CHECK_GT(compress_ratio, 0);
  CHECK(global_block_table.defined());
  CHECK(global_block_table.device().is_cpu())
      << "DSV4 owner QLI requires host block-table metadata";
  CHECK_EQ(global_block_table.dim(), 2);
  CHECK_EQ(global_block_table.size(0),
           static_cast<int64_t>(global_q_seq_lens.size()));
  CHECK(global_block_table.scalar_type() == torch::kInt ||
        global_block_table.scalar_type() == torch::kLong);

  std::vector<int32_t> completed_compressed_seq_lens;
  completed_compressed_seq_lens.reserve(global_kv_seq_lens.size());
  for (int32_t kv_seq_len : global_kv_seq_lens) {
    CHECK_GE(kv_seq_len, 0);
    completed_compressed_seq_lens.emplace_back(
        static_cast<int32_t>(kv_seq_len / compress_ratio));
  }

  Dsv4CpOwnerQliMetadata metadata;
  metadata.index_map = build_owner_index_map(global_block_table,
                                             completed_compressed_seq_lens,
                                             compressed_block_size,
                                             cp_size,
                                             cp_rank,
                                             device,
                                             cache_addressing);

  const torch::Tensor block_table =
      global_block_table.to(torch::kInt64).contiguous();
  const auto table = block_table.accessor<int64_t, 2>();
  std::vector<int64_t> query_sequence_indices;
  std::vector<int32_t> local_key_seq_lens;
  for (int64_t sequence = 0;
       sequence < static_cast<int64_t>(global_q_seq_lens.size());
       ++sequence) {
    const int32_t q_seq_len = global_q_seq_lens[sequence];
    const int32_t kv_seq_len = global_kv_seq_lens[sequence];
    CHECK_GE(q_seq_len, 0);
    CHECK_GE(kv_seq_len, q_seq_len);
    const int32_t completed_compressed_len =
        completed_compressed_seq_lens[sequence];
    std::vector<int32_t> owner_prefix_counts(
        static_cast<size_t>(completed_compressed_len + 1), 0);
    for (int32_t compressed_position = 0;
         compressed_position < completed_compressed_len;
         ++compressed_position) {
      const int64_t block_index =
          compressed_position / compressed_block_size;
      CHECK_LT(block_index, block_table.size(1));
      const int64_t global_block_id = table[sequence][block_index];
      CHECK_GE(global_block_id, 0)
          << "DSV4 owner QLI block table contains an invalid live block";
      owner_prefix_counts[static_cast<size_t>(compressed_position + 1)] =
          owner_prefix_counts[static_cast<size_t>(compressed_position)] +
          (global_block_id % cp_size == cp_rank ? 1 : 0);
    }

    const int32_t start_position = kv_seq_len - q_seq_len;
    for (int32_t query_offset = 0; query_offset < q_seq_len; ++query_offset) {
      const int64_t absolute_position = start_position + query_offset;
      const int64_t visible_compressed =
          (absolute_position + 1) / compress_ratio;
      CHECK_GE(visible_compressed, 0);
      CHECK_LE(visible_compressed, completed_compressed_len);
      const int32_t owner_key_count =
          owner_prefix_counts[static_cast<size_t>(visible_compressed)];
      CHECK_LE(static_cast<int64_t>(owner_key_count) * compress_ratio,
               std::numeric_limits<int32_t>::max());
      query_sequence_indices.emplace_back(sequence);
      local_key_seq_lens.emplace_back(
          static_cast<int32_t>(owner_key_count * compress_ratio));
      if (owner_key_count > 0) {
        ++metadata.valid_query_row_count;
      }
    }
  }

  const int64_t query_count =
      static_cast<int64_t>(query_sequence_indices.size());
  metadata.query_sequence_indices =
      torch::tensor(query_sequence_indices,
                    torch::TensorOptions().dtype(torch::kInt64))
          .to(device, /*non_blocking=*/true);
  metadata.local_key_seq_lens =
      torch::tensor(local_key_seq_lens,
                    torch::TensorOptions().dtype(torch::kInt32))
          .to(device, /*non_blocking=*/true);
  if (!local_key_seq_lens.empty()) {
    metadata.max_local_key_seq_len =
        *std::max_element(local_key_seq_lens.begin(), local_key_seq_lens.end());
  }
  metadata.query_seq_endpoints =
      torch::arange(1,
                    query_count + 1,
                    torch::TensorOptions().dtype(torch::kInt32).device(device));
  metadata.valid_query_rows = metadata.local_key_seq_lens.gt(0);
  if (query_count == 0) {
    metadata.query_block_table = metadata.index_map.local_block_table.slice(
        /*dim=*/0, /*start=*/0, /*end=*/0);
  } else {
    metadata.query_block_table = metadata.index_map.local_block_table
                                     .index_select(
                                         /*dim=*/0,
                                         metadata.query_sequence_indices)
                                     .contiguous();
  }
  return metadata;
}

torch::Tensor map_indices_with_owner_table(
    const torch::Tensor& indices,
    const torch::Tensor& query_sequence_indices,
    const torch::Tensor& owner_table) {
  CHECK(indices.defined());
  CHECK(query_sequence_indices.defined());
  CHECK(owner_table.defined());
  CHECK_GE(indices.dim(), 2);
  CHECK_EQ(query_sequence_indices.dim(), 1);
  CHECK_EQ(indices.size(0), query_sequence_indices.size(0));
  CHECK_EQ(query_sequence_indices.device(), indices.device());
  CHECK_EQ(owner_table.device(), indices.device());
  CHECK_EQ(owner_table.dim(), 2);
  CHECK(query_sequence_indices.scalar_type() == torch::kInt32 ||
        query_sequence_indices.scalar_type() == torch::kInt64);
  CHECK(indices.scalar_type() == torch::kInt32 ||
        indices.scalar_type() == torch::kInt64);

  const int64_t table_width = owner_table.size(1);
  torch::Tensor result = torch::full_like(indices, -1);
  if (indices.numel() == 0 || table_width == 0) {
    return result;
  }
  torch::Tensor sequence_indices = query_sequence_indices.to(torch::kLong);
  if (sequence_indices.device().is_cpu()) {
    CHECK(sequence_indices.ge(0).all().item<bool>());
    CHECK(sequence_indices.lt(owner_table.size(0)).all().item<bool>());
  }
  torch::Tensor selected = owner_table.index_select(0, sequence_indices);
  std::vector<int64_t> expanded_shape;
  expanded_shape.reserve(indices.dim());
  expanded_shape.push_back(indices.size(0));
  for (int64_t dim = 1; dim < indices.dim() - 1; ++dim) {
    expanded_shape.push_back(indices.size(dim));
  }
  expanded_shape.push_back(table_width);
  while (selected.dim() < indices.dim()) {
    selected = selected.unsqueeze(1);
  }
  selected = selected.expand(expanded_shape);
  torch::Tensor safe_indices =
      indices.to(torch::kLong).clamp(0, table_width - 1);
  torch::Tensor mapped = selected.gather(/*dim=*/-1, safe_indices);
  return torch::where(indices.ge(0), mapped, result).to(indices.scalar_type());
}

torch::Tensor Dsv4CpAttentionExchange::map_owner_local_indices_to_global(
    const torch::Tensor& local_indices,
    const torch::Tensor& query_sequence_indices,
    const Dsv4CpOwnerIndexMap& index_map) {
  return map_indices_with_owner_table(
      local_indices, query_sequence_indices, index_map.local_to_global_indices);
}

torch::Tensor Dsv4CpAttentionExchange::map_global_indices_to_owner_local(
    const torch::Tensor& global_indices,
    const torch::Tensor& query_sequence_indices,
    const Dsv4CpOwnerIndexMap& index_map) {
  return map_indices_with_owner_table(global_indices,
                                      query_sequence_indices,
                                      index_map.global_to_local_indices);
}

torch::Tensor Dsv4CpAttentionExchange::compact_valid_indices(
    const torch::Tensor& owner_local_indices) {
  CHECK(owner_local_indices.defined());
  CHECK_GE(owner_local_indices.dim(), 1);
  CHECK(owner_local_indices.scalar_type() == torch::kInt32 ||
        owner_local_indices.scalar_type() == torch::kInt64);
  const int64_t width = owner_local_indices.size(-1);
  if (width == 0) {
    return owner_local_indices;
  }

  std::vector<int64_t> positions_shape(
      static_cast<size_t>(owner_local_indices.dim()), 1);
  positions_shape.back() = width;
  const torch::Tensor positions =
      torch::arange(width,
                    torch::TensorOptions()
                        .dtype(torch::kInt64)
                        .device(owner_local_indices.device()))
          .view(positions_shape);
  const torch::Tensor sort_keys = torch::where(
      owner_local_indices.ge(0), positions, positions + width);
  const torch::Tensor permutation =
      std::get<1>(torch::sort(sort_keys, /*dim=*/-1, /*descending=*/false));
  const torch::Tensor compacted =
      owner_local_indices.gather(/*dim=*/-1, permutation);
  return torch::where(compacted.ge(0),
                      compacted,
                      torch::full_like(compacted, -1));
}

Dsv4CpAttentionMergeResult Dsv4CpAttentionExchange::merge_attention_partials(
    const torch::Tensor& local_partial_output,
    const torch::Tensor& local_partial_lse) const {
  CHECK(local_partial_output.defined());
  CHECK(local_partial_lse.defined());
  CHECK_EQ(local_partial_output.device(), cp_group_->device());
  CHECK_EQ(local_partial_lse.device(), cp_group_->device());
  torch::Tensor gathered_lse =
      cp_group_->allgather_base_sync(local_partial_lse);
  torch::Tensor global_lse = torch::logsumexp(gathered_lse, /*dim=*/0);
  torch::Tensor has_finite_partition = torch::isfinite(global_lse);
  torch::Tensor local_partition_is_finite =
      torch::isfinite(local_partial_lse);
  torch::Tensor local_weights =
      torch::where(has_finite_partition.logical_and(local_partition_is_finite),
                   torch::exp(local_partial_lse - global_lse),
                   torch::zeros_like(local_partial_lse));
  torch::Tensor merged_output = torch::where(
      local_partition_is_finite,
      local_partial_output.to(torch::kFloat32),
      torch::zeros_like(local_partial_output, torch::kFloat32));
  merged_output.mul_(local_weights);
  cp_group_->allreduce(merged_output);
  return {merged_output.to(local_partial_output.scalar_type()),
          std::move(global_lse)};
}

Dsv4CpQliMergeResult Dsv4CpAttentionExchange::global_topk(
    const torch::Tensor& local_candidate_indices,
    const torch::Tensor& local_candidate_scores,
    int64_t global_topk) const {
  CHECK(local_candidate_indices.defined());
  CHECK(local_candidate_scores.defined());
  CHECK_EQ(local_candidate_indices.device(), cp_group_->device());
  CHECK_EQ(local_candidate_scores.device(), cp_group_->device());
  torch::Tensor gathered_indices =
      cp_group_->allgather_base_sync(local_candidate_indices);
  torch::Tensor gathered_scores =
      cp_group_->allgather_base_sync(local_candidate_scores);
  return merge_gathered_qli_candidates(
      gathered_indices, gathered_scores, global_topk);
}

Dsv4CpAttentionMergeResult
Dsv4CpAttentionExchange::merge_gathered_attention_partials(
    const torch::Tensor& partial_outputs,
    const torch::Tensor& partial_lse) {
  CHECK(partial_outputs.defined());
  CHECK(partial_lse.defined());
  CHECK_GE(partial_outputs.dim(), 2);
  CHECK_EQ(partial_lse.dim(), partial_outputs.dim());
  CHECK_EQ(partial_outputs.size(0), partial_lse.size(0));
  CHECK_EQ(partial_lse.size(partial_lse.dim() - 1), 1);
  CHECK_EQ(partial_lse.scalar_type(), torch::kFloat32)
      << "DSV4 distributed attention LSE must use FP32";
  CHECK_EQ(partial_outputs.device(), partial_lse.device());
  for (int64_t dim = 1; dim < partial_outputs.dim() - 1; ++dim) {
    CHECK_EQ(partial_outputs.size(dim), partial_lse.size(dim));
  }

  torch::Tensor global_lse = torch::logsumexp(partial_lse, /*dim=*/0);
  torch::Tensor has_finite_partition = torch::isfinite(global_lse);
  torch::Tensor merged_output =
      torch::zeros(partial_outputs.sizes().slice(/*start=*/1),
                   partial_outputs.options().dtype(torch::kFloat32));
  for (int64_t partition = 0; partition < partial_outputs.size(0);
       ++partition) {
    const torch::Tensor partition_lse = partial_lse.select(0, partition);
    const torch::Tensor partition_is_finite = torch::isfinite(partition_lse);
    torch::Tensor partition_weights =
        torch::where(has_finite_partition.logical_and(partition_is_finite),
                     torch::exp(partition_lse - global_lse),
                     torch::zeros_like(partition_lse));
    torch::Tensor weighted_output = torch::where(
        partition_is_finite,
        partial_outputs.select(0, partition).to(torch::kFloat32),
        torch::zeros_like(partial_outputs.select(0, partition),
                          torch::kFloat32));
    weighted_output.mul_(partition_weights);
    merged_output.add_(weighted_output);
  }
  return {merged_output.to(partial_outputs.scalar_type()),
          std::move(global_lse)};
}

Dsv4CpQliMergeResult Dsv4CpAttentionExchange::merge_gathered_qli_candidates(
    const torch::Tensor& candidate_indices,
    const torch::Tensor& candidate_scores,
    int64_t global_topk) {
  CHECK(candidate_indices.defined());
  CHECK(candidate_scores.defined());
  CHECK_EQ(candidate_indices.sizes(), candidate_scores.sizes());
  CHECK_GE(candidate_indices.dim(), 2);
  CHECK(candidate_indices.scalar_type() == torch::kInt32 ||
        candidate_indices.scalar_type() == torch::kInt64);
  CHECK(candidate_scores.is_floating_point());
  CHECK_EQ(candidate_indices.device(), candidate_scores.device());
  CHECK_GT(global_topk, 0);

  std::vector<int64_t> flattened_shape = candidate_indices.sizes().vec();
  const int64_t candidate_count =
      flattened_shape.front() * flattened_shape.back();
  CHECK_LE(global_topk, candidate_count);
  flattened_shape.erase(flattened_shape.begin());
  flattened_shape.back() = candidate_count;
  torch::Tensor indices = candidate_indices.movedim(/*source=*/0, /*dest=*/-2)
                              .contiguous()
                              .view(flattened_shape);
  torch::Tensor scores = candidate_scores.movedim(/*source=*/0, /*dest=*/-2)
                             .contiguous()
                             .view(flattened_shape);
  scores = torch::where(
      indices.ge(0),
      scores,
      torch::full_like(scores, -std::numeric_limits<float>::infinity()));

  torch::Tensor id_order = torch::argsort(
      indices, /*stable=*/true, /*dim=*/-1, /*descending=*/false);
  indices = indices.gather(/*dim=*/-1, id_order);
  scores = scores.gather(/*dim=*/-1, id_order);
  torch::Tensor score_order =
      torch::argsort(scores, /*stable=*/true, /*dim=*/-1, /*descending=*/true);
  torch::Tensor top_order =
      score_order.slice(/*dim=*/-1, /*start=*/0, /*end=*/global_topk);
  return {indices.gather(/*dim=*/-1, top_order).contiguous(),
          scores.gather(/*dim=*/-1, top_order).contiguous()};
}

torch::Tensor Dsv4CpAttentionExchange::route_owner_rows(
    const torch::Tensor& local_padded_rows,
    const Dsv4CpRouteDescriptor& route) const {
  CHECK_EQ(cp_group_->world_size(), route.send_real_counts.numel());
  CHECK_EQ(cp_group_->world_size(), route.recv_real_counts.numel());
  torch::Tensor send_rows = pack_send_rows(local_padded_rows, route);
  torch::Tensor received_rows = torch::empty_like(send_rows);
  if (send_rows.size(0) > 0) {
    cp_group_->all_to_all_single(received_rows, send_rows);
  }
  return unpack_received_rows(received_rows, route);
}

}  // namespace xllm::layer
