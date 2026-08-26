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

#include "layers/npu_torch/deepseek_v4_cp_owner_attention.h"

#include <glog/logging.h>

#include <limits>
#include <unordered_set>
#include <utility>

namespace xllm::layer {
namespace {

torch::Tensor to_cpu_int64(const torch::Tensor& tensor) {
  CHECK(tensor.defined());
  CHECK(tensor.scalar_type() == torch::kInt32 ||
        tensor.scalar_type() == torch::kInt64);
  return tensor.to(torch::kCPU).to(torch::kInt64).contiguous();
}

void validate_key_shard(const Dsv4CpOwnerKeyShard& shard) {
  CHECK(shard.values.defined());
  CHECK_EQ(shard.values.dim(), 3) << "owner key values must be [K,H,D]";
  const int64_t key_count = shard.values.size(0);
  for (const torch::Tensor* metadata : {&shard.sequence_indices,
                                        &shard.logical_indices,
                                        &shard.absolute_positions}) {
    CHECK(metadata->defined());
    CHECK_EQ(metadata->dim(), 1);
    CHECK_EQ(metadata->size(0), key_count);
    CHECK(metadata->scalar_type() == torch::kInt32 ||
          metadata->scalar_type() == torch::kInt64);
  }
  for (const torch::Tensor* mask : {&shard.windowed,
                                    &shard.sparse_selected}) {
    CHECK(mask->defined());
    CHECK_EQ(mask->dim(), 1);
    CHECK_EQ(mask->size(0), key_count);
    CHECK_EQ(mask->scalar_type(), torch::kBool);
  }
}

std::unordered_set<int64_t> sparse_candidates_for_query(
    const torch::Tensor& candidates,
    int64_t query_row,
    int64_t head) {
  std::unordered_set<int64_t> result;
  if (!candidates.defined()) {
    return result;
  }
  CHECK(candidates.scalar_type() == torch::kInt32 ||
        candidates.scalar_type() == torch::kInt64);
  CHECK(candidates.dim() == 2 || candidates.dim() == 3)
      << "sparse candidates must be [T,K] or [T,H,K]";
  CHECK_LT(query_row, candidates.size(0));
  torch::Tensor row;
  if (candidates.dim() == 2) {
    row = candidates.select(/*dim=*/0, query_row);
  } else {
    CHECK_LT(head, candidates.size(1));
    row = candidates.select(/*dim=*/0, query_row).select(/*dim=*/0, head);
  }
  row = to_cpu_int64(row);
  const int64_t* values = row.data_ptr<int64_t>();
  for (int64_t i = 0; i < row.numel(); ++i) {
    if (values[i] >= 0) {
      result.insert(values[i]);
    }
  }
  return result;
}

}  // namespace

Dsv4CpOwnerPaCache build_dsv4_cp_owner_pa_cache(
    const torch::Tensor& global_cache,
    const torch::Tensor& global_block_table,
    const std::vector<int32_t>& global_seq_lens,
    int32_t cp_size,
    int32_t cp_rank,
    Dsv4CpBlockTableHolePolicy hole_policy) {
  CHECK(global_cache.defined());
  CHECK_EQ(global_cache.dim(), 4);
  CHECK(global_block_table.defined());
  CHECK_EQ(global_block_table.dim(), 2);
  CHECK_EQ(global_block_table.size(0),
           static_cast<int64_t>(global_seq_lens.size()));
  CHECK(global_block_table.scalar_type() == torch::kInt32 ||
        global_block_table.scalar_type() == torch::kInt64)
      << "DSV4 owner cache block table must be int32 or int64";
  CHECK_GT(cp_size, 1);
  CHECK_GE(cp_rank, 0);
  CHECK_LT(cp_rank, cp_size);

  const int64_t block_size = global_cache.size(1);
  const torch::Tensor table_cpu = to_cpu_int64(global_block_table);
  const auto table = table_cpu.accessor<int64_t, 2>();
  const int64_t batch_size = table_cpu.size(0);
  const int64_t global_table_columns = table_cpu.size(1);

  std::vector<int64_t> owner_global_blocks;
  std::vector<std::vector<int32_t>> physical_blocks_by_sequence(
      static_cast<size_t>(batch_size));
  std::vector<int32_t> local_seq_lens;
  std::vector<std::vector<int32_t>> global_to_local(global_seq_lens.size());
  local_seq_lens.reserve(global_seq_lens.size());

  int64_t max_local_block_count = 0;
  for (int64_t sequence = 0; sequence < batch_size; ++sequence) {
    const int32_t seq_len = global_seq_lens[static_cast<size_t>(sequence)];
    CHECK_GE(seq_len, 0);
    auto& position_map = global_to_local[static_cast<size_t>(sequence)];
    position_map.assign(static_cast<size_t>(seq_len), -1);
    const int64_t block_count =
        (static_cast<int64_t>(seq_len) + block_size - 1) / block_size;
    CHECK_LE(block_count, global_table_columns);

    int32_t local_token_count = 0;
    for (int64_t block = 0; block < block_count; ++block) {
      const int64_t global_block = table[sequence][block];
      if (global_block < 0) {
        CHECK(hole_policy == Dsv4CpBlockTableHolePolicy::SKIP_EVICTED)
            << "full-history DSV4 block table entry cannot be negative";
        continue;
      }
      CHECK_LT(global_block, global_cache.size(0))
          << "DSV4 block table entry exceeds the cache block count";
      if (global_block % cp_size != cp_rank) {
        continue;
      }
      CHECK_LE(owner_global_blocks.size(),
               static_cast<size_t>(std::numeric_limits<int32_t>::max()));
      const int32_t physical_block =
          static_cast<int32_t>(owner_global_blocks.size());
      auto& sequence_blocks =
          physical_blocks_by_sequence[static_cast<size_t>(sequence)];
      const int32_t logical_block =
          static_cast<int32_t>(sequence_blocks.size());
      owner_global_blocks.push_back(global_block);
      sequence_blocks.push_back(physical_block);
      const int32_t valid_tokens = static_cast<int32_t>(std::min<int64_t>(
          block_size, static_cast<int64_t>(seq_len) - block * block_size));
      for (int32_t offset = 0; offset < valid_tokens; ++offset) {
        position_map[static_cast<size_t>(block * block_size + offset)] =
            logical_block * static_cast<int32_t>(block_size) + offset;
      }
      local_token_count = std::max(
          local_token_count,
          logical_block * static_cast<int32_t>(block_size) + valid_tokens);
    }
    max_local_block_count = std::max<int64_t>(
        max_local_block_count,
        physical_blocks_by_sequence[static_cast<size_t>(sequence)].size());
    local_seq_lens.emplace_back(local_token_count);
  }

  const int64_t local_table_columns =
      std::max<int64_t>(max_local_block_count, 1);
  std::vector<int32_t> local_table(
      static_cast<size_t>(batch_size * local_table_columns), 0);
  for (int64_t sequence = 0; sequence < batch_size; ++sequence) {
    const auto& sequence_blocks =
        physical_blocks_by_sequence[static_cast<size_t>(sequence)];
    for (int64_t block = 0;
         block < static_cast<int64_t>(sequence_blocks.size());
         ++block) {
      local_table[static_cast<size_t>(sequence * local_table_columns + block)] =
          sequence_blocks[static_cast<size_t>(block)];
    }
  }

  torch::Tensor owner_cache;
  if (owner_global_blocks.empty()) {
    owner_cache = torch::zeros(
        {1, block_size, global_cache.size(2), global_cache.size(3)},
        global_cache.options());
  } else {
    const torch::Tensor block_indices =
        torch::tensor(owner_global_blocks,
                      torch::TensorOptions().dtype(torch::kInt64))
            .to(global_cache.device(), /*non_blocking=*/true);
    owner_cache = global_cache.index_select(/*dim=*/0, block_indices)
                      .contiguous();
  }
  const torch::Tensor owner_table_cpu =
      torch::tensor(local_table, torch::TensorOptions().dtype(torch::kInt32))
          .view({batch_size, local_table_columns});
  return {std::move(owner_cache),
          owner_table_cpu.to(global_cache.device(), /*non_blocking=*/true),
          torch::tensor(local_seq_lens,
                        torch::TensorOptions().dtype(torch::kInt32))
              .to(global_cache.device(), /*non_blocking=*/true),
          std::move(global_to_local)};
}

Dsv4CpOwnerKeyShard Dsv4CpOwnerAttentionReference::build_paged_key_shard(
    const torch::Tensor& cache,
    const Dsv4CpOwnerIndexMap& index_map,
    int64_t block_size,
    int64_t absolute_position_stride,
    int64_t absolute_position_offset,
    bool windowed,
    bool sparse_selected) {
  CHECK(cache.defined());
  CHECK_EQ(cache.dim(), 4) << "owner PA cache must be [B,block,H,D]";
  CHECK_EQ(cache.size(1), block_size);
  CHECK_GT(block_size, 0);
  CHECK_GT(absolute_position_stride, 0);
  CHECK(index_map.local_block_table.defined());
  CHECK(index_map.local_to_global_indices.defined());
  CHECK_EQ(index_map.local_block_table.dim(), 2);
  CHECK_EQ(index_map.local_to_global_indices.dim(), 2);
  CHECK_EQ(index_map.local_block_table.size(0),
           index_map.local_to_global_indices.size(0));

  const torch::Tensor block_table =
      to_cpu_int64(index_map.local_block_table);
  const torch::Tensor local_to_global =
      to_cpu_int64(index_map.local_to_global_indices);
  const auto block_accessor = block_table.accessor<int64_t, 2>();
  const auto global_accessor = local_to_global.accessor<int64_t, 2>();

  std::vector<int64_t> physical_slots;
  std::vector<int64_t> sequence_indices;
  std::vector<int64_t> logical_indices;
  std::vector<int64_t> absolute_positions;
  for (int64_t sequence = 0; sequence < local_to_global.size(0); ++sequence) {
    for (int64_t local_index = 0;
         local_index < local_to_global.size(1);
         ++local_index) {
      const int64_t global_index = global_accessor[sequence][local_index];
      if (global_index < 0) {
        continue;
      }
      const int64_t local_block_column = local_index / block_size;
      CHECK_LT(local_block_column, block_table.size(1));
      const int64_t local_block =
          block_accessor[sequence][local_block_column];
      CHECK_GE(local_block, 0);
      CHECK_LT(local_block, cache.size(0));
      physical_slots.push_back(local_block * block_size +
                               local_index % block_size);
      sequence_indices.push_back(sequence);
      logical_indices.push_back(global_index);
      absolute_positions.push_back(global_index * absolute_position_stride +
                                   absolute_position_offset);
    }
  }

  const torch::TensorOptions device_int64 =
      torch::TensorOptions().dtype(torch::kInt64).device(cache.device());
  const torch::TensorOptions device_bool =
      torch::TensorOptions().dtype(torch::kBool).device(cache.device());
  torch::Tensor slots =
      torch::tensor(physical_slots, torch::TensorOptions().dtype(torch::kInt64))
          .to(cache.device(), /*non_blocking=*/true);
  const torch::Tensor cache_rows =
      cache.view({-1, cache.size(2), cache.size(3)});

  Dsv4CpOwnerKeyShard shard;
  shard.values = cache_rows.index_select(/*dim=*/0, slots).contiguous();
  shard.sequence_indices =
      torch::tensor(sequence_indices, torch::kInt64)
          .to(device_int64.device(), /*non_blocking=*/true);
  shard.logical_indices =
      torch::tensor(logical_indices, torch::kInt64)
          .to(device_int64.device(), /*non_blocking=*/true);
  shard.absolute_positions =
      torch::tensor(absolute_positions, torch::kInt64)
          .to(device_int64.device(), /*non_blocking=*/true);
  shard.windowed = torch::full(
      {static_cast<int64_t>(physical_slots.size())}, windowed, device_bool);
  shard.sparse_selected = torch::full(
      {static_cast<int64_t>(physical_slots.size())}, sparse_selected, device_bool);
  return shard;
}

Dsv4CpOwnerKeyShard Dsv4CpOwnerAttentionReference::concatenate_key_shards(
    const std::vector<Dsv4CpOwnerKeyShard>& shards) {
  CHECK(!shards.empty());
  std::vector<torch::Tensor> values;
  std::vector<torch::Tensor> sequence_indices;
  std::vector<torch::Tensor> logical_indices;
  std::vector<torch::Tensor> absolute_positions;
  std::vector<torch::Tensor> windowed;
  std::vector<torch::Tensor> sparse_selected;
  for (const Dsv4CpOwnerKeyShard& shard : shards) {
    validate_key_shard(shard);
    values.push_back(shard.values);
    sequence_indices.push_back(shard.sequence_indices);
    logical_indices.push_back(shard.logical_indices);
    absolute_positions.push_back(shard.absolute_positions);
    windowed.push_back(shard.windowed);
    sparse_selected.push_back(shard.sparse_selected);
  }

  Dsv4CpOwnerKeyShard result;
  result.values = torch::cat(values, /*dim=*/0).contiguous();
  result.sequence_indices = torch::cat(sequence_indices, /*dim=*/0);
  result.logical_indices = torch::cat(logical_indices, /*dim=*/0);
  result.absolute_positions = torch::cat(absolute_positions, /*dim=*/0);
  result.windowed = torch::cat(windowed, /*dim=*/0);
  result.sparse_selected = torch::cat(sparse_selected, /*dim=*/0);
  return result;
}

Dsv4CpAttentionMergeResult Dsv4CpOwnerAttentionReference::run(
    const Dsv4CpOwnerAttentionInput& input) {
  CHECK(input.query.defined());
  CHECK_EQ(input.query.dim(), 3) << "owner query must be [T,H,D]";
  CHECK(input.query.scalar_type() == torch::kFloat16 ||
        input.query.scalar_type() == torch::kBFloat16 ||
        input.query.scalar_type() == torch::kFloat32);
  CHECK_GT(input.softmax_scale, 0.0);
  validate_key_shard(input.keys);
  CHECK_EQ(input.keys.values.device(), input.query.device());
  CHECK_EQ(input.keys.values.size(2), input.query.size(2));
  CHECK(input.keys.values.size(1) == 1 ||
        input.keys.values.size(1) == input.query.size(1))
      << "owner key heads must be shared or match query heads";

  const int64_t query_count = input.query.size(0);
  const int64_t head_count = input.query.size(1);
  for (const torch::Tensor* metadata : {&input.query_sequence_indices,
                                        &input.query_absolute_positions}) {
    CHECK(metadata->defined());
    CHECK_EQ(metadata->dim(), 1);
    CHECK_EQ(metadata->size(0), query_count);
  }
  if (input.sparse_global_indices.has_value()) {
    CHECK_EQ(input.sparse_global_indices->size(0), query_count);
    if (input.sparse_global_indices->dim() == 3) {
      CHECK_EQ(input.sparse_global_indices->size(1), head_count);
    }
  }
  if (input.include_sinks) {
    CHECK(input.sinks.has_value());
    CHECK(input.sinks->defined());
    CHECK_EQ(input.sinks->numel(), head_count);
  }

  const torch::Tensor query_sequence =
      to_cpu_int64(input.query_sequence_indices);
  const torch::Tensor query_positions =
      to_cpu_int64(input.query_absolute_positions);
  const torch::Tensor key_sequence =
      to_cpu_int64(input.keys.sequence_indices);
  const torch::Tensor key_logical =
      to_cpu_int64(input.keys.logical_indices);
  const torch::Tensor key_positions =
      to_cpu_int64(input.keys.absolute_positions);
  const torch::Tensor key_windowed =
      input.keys.windowed.to(torch::kCPU).contiguous();
  const torch::Tensor key_sparse =
      input.keys.sparse_selected.to(torch::kCPU).contiguous();
  torch::Tensor candidates;
  if (input.sparse_global_indices.has_value()) {
    candidates = input.sparse_global_indices->to(torch::kCPU).contiguous();
  }

  torch::Tensor output = torch::zeros_like(input.query);
  torch::Tensor lse = torch::full(
      {query_count, head_count, 1},
      -std::numeric_limits<float>::infinity(),
      input.query.options().dtype(torch::kFloat32));
  const torch::Tensor query_float = input.query.to(torch::kFloat32);
  const torch::Tensor key_float = input.keys.values.to(torch::kFloat32);
  const int64_t* query_sequence_values = query_sequence.data_ptr<int64_t>();
  const int64_t* query_position_values = query_positions.data_ptr<int64_t>();
  const int64_t* key_sequence_values = key_sequence.data_ptr<int64_t>();
  const int64_t* key_logical_values = key_logical.data_ptr<int64_t>();
  const int64_t* key_position_values = key_positions.data_ptr<int64_t>();
  const bool* key_windowed_values = key_windowed.data_ptr<bool>();
  const bool* key_sparse_values = key_sparse.data_ptr<bool>();

  for (int64_t query_row = 0; query_row < query_count; ++query_row) {
    std::vector<std::unordered_set<int64_t>> candidates_by_head;
    candidates_by_head.reserve(head_count);
    for (int64_t head = 0; head < head_count; ++head) {
      candidates_by_head.emplace_back(
          sparse_candidates_for_query(candidates, query_row, head));
    }

    std::vector<int64_t> valid_indices;
    valid_indices.reserve(input.keys.values.size(0));
    for (int64_t key_row = 0; key_row < input.keys.values.size(0); ++key_row) {
      if (key_sequence_values[key_row] != query_sequence_values[query_row] ||
          key_position_values[key_row] > query_position_values[query_row]) {
        continue;
      }
      if (key_windowed_values[key_row] && input.window_left >= 0 &&
          key_position_values[key_row] <
              query_position_values[query_row] - input.window_left) {
        continue;
      }
      // Head-specific QLI filtering is applied below. Keep the row when at
      // least one head selected it so all head outputs retain one tensor shape.
      if (key_sparse_values[key_row]) {
        bool selected_by_any_head = false;
        for (const auto& head_candidates : candidates_by_head) {
          selected_by_any_head |=
              head_candidates.count(key_logical_values[key_row]) > 0;
        }
        if (!selected_by_any_head) {
          continue;
        }
      }
      valid_indices.push_back(key_row);
    }

    torch::Tensor selected_values;
    if (!valid_indices.empty()) {
      torch::Tensor selected_indices =
          torch::tensor(valid_indices,
                        torch::TensorOptions().dtype(torch::kInt64))
              .to(input.query.device(), /*non_blocking=*/true);
      selected_values = key_float.index_select(/*dim=*/0, selected_indices);
      if (selected_values.size(1) == 1 && head_count > 1) {
        selected_values = selected_values.expand(
            {selected_values.size(0), head_count, selected_values.size(2)});
      }
    } else {
      selected_values = torch::empty(
          {0, head_count, input.query.size(2)}, key_float.options());
    }

    torch::Tensor scores =
        (selected_values.permute({1, 0, 2}) *
         query_float.select(/*dim=*/0, query_row).unsqueeze(/*dim=*/1))
            .sum(/*dim=*/-1) *
        input.softmax_scale;
    if (!valid_indices.empty()) {
      std::vector<float> sparse_mask_values(
          static_cast<size_t>(head_count * valid_indices.size()), 0.0f);
      for (int64_t head = 0; head < head_count; ++head) {
        for (int64_t selected_row = 0;
             selected_row < static_cast<int64_t>(valid_indices.size());
             ++selected_row) {
          const int64_t key_row = valid_indices[selected_row];
          if (key_sparse_values[key_row] &&
              candidates_by_head[head].count(key_logical_values[key_row]) == 0) {
            sparse_mask_values[head * valid_indices.size() + selected_row] =
                -std::numeric_limits<float>::infinity();
          }
        }
      }
      scores = scores +
               torch::tensor(sparse_mask_values, torch::kFloat32)
                   .view({head_count,
                          static_cast<int64_t>(valid_indices.size())})
                   .to(input.query.device(), /*non_blocking=*/true);
    }

    torch::Tensor reduction_scores = scores;
    if (input.include_sinks) {
      reduction_scores = torch::cat(
          {scores,
           input.sinks->to(input.query.device())
               .to(torch::kFloat32)
               .view({head_count, 1})},
          /*dim=*/1);
    }
    if (reduction_scores.size(1) == 0) {
      continue;
    }

    torch::Tensor row_lse =
        torch::logsumexp(reduction_scores, /*dim=*/1, /*keepdim=*/true);
    lse.select(/*dim=*/0, query_row).copy_(row_lse);
    if (scores.size(1) == 0) {
      continue;
    }
    const torch::Tensor finite_lse = torch::isfinite(row_lse);
    torch::Tensor weights = torch::where(
        finite_lse, torch::exp(scores - row_lse), torch::zeros_like(scores));
    torch::Tensor row_output =
        (weights.unsqueeze(/*dim=*/-1) * selected_values.permute({1, 0, 2}))
            .sum(/*dim=*/1);
    output.select(/*dim=*/0, query_row)
        .copy_(row_output.to(input.query.scalar_type()));
  }

  return {std::move(output), std::move(lse)};
}

}  // namespace xllm::layer
