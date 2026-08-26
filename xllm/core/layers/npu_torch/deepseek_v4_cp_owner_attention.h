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

#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <optional>
#include <vector>

#include "layers/npu_torch/deepseek_v4_cp_attention_exchange.h"

namespace xllm::layer {

enum class Dsv4CpBlockTableHolePolicy : uint8_t {
  REJECT = 0,
  SKIP_EVICTED = 1,
};

struct Dsv4CpOwnerPaCache {
  torch::Tensor cache;
  torch::Tensor block_table;
  torch::Tensor local_seq_lens;
  std::vector<std::vector<int32_t>> global_to_local;
};

// Compacts the blocks owned by one CP rank into a transient PA cache. Sliding
// window tables retain -1 placeholders for evicted history, while full-history
// compressed tables must be complete.
Dsv4CpOwnerPaCache build_dsv4_cp_owner_pa_cache(
    const torch::Tensor& global_cache,
    const torch::Tensor& global_block_table,
    const std::vector<int32_t>& global_seq_lens,
    int32_t cp_size,
    int32_t cp_rank,
    Dsv4CpBlockTableHolePolicy hole_policy);

// Logical rows owned by one CP rank. values is the shared K/V tensor. The
// remaining tensors describe the global position of every compact owner row.
struct Dsv4CpOwnerKeyShard {
  torch::Tensor values;
  torch::Tensor sequence_indices;
  torch::Tensor logical_indices;
  torch::Tensor absolute_positions;
  torch::Tensor windowed;
  torch::Tensor sparse_selected;
};

struct Dsv4CpOwnerAttentionInput {
  torch::Tensor query;
  torch::Tensor query_sequence_indices;
  torch::Tensor query_absolute_positions;
  Dsv4CpOwnerKeyShard keys;

  // C4 QLI candidates use global compressed logical indices. The tensor may
  // be [T, K] or [T, H, K]; -1 is padding.
  std::optional<torch::Tensor> sparse_global_indices;

  // Attention sinks are logits with zero-valued V. Exactly one owner must set
  // include_sinks=true before CP partials are merged.
  std::optional<torch::Tensor> sinks;
  bool include_sinks = false;
  int64_t window_left = -1;
  double softmax_scale = 1.0;
};

// Correctness implementation for the owner-local sparse-attention contract.
// It is intentionally position-aware and independent from PA block-table mask
// semantics, making it suitable as the C1/C4/C128 oracle for the production
// owner-aware NPU operator.
class Dsv4CpOwnerAttentionReference final {
 public:
  static Dsv4CpOwnerKeyShard build_paged_key_shard(
      const torch::Tensor& cache,
      const Dsv4CpOwnerIndexMap& index_map,
      int64_t block_size,
      int64_t absolute_position_stride,
      int64_t absolute_position_offset,
      bool windowed,
      bool sparse_selected);

  static Dsv4CpOwnerKeyShard concatenate_key_shards(
      const std::vector<Dsv4CpOwnerKeyShard>& shards);

  static Dsv4CpAttentionMergeResult run(
      const Dsv4CpOwnerAttentionInput& input);
};

}  // namespace xllm::layer
