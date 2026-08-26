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
#include <vector>

#include "framework/kv_cache/deepseek_v4_cp_cache_layout.h"
#include "framework/parallel_state/npu_cp_plan.h"

namespace xllm::layer {

// Fixed-capacity source/destination descriptor for one owner exchange. Tensor
// storage is value-only; the descriptor owns no ProcessGroup, stream, or cache.
struct Dsv4CpRouteDescriptor {
  torch::Tensor send_row_indices;
  torch::Tensor send_valid_indices;
  torch::Tensor send_owner_ranks;
  torch::Tensor send_sequence_ids;
  torch::Tensor send_window_ids;
  torch::Tensor send_window_offsets;
  torch::Tensor send_cache_write_flags;
  torch::Tensor send_real_counts;

  torch::Tensor recv_source_ranks;
  torch::Tensor recv_sequence_ids;
  torch::Tensor recv_window_ids;
  torch::Tensor recv_window_offsets;
  torch::Tensor recv_cache_write_flags;
  torch::Tensor recv_real_counts;
  torch::Tensor recv_canonical_indices;
  torch::Tensor recv_canonical_local_slots;

  int64_t padded_rows_per_peer = 0;
  int64_t send_real_row_count = 0;
  int64_t recv_real_row_count = 0;
};

// Core metadata for the compression windows owned by one CP rank. Rows are
// compact and canonical; fixed-capacity transport remains in the route.
struct Dsv4CpOwnerMetadata {
  torch::Tensor recv_canonical_indices;
  torch::Tensor row_sequence_ids;
  torch::Tensor row_window_ids;
  torch::Tensor row_window_offsets;
  torch::Tensor row_absolute_positions;

  torch::Tensor window_sequence_ids;
  torch::Tensor window_ids;
  torch::Tensor q_cu_seq_lens;
  torch::Tensor start_positions;
  torch::Tensor local_state_block_table;
  torch::Tensor output_cache_write_flags;
  torch::Tensor output_cache_write_row_indices;
  torch::Tensor output_rope_indices;

  int64_t row_capacity = 0;
  int64_t real_row_count = 0;
  int64_t segment_count = 0;
  int64_t output_row_count = 0;
  int64_t committed_output_row_count = 0;
};

class Dsv4CpOwnershipPlan final {
 public:
  const Dsv4CpRouteDescriptor& main_route() const;
  const Dsv4CpRouteDescriptor& index_route() const;
  const Dsv4CpRouteDescriptor& swa_route() const;
  const Dsv4CpRouteDescriptor& compressed_cache_route() const;
  const Dsv4CpOwnerMetadata& owner_metadata() const;
  uint64_t signature() const;

 private:
  friend class Dsv4CpOwnershipPlanner;

  Dsv4CpRouteDescriptor main_route_;
  Dsv4CpRouteDescriptor index_route_;
  Dsv4CpRouteDescriptor swa_route_;
  Dsv4CpRouteDescriptor compressed_cache_route_;
  Dsv4CpOwnerMetadata owner_metadata_;
  uint64_t signature_ = 0;
};

class Dsv4CpOwnershipPlanner final {
 public:
  Dsv4CpRouteDescriptor build_swa_route(
      const CpRowLayout& row_layout,
      const std::vector<int32_t>& global_q_seq_lens,
      const std::vector<int32_t>& global_kv_seq_lens,
      const torch::Tensor& swa_global_block_table,
      int64_t swa_block_size,
      const torch::Device& device,
      Dsv4CpCacheAddressing cache_addressing =
          Dsv4CpCacheAddressing::OWNER_LOCAL) const;

  Dsv4CpOwnershipPlan build(
      const CpRowLayout& row_layout,
      const std::vector<int32_t>& global_q_seq_lens,
      const std::vector<int32_t>& global_kv_seq_lens,
      const torch::Tensor& state_global_block_table,
      const torch::Tensor& compressed_global_block_table,
      int64_t compress_ratio,
      int64_t state_block_size,
      int64_t compressed_block_size,
      const torch::Device& device,
      Dsv4CpCacheAddressing cache_addressing =
          Dsv4CpCacheAddressing::OWNER_LOCAL) const;
};

}  // namespace xllm::layer
