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

#include "framework/parallel_state/npu_cp_plan.h"

namespace xllm::layer {

// Query-only metadata for one contiguous half of a rank's zigzag CP rows.
// Cache and compressor metadata remain global and are intentionally not owned
// by this structure.
struct Dsv4CpHalfMetadata {
  torch::Tensor pack_indices;
  torch::Tensor destination_indices;
  torch::Tensor active_sequence_indices;
  torch::Tensor q_seq_lens;
  torch::Tensor q_cu_seq_lens;
  torch::Tensor q_seq_endpoints;
  torch::Tensor kv_seq_lens;
  torch::Tensor c1_sparse_metadata;
  torch::Tensor c4_sparse_metadata;
  torch::Tensor c128_sparse_metadata;
  torch::Tensor qli_metadata;
  std::vector<int32_t> host_q_seq_lens;
  std::vector<int32_t> host_kv_seq_lens;
  int64_t real_row_count = 0;
};

struct Dsv4CpMetadata {
  Dsv4CpHalfMetadata front;
  Dsv4CpHalfMetadata back;
  int64_t local_real_row_count = 0;
  uint64_t layout_signature = 0;
};

struct Dsv4CpAttentionMetadataConfig {
  int64_t num_heads_q = 0;
  int64_t head_dim = 0;
  int64_t window_size = 0;
  int64_t index_num_heads = 0;
  int64_t index_head_dim = 0;
  int64_t index_topk = 0;
};

struct Dsv4CpModelInputBundle {
  torch::Tensor* hidden_states = nullptr;
  torch::Tensor* positions = nullptr;
  torch::Tensor* token_ids = nullptr;
  torch::Tensor* auxiliary_hidden_states = nullptr;
};

class Dsv4CpMetadataBuilder final {
 public:
  static Dsv4CpMetadata build(const CpRowLayout& row_layout,
                              const std::vector<int32_t>& global_q_seq_lens,
                              const std::vector<int32_t>& global_kv_seq_lens,
                              const torch::Device& device);

  static void populate_attention_metadata(
      Dsv4CpMetadata* metadata,
      const Dsv4CpAttentionMetadataConfig& config);
};

class Dsv4CpModelInputPreparer final {
 public:
  static Dsv4CpMetadata prepare(
      const NpuCpPlan& cp_plan,
      const Dsv4CpAttentionMetadataConfig& metadata_config,
      const torch::Device& device,
      const Dsv4CpModelInputBundle& inputs);
};

}  // namespace xllm::layer
