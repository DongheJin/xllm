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

#include "layers/npu_torch/deepseek_v4_cp_ownership.h"

namespace xllm {

class ProcessGroup;

namespace layer {

struct Dsv4CpAttentionMergeResult {
  torch::Tensor output;
  torch::Tensor lse;
};

struct Dsv4CpQliMergeResult {
  torch::Tensor indices;
  torch::Tensor scores;
};

// Per-owner logical index translation for a paged cache. The block table is
// compacted in logical token order so a consumer never has to dereference a
// non-owner physical block or a -1 padding entry.
struct Dsv4CpOwnerIndexMap {
  torch::Tensor local_block_table;
  torch::Tensor local_key_seq_lens;
  torch::Tensor local_to_global_indices;
  torch::Tensor global_to_local_indices;
};

// QLI sees each global query as a one-token virtual sequence. This preserves
// its causal boundary after non-owner compressed blocks are removed from the
// logical key sequence.
struct Dsv4CpOwnerQliMetadata {
  Dsv4CpOwnerIndexMap index_map;
  torch::Tensor query_block_table;
  torch::Tensor query_seq_endpoints;
  torch::Tensor local_key_seq_lens;
  torch::Tensor query_sequence_indices;
  torch::Tensor valid_query_rows;
  int64_t valid_query_row_count = 0;
  int64_t max_local_key_seq_len = 0;
};

// Stateless data-plane adapter for DSV4 owner routes. The process group is
// non-owning and all route capacity/order decisions remain in the plan.
class Dsv4CpAttentionExchange final {
 public:
  explicit Dsv4CpAttentionExchange(ProcessGroup* cp_group);

  torch::Tensor route_owner_rows(const torch::Tensor& local_padded_rows,
                                 const Dsv4CpRouteDescriptor& route) const;

  // Replicate local query rows to every cache owner. The leading dimension of
  // the result is the source CP rank and all remaining dimensions match the
  // input rows.
  torch::Tensor replicate_query_rows(
      const torch::Tensor& local_padded_rows) const;

  static torch::Tensor pack_send_rows(const torch::Tensor& local_padded_rows,
                                      const Dsv4CpRouteDescriptor& route);

  static torch::Tensor unpack_received_rows(const torch::Tensor& received_rows,
                                            const Dsv4CpRouteDescriptor& route);

  static torch::Tensor select_owner_output_rows(
      const torch::Tensor& global_rows,
      const Dsv4CpOwnerMetadata& owner_metadata);

  static void write_received_cache_rows(
      torch::Tensor& cache,
      const torch::Tensor& received_rows,
      const Dsv4CpRouteDescriptor& route);

  static torch::Tensor select_query_rank_rows(
      const torch::Tensor& replicated_rows,
      int32_t query_rank);

  static torch::Tensor build_query_sequence_indices(
      const std::vector<int32_t>& local_padded_seq_lens,
      const torch::Device& device);

  static Dsv4CpOwnerIndexMap build_owner_index_map(
      const torch::Tensor& global_block_table,
      const std::vector<int32_t>& global_seq_lens,
      int64_t block_size,
      int32_t cp_size,
      int32_t cp_rank,
      const torch::Device& device,
      Dsv4CpCacheAddressing cache_addressing =
          Dsv4CpCacheAddressing::OWNER_LOCAL);

  static Dsv4CpOwnerQliMetadata build_owner_qli_metadata(
      const torch::Tensor& global_block_table,
      const std::vector<int32_t>& global_q_seq_lens,
      const std::vector<int32_t>& global_kv_seq_lens,
      int64_t compressed_block_size,
      int64_t compress_ratio,
      int32_t cp_size,
      int32_t cp_rank,
      const torch::Device& device,
      Dsv4CpCacheAddressing cache_addressing =
          Dsv4CpCacheAddressing::OWNER_LOCAL);

  static torch::Tensor map_owner_local_indices_to_global(
      const torch::Tensor& local_indices,
      const torch::Tensor& query_sequence_indices,
      const Dsv4CpOwnerIndexMap& index_map);

  static torch::Tensor map_global_indices_to_owner_local(
      const torch::Tensor& global_indices,
      const torch::Tensor& query_sequence_indices,
      const Dsv4CpOwnerIndexMap& index_map);

  // Owner sparse kernels stop at the first negative sentinel. Compact valid
  // entries along the last dimension while preserving their original order.
  static torch::Tensor compact_valid_indices(
      const torch::Tensor& owner_local_indices);

  Dsv4CpAttentionMergeResult merge_attention_partials(
      const torch::Tensor& local_partial_output,
      const torch::Tensor& local_partial_lse) const;

  Dsv4CpQliMergeResult global_topk(const torch::Tensor& local_candidate_indices,
                                   const torch::Tensor& local_candidate_scores,
                                   int64_t global_topk) const;

  static Dsv4CpAttentionMergeResult merge_gathered_attention_partials(
      const torch::Tensor& partial_outputs,
      const torch::Tensor& partial_lse);

  static Dsv4CpQliMergeResult merge_gathered_qli_candidates(
      const torch::Tensor& candidate_indices,
      const torch::Tensor& candidate_scores,
      int64_t global_topk);

 private:
  ProcessGroup* cp_group_ = nullptr;
};

}  // namespace layer
}  // namespace xllm
