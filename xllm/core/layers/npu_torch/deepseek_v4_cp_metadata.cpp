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

#include "layers/npu_torch/deepseek_v4_cp_metadata.h"

#include <glog/logging.h>

#include <algorithm>
#include <limits>
#include <numeric>
#include <utility>

#include "kernels/ops_api.h"

namespace xllm::layer {
namespace {

struct Dsv4CpHalfHostMetadata {
  std::vector<int64_t> pack_indices;
  std::vector<int64_t> active_sequence_indices;
  std::vector<int32_t> q_seq_lens;
  std::vector<int32_t> kv_seq_lens;
};

torch::Tensor make_int32_tensor(const std::vector<int32_t>& values,
                                const torch::Device& device) {
  return torch::tensor(
             values,
             torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU))
      .to(device, /*non_blocking=*/true);
}

torch::Tensor make_int64_tensor(const std::vector<int64_t>& values,
                                const torch::Device& device) {
  return torch::tensor(
             values,
             torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU))
      .to(device, /*non_blocking=*/true);
}

std::vector<int32_t> make_cumulative_lengths(
    const std::vector<int32_t>& lengths) {
  std::vector<int32_t> cumulative;
  cumulative.reserve(lengths.size() + 1);
  cumulative.push_back(0);
  for (int32_t length : lengths) {
    cumulative.push_back(cumulative.back() + length);
  }
  return cumulative;
}

void append_half(int32_t sequence_index,
                 int32_t prefix_length,
                 int32_t fragment_start,
                 int32_t fragment_end,
                 int64_t local_fragment_offset,
                 Dsv4CpHalfHostMetadata* metadata) {
  CHECK(metadata != nullptr);
  const int32_t fragment_length = fragment_end - fragment_start;
  if (fragment_length <= 0) {
    return;
  }

  metadata->active_sequence_indices.push_back(sequence_index);
  metadata->q_seq_lens.push_back(fragment_length);
  metadata->kv_seq_lens.push_back(prefix_length + fragment_end);
  for (int32_t row = 0; row < fragment_length; ++row) {
    metadata->pack_indices.push_back(local_fragment_offset + row);
  }
}

Dsv4CpHalfMetadata to_device_metadata(Dsv4CpHalfHostMetadata host_metadata,
                                      const torch::Device& device) {
  Dsv4CpHalfMetadata metadata;
  metadata.real_row_count =
      static_cast<int64_t>(host_metadata.pack_indices.size());
  metadata.host_q_seq_lens = std::move(host_metadata.q_seq_lens);
  metadata.host_kv_seq_lens = std::move(host_metadata.kv_seq_lens);
  metadata.pack_indices = make_int64_tensor(host_metadata.pack_indices, device);
  metadata.destination_indices = metadata.pack_indices;
  metadata.active_sequence_indices =
      make_int64_tensor(host_metadata.active_sequence_indices, device);
  metadata.q_seq_lens = make_int32_tensor(metadata.host_q_seq_lens, device);
  const std::vector<int32_t> cumulative_q_seq_lens =
      make_cumulative_lengths(metadata.host_q_seq_lens);
  metadata.q_cu_seq_lens = make_int32_tensor(cumulative_q_seq_lens, device);
  metadata.q_seq_endpoints =
      make_int32_tensor(std::vector<int32_t>(cumulative_q_seq_lens.begin() + 1,
                                             cumulative_q_seq_lens.end()),
                        device);
  metadata.kv_seq_lens = make_int32_tensor(metadata.host_kv_seq_lens, device);
  return metadata;
}

int64_t max_or_zero(const std::vector<int32_t>& values) {
  if (values.empty()) {
    return 0;
  }
  return *std::max_element(values.begin(), values.end());
}

void populate_half_attention_metadata(
    Dsv4CpHalfMetadata* half,
    const Dsv4CpAttentionMetadataConfig& config) {
  CHECK(half != nullptr);
  if (half->real_row_count == 0) {
    return;
  }
  CHECK_GT(config.num_heads_q, 0);
  CHECK_GT(config.head_dim, 0);
  CHECK_GT(config.window_size, 0);
  const int64_t batch_size = static_cast<int64_t>(half->host_q_seq_lens.size());
  CHECK_EQ(batch_size, static_cast<int64_t>(half->host_kv_seq_lens.size()));
  const int64_t max_query_len = max_or_zero(half->host_q_seq_lens);
  const int64_t max_kv_len = max_or_zero(half->host_kv_seq_lens);

  auto build_sparse_metadata =
      [&](int64_t cmp_ratio, int64_t cmp_topk, bool has_cmp_kv) {
        xllm::kernel::SparseAttnSharedkvMetadataParams params;
        params.num_heads_q = config.num_heads_q;
        params.num_heads_kv = 1;
        params.head_dim = config.head_dim;
        params.cu_seqlens_q = half->q_cu_seq_lens;
        params.seqused_q = half->q_seq_lens;
        params.seqused_kv = half->kv_seq_lens;
        params.batch_size = batch_size;
        params.max_seqlen_q = max_query_len;
        params.max_seqlen_kv = max_kv_len;
        params.ori_topk = 0;
        params.cmp_topk = cmp_topk;
        params.cmp_ratio = cmp_ratio;
        params.ori_mask_mode = 4;
        params.cmp_mask_mode = 3;
        params.ori_win_left = config.window_size - 1;
        params.ori_win_right = 0;
        params.layout_q = "TND";
        params.layout_kv = "PA_ND";
        params.has_ori_kv = true;
        params.has_cmp_kv = has_cmp_kv;
        return xllm::kernel::sparse_attn_sharedkv_metadata(params);
      };

  half->c1_sparse_metadata = build_sparse_metadata(
      /*cmp_ratio=*/1, /*cmp_topk=*/0, /*has_cmp_kv=*/false);
  half->c4_sparse_metadata = build_sparse_metadata(
      /*cmp_ratio=*/4, config.index_topk, /*has_cmp_kv=*/true);
  half->c128_sparse_metadata = build_sparse_metadata(
      /*cmp_ratio=*/128, /*cmp_topk=*/0, /*has_cmp_kv=*/true);

  CHECK_GT(config.index_num_heads, 0);
  CHECK_GT(config.index_head_dim, 0);
  CHECK_GT(config.index_topk, 0);
  xllm::kernel::QuantLightningIndexerMetadataParams qli_params;
  qli_params.num_heads_q = config.index_num_heads;
  qli_params.num_heads_k = 1;
  qli_params.head_dim = config.index_head_dim;
  qli_params.query_quant_mode = 0;
  qli_params.key_quant_mode = 0;
  // QuantLightningIndexer uses cumulative query endpoints for TND, without
  // the leading zero required by sparse attention's cu_seqlens_q.
  qli_params.actual_seq_lengths_query = half->q_seq_endpoints;
  qli_params.actual_seq_lengths_key = half->kv_seq_lens;
  qli_params.batch_size = batch_size;
  qli_params.max_seqlen_q = max_query_len;
  qli_params.max_seqlen_k = max_kv_len;
  qli_params.layout_query = "TND";
  qli_params.layout_key = "PA_BSND";
  qli_params.sparse_count = config.index_topk;
  qli_params.sparse_mode = 3;
  qli_params.pre_tokens = std::numeric_limits<int64_t>::max();
  qli_params.next_tokens = std::numeric_limits<int64_t>::max();
  qli_params.cmp_ratio = 4;
  qli_params.device = half->q_seq_lens.device().str();
  half->qli_metadata =
      xllm::kernel::quant_lightning_indexer_metadata(qli_params);
}

}  // namespace

Dsv4CpMetadata Dsv4CpMetadataBuilder::build(
    const CpRowLayout& row_layout,
    const std::vector<int32_t>& global_q_seq_lens,
    const std::vector<int32_t>& global_kv_seq_lens,
    const torch::Device& device) {
  CHECK_GT(row_layout.cp_size(), 1);
  CHECK_EQ(global_q_seq_lens.size(), global_kv_seq_lens.size())
      << "DSV4 CP requires one KV length for every query length";
  CHECK_EQ(global_q_seq_lens.size(),
           row_layout.input_shard_meta().local_padded_seq_lens.size())
      << "DSV4 CP sequence count does not match the row layout";

  const int32_t cp_size = row_layout.cp_size();
  const int32_t cp_rank = row_layout.cp_rank();
  const int32_t chunk_count = 2 * cp_size;
  Dsv4CpHalfHostMetadata front;
  Dsv4CpHalfHostMetadata back;
  const size_t sequence_count = global_q_seq_lens.size();
  front.pack_indices.reserve(row_layout.local_real_token_count());
  back.pack_indices.reserve(row_layout.local_real_token_count());
  front.active_sequence_indices.reserve(sequence_count);
  back.active_sequence_indices.reserve(sequence_count);
  front.q_seq_lens.reserve(sequence_count);
  back.q_seq_lens.reserve(sequence_count);
  front.kv_seq_lens.reserve(sequence_count);
  back.kv_seq_lens.reserve(sequence_count);
  int64_t local_sequence_offset = 0;

  for (size_t sequence_index = 0; sequence_index < global_q_seq_lens.size();
       ++sequence_index) {
    const int32_t q_length = global_q_seq_lens[sequence_index];
    const int32_t kv_length = global_kv_seq_lens[sequence_index];
    CHECK_GE(q_length, 0);
    CHECK_GE(kv_length, q_length)
        << "DSV4 CP requires kv_seq_len >= q_seq_len, sequence="
        << sequence_index;

    const int32_t padded_length =
        ((q_length + chunk_count - 1) / chunk_count) * chunk_count;
    const int32_t chunk_length = padded_length / chunk_count;
    const int32_t expected_local_padded_length = 2 * chunk_length;
    CHECK_EQ(
        row_layout.input_shard_meta().local_padded_seq_lens[sequence_index],
        expected_local_padded_length);

    const int32_t prefix_length = kv_length - q_length;
    const int32_t front_start = cp_rank * chunk_length;
    const int32_t front_end = std::min(front_start + chunk_length, q_length);
    append_half(static_cast<int32_t>(sequence_index),
                prefix_length,
                std::min(front_start, q_length),
                front_end,
                local_sequence_offset,
                &front);

    const int32_t back_chunk = chunk_count - 1 - cp_rank;
    const int32_t back_start = back_chunk * chunk_length;
    const int32_t back_end = std::min(back_start + chunk_length, q_length);
    append_half(static_cast<int32_t>(sequence_index),
                prefix_length,
                std::min(back_start, q_length),
                back_end,
                local_sequence_offset + chunk_length,
                &back);

    local_sequence_offset += expected_local_padded_length;
  }

  Dsv4CpMetadata metadata;
  metadata.front = to_device_metadata(std::move(front), device);
  metadata.back = to_device_metadata(std::move(back), device);
  metadata.local_real_row_count =
      metadata.front.real_row_count + metadata.back.real_row_count;
  metadata.layout_signature = row_layout.signature();

  CHECK_EQ(local_sequence_offset, row_layout.local_padded_token_count());
  CHECK_EQ(metadata.local_real_row_count, row_layout.local_real_token_count())
      << "DSV4 CP halves must cover every local-real row exactly once";
  return metadata;
}

void Dsv4CpMetadataBuilder::populate_attention_metadata(
    Dsv4CpMetadata* metadata,
    const Dsv4CpAttentionMetadataConfig& config) {
  CHECK(metadata != nullptr);
  populate_half_attention_metadata(&metadata->front, config);
  populate_half_attention_metadata(&metadata->back, config);
}

Dsv4CpMetadata Dsv4CpModelInputPreparer::prepare(
    const NpuCpPlan& cp_plan,
    const Dsv4CpAttentionMetadataConfig& metadata_config,
    const torch::Device& device,
    const Dsv4CpModelInputBundle& inputs) {
  CHECK(cp_plan.enabled());
  CHECK(cp_plan.process_group() != nullptr)
      << "DeepSeek V4 CP requires a bound CP process group";
  CHECK(inputs.hidden_states != nullptr);
  CHECK(inputs.positions != nullptr);
  CHECK(inputs.token_ids != nullptr);

  const CpRowLayout& row_layout = cp_plan.row_layout();
  *inputs.hidden_states =
      row_layout.shard_rows(*inputs.hidden_states, /*pad_value=*/0);
  *inputs.positions = row_layout.input_shard_meta().local_position_ids;
  *inputs.token_ids = row_layout.shard_rows(*inputs.token_ids, /*pad_value=*/0);
  if (inputs.auxiliary_hidden_states != nullptr) {
    *inputs.auxiliary_hidden_states =
        row_layout.shard_rows(*inputs.auxiliary_hidden_states, /*pad_value=*/0);
  }

  Dsv4CpMetadata metadata =
      Dsv4CpMetadataBuilder::build(row_layout,
                                   cp_plan.global_q_seq_lens(),
                                   cp_plan.global_kv_seq_lens(),
                                   device);
  Dsv4CpMetadataBuilder::populate_attention_metadata(&metadata,
                                                     metadata_config);
  return metadata;
}

}  // namespace xllm::layer
