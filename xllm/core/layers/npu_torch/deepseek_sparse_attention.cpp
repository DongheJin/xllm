/* Copyright 2025-2026 The xLLM Authors.

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
#include "deepseek_sparse_attention.h"

#include <glog/logging.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "common/flash_comm1_context.h"
#include "framework/parallel_state/npu_cp_plan.h"
#include "kernels/ops_api.h"
#include "layers/npu_torch/deepseek_v4_cp_attention_exchange.h"
#include "layers/npu_torch/deepseek_v4_cp_execution.h"
#include "layers/npu_torch/deepseek_v4_cp_owner_attention.h"
#include "xllm/core/kernels/npu/xllm_ops/xllm_ops_api.h"

DECLARE_bool(enable_chunked_prefill);
namespace xllm {
namespace layer {
namespace {

constexpr const char* kCpTensorDebugPrefix = "[DEBUG-DSV4-CP-TENSOR]";

bool cp_tensor_debug_enabled(int32_t layer_id) {
  const char* enabled = std::getenv("XLLM_DSV4_CP_TENSOR_DEBUG");
  if (enabled == nullptr || std::string(enabled) != "1") {
    return false;
  }
  const char* layer_filter =
      std::getenv("XLLM_DSV4_CP_TENSOR_DEBUG_LAYER");
  if (layer_filter == nullptr || *layer_filter == '\0') {
    return true;
  }
  char* end = nullptr;
  const long requested_layer = std::strtol(layer_filter, &end, 10);
  return end != layer_filter && *end == '\0' && requested_layer == layer_id;
}

void log_cp_tensor_summary(int32_t layer_id,
                           int32_t cp_rank,
                           const char* stage,
                           const char* name,
                           const torch::Tensor& tensor) {
  if (!tensor.defined()) {
    LOG(INFO) << kCpTensorDebugPrefix << " layer=" << layer_id
              << " cp_rank=" << cp_rank << " stage=" << stage
              << " name=" << name << " defined=0";
    return;
  }

  int64_t nan_count = 0;
  int64_t pos_inf_count = 0;
  int64_t neg_inf_count = 0;
  int64_t finite_count = 0;
  double finite_min = 0.0;
  double finite_max = 0.0;
  double finite_mean = 0.0;
  int64_t nonnegative_count = 0;
  int64_t integer_min = 0;
  int64_t integer_max = 0;

  if (tensor.numel() > 0 && tensor.is_floating_point()) {
    const torch::Tensor nan_mask = torch::isnan(tensor);
    const torch::Tensor pos_inf_mask = torch::isposinf(tensor);
    const torch::Tensor neg_inf_mask = torch::isneginf(tensor);
    const torch::Tensor finite_mask = torch::isfinite(tensor);
    nan_count = nan_mask.sum().item<int64_t>();
    pos_inf_count = pos_inf_mask.sum().item<int64_t>();
    neg_inf_count = neg_inf_mask.sum().item<int64_t>();
    finite_count = finite_mask.sum().item<int64_t>();
    if (finite_count > 0) {
      const torch::Tensor finite_values =
          tensor.masked_select(finite_mask).to(torch::kFloat32);
      finite_min = finite_values.min().item<double>();
      finite_max = finite_values.max().item<double>();
      finite_mean = finite_values.mean().item<double>();
    }
  } else if (tensor.numel() > 0 && tensor.scalar_type() != torch::kBool) {
    const torch::Tensor values = tensor.to(torch::kInt64);
    nonnegative_count = values.ge(0).sum().item<int64_t>();
    integer_min = values.min().item<int64_t>();
    integer_max = values.max().item<int64_t>();
  }

  LOG(INFO) << kCpTensorDebugPrefix << " layer=" << layer_id
            << " cp_rank=" << cp_rank << " stage=" << stage
            << " name=" << name << " shape=" << tensor.sizes()
            << " dtype=" << tensor.scalar_type()
            << " contiguous=" << tensor.is_contiguous()
            << " numel=" << tensor.numel() << " nan=" << nan_count
            << " pos_inf=" << pos_inf_count
            << " neg_inf=" << neg_inf_count << " finite=" << finite_count
            << " finite_min=" << finite_min << " finite_max=" << finite_max
            << " finite_mean=" << finite_mean
            << " nonnegative=" << nonnegative_count
            << " integer_min=" << integer_min
            << " integer_max=" << integer_max;
}

struct Dsv4PreprocessOutputs {
  torch::Tensor qr;
  std::optional<torch::Tensor> qr_pertoken_scale;
  torch::Tensor q;
  torch::Tensor kv;
};

torch::Tensor w8a8_dynamic_linear_forward(
    const torch::Tensor& quantized_input,
    const torch::Tensor& weight,
    const torch::Tensor& weight_scale,
    const torch::Tensor& pertoken_scale,
    const std::optional<torch::Tensor>& bias,
    at::ScalarType output_dtype) {
  xllm::kernel::QuantMatmulParams params;
  params.x1 = quantized_input;
  params.x2 = weight;
  params.transpose2 = true;
  params.scale = weight_scale;
  params.pertoken_scale = pertoken_scale;
  params.bias = bias;
  params.output_dtype = output_dtype;
  return xllm::kernel::quant_matmul(params);
}

struct DsaCacheMapping {
  int64_t cmp_cache_idx = -1;
  int64_t index_cache_idx = -1;
  int64_t indexer_scale_cache_idx = -1;
  int64_t ori_cache_idx = -1;
  int64_t kv_state_cache_idx = -1;
  int64_t score_state_cache_idx = -1;
  int64_t index_kv_state_cache_idx = -1;
  int64_t index_score_state_cache_idx = -1;
};

std::optional<torch::Tensor> as_optional(const torch::Tensor& tensor) {
  if (tensor.defined() && tensor.numel() > 0) {
    return std::optional<torch::Tensor>(tensor);
  }
  return std::nullopt;
}

torch::Tensor get_layer_cache_tensor(
    const std::vector<std::vector<torch::Tensor>>& layer_tensors,
    int32_t layer_id,
    int64_t cache_idx) {
  if (layer_id < 0 || layer_id >= static_cast<int32_t>(layer_tensors.size()) ||
      cache_idx < 0 ||
      cache_idx >= static_cast<int64_t>(layer_tensors[layer_id].size())) {
    return torch::Tensor();
  }
  return layer_tensors[layer_id][cache_idx];
}

DsaCacheMapping resolve_cache_mapping(const DSAMetadata& attn_metadata,
                                      int64_t compress_ratio) {
  DsaCacheMapping mapping;
  if (!attn_metadata.caches_info || attn_metadata.layer_id < 0 ||
      attn_metadata.layer_id >=
          static_cast<int32_t>(attn_metadata.caches_info->size())) {
    return mapping;
  }

  const auto& layer_caches =
      (*(attn_metadata.caches_info))[attn_metadata.layer_id];

  std::vector<int64_t> token_ratio_indices;
  std::vector<int64_t> swa_indices;
  token_ratio_indices.reserve(layer_caches.size());
  swa_indices.reserve(layer_caches.size());

  for (int64_t cache_idx = 0;
       cache_idx < static_cast<int64_t>(layer_caches.size());
       ++cache_idx) {
    const auto& cache_info = layer_caches[cache_idx];
    if (cache_info.type == DSACacheType::TOKEN &&
        cache_info.ratio == static_cast<int32_t>(compress_ratio)) {
      token_ratio_indices.push_back(cache_idx);
    }
    if (cache_info.type == DSACacheType::SLIDING_WINDOW) {
      swa_indices.push_back(cache_idx);
    }
  }

  if (!token_ratio_indices.empty() && compress_ratio > 1) {
    mapping.cmp_cache_idx = token_ratio_indices[0];
  }
  if (token_ratio_indices.size() > 1) {
    mapping.index_cache_idx = token_ratio_indices[1];
  }
  if (token_ratio_indices.size() > 2) {
    mapping.indexer_scale_cache_idx = token_ratio_indices[2];
  }

  if (!swa_indices.empty()) {
    mapping.ori_cache_idx = swa_indices[0];
  }
  if (swa_indices.size() > 1) {
    mapping.kv_state_cache_idx = swa_indices[1];
  }
  if (swa_indices.size() > 2) {
    mapping.score_state_cache_idx = swa_indices[2];
  }
  if (swa_indices.size() > 3) {
    mapping.index_kv_state_cache_idx = swa_indices[3];
  }
  if (swa_indices.size() > 4) {
    mapping.index_score_state_cache_idx = swa_indices[4];
  }

  return mapping;
}

void apply_partial_rope(torch::Tensor& input,
                        int64_t rope_start_dim,
                        int64_t rope_head_dim,
                        const torch::Tensor& cos,
                        const torch::Tensor& sin,
                        bool inverse = false) {
  if (!input.defined() || !cos.defined() || !sin.defined() ||
      rope_head_dim <= 0 || rope_start_dim < 0) {
    return;
  }

  const int64_t input_last_dim = input.size(input.dim() - 1);
  if (input_last_dim < rope_start_dim + rope_head_dim) {
    return;
  }

  auto sin_cache = inverse ? -sin : sin;
  auto cos_cache = cos;
  CHECK(input.dim() == 2 || input.dim() == 3)
      << "apply_partial_rope only supports input dim 2/3, got: " << input.dim();
  CHECK(cos_cache.dim() == 2 && sin_cache.dim() == 2)
      << "apply_partial_rope expects cos/sin dim=2, got cos dim "
      << cos_cache.dim() << ", sin dim " << sin_cache.dim();
  CHECK(cos_cache.size(0) == input.size(0) &&
        sin_cache.size(0) == input.size(0))
      << "apply_partial_rope expects cos/sin batch == input.size(0), got cos "
      << cos_cache.size(0) << ", sin " << sin_cache.size(0) << ", input "
      << input.size(0);
  CHECK(cos_cache.size(1) == rope_head_dim &&
        sin_cache.size(1) == rope_head_dim)
      << "apply_partial_rope expects cos/sin last dim == rope_head_dim("
      << rope_head_dim << "), got cos " << cos_cache.size(1) << ", sin "
      << sin_cache.size(1);

  auto cos_4d = cos_cache.view({cos_cache.size(0), 1, 1, rope_head_dim});
  auto sin_4d = sin_cache.view({sin_cache.size(0), 1, 1, rope_head_dim});
  auto input_4d =
      (input.dim() == 3) ? input.unsqueeze(2) : input.unsqueeze(1).unsqueeze(1);
  xllm::kernel::NpuInplacePartialRotaryMulParams rope_params;
  rope_params.x = input_4d;
  rope_params.r1 = cos_4d;
  rope_params.r2 = sin_4d;
  rope_params.rotary_mode = "interleave";
  rope_params.partial_slice = {rope_start_dim, rope_start_dim + rope_head_dim};
  xllm::kernel::npu_inplace_partial_rotary_mul(rope_params);
  input =
      (input.dim() == 3) ? input_4d.squeeze(2) : input_4d.squeeze(1).squeeze(1);
}

void scatter_by_slot(torch::Tensor& cache,
                     const torch::Tensor& slot_mapping,
                     const torch::Tensor& value,
                     bool require_exact_rows = false) {
  if (!cache.defined() || !slot_mapping.defined() || !value.defined()) {
    return;
  }

  auto value_2d = value.reshape({-1, value.size(value.dim() - 1)});
  auto cache_2d = cache.view({-1, value_2d.size(1)});

  auto slots = slot_mapping.reshape({-1}).to(torch::kLong).to(cache.device());
  if (require_exact_rows) {
    CHECK_EQ(slots.size(0), value_2d.size(0))
        << "DeepSeek V4 CP cache rows must match slot rows exactly, slots="
        << slots.size(0) << ", values=" << value_2d.size(0);
  }
  const int64_t update_rows =
      std::min<int64_t>(slots.size(0), value_2d.size(0));
  if (update_rows <= 0) {
    return;
  }

  auto slots_slice = slots.slice(/*dim=*/0, /*start=*/0, /*end=*/update_rows);
  auto value_slice =
      value_2d.slice(/*dim=*/0, /*start=*/0, /*end=*/update_rows);

  const int64_t cache_rows = cache_2d.size(0);
  CHECK_GT(cache_rows, 0) << "scatter_by_slot requires cache rows > 0, cache "
                          << cache.sizes();

  if (!cache.device().is_cpu()) {
    auto safe_slots = slots_slice.clamp_min(0);
    auto valid_mask = slots_slice.ge(0).unsqueeze(1);
    auto old_values = cache_2d.index_select(/*dim=*/0, safe_slots);
    auto safe_values = torch::where(valid_mask, value_slice, old_values);
    xllm::kernel::npu::scatter_nd_update(
        cache_2d, safe_slots.reshape({-1, 1}), safe_values);
    return;
  }

  auto valid_mask = slots_slice.ge(0);
  auto valid_slots = slots_slice.index({valid_mask});
  if (valid_slots.numel() == 0) {
    return;
  }
  auto valid_values = value_slice.index({valid_mask});

  const int64_t max_slot = valid_slots.max().item<int64_t>();
  CHECK_LT(max_slot, cache_rows)
      << "scatter_by_slot slot index out of range: max_slot=" << max_slot
      << ", cache_rows=" << cache_rows
      << ", slot_mapping_shape=" << slot_mapping.sizes()
      << ", value_shape=" << value.sizes()
      << ", value_rows=" << value_2d.size(0) << ", update_rows=" << update_rows
      << ", cache_shape=" << cache.sizes();
  cache_2d.index_copy_(/*dim=*/0, valid_slots, valid_values);
}

// Pack prefill TND KV [total_tokens, n, d] into temporary PA_ND blocks
// [num_blocks + 1, block_size, n, d]. This mirrors vllm-ascend's
// pad_to_blocks path and lets sparse_attn_sharedkv read prefill KV through a
// block table without depending on the persistent SWA ring cache.
std::tuple<torch::Tensor, torch::Tensor> build_prefill_pa_nd_kv(
    const torch::Tensor& kv,
    const torch::Tensor& cu_seqlens_q,
    const torch::Tensor& block_table_hint,
    int64_t block_size) {
  if (!kv.defined() || !cu_seqlens_q.defined() || cu_seqlens_q.numel() <= 1 ||
      block_size <= 0) {
    return {torch::Tensor(), torch::Tensor()};
  }

  const int64_t batch_size = cu_seqlens_q.numel() - 1;
  const auto cu_cpu = cu_seqlens_q.to(torch::kCPU).to(torch::kInt64);
  std::vector<int64_t> q_starts(batch_size + 1);
  auto cu_acc = cu_cpu.accessor<int64_t, 1>();
  int64_t total_blocks = 0;
  int64_t max_blocks_per_req = 0;
  // cu_seqlens_q describes the packed token ranges for each request. Convert
  // those ranges into the number of PA_ND blocks each request needs.
  for (int64_t i = 0; i <= batch_size; ++i) {
    q_starts[i] = cu_acc[i];
    if (i > 0) {
      const int64_t q_len = q_starts[i] - q_starts[i - 1];
      const int64_t blocks = (q_len + block_size - 1) / block_size;
      total_blocks += blocks;
      max_blocks_per_req = std::max(max_blocks_per_req, blocks);
    }
  }
  if (total_blocks <= 0) {
    return {torch::Tensor(), torch::Tensor()};
  }

  const int64_t table_cols =
      std::max<int64_t>(block_table_hint.defined() && block_table_hint.dim() > 1
                            ? block_table_hint.size(1)
                            : 0,
                        max_blocks_per_req);
  std::vector<int32_t> table_data(static_cast<size_t>(batch_size * table_cols),
                                  0);

  auto packed_kv = torch::zeros(
      {total_blocks + 1, block_size, kv.size(1), kv.size(2)}, kv.options());

  // Block 0 stays zero-filled as the padding block; real requests use 1-based
  // block ids, matching the block tables consumed by sparse_attn_sharedkv.
  int64_t next_block = 1;
  for (int64_t req = 0; req < batch_size; ++req) {
    const int64_t q_start = q_starts[req];
    const int64_t q_len = q_starts[req + 1] - q_start;
    const int64_t blocks = (q_len + block_size - 1) / block_size;
    if (q_len <= 0 || blocks <= 0) {
      continue;
    }
    for (int64_t j = 0; j < blocks; ++j) {
      table_data[static_cast<size_t>(req * table_cols + j)] =
          static_cast<int32_t>(next_block + j);
    }
    // Copy this request's contiguous prefill KV into its temporary PA_ND block
    // range. The zero-initialized tail of the last block is padding.
    auto target = packed_kv
                      .slice(/*dim=*/0,
                             /*start=*/next_block,
                             /*end=*/next_block + blocks)
                      .view({blocks * block_size, kv.size(1), kv.size(2)});
    target.narrow(/*dim=*/0, /*start=*/0, /*length=*/q_len)
        .copy_(kv.narrow(/*dim=*/0, /*start=*/q_start, /*length=*/q_len));
    next_block += blocks;
  }

  auto table_cpu =
      torch::tensor(table_data, torch::TensorOptions().dtype(torch::kInt32))
          .view({batch_size, table_cols});
  auto table = table_cpu.to(
      block_table_hint.defined() ? block_table_hint.device() : kv.device());
  return {packed_kv, table};
}

torch::Tensor make_owner_window_indices(
    const std::vector<int32_t>& q_seq_lens,
    const std::vector<int32_t>& kv_seq_lens,
    const std::vector<std::vector<int32_t>>& global_to_local,
    int64_t window_left,
    int64_t capacity,
    const torch::Device& device) {
  CHECK_EQ(q_seq_lens.size(), kv_seq_lens.size());
  CHECK_EQ(q_seq_lens.size(), global_to_local.size());
  CHECK_GT(capacity, 0);
  std::vector<int32_t> values;
  int64_t total_rows = 0;
  for (size_t sequence = 0; sequence < q_seq_lens.size(); ++sequence) {
    const int32_t q_len = q_seq_lens[sequence];
    const int32_t kv_len = kv_seq_lens[sequence];
    CHECK_GE(q_len, 0);
    CHECK_GE(kv_len, q_len);
    total_rows += q_len;
    for (int32_t query = 0; query < q_len; ++query) {
      const int32_t absolute = kv_len - q_len + query;
      const int32_t start = std::max<int32_t>(
          0, absolute - static_cast<int32_t>(window_left));
      const int32_t end = std::min(absolute, kv_len - 1);
      int64_t written = 0;
      for (int32_t position = start;
           position <= end && written < capacity;
           ++position) {
        const int32_t local = global_to_local[sequence][position];
        if (local >= 0) {
          values.emplace_back(local);
          ++written;
        }
      }
      values.insert(values.end(),
                    static_cast<size_t>(capacity - written),
                    -1);
    }
  }
  CHECK_EQ(static_cast<int64_t>(values.size()), total_rows * capacity);
  return torch::tensor(values, torch::TensorOptions().dtype(torch::kInt32))
      .view({total_rows, 1, capacity})
      .to(device, /*non_blocking=*/true);
}

torch::Tensor make_owner_compressed_indices(
    const std::vector<int32_t>& q_seq_lens,
    const std::vector<int32_t>& kv_seq_lens,
    const std::vector<std::vector<int32_t>>& global_to_local,
    int64_t compress_ratio,
    int64_t capacity,
    const torch::Tensor& global_candidates,
    const torch::Device& device) {
  CHECK_EQ(q_seq_lens.size(), kv_seq_lens.size());
  CHECK_EQ(q_seq_lens.size(), global_to_local.size());
  CHECK_GT(compress_ratio, 0);
  CHECK_GT(capacity, 0);
  const torch::Tensor candidates_cpu =
      global_candidates.defined()
          ? global_candidates.to(torch::kCPU).to(torch::kInt64).contiguous()
          : torch::Tensor();
  int64_t expected_rows = 0;
  for (int32_t q_len : q_seq_lens) {
    CHECK_GE(q_len, 0);
    expected_rows += q_len;
  }
  torch::Tensor candidate_rows;
  int64_t candidate_width = 0;
  if (candidates_cpu.defined()) {
    CHECK_GE(candidates_cpu.dim(), 2)
        << "DSV4 owner QLI candidates must be [T,...,K]";
    CHECK_EQ(candidates_cpu.size(0), expected_rows)
        << "DSV4 owner QLI candidates do not cover all global query rows";
    candidate_rows = candidates_cpu.view({expected_rows, -1});
    candidate_width = candidate_rows.size(1);
  }
  std::vector<int32_t> values;
  int64_t total_rows = 0;
  int64_t row_index = 0;
  for (size_t sequence = 0; sequence < q_seq_lens.size(); ++sequence) {
    const int32_t q_len = q_seq_lens[sequence];
    const int32_t kv_len = kv_seq_lens[sequence];
    const int32_t compressed_len =
        static_cast<int32_t>(kv_len / compress_ratio);
    total_rows += q_len;
    for (int32_t query = 0; query < q_len; ++query, ++row_index) {
      const int32_t available = static_cast<int32_t>(
          (kv_len - q_len + query + 1) / compress_ratio);
      int64_t written = 0;
      auto append_global = [&](int64_t global_index) {
        if (written >= capacity || global_index < 0 ||
            global_index >= compressed_len || global_index >= available ||
            global_index >=
                static_cast<int64_t>(global_to_local[sequence].size())) {
          return;
        }
        const int32_t local =
            global_to_local[sequence][static_cast<size_t>(global_index)];
        if (local >= 0) {
          values.emplace_back(local);
          ++written;
        }
      };
      if (candidate_rows.defined() && candidate_width > 0) {
        const auto row = candidate_rows.select(/*dim=*/0, row_index);
        for (int64_t candidate = 0;
             candidate < candidate_width && written < capacity;
             ++candidate) {
          append_global(row[candidate].item<int64_t>());
        }
      } else {
        for (int32_t global_index = 0;
             global_index < available && written < capacity;
             ++global_index) {
          append_global(global_index);
        }
      }
      values.insert(values.end(),
                    static_cast<size_t>(capacity - written),
                    -1);
    }
  }
  CHECK_EQ(static_cast<int64_t>(values.size()), total_rows * capacity);
  return torch::tensor(values, torch::TensorOptions().dtype(torch::kInt32))
      .view({total_rows, 1, capacity})
      .to(device, /*non_blocking=*/true);
}

torch::Tensor make_global_cumulative_lengths(
    const std::vector<int32_t>& lengths,
    const torch::Device& device) {
  std::vector<int32_t> cumulative(1, 0);
  cumulative.reserve(lengths.size() + 1);
  for (int32_t length : lengths) {
    CHECK_GE(length, 0);
    CHECK_LE(static_cast<int64_t>(cumulative.back()) + length,
             std::numeric_limits<int32_t>::max());
    cumulative.emplace_back(cumulative.back() + length);
  }
  return torch::tensor(cumulative,
                       torch::TensorOptions().dtype(torch::kInt32))
      .to(device, /*non_blocking=*/true);
}

CpRowLayout build_owner_row_layout(
    const std::vector<int32_t>& global_q_seq_lens,
    int32_t cp_size,
    int32_t cp_rank,
    const torch::Device& device) {
  int64_t global_token_count = 0;
  for (int32_t query_length : global_q_seq_lens) {
    CHECK_GE(query_length, 0);
    global_token_count += query_length;
  }
  CHECK_LE(global_token_count, std::numeric_limits<int32_t>::max());
  CpPlanInput input;
  input.q_seq_lens = global_q_seq_lens;
  // Owner routing only consumes row indices. Packed ordinal positions keep
  // CpRowLayout construction host-only without reading position tensors back
  // from the device.
  input.position_ids = torch::arange(
      global_token_count,
      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
  return CpRowLayout::build(input, cp_size, cp_rank, device);
}

Dsv4PreprocessOutputs run_dsv4_preprocess_fallback(
    ReplicatedLinear& q_a_proj,
    RMSNorm& q_layernorm,
    ColumnParallelLinear& q_b_proj,
    ReplicatedLinear& kv_proj,
    RMSNorm& kv_layernorm,
    const torch::Tensor& hidden_states,
    int64_t n_local_heads,
    int64_t head_dim,
    int64_t qk_head_dim,
    int64_t nope_head_dim,
    int64_t rope_head_dim,
    double eps,
    const torch::Tensor& q_rms_gamma,
    const torch::Tensor& cos,
    const torch::Tensor& sin) {
  Dsv4PreprocessOutputs outputs;

  auto q_down = q_a_proj->forward(hidden_states);
  if (q_b_proj->uses_w8a8_dynamic_quant()) {
    xllm::kernel::RmsNormDynamicQuantParams rms_quant_params;
    rms_quant_params.input = q_down;
    rms_quant_params.weight = q_layernorm->weight();
    rms_quant_params.eps = eps;
    torch::Tensor q_pertoken_scale;
    std::tie(outputs.qr, q_pertoken_scale) =
        xllm::kernel::rms_norm_dynamic_quant(rms_quant_params);
    outputs.qr_pertoken_scale = q_pertoken_scale;
    outputs.q =
        w8a8_dynamic_linear_forward(outputs.qr,
                                    q_b_proj->weight(),
                                    q_b_proj->w8a8_dynamic_weight_scale(),
                                    q_pertoken_scale,
                                    q_b_proj->bias(),
                                    hidden_states.scalar_type())
            .view({-1, n_local_heads, head_dim});
  } else {
    outputs.qr = std::get<0>(q_layernorm->forward(q_down));
    outputs.q =
        q_b_proj->forward(outputs.qr).view({-1, n_local_heads, head_dim});
  }

  xllm::kernel::FusedLayerNormParams q_rmsnorm_params;
  q_rmsnorm_params.input = outputs.q;
  q_rmsnorm_params.weight = q_rms_gamma;
  q_rmsnorm_params.mode = "rmsnorm";
  q_rmsnorm_params.eps = eps;
  xllm::kernel::fused_layernorm(q_rmsnorm_params);
  outputs.q = q_rmsnorm_params.output;

  auto kv_down = kv_proj->forward(hidden_states);
  outputs.kv = std::get<0>(kv_layernorm->forward(kv_down));
  outputs.kv = outputs.kv.view({-1, 1, qk_head_dim});

  apply_partial_rope(outputs.q, nope_head_dim, rope_head_dim, cos, sin);
  apply_partial_rope(outputs.kv, nope_head_dim, rope_head_dim, cos, sin);

  return outputs;
}

AttentionMetadata build_indexer_attention_metadata(
    const DSAMetadata& attn_metadata,
    const torch::Tensor& block_table,
    const torch::Tensor& slot_mapping,
    bool is_prefill,
    int64_t max_query_len,
    int64_t max_seq_len) {
  AttentionMetadata metadata;
  metadata.is_prefill = is_prefill;
  metadata.is_chunked_prefill = false;
  metadata.is_dummy = false;
  metadata.is_causal = true;

  metadata.block_table = block_table;
  metadata.slot_mapping = slot_mapping;

  metadata.q_cu_seq_lens = attn_metadata.actual_seq_lengths_query;
  metadata.kv_seq_lens = attn_metadata.actual_seq_lengths_kv;
  if (!metadata.kv_seq_lens.defined()) {
    metadata.kv_seq_lens = attn_metadata.seq_lens;
  }

  if (attn_metadata.actual_seq_lengths_query.defined() &&
      attn_metadata.actual_seq_lengths_query.dim() > 0 &&
      attn_metadata.actual_seq_lengths_query.size(0) > 1) {
    const torch::Tensor& q_cu = attn_metadata.actual_seq_lengths_query;
    metadata.q_seq_lens =
        q_cu.slice(/*dim=*/0, /*start=*/1, /*end=*/q_cu.size(0)) -
        q_cu.slice(/*dim=*/0, /*start=*/0, /*end=*/q_cu.size(0) - 1);
  } else {
    metadata.q_seq_lens = attn_metadata.seq_lens_q;
  }

  if (attn_metadata.kv_cu_seq_lens.defined() &&
      attn_metadata.kv_cu_seq_lens.numel() > 0) {
    // Reuse the kv cumulative sequence lengths computed once per forward in
    // DSAMetadataBuilder, avoiding a redundant host-side cumsum on every DSA
    // layer.
    metadata.kv_cu_seq_lens = attn_metadata.kv_cu_seq_lens;
  } else if (metadata.kv_seq_lens.defined()) {
    auto kv_seq_lens = metadata.kv_seq_lens.to(torch::kInt32);
    auto kv_cumsum = torch::cumsum(kv_seq_lens, /*dim=*/0);
    metadata.kv_cu_seq_lens =
        torch::cat({torch::zeros({1}, kv_seq_lens.options()), kv_cumsum});
  }

  metadata.max_query_len = max_query_len;
  metadata.max_seq_len = max_seq_len;

  metadata.dsa_metadata = std::make_shared<DSAMetadata>(attn_metadata);

  return metadata;
}

torch::Tensor select_active_sequences(const torch::Tensor& tensor,
                                      const Dsv4CpHalfMetadata& half_metadata) {
  if (!tensor.defined() || tensor.dim() == 0) {
    return tensor;
  }
  return tensor.index_select(/*dim=*/0, half_metadata.active_sequence_indices);
}

AttentionMetadata build_cp_half_indexer_metadata(
    const DSAMetadata& attn_metadata,
    const Dsv4CpHalfMetadata& half_metadata,
    const torch::Tensor& block_table,
    const torch::Tensor& slot_mapping) {
  AttentionMetadata metadata;
  metadata.is_prefill = true;
  metadata.is_chunked_prefill = false;
  metadata.is_dummy = false;
  metadata.is_causal = true;
  metadata.block_table = select_active_sequences(block_table, half_metadata);
  metadata.slot_mapping = select_active_sequences(slot_mapping, half_metadata);
  metadata.q_cu_seq_lens = half_metadata.q_cu_seq_lens;
  metadata.q_seq_lens = half_metadata.q_seq_lens;
  metadata.kv_seq_lens = half_metadata.kv_seq_lens;
  metadata.max_query_len =
      half_metadata.host_q_seq_lens.empty()
          ? 0
          : *std::max_element(half_metadata.host_q_seq_lens.begin(),
                              half_metadata.host_q_seq_lens.end());
  metadata.max_seq_len =
      half_metadata.host_kv_seq_lens.empty()
          ? 0
          : *std::max_element(half_metadata.host_kv_seq_lens.begin(),
                              half_metadata.host_kv_seq_lens.end());
  metadata.dsa_metadata = std::make_shared<DSAMetadata>(attn_metadata);
  return metadata;
}

torch::Tensor half_sparse_metadata(const Dsv4CpHalfMetadata& half_metadata,
                                   int64_t compress_ratio) {
  if (compress_ratio == 1) {
    return half_metadata.c1_sparse_metadata;
  }
  if (compress_ratio == 4) {
    return half_metadata.c4_sparse_metadata;
  }
  if (compress_ratio == 128) {
    return half_metadata.c128_sparse_metadata;
  }
  LOG(FATAL) << "Unsupported DeepSeek V4 CP compression ratio "
             << compress_ratio;
  return torch::Tensor();
}

}  // namespace

DSAttentionImpl::DSAttentionImpl(const ModelContext& context, int32_t layer_id)
    : DSAttentionImpl(context.get_model_args(),
                      context.get_quant_args(),
                      context.get_parallel_args(),
                      context.get_tensor_options(),
                      layer_id) {}

DSAttentionImpl::DSAttentionImpl(const ModelArgs& args,
                                 const QuantArgs& quant_args,
                                 const ParallelArgs& parallel_args,
                                 const torch::TensorOptions& options,
                                 int32_t layer_id)
    : num_heads_(args.n_heads()),
      head_size_(args.head_dim()),
      head_dim_(args.head_dim()),
      n_kv_heads_(args.n_kv_heads().value()),
      sliding_window_(-1),
      q_lora_rank_(args.q_lora_rank()),
      o_lora_rank_(args.o_lora_rank()),
      o_groups_(args.o_groups()),
      rope_head_dim_(args.rope_head_dim()),
      window_size_(args.window_size()),
      compress_ratio_(1.0),
      index_n_heads_(args.index_n_heads()),
      index_head_dim_(args.index_head_dim()),
      index_topk_(args.index_topk()),
      eps_(args.rms_norm_eps()) {
  const auto& compress_ratios = args.compress_ratios();
  CHECK(!compress_ratios.empty())
      << "DSAttention requires non-empty compress_ratios for DeepSeek V4";
  CHECK_GE(layer_id, 0) << "DSAttention requires valid layer_id, got "
                        << layer_id;
  CHECK_LT(layer_id, static_cast<int32_t>(compress_ratios.size()))
      << "DSAttention layer_id " << layer_id << " exceeds compress_ratios size "
      << compress_ratios.size();
  int64_t compress_ratio = compress_ratios[static_cast<size_t>(layer_id)];

  CHECK(compress_ratio == 1 || compress_ratio == 4 || compress_ratio == 128)
      << "DSAttention unsupported compress_ratio " << compress_ratio
      << " at layer " << layer_id;

  compress_ratio_ = static_cast<double>(compress_ratio);

  softmax_scale_ = std::pow(head_dim_, static_cast<double>(-0.5));
  scale_ = static_cast<float>(softmax_scale_);
  nope_head_dim_ = head_dim_ - rope_head_dim_;
  qk_head_dim_ = nope_head_dim_ + rope_head_dim_;

  const int64_t tp_size = parallel_args.tp_group_->world_size();
  tp_rank_ = parallel_args.tp_group_->rank();
  tp_size_ = tp_size;
  cp_size_ = parallel_args.cp_size();
  cp_rank_ = parallel_args.cp_rank();
  cp_group_ = parallel_args.cp_group_;
  int64_t hidden_size = args.hidden_size();
  int64_t num_heads = args.n_heads();

  CHECK_EQ(o_groups_ % tp_size, 0)
      << "o_groups must be divisible by tensor parallel size";
  CHECK_EQ(num_heads % tp_size, 0)
      << "num_heads must be divisible by tensor parallel size";
  if (parallel_args.cp_size() > 1) {
    CHECK(cp_group_ != nullptr)
        << "DeepSeek V4 CP requires a dedicated CP process group";
    CHECK_EQ(cp_group_->world_size(), cp_size_);
    CHECK_EQ(cp_group_->rank(), cp_rank_);
    CHECK(xllm::kernel::has_split_compressor())
        << "DeepSeek V4 CP requires CompressorProjection and CompressorCore "
           "custom operators during model initialization";
  }
  n_local_heads_ = num_heads / tp_size;
  n_local_groups_ = o_groups_ / tp_size;

  attn_sink_ = register_parameter(
      "attn_sink",
      torch::zeros({n_local_heads_}, options.dtype(torch::kFloat32)),
      /*requires_grad=*/false);

  q_a_proj_ = register_module(
      "q_a_proj",
      ReplicatedLinear(
          hidden_size, q_lora_rank_, /*bias=*/false, quant_args, options));

  q_layernorm_ =
      register_module("q_a_layernorm", RMSNorm(q_lora_rank_, eps_, options));

  q_b_proj_ = register_module("q_b_proj",
                              ColumnParallelLinear(q_lora_rank_,
                                                   num_heads * head_dim_,
                                                   false,
                                                   false,
                                                   quant_args,
                                                   parallel_args.tp_group_,
                                                   options));

  kv_proj_ = register_module(
      "kv_proj",
      ReplicatedLinear(
          hidden_size, head_dim_, /*bias=*/false, quant_args, options));
  kv_layernorm_ =
      register_module("kv_layernorm", RMSNorm(head_dim_, eps_, options));

  if (compress_ratio_ > 1) {
    compressor_ =
        register_module("compressor",
                        Compressor(static_cast<int64_t>(compress_ratio_),
                                   head_dim_,
                                   rope_head_dim_,
                                   /*rot_mode=*/2,
                                   eps_,
                                   options));
  }

  if (compress_ratio_ == 4) {
    if (index_n_heads_ <= 0) {
      index_n_heads_ = num_heads_;
    }
    if (index_head_dim_ <= 0) {
      index_head_dim_ = head_dim_;
    }
    if (index_topk_ <= 0) {
      index_topk_ = 512;
    }

    if (index_n_heads_ > 0 && index_head_dim_ > 0 && index_topk_ > 0) {
      indexer_ = register_module(
          "indexer",
          DeepseekV4Indexer(hidden_size,
                            index_n_heads_,
                            index_head_dim_,
                            rope_head_dim_,
                            index_topk_,
                            q_lora_rank_,
                            static_cast<int64_t>(compress_ratio_),
                            eps_,
                            quant_args,
                            options));
    } else {
      LOG(FATAL) << "DSAttention indexer disabled due to invalid config: "
                 << "index_n_heads=" << index_n_heads_
                 << ", index_head_dim=" << index_head_dim_
                 << ", index_topk=" << index_topk_;
    }
  }

  q_rms_gamma_ = register_buffer("q_rms_gamma",
                                 torch::ones({head_dim_},
                                             torch::TensorOptions()
                                                 .dtype(options.dtype())
                                                 .device(options.device())));

  o_a_proj_ =
      register_module("o_a_proj",
                      ColumnParallelLinear(num_heads * head_dim_ / o_groups_,
                                           o_groups_ * o_lora_rank_,
                                           false,
                                           true,
                                           quant_args,
                                           parallel_args.tp_group_,
                                           options));

  o_b_proj_ = register_module("o_b_proj",
                              RowParallelLinear(o_groups_ * o_lora_rank_,
                                                hidden_size,
                                                false,
                                                true,
                                                /*reduce=*/true,
                                                quant_args,
                                                parallel_args.tp_group_,
                                                options));
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>>
DSAttentionImpl::forward(const DSAMetadata& attn_metadata,
                         torch::Tensor& hidden_states,
                         KVCache& kv_cache,
                         KVState& kv_state,
                         bool is_prefill,
                         bool is_chunked_prefill,
                         const std::tuple<torch::Tensor,
                                          torch::Tensor,
                                          torch::Tensor,
                                          torch::Tensor>& compress_metadata,
                         const NpuCpPlan& cp_plan,
                         const Dsv4CpMetadata* cp_metadata) {
  auto [c1_metadata, c4_metadata, c128_metadata, qli_metadata] =
      compress_metadata;
  const bool model_rows_are_sharded = cp_plan.enabled();
  std::optional<Dsv4CpExecutionContext> cp_execution;
  std::optional<CpRowLayout> synthetic_owner_layout;
  const CpRowLayout* owner_row_layout = nullptr;
  ProcessGroup* owner_cp_group = nullptr;
  const std::vector<int32_t>* global_q_seq_lens = nullptr;
  const std::vector<int32_t>* global_kv_seq_lens = nullptr;
  torch::Tensor preprocess_cos = attn_metadata.cos;
  torch::Tensor preprocess_sin = attn_metadata.sin;
  if (model_rows_are_sharded) {
    CHECK(is_prefill || is_chunked_prefill)
        << "DeepSeek V4 split compressor is restricted to prefill phases";
    CHECK(cp_plan.process_group() != nullptr);
    CHECK(cp_metadata != nullptr);
    CHECK_EQ(cp_metadata->layout_signature, cp_plan.row_layout().signature());
    cp_execution.emplace(cp_plan.process_group(),
                         cp_plan.projection_gather_mode());
    preprocess_cos =
        cp_plan.row_layout().shard_rows(preprocess_cos, /*pad_value=*/0);
    preprocess_sin =
        cp_plan.row_layout().shard_rows(preprocess_sin, /*pad_value=*/0);
    owner_row_layout = &cp_plan.row_layout();
    owner_cp_group = cp_plan.process_group();
    global_q_seq_lens = &cp_plan.global_q_seq_lens();
    global_kv_seq_lens = &cp_plan.global_kv_seq_lens();
  } else if (cp_size_ > 1) {
    CHECK_EQ(attn_metadata.host_q_seq_lens.size(),
             attn_metadata.host_kv_seq_lens.size())
        << "DeepSeek V4 owner CP requires one KV length per query length";
    synthetic_owner_layout.emplace(build_owner_row_layout(
        attn_metadata.host_q_seq_lens,
        cp_size_,
        cp_rank_,
        hidden_states.device()));
    CHECK_EQ(synthetic_owner_layout->global_real_token_count(),
             hidden_states.size(0))
        << "DeepSeek V4 synthetic owner layout requires global-real model "
           "rows";
    owner_row_layout = &synthetic_owner_layout.value();
    owner_cp_group = cp_group_;
    global_q_seq_lens = &attn_metadata.host_q_seq_lens;
    global_kv_seq_lens = &attn_metadata.host_kv_seq_lens;
  }
  const bool owner_cp_enabled = owner_row_layout != nullptr;
  const bool use_owner_fused_decode =
      owner_cp_enabled && !is_prefill && !is_chunked_prefill;
  torch::Tensor owner_local_hidden = hidden_states;
  if (owner_cp_enabled && !model_rows_are_sharded) {
    owner_local_hidden =
        owner_row_layout->shard_rows(hidden_states, /*pad_value=*/0);
  }

  Dsv4PreprocessOutputs preprocess_outputs =
      run_dsv4_preprocess_fallback(q_a_proj_,
                                   q_layernorm_,
                                   q_b_proj_,
                                   kv_proj_,
                                   kv_layernorm_,
                                   hidden_states,
                                   n_local_heads_,
                                   head_dim_,
                                   qk_head_dim_,
                                   nope_head_dim_,
                                   rope_head_dim_,
                                   eps_,
                                   q_rms_gamma_,
                                   preprocess_cos,
                                   preprocess_sin);
  auto qr = preprocess_outputs.qr;
  auto qr_pertoken_scale = preprocess_outputs.qr_pertoken_scale;
  auto q = preprocess_outputs.q;
  auto kv = preprocess_outputs.kv;
  torch::Tensor local_q = q;
  torch::Tensor local_kv = kv;
  if (owner_cp_enabled && !model_rows_are_sharded) {
    local_kv = owner_row_layout->shard_rows(kv, /*pad_value=*/0);
  }

  const int64_t compress_ratio_i = static_cast<int64_t>(compress_ratio_);
  torch::Tensor local_main_packed_projection;
  torch::Tensor index_packed_projection;
  if (owner_cp_enabled && compress_ratio_i > 1 && compressor_ &&
      !use_owner_fused_decode) {
    local_main_packed_projection = compressor_->project(owner_local_hidden);
    CHECK_EQ(local_main_packed_projection.size(0),
             owner_row_layout->local_padded_token_count())
        << "DeepSeek V4 CP main projection lost local-padded rows";
  }
  if (model_rows_are_sharded) {
    if (local_main_packed_projection.defined()) {
      CHECK_EQ(local_main_packed_projection.size(0),
               cp_plan.local_padded_token_count())
          << "DeepSeek V4 CP main projection lost local-padded rows";
    }
    if (cp_execution->gather_mode() == CpProjectionGatherMode::BUNDLED &&
        compress_ratio_i == 4 && indexer_) {
      const torch::Tensor local_index_projection =
          indexer_->project_kv(hidden_states);
      const std::vector<torch::Tensor> global_projections =
          cp_execution->gather_projection_bundle(cp_plan.row_layout(),
                                                 {local_index_projection});
      CHECK_EQ(global_projections.size(), 1);
      index_packed_projection = global_projections.front();
    }

    kv = cp_execution->gather_global_rows(cp_plan.row_layout(), kv);
    CHECK_EQ(kv.size(0), cp_plan.global_real_token_count())
        << "DeepSeek V4 CP SWA KV gather lost global rows";
    if (index_packed_projection.defined()) {
      CHECK_EQ(index_packed_projection.size(0),
               cp_plan.global_real_token_count())
          << "DeepSeek V4 CP index projection gather lost global rows";
    }
  }

  torch::Tensor cos = attn_metadata.cos;
  torch::Tensor sin = attn_metadata.sin;

  // 4) resolve per-layer cache mapping
  DsaCacheMapping mapping =
      resolve_cache_mapping(attn_metadata, compress_ratio_i);

  auto cmp_block_table = get_layer_cache_tensor(attn_metadata.block_tables,
                                                attn_metadata.layer_id,
                                                mapping.cmp_cache_idx);
  auto ori_block_table = get_layer_cache_tensor(attn_metadata.block_tables,
                                                attn_metadata.layer_id,
                                                mapping.ori_cache_idx);
  auto kv_block_table = get_layer_cache_tensor(attn_metadata.block_tables,
                                               attn_metadata.layer_id,
                                               mapping.kv_state_cache_idx);
  auto score_block_table =
      get_layer_cache_tensor(attn_metadata.block_tables,
                             attn_metadata.layer_id,
                             mapping.score_state_cache_idx);
  auto index_kv_block_table =
      get_layer_cache_tensor(attn_metadata.block_tables,
                             attn_metadata.layer_id,
                             mapping.index_kv_state_cache_idx);
  auto index_score_block_table =
      get_layer_cache_tensor(attn_metadata.block_tables,
                             attn_metadata.layer_id,
                             mapping.index_score_state_cache_idx);
  auto index_block_table = get_layer_cache_tensor(attn_metadata.block_tables,
                                                  attn_metadata.layer_id,
                                                  mapping.index_cache_idx);

  auto cmp_slot = get_layer_cache_tensor(attn_metadata.slot_mappings,
                                         attn_metadata.layer_id,
                                         mapping.cmp_cache_idx);
  auto ori_slot = get_layer_cache_tensor(attn_metadata.slot_mappings,
                                         attn_metadata.layer_id,
                                         mapping.ori_cache_idx);
  auto index_slot = get_layer_cache_tensor(attn_metadata.slot_mappings,
                                           attn_metadata.layer_id,
                                           mapping.index_cache_idx);

  auto ori_kv = std::get<0>(kv_state);
  if (!ori_kv.defined()) {
    ori_kv = kv_cache.get_swa_cache();
  }

  auto compressor_kv_state = std::get<1>(kv_state);
  if (!compressor_kv_state.defined()) {
    compressor_kv_state = kv_cache.get_compress_kv_state();
  }

  auto compressor_score_state = std::get<2>(kv_state);
  if (!compressor_score_state.defined()) {
    compressor_score_state = kv_cache.get_compress_score_state();
  }

  auto index_kv_state = std::get<3>(kv_state);
  if (!index_kv_state.defined()) {
    index_kv_state = kv_cache.get_compress_index_kv_state();
  }

  auto index_score_state = std::get<4>(kv_state);
  if (!index_score_state.defined()) {
    index_score_state = kv_cache.get_compress_index_score_state();
  }

  auto cmp_kv = kv_cache.get_k_cache();
  std::optional<Dsv4CpOwnershipPlan> cp_ownership_plan;
  std::optional<Dsv4CpRouteDescriptor> cp_swa_route;
  std::optional<Dsv4CpAttentionExchange> cp_owner_exchange;
  if (owner_cp_enabled) {
    CHECK(ori_kv.defined());
    CHECK_GE(ori_kv.dim(), 2);
    Dsv4CpOwnershipPlanner ownership_planner;
    if (compress_ratio_i > 1) {
      CHECK(cmp_kv.defined());
      CHECK(compressor_kv_state.defined());
      CHECK(compressor_score_state.defined());
      CHECK_GE(cmp_kv.dim(), 2);
      CHECK_GE(compressor_kv_state.dim(), 2);
      CHECK_EQ(ori_kv.size(1), compressor_kv_state.size(1))
          << "DSV4 SWA and compressor state must share one block layout";

      const torch::Tensor state_global_block_table = get_layer_cache_tensor(
          attn_metadata.host_block_tables,
          attn_metadata.layer_id,
          mapping.kv_state_cache_idx);
      const torch::Tensor compressed_global_block_table =
          get_layer_cache_tensor(attn_metadata.host_block_tables,
                                 attn_metadata.layer_id,
                                 mapping.cmp_cache_idx);
      CHECK(state_global_block_table.defined())
          << "DSV4 CP owner planning requires the host state block table";
      CHECK(compressed_global_block_table.defined())
          << "DSV4 CP owner planning requires the host compressed block table";

      cp_ownership_plan.emplace(ownership_planner.build(
          *owner_row_layout,
          *global_q_seq_lens,
          *global_kv_seq_lens,
          state_global_block_table,
          compressed_global_block_table,
          compress_ratio_i,
          compressor_kv_state.size(1),
          cmp_kv.size(1),
          hidden_states.device(),
          Dsv4CpCacheAddressing::REPLICATED));
      cp_swa_route = cp_ownership_plan->swa_route();
    } else {
      const torch::Tensor swa_global_block_table = get_layer_cache_tensor(
          attn_metadata.host_block_tables,
          attn_metadata.layer_id,
          mapping.ori_cache_idx);
      CHECK(swa_global_block_table.defined())
          << "DSV4 C1 owner planning requires the host SWA block table";
      cp_swa_route.emplace(ownership_planner.build_swa_route(
          *owner_row_layout,
          *global_q_seq_lens,
          *global_kv_seq_lens,
          swa_global_block_table,
          ori_kv.size(1),
          hidden_states.device(),
          Dsv4CpCacheAddressing::REPLICATED));
    }
    cp_owner_exchange.emplace(owner_cp_group);

    const torch::Tensor owner_swa_rows = cp_owner_exchange->route_owner_rows(
        local_kv.reshape({local_kv.size(0), -1}).contiguous(),
        cp_swa_route.value());
    Dsv4CpAttentionExchange::write_received_cache_rows(
        ori_kv, owner_swa_rows, cp_swa_route.value());
  }

  // 5) Prepare ori_kv for attention.
  // Full prefill can read a temporary PA_ND cache built from current KV.
  // Chunked prefill needs prefix KV, so it reads the persistent SWA cache.
  torch::Tensor ori_kv_for_attn;
  torch::Tensor ori_block_table_for_attn = ori_block_table;
  const std::string ori_kv_layout = "PA_ND";
  const bool use_prefill_attn = is_prefill || is_chunked_prefill;
  const bool use_temporary_prefill_kv = is_prefill && !is_chunked_prefill;
  if (use_temporary_prefill_kv) {
    const int64_t block_size =
        ori_kv.defined() && ori_kv.dim() > 1 ? ori_kv.size(1) : 128;
    std::tie(ori_kv_for_attn, ori_block_table_for_attn) =
        build_prefill_pa_nd_kv(kv,
                               attn_metadata.actual_seq_lengths_query,
                               ori_block_table,
                               block_size);
    CHECK(ori_kv_for_attn.defined())
        << "Failed to build PA_ND KV for DeepSeek V4 prefill attention.";
  } else {
    if (!cp_swa_route.has_value()) {
      scatter_by_slot(ori_kv,
                      ori_slot,
                      kv,
                      /*require_exact_rows=*/model_rows_are_sharded);
    }
    ori_kv_for_attn = ori_kv;
  }

  // 6) optional compressor for cmp cache
  // Token compressed cache is PA_ND for both prefill and decode.
  torch::Tensor cmp_kv_for_attn;
  if (compress_ratio_i > 1 && compressor_ && cmp_kv.defined() &&
      cmp_slot.defined() && compressor_kv_state.defined() &&
      compressor_score_state.defined()) {
    torch::Tensor compress_cos;
    torch::Tensor compress_sin;
    if (compress_ratio_i == 4) {
      compress_cos = attn_metadata.c4_cos;
      compress_sin = attn_metadata.c4_sin;
    } else if (compress_ratio_i == 128) {
      compress_cos = attn_metadata.c128_cos;
      compress_sin = attn_metadata.c128_sin;
    }

    std::tuple<torch::Tensor, torch::Tensor> compressor_states{
        compressor_kv_state, compressor_score_state};
    std::tuple<torch::Tensor, torch::Tensor> compressor_block_tables{
        kv_block_table, score_block_table};

    torch::Tensor compressed_kv;
    if (owner_cp_enabled) {
      CHECK(cp_ownership_plan.has_value());
      CHECK(cp_owner_exchange.has_value());
      const Dsv4CpOwnerMetadata& owner_metadata =
          cp_ownership_plan->owner_metadata();
      torch::Tensor owner_compressor_input =
          cp_owner_exchange->route_owner_rows(
              use_owner_fused_decode ? owner_local_hidden
                                     : local_main_packed_projection,
              cp_ownership_plan->main_route());
      if (owner_metadata.real_row_count == 0) {
        CHECK_EQ(owner_metadata.output_row_count, 0);
        compressed_kv = torch::empty(
            {0, cmp_kv.size(cmp_kv.dim() - 1)}, cmp_kv.options());
      } else {
        torch::Tensor owner_compress_sin =
            Dsv4CpAttentionExchange::select_owner_output_rows(
                compress_sin, owner_metadata);
        torch::Tensor owner_compress_cos =
            Dsv4CpAttentionExchange::select_owner_output_rows(
                compress_cos, owner_metadata);
        std::tuple<torch::Tensor, torch::Tensor> owner_block_tables{
            owner_metadata.local_state_block_table,
            owner_metadata.local_state_block_table};
        if (use_owner_fused_decode) {
          compressed_kv = compressor_->forward_owner_decode(
              owner_metadata,
              owner_compressor_input,
              compressor_states,
              owner_block_tables,
              owner_compress_sin,
              owner_compress_cos);
        } else {
          compressed_kv = compressor_->forward_owner_core(
              owner_metadata,
              owner_compressor_input,
              compressor_states,
              owner_block_tables,
              owner_compress_sin,
              owner_compress_cos);
        }
      }

      const torch::Tensor received_compressed_kv =
          cp_owner_exchange->route_owner_rows(
              compressed_kv,
              cp_ownership_plan->compressed_cache_route());
      Dsv4CpAttentionExchange::write_received_cache_rows(
          cmp_kv,
          received_compressed_kv,
          cp_ownership_plan->compressed_cache_route());
    } else {
      compressed_kv =
          compressor_->forward(attn_metadata,
                               hidden_states,
                               compressor_states,
                               compressor_block_tables,
                               compress_sin,
                               compress_cos,
                               attn_metadata.actual_seq_lengths_query);
      scatter_by_slot(cmp_kv,
                      cmp_slot,
                      compressed_kv,
                      /*require_exact_rows=*/false);
    }
    cmp_kv_for_attn = cmp_kv;
  }

  torch::Tensor compress_topk_idxs;
  if (compress_ratio_i == 4 && cmp_kv.defined()) {
    auto index_cache = kv_cache.get_index_cache();
    std::optional<torch::Tensor> indexer_cache_scale =
        kv_cache.get_indexer_cache_scale();
    torch::Tensor& indexer_cache_scale_tensor = indexer_cache_scale.value();

    std::tuple<torch::Tensor, torch::Tensor> indexer_states{index_kv_state,
                                                            index_score_state};
    std::tuple<torch::Tensor, torch::Tensor> indexer_block_tables{
        index_kv_block_table, index_score_block_table};
    auto indexer_metadata =
        build_indexer_attention_metadata(attn_metadata,
                                         index_block_table,
                                         index_slot,
                                         use_prefill_attn,
                                         attn_metadata.max_query_len,
                                         attn_metadata.max_seq_len);
    CHECK(qli_metadata.defined()) << "DSAttention requires precomputed "
                                     "qli_metadata for compress_ratio==4.";
    auto qli_metadata_opt = std::optional<torch::Tensor>(qli_metadata);
    if (model_rows_are_sharded) {
      if (cp_execution->gather_mode() == CpProjectionGatherMode::SEQUENTIAL) {
        torch::Tensor local_index_projection =
            indexer_->project_kv(hidden_states);
        index_packed_projection = cp_execution->gather_global_rows(
            cp_plan.row_layout(), local_index_projection);
      }
      CHECK(index_packed_projection.defined());
      torch::Tensor precomputed_index_kv =
          indexer_->compress_kv_core(index_packed_projection,
                                     indexer_metadata,
                                     attn_metadata.c4_cos,
                                     attn_metadata.c4_sin,
                                     attn_metadata.actual_seq_lengths_query,
                                     &indexer_states,
                                     &indexer_block_tables);
      if (cp_execution->gather_mode() == CpProjectionGatherMode::SEQUENTIAL) {
        index_packed_projection = torch::Tensor();
      }
      indexer_->update_kv_cache(precomputed_index_kv,
                                index_cache,
                                &indexer_cache_scale_tensor,
                                indexer_metadata,
                                /*require_exact_rows=*/true);

      // Each cache owner scores only its compressed index blocks. Global
      // query rows are one-token virtual sequences so compacting owner blocks
      // does not change the causal boundary of an earlier prefill query.
      torch::Tensor global_hidden = cp_execution->gather_global_rows(
          cp_plan.row_layout(), hidden_states);
      torch::Tensor global_qr =
          cp_execution->gather_global_rows(cp_plan.row_layout(), qr);
      std::optional<torch::Tensor> global_qr_scale;
      if (qr_pertoken_scale.has_value()) {
        global_qr_scale = cp_execution->gather_global_rows(
            cp_plan.row_layout(), qr_pertoken_scale.value());
      }
      torch::Tensor global_index_query = indexer_->prepare_query(global_qr,
                                                                 global_qr_scale,
                                                                 indexer_metadata,
                                                                 cos,
                                                                 sin);
      torch::Tensor global_index_weights =
          indexer_->build_weights(global_hidden);
      const torch::Tensor index_global_block_table = get_layer_cache_tensor(
          attn_metadata.host_block_tables,
          attn_metadata.layer_id,
          mapping.index_cache_idx);
      CHECK(index_global_block_table.defined())
          << "DSV4 CP owner QLI requires the host index block table";
      const Dsv4CpOwnerQliMetadata owner_qli_metadata =
          Dsv4CpAttentionExchange::build_owner_qli_metadata(
              index_global_block_table,
              cp_plan.global_q_seq_lens(),
              cp_plan.global_kv_seq_lens(),
              index_cache.size(1),
              compress_ratio_i,
              cp_plan.size(),
              cp_plan.rank(),
              global_index_query.device(),
              Dsv4CpCacheAddressing::REPLICATED);
      CHECK_EQ(owner_qli_metadata.query_sequence_indices.numel(),
               global_index_query.size(0));

      DeepseekV4QliResult local_candidates;
      if (owner_qli_metadata.valid_query_row_count > 0) {
        const torch::Tensor safe_key_seq_lens = torch::where(
            owner_qli_metadata.valid_query_rows,
            owner_qli_metadata.local_key_seq_lens,
            torch::full_like(owner_qli_metadata.local_key_seq_lens,
                             compress_ratio_i));
        const int64_t max_owner_key_seq_len = std::max<int64_t>(
            owner_qli_metadata.max_local_key_seq_len, compress_ratio_i);
        const torch::Tensor owner_qli_tiling_metadata =
            indexer_->build_qli_metadata(global_index_query,
                                         owner_qli_metadata.query_seq_endpoints,
                                         safe_key_seq_lens,
                                         /*max_query_len=*/1,
                                         max_owner_key_seq_len);
        local_candidates = indexer_->select_qli_candidates(
            global_index_query,
            global_index_weights,
            index_cache,
            &indexer_cache_scale_tensor,
            owner_qli_metadata.query_seq_endpoints,
            safe_key_seq_lens,
            owner_qli_metadata.query_block_table,
            owner_qli_tiling_metadata);
        torch::Tensor valid_rows =
            owner_qli_metadata.valid_query_rows.view({-1, 1, 1});
        local_candidates.indices = torch::where(
            valid_rows,
            local_candidates.indices,
            torch::full_like(local_candidates.indices, -1));
        local_candidates.scores = torch::where(
            valid_rows,
            local_candidates.scores,
            torch::full_like(local_candidates.scores,
                             -std::numeric_limits<float>::infinity()));
      } else {
        const std::vector<int64_t> candidate_shape = {
            global_index_query.size(0), index_cache.size(2), index_topk_};
        local_candidates.indices = torch::full(
            candidate_shape,
            -1,
            global_index_query.options().dtype(torch::kInt32));
        local_candidates.scores = torch::full(
            candidate_shape,
            -std::numeric_limits<float>::infinity(),
            global_index_query.options().dtype(torch::kFloat32));
      }

      local_candidates.indices =
          Dsv4CpAttentionExchange::map_owner_local_indices_to_global(
              local_candidates.indices,
              owner_qli_metadata.query_sequence_indices,
              owner_qli_metadata.index_map);
      CHECK(cp_owner_exchange.has_value());
      const Dsv4CpQliMergeResult global_candidates =
          cp_owner_exchange->global_topk(local_candidates.indices,
                                         local_candidates.scores,
                                         index_topk_);
      compress_topk_idxs = global_candidates.indices;
      CHECK(compress_topk_idxs.defined())
          << "DSAttention owner QLI returned undefined topk indices.";
    } else {
      compress_topk_idxs =
          indexer_->select_qli(hidden_states,
                               qr,
                               qr_pertoken_scale,
                               index_cache,
                               &indexer_cache_scale_tensor,
                               indexer_metadata,
                               cos,
                               sin,
                               attn_metadata.c4_cos,
                               attn_metadata.c4_sin,
                               attn_metadata.actual_seq_lengths_query,
                               attn_metadata.actual_seq_lengths_kv,
                               qli_metadata_opt,
                               use_prefill_attn,
                               &indexer_states,
                               &indexer_block_tables);
      CHECK(compress_topk_idxs.defined())
          << "DSAttention indexer returned undefined topk indices for "
             "compress_ratio==4.";
    }
  }

  // 7) sparse shared-kv attention
  std::optional<torch::Tensor> sparse_metadata = std::nullopt;
  if (compress_ratio_i == 1) {
    sparse_metadata = as_optional(c1_metadata);
  } else if (compress_ratio_i == 4) {
    sparse_metadata = as_optional(c4_metadata);
  } else if (compress_ratio_i == 128) {
    sparse_metadata = as_optional(c128_metadata);
  }

  torch::Tensor attn_output;
  torch::Tensor output_lse;
  if (owner_cp_enabled) {
    const std::vector<int32_t>& owner_global_q_seq_lens =
        *global_q_seq_lens;
    const std::vector<int32_t>& owner_global_kv_seq_lens =
        *global_kv_seq_lens;
    const torch::Tensor global_q =
        model_rows_are_sharded
            ? cp_execution->gather_global_rows(*owner_row_layout, local_q)
            : q;
    const int32_t owner_cp_size = owner_row_layout->cp_size();
    const int32_t owner_cp_rank = owner_row_layout->cp_rank();

    Dsv4CpOwnerPaCache owner_ori = build_dsv4_cp_owner_pa_cache(
        ori_kv_for_attn,
        ori_block_table_for_attn,
        owner_global_kv_seq_lens,
        owner_cp_size,
        owner_cp_rank,
        Dsv4CpBlockTableHolePolicy::SKIP_EVICTED);
    const int64_t owner_sparse_capacity = std::min<int64_t>(
        1024,
        std::max<int64_t>(512,
                          (std::max<int64_t>(window_size_, 1) +
                           owner_cp_size * ori_kv_for_attn.size(1) - 1) /
                              (owner_cp_size * ori_kv_for_attn.size(1)) *
                              ori_kv_for_attn.size(1)));
    const torch::Tensor owner_ori_indices = make_owner_window_indices(
        owner_global_q_seq_lens,
        owner_global_kv_seq_lens,
        owner_ori.global_to_local,
        std::max<int64_t>(window_size_ - 1, 0),
        owner_sparse_capacity,
        global_q.device());

    std::optional<Dsv4CpOwnerPaCache> owner_cmp;
    std::optional<torch::Tensor> owner_cmp_indices;
    int64_t cmp_topk = 0;
    if (compress_ratio_i > 1) {
      CHECK(cmp_kv_for_attn.defined());
      CHECK(cmp_block_table.defined());
      std::vector<int32_t> global_cmp_seq_lens;
      global_cmp_seq_lens.reserve(owner_global_kv_seq_lens.size());
      for (int32_t length : owner_global_kv_seq_lens) {
        global_cmp_seq_lens.emplace_back(
            static_cast<int32_t>(length / compress_ratio_i));
      }
      owner_cmp.emplace(build_dsv4_cp_owner_pa_cache(
          cmp_kv_for_attn,
          cmp_block_table,
          global_cmp_seq_lens,
          owner_cp_size,
          owner_cp_rank,
          Dsv4CpBlockTableHolePolicy::REJECT));
      const int64_t cmp_capacity = std::min<int64_t>(
          1024, std::max<int64_t>(512, index_topk_));
      // C4 uses the QLI-selected compressed-token bucket. C128 uses the
      // operator's CFA path and its zero cmp_topk means that the compressed
      // cache is consumed in full; the owner index tensor remains present to
      // describe the rank-local physical cache layout.
      cmp_topk = compress_ratio_i == 4 ? cmp_capacity : 0;
      if (compress_ratio_i == 4) {
        CHECK(compress_topk_idxs.defined());
        const torch::Tensor compressed_global_block_table =
            get_layer_cache_tensor(attn_metadata.host_block_tables,
                                   attn_metadata.layer_id,
                                   mapping.cmp_cache_idx);
        CHECK(compressed_global_block_table.defined())
            << "DSV4 CP owner attention requires the host compressed block "
               "table";
        const Dsv4CpOwnerIndexMap compressed_index_map =
            Dsv4CpAttentionExchange::build_owner_index_map(
                compressed_global_block_table,
                global_cmp_seq_lens,
                cmp_kv_for_attn.size(1),
                owner_cp_size,
                owner_cp_rank,
                global_q.device());
        const torch::Tensor query_sequence_indices =
            Dsv4CpAttentionExchange::build_query_sequence_indices(
                owner_global_q_seq_lens, global_q.device());
        torch::Tensor local_candidates =
            Dsv4CpAttentionExchange::map_global_indices_to_owner_local(
                compress_topk_idxs,
                query_sequence_indices,
                compressed_index_map);
        local_candidates =
            Dsv4CpAttentionExchange::compact_valid_indices(local_candidates);
        const int64_t local_candidate_count =
            local_candidates.size(local_candidates.dim() - 1);
        if (local_candidate_count < cmp_capacity) {
          std::vector<int64_t> padding_shape = local_candidates.sizes().vec();
          padding_shape.back() = cmp_capacity - local_candidate_count;
          local_candidates = torch::cat(
              {local_candidates,
               torch::full(padding_shape, -1, local_candidates.options())},
              /*dim=*/local_candidates.dim() - 1);
        } else if (local_candidate_count > cmp_capacity) {
          local_candidates = local_candidates.slice(
              /*dim=*/local_candidates.dim() - 1,
              /*start=*/0,
              /*end=*/cmp_capacity);
        }
        owner_cmp_indices = local_candidates.to(torch::kInt32).contiguous();
      } else {
        owner_cmp_indices = make_owner_compressed_indices(
            owner_global_q_seq_lens,
            owner_global_kv_seq_lens,
            owner_cmp->global_to_local,
            compress_ratio_i,
            cmp_capacity,
            /*global_candidates=*/torch::Tensor(),
            global_q.device());
      }
    }

    torch::Tensor owner_partial_output = torch::zeros_like(global_q);
    torch::Tensor owner_partial_lse = torch::full(
        {global_q.size(0), global_q.size(1), 1},
        -std::numeric_limits<float>::infinity(),
        global_q.options().dtype(torch::kFloat32));
    const bool owner_has_ori_keys =
        owner_ori.local_seq_lens.numel() > 0 &&
        owner_ori.local_seq_lens.max().item<int64_t>() > 0;
    const bool owner_has_cmp_keys =
        owner_cmp.has_value() && owner_cmp->local_seq_lens.numel() > 0 &&
        owner_cmp->local_seq_lens.max().item<int64_t>() > 0;
    torch::Tensor owner_sinks = torch::full(
        {n_local_heads_},
        std::numeric_limits<float>::lowest(),
        torch::TensorOptions().dtype(torch::kFloat32).device(global_q.device()));
    const bool owner_has_sink = attn_sink_loaded_ && owner_cp_rank == 0;
    if (owner_has_sink) {
      owner_sinks.copy_(attn_sink_);
    }
    if (owner_has_ori_keys || owner_has_cmp_keys) {
      if (cp_tensor_debug_enabled(attn_metadata.layer_id)) {
        const int32_t debug_cp_rank = owner_cp_rank;
        log_cp_tensor_summary(attn_metadata.layer_id,
                              debug_cp_rank,
                              "owner_attention_input",
                              "global_q",
                              global_q);
        log_cp_tensor_summary(attn_metadata.layer_id,
                              debug_cp_rank,
                              "owner_attention_input",
                              "owner_ori_cache",
                              owner_ori.cache);
        log_cp_tensor_summary(attn_metadata.layer_id,
                              debug_cp_rank,
                              "owner_attention_input",
                              "owner_cmp_cache",
                              owner_cmp.has_value()
                                  ? owner_cmp->cache
                                  : torch::Tensor());
        log_cp_tensor_summary(attn_metadata.layer_id,
                              debug_cp_rank,
                              "owner_attention_input",
                              "owner_ori_indices",
                              owner_ori_indices);
        log_cp_tensor_summary(attn_metadata.layer_id,
                              debug_cp_rank,
                              "owner_attention_input",
                              "owner_cmp_indices",
                              owner_cmp_indices.has_value()
                                  ? owner_cmp_indices.value()
                                  : torch::Tensor());
        log_cp_tensor_summary(attn_metadata.layer_id,
                              debug_cp_rank,
                              "owner_attention_input",
                              "owner_ori_block_table",
                              owner_ori.block_table);
        log_cp_tensor_summary(attn_metadata.layer_id,
                              debug_cp_rank,
                              "owner_attention_input",
                              "owner_cmp_block_table",
                              owner_cmp.has_value()
                                  ? owner_cmp->block_table
                                  : torch::Tensor());
        log_cp_tensor_summary(
            attn_metadata.layer_id,
            debug_cp_rank,
            "owner_attention_input",
            "cu_seqlens_q",
            make_global_cumulative_lengths(owner_global_q_seq_lens,
                                           global_q.device()));
        log_cp_tensor_summary(attn_metadata.layer_id,
                              debug_cp_rank,
                              "owner_attention_input",
                              "owner_local_seq_lens",
                              owner_ori.local_seq_lens);
        log_cp_tensor_summary(attn_metadata.layer_id,
                              debug_cp_rank,
                              "owner_attention_input",
                              "sinks",
                              owner_sinks);
      }
      std::tie(owner_partial_output, owner_partial_lse) =
          xllm::kernel::npu::sparse_attn_sharedkv_owner(
              global_q,
              owner_ori.cache,
              owner_cmp.has_value() ? std::optional<torch::Tensor>(
                                          owner_cmp->cache)
                                    : std::nullopt,
              owner_ori_indices,
              owner_cmp_indices,
              owner_ori.block_table,
              owner_cmp.has_value() ? std::optional<torch::Tensor>(
                                          owner_cmp->block_table)
                                    : std::nullopt,
              make_global_cumulative_lengths(owner_global_q_seq_lens,
                                             global_q.device()),
              owner_ori.local_seq_lens,
              owner_sinks,
              owner_sparse_capacity,
              cmp_topk,
              compress_ratio_i,
              softmax_scale_,
              std::max<int64_t>(window_size_ - 1, 0));
      if (cp_tensor_debug_enabled(attn_metadata.layer_id)) {
        const int32_t debug_cp_rank = owner_cp_rank;
        log_cp_tensor_summary(attn_metadata.layer_id,
                              debug_cp_rank,
                              "owner_attention_output",
                              "owner_partial_output",
                              owner_partial_output);
        log_cp_tensor_summary(attn_metadata.layer_id,
                              debug_cp_rank,
                              "owner_attention_output",
                              "owner_partial_lse",
                              owner_partial_lse);
      }
    } else if (owner_has_sink) {
      owner_partial_lse.copy_(
          owner_sinks.view({1, n_local_heads_, 1})
              .expand(owner_partial_lse.sizes()));
    }

    Dsv4CpAttentionExchange owner_exchange(owner_cp_group);
    const Dsv4CpAttentionMergeResult merged =
        owner_exchange.merge_attention_partials(owner_partial_output,
                                                owner_partial_lse);
    if (cp_tensor_debug_enabled(attn_metadata.layer_id)) {
      log_cp_tensor_summary(attn_metadata.layer_id,
                            owner_cp_rank,
                            "owner_attention_merged",
                            "merged_output",
                            merged.output);
      log_cp_tensor_summary(attn_metadata.layer_id,
                            owner_cp_rank,
                            "owner_attention_merged",
                            "merged_lse",
                            merged.lse);
    }
    attn_output = model_rows_are_sharded
                      ? owner_row_layout->shard_rows(merged.output, 0)
                      : merged.output;
    output_lse = std::move(merged.lse);
    if (model_rows_are_sharded) {
      cos = preprocess_cos;
      sin = preprocess_sin;
    }
  } else {
    CHECK(sparse_metadata.has_value())
        << "DSAttention requires precomputed sparse metadata for "
           "compress_ratio="
        << compress_ratio_i;
    std::optional<torch::Tensor> cu_seqlens_ori_kv_for_attn = std::nullopt;
    if (use_prefill_attn) {
      cu_seqlens_ori_kv_for_attn =
          as_optional(attn_metadata.actual_seq_lengths_query);
    }
    std::tie(attn_output, output_lse) = xllm::kernel::npu::sparse_attn_sharedkv(
        /*q=*/q,
        /*ori_kv=*/as_optional(ori_kv_for_attn),
        /*cmp_kv=*/compress_ratio_i > 1 ? as_optional(cmp_kv_for_attn)
                                        : std::nullopt,
        /*ori_sparse_indices=*/std::nullopt,
        /*cmp_sparse_indices=*/compress_ratio_i == 4
            ? as_optional(compress_topk_idxs)
            : std::nullopt,
        /*ori_block_table=*/as_optional(ori_block_table_for_attn),
        /*cmp_block_table=*/compress_ratio_i > 1 ? as_optional(cmp_block_table)
                                                 : std::nullopt,
        /*cu_seqlens_q=*/
        as_optional(attn_metadata.actual_seq_lengths_query),
        /*cu_seqlens_ori_kv=*/cu_seqlens_ori_kv_for_attn,
        /*cu_seqlens_cmp_kv=*/std::nullopt,
        /*seqused_q=*/std::nullopt,
        /*seqused_kv=*/as_optional(attn_metadata.actual_seq_lengths_kv),
        /*sinks=*/attn_sink_loaded_ ? as_optional(attn_sink_) : std::nullopt,
        /*metadata=*/sparse_metadata,
        /*softmax_scale=*/softmax_scale_,
        /*cmp_ratio=*/compress_ratio_i,
        /*ori_mask_mode=*/4,
        /*cmp_mask_mode=*/3,
        /*ori_win_left=*/std::max<int64_t>(window_size_ - 1, 0),
        /*ori_win_right=*/0,
        /*layout_q=*/"TND",
        /*layout_kv=*/ori_kv_layout,
        /*return_softmax_lse=*/false);
  }

  // 8) Deferred cache write for full prefill.
  if (use_temporary_prefill_kv && !cp_swa_route.has_value()) {
    scatter_by_slot(ori_kv,
                    ori_slot,
                    kv,
                    /*require_exact_rows=*/model_rows_are_sharded);
  }

  // 9) output RoPE + projection
  auto o = attn_output.view({-1, n_local_heads_, head_dim_});
  apply_partial_rope(
      o, nope_head_dim_, rope_head_dim_, cos, sin, /*inverse=*/true);

  const int64_t num_tokens = o.size(0);
  auto o_group = o.view({num_tokens, n_local_groups_, -1});
  auto wo_a = o_a_proj_->weight().view({n_local_groups_, o_lora_rank_, -1});
  auto o_low_rank = torch::einsum("tgd,grd->tgr", {o_group, wo_a});
  torch::Tensor output;
  const FlashComm1Context* fc1_ctx = get_current_flash_comm1_context();
  if (fc1_ctx && is_sequence_sharded(*fc1_ctx)) {
    output = o_b_proj_->forward(o_low_rank.reshape({num_tokens, -1}),
                                row_parallel_reduce_mode_for_fc1(*fc1_ctx));
  } else {
    output = o_b_proj_->forward(o_low_rank.reshape({num_tokens, -1}));
  }
  std::optional<torch::Tensor> final_lse = std::nullopt;
  (void)output_lse;

  return std::make_tuple(output, final_lse);
}

void DSAttentionImpl::load_state_dict(const StateDict& state_dict) {
  q_a_proj_->load_state_dict(state_dict.get_dict_with_prefix("wq_a."));
  q_b_proj_->load_state_dict(state_dict.get_dict_with_prefix("wq_b."));
  q_layernorm_->load_state_dict(state_dict.get_dict_with_prefix("q_norm."));

  kv_proj_->load_state_dict(state_dict.get_dict_with_prefix("wkv."));
  kv_layernorm_->load_state_dict(state_dict.get_dict_with_prefix("kv_norm."));
  o_a_proj_->load_state_dict(state_dict.get_dict_with_prefix("wo_a."));
  o_b_proj_->load_state_dict(state_dict.get_dict_with_prefix("wo_b."));

  auto attn_sink = state_dict.get_tensor("attn_sink");
  if (!attn_sink.defined()) {
    attn_sink = state_dict.get_tensor("attn_sink.weight");
  }
  if (attn_sink.defined()) {
    if (attn_sink.dim() == 1 && attn_sink.size(0) == num_heads_ &&
        tp_size_ > 1) {
      CHECK_EQ(num_heads_ % tp_size_, 0)
          << "attn_sink full-head tensor size is not divisible by tp_size.";
      const int64_t shard_size = num_heads_ / tp_size_;
      const int64_t shard_start = tp_rank_ * shard_size;
      attn_sink = attn_sink.slice(/*dim=*/0,
                                  /*start=*/shard_start,
                                  /*end=*/shard_start + shard_size);
    }

    CHECK(attn_sink.dim() == 1 && attn_sink.size(0) == n_local_heads_)
        << "attn_sink shape mismatch, expected [" << n_local_heads_ << "], got "
        << attn_sink.sizes();

    torch::NoGradGuard no_grad;
    attn_sink_.copy_(attn_sink.to(attn_sink_.device()).to(attn_sink_.dtype()));
    attn_sink_loaded_ = true;
  }

  if (compressor_ && compress_ratio_ >= 4) {
    auto compressor_state = state_dict.get_dict_with_prefix("compressor.");
    if (compressor_state.size() == 0) {
      compressor_state = state_dict.get_dict_with_prefix("compress.");
    }
    if (compressor_state.size() > 0) {
      compressor_->load_state_dict(compressor_state);
    }
  }

  if (indexer_ && compress_ratio_ == 4) {
    indexer_->load_state_dict(state_dict.get_dict_with_prefix("indexer."));
  }
}

int64_t DSAttentionImpl::non_registered_weight_bytes() const {
  if (!compressor_) {
    return 0;
  }
  return compressor_->weight_bytes();
}

}  // namespace layer
}  // namespace xllm
