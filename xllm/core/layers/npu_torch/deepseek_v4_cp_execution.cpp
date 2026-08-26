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

#include "layers/npu_torch/deepseek_v4_cp_execution.h"

#include <glog/logging.h>

#include <algorithm>
#include <initializer_list>
#include <limits>
#include <vector>

#include "framework/model/model_args.h"
#include "framework/parallel_state/process_group.h"

namespace xllm::layer {
namespace {

int64_t checked_product(std::initializer_list<int64_t> factors,
                        const char* name) {
  __int128 result = 1;
  for (int64_t factor : factors) {
    CHECK_GE(factor, 0) << name << " contains a negative factor";
    result *= static_cast<__int128>(factor);
    if (result > static_cast<__int128>(std::numeric_limits<int64_t>::max())) {
      LOG(FATAL) << name << " exceeds int64 byte accounting";
    }
  }
  return static_cast<int64_t>(result);
}

int64_t checked_sum(const std::vector<int64_t>& values, const char* name) {
  __int128 result = 0;
  for (int64_t value : values) {
    CHECK_GE(value, 0) << name << " contains a negative value";
    result += static_cast<__int128>(value);
    if (result > static_cast<__int128>(std::numeric_limits<int64_t>::max())) {
      LOG(FATAL) << name << " exceeds int64 byte accounting";
    }
  }
  return static_cast<int64_t>(result);
}

int64_t projection_gather_bytes(int64_t global_token_count,
                                int64_t gathered_token_count,
                                int64_t projection_width,
                                int64_t dtype_size) {
  const int64_t global_bytes =
      checked_product({global_token_count, projection_width, dtype_size},
                      "global projection gather bytes");
  const int64_t gathered_bytes =
      checked_product({gathered_token_count, projection_width, dtype_size},
                      "padded projection gather bytes");
  return std::max(
      checked_product({/*stacked and concatenated=*/2, gathered_bytes},
                      "projection collective bytes"),
      checked_sum({gathered_bytes, global_bytes}, "projection restore bytes"));
}

int64_t model_row_gather_bytes(int64_t local_token_count,
                               int64_t global_token_count,
                               int64_t gathered_token_count,
                               int64_t width,
                               int64_t dtype_size) {
  const int64_t local_bytes = checked_product(
      {local_token_count, width, dtype_size}, "local row gather bytes");
  const int64_t global_bytes = checked_product(
      {global_token_count, width, dtype_size}, "global row gather bytes");
  const int64_t gathered_bytes = checked_product(
      {gathered_token_count, width, dtype_size}, "padded row gather bytes");
  return checked_sum(
      {local_bytes,
       std::max(
           checked_product({/*stacked and concatenated=*/2, gathered_bytes},
                           "row collective bytes"),
           checked_sum({gathered_bytes, global_bytes}, "row restore bytes"))},
      "row gather peak bytes");
}

}  // namespace

Dsv4CpMemoryBudget Dsv4CpMemoryBudgetEstimator::estimate(
    const Dsv4CpMemoryBudgetConfig& config) {
  CHECK_GE(config.persistent_cache_bytes, 0);
  CHECK_GE(config.global_real_token_count, 0);
  CHECK_GE(config.local_padded_token_count, 0);
  CHECK_GT(config.cp_size, 1);
  CHECK_GT(config.model_dtype_size, 0);
  CHECK_GE(config.collective_workspace_bytes, 0);
  CHECK_GE(config.moe_operator_workspace_bytes, 0);

  const int64_t gathered_padded_token_count =
      checked_product({config.cp_size, config.local_padded_token_count},
                      "gathered padded token count");

  std::vector<int64_t> local_projection_bytes;
  std::vector<int64_t> global_projection_bytes;
  local_projection_bytes.reserve(config.projection_widths.size());
  global_projection_bytes.reserve(config.projection_widths.size());
  for (int64_t width : config.projection_widths) {
    CHECK_GT(width, 0) << "DSV4 CP projection width must be positive";
    local_projection_bytes.emplace_back(checked_product(
        {config.local_padded_token_count, width, config.model_dtype_size},
        "local projection bytes"));
    global_projection_bytes.emplace_back(checked_product(
        {config.global_real_token_count, width, config.model_dtype_size},
        "global projection bytes"));
  }

  Dsv4CpMemoryBudget budget;
  budget.persistent_cache_bytes = config.persistent_cache_bytes;
  if (config.gather_mode == CpProjectionGatherMode::BUNDLED) {
    budget.local_projection_bytes =
        checked_sum(local_projection_bytes, "local projection bundle bytes");
    budget.global_projection_bytes =
        checked_sum(global_projection_bytes, "global projection bundle bytes");
    const int64_t bundled_width =
        checked_sum(config.projection_widths, "projection bundle width");
    if (bundled_width > 0) {
      // torch::cat creates a local bundle in addition to the individual local
      // projection tensors before the bundled AllGather starts.
      budget.projection_gather_bytes =
          checked_sum({budget.local_projection_bytes,
                       projection_gather_bytes(config.global_real_token_count,
                                               gathered_padded_token_count,
                                               bundled_width,
                                               config.model_dtype_size)},
                      "bundled projection gather bytes");
    }
  } else {
    budget.local_projection_bytes =
        local_projection_bytes.empty()
            ? 0
            : *std::max_element(local_projection_bytes.begin(),
                                local_projection_bytes.end());
    budget.global_projection_bytes =
        global_projection_bytes.empty()
            ? 0
            : *std::max_element(global_projection_bytes.begin(),
                                global_projection_bytes.end());
    const int64_t largest_width =
        config.projection_widths.empty()
            ? 0
            : *std::max_element(config.projection_widths.begin(),
                                config.projection_widths.end());
    if (largest_width > 0) {
      budget.projection_gather_bytes =
          projection_gather_bytes(config.global_real_token_count,
                                  gathered_padded_token_count,
                                  largest_width,
                                  config.model_dtype_size);
    }
  }

  budget.swa_gather_bytes =
      model_row_gather_bytes(config.local_padded_token_count,
                             config.global_real_token_count,
                             gathered_padded_token_count,
                             config.swa_width,
                             config.model_dtype_size);
  if (config.requires_moe_bridge) {
    const int64_t global_input_output_bytes =
        checked_product({/*global input and output=*/2,
                         config.global_real_token_count,
                         config.hidden_size,
                         config.model_dtype_size},
                        "MoE global input/output bytes");
    budget.moe_bridge_bytes =
        checked_sum({model_row_gather_bytes(config.local_padded_token_count,
                                            config.global_real_token_count,
                                            gathered_padded_token_count,
                                            config.hidden_size,
                                            config.model_dtype_size),
                     global_input_output_bytes},
                    "MoE bridge bytes");
  }
  budget.collective_workspace_bytes = config.collective_workspace_bytes;
  budget.moe_operator_workspace_bytes = config.moe_operator_workspace_bytes;
  budget.peak_transient_bytes = checked_sum({budget.local_projection_bytes,
                                             budget.projection_gather_bytes,
                                             budget.global_projection_bytes,
                                             budget.swa_gather_bytes,
                                             budget.moe_bridge_bytes,
                                             budget.collective_workspace_bytes,
                                             budget.moe_operator_workspace_bytes},
                                            "DSV4 CP peak transient bytes");
  budget.required_bytes =
      checked_sum({budget.persistent_cache_bytes, budget.peak_transient_bytes},
                  "DSV4 CP required bytes");
  return budget;
}

Dsv4CpMemoryBudgetConfig Dsv4CpMemoryBudgetEstimator::make_worst_case_config(
    const ModelArgs& model_args,
    int64_t max_tokens_per_batch,
    int64_t max_seqs_per_batch,
    int32_t cp_size,
    int64_t model_dtype_size,
    bool requires_moe_bridge) {
  CHECK_GT(model_args.head_dim(), 0);
  CHECK_GT(model_args.hidden_size(), 0);
  CHECK_GT(model_dtype_size, 0);

  bool has_c4_layer = false;
  bool has_c128_layer = false;
  for (int32_t ratio : model_args.compress_ratios()) {
    has_c4_layer = has_c4_layer || ratio == 4;
    has_c128_layer = has_c128_layer || ratio == 128;
  }

  std::vector<int64_t> projection_widths;
  const int64_t c128_width = has_c128_layer ? 2 * model_args.head_dim() : 0;
  const int64_t c4_main_width = has_c4_layer ? 4 * model_args.head_dim() : 0;
  const int64_t c4_index_width =
      has_c4_layer ? 4 * model_args.index_head_dim() : 0;
  if (has_c4_layer && c4_main_width + c4_index_width >= c128_width) {
    CHECK_GT(model_args.index_head_dim(), 0);
    projection_widths = {c4_main_width, c4_index_width};
  } else if (has_c128_layer) {
    projection_widths = {c128_width};
  }

  Dsv4CpMemoryBudgetConfig config;
  // Persistent DSV4 cache/state is already owned by kv_cache_estimation. This
  // budget reserves only per-forward CP transient memory before that estimator
  // consumes the remaining capacity.
  config.persistent_cache_bytes = 0;
  config.global_real_token_count = max_tokens_per_batch;
  config.local_padded_token_count = max_local_padded_token_count(
      max_tokens_per_batch, max_seqs_per_batch, cp_size);
  config.cp_size = cp_size;
  config.projection_widths = std::move(projection_widths);
  config.swa_width = model_args.head_dim();
  config.hidden_size = model_args.hidden_size();
  config.model_dtype_size = model_dtype_size;
  // ProcessGroup AllGather has no caller-owned workspace. Its persistent HCCL
  // buffers are initialized before WorkerImpl takes the free-memory snapshot;
  // gather outputs and concatenation tensors are accounted above.
  config.collective_workspace_bytes = 0;
  config.requires_moe_bridge = requires_moe_bridge;
  return config;
}

int64_t Dsv4CpMemoryBudgetEstimator::max_local_padded_token_count(
    int64_t max_tokens_per_batch,
    int64_t max_seqs_per_batch,
    int32_t cp_size) {
  CHECK_GT(max_tokens_per_batch, 0);
  CHECK_GT(max_seqs_per_batch, 0);
  CHECK_GT(cp_size, 1);
  const int64_t active_sequences =
      std::min(max_tokens_per_batch, max_seqs_per_batch);
  const int64_t extra_chunks =
      (max_tokens_per_batch - active_sequences) / (2 * cp_size);
  return checked_product(
      {/*front and back chunks=*/2, active_sequences + extra_chunks},
      "maximum local padded token count");
}

Dsv4CpMemoryBudgetDecision Dsv4CpMemoryBudgetEstimator::select_gather_mode(
    const Dsv4CpMemoryBudgetConfig& config,
    int64_t available_bytes,
    int64_t reserved_bytes) {
  CHECK_GE(available_bytes, 0);
  CHECK_GE(reserved_bytes, 0);
  CHECK_LE(reserved_bytes, available_bytes);
  const int64_t usable_bytes = available_bytes - reserved_bytes;

  Dsv4CpMemoryBudgetConfig selected_config = config;
  selected_config.gather_mode = CpProjectionGatherMode::BUNDLED;
  Dsv4CpMemoryBudget bundled_budget = estimate(selected_config);
  if (bundled_budget.required_bytes <= usable_bytes) {
    return {CpProjectionGatherMode::BUNDLED, std::move(bundled_budget)};
  }

  selected_config.gather_mode = CpProjectionGatherMode::SEQUENTIAL;
  Dsv4CpMemoryBudget sequential_budget = estimate(selected_config);
  require_fits(sequential_budget, available_bytes, reserved_bytes);
  return {CpProjectionGatherMode::SEQUENTIAL, std::move(sequential_budget)};
}

void Dsv4CpMemoryBudgetEstimator::require_fits(const Dsv4CpMemoryBudget& budget,
                                               int64_t available_bytes,
                                               int64_t reserved_bytes) {
  CHECK_GE(available_bytes, 0);
  CHECK_GE(reserved_bytes, 0);
  CHECK_LE(reserved_bytes, available_bytes);
  const int64_t usable_bytes = available_bytes - reserved_bytes;
  CHECK_LE(budget.required_bytes, usable_bytes)
      << "DeepSeek V4 CP memory budget does not fit: required="
      << budget.required_bytes << ", usable=" << usable_bytes
      << ", available=" << available_bytes << ", reserved=" << reserved_bytes;
}

Dsv4CpExecutionContext::Dsv4CpExecutionContext(
    ProcessGroup* cp_group,
    CpProjectionGatherMode gather_mode)
    : cp_group_(cp_group), gather_mode_(gather_mode) {
  CHECK(cp_group_ != nullptr)
      << "DeepSeek V4 CP execution requires a process group";
}

torch::Tensor Dsv4CpExecutionContext::gather_global_rows(
    const CpRowLayout& layout,
    const torch::Tensor& local_rows) const {
  return layout.gather_global_rows(local_rows, cp_group_);
}

std::vector<torch::Tensor> Dsv4CpExecutionContext::gather_projection_bundle(
    const CpRowLayout& layout,
    torch::TensorList local_projections) const {
  CHECK(!local_projections.empty());
  if (gather_mode_ == CpProjectionGatherMode::BUNDLED) {
    return layout.gather_global_rows_bundle(local_projections, cp_group_);
  }

  std::vector<torch::Tensor> global_projections;
  global_projections.reserve(local_projections.size());
  for (const torch::Tensor& projection : local_projections) {
    global_projections.emplace_back(
        layout.gather_global_rows(projection, cp_group_));
  }
  return global_projections;
}

}  // namespace xllm::layer
