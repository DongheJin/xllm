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

namespace xllm {

struct ModelArgs;
class ProcessGroup;

namespace layer {

struct Dsv4CpMemoryBudgetConfig {
  int64_t persistent_cache_bytes = 0;
  int64_t global_real_token_count = 0;
  int64_t local_padded_token_count = 0;
  int32_t cp_size = 1;
  std::vector<int64_t> projection_widths;
  int64_t swa_width = 0;
  int64_t hidden_size = 0;
  int64_t model_dtype_size = 0;
  int64_t collective_workspace_bytes = 0;
  int64_t moe_operator_workspace_bytes = 0;
  bool requires_moe_bridge = false;
  CpProjectionGatherMode gather_mode = CpProjectionGatherMode::BUNDLED;
};

struct Dsv4CpMemoryBudget {
  int64_t persistent_cache_bytes = 0;
  int64_t local_projection_bytes = 0;
  int64_t projection_gather_bytes = 0;
  int64_t global_projection_bytes = 0;
  int64_t swa_gather_bytes = 0;
  int64_t moe_bridge_bytes = 0;
  int64_t collective_workspace_bytes = 0;
  int64_t moe_operator_workspace_bytes = 0;
  int64_t peak_transient_bytes = 0;
  int64_t required_bytes = 0;
};

struct Dsv4CpMemoryBudgetDecision {
  CpProjectionGatherMode gather_mode = CpProjectionGatherMode::BUNDLED;
  Dsv4CpMemoryBudget budget;
};

class Dsv4CpMemoryBudgetEstimator final {
 public:
  static Dsv4CpMemoryBudget estimate(const Dsv4CpMemoryBudgetConfig& config);

  static Dsv4CpMemoryBudgetConfig make_worst_case_config(
      const ModelArgs& model_args,
      int64_t max_tokens_per_batch,
      int64_t max_seqs_per_batch,
      int32_t cp_size,
      int64_t model_dtype_size,
      bool requires_moe_bridge);

  static int64_t max_local_padded_token_count(int64_t max_tokens_per_batch,
                                              int64_t max_seqs_per_batch,
                                              int32_t cp_size);

  static Dsv4CpMemoryBudgetDecision select_gather_mode(
      const Dsv4CpMemoryBudgetConfig& config,
      int64_t available_bytes,
      int64_t reserved_bytes = 0);

  static void require_fits(const Dsv4CpMemoryBudget& budget,
                           int64_t available_bytes,
                           int64_t reserved_bytes = 0);
};

// Per-forward collective adapter. It binds a non-owning CP process group and
// deliberately owns no model weight, cache, state, or long-lived tensor.
class Dsv4CpExecutionContext final {
 public:
  explicit Dsv4CpExecutionContext(
      ProcessGroup* cp_group,
      CpProjectionGatherMode gather_mode = CpProjectionGatherMode::BUNDLED);

  torch::Tensor gather_global_rows(const CpRowLayout& layout,
                                   const torch::Tensor& local_rows) const;

  std::vector<torch::Tensor> gather_projection_bundle(
      const CpRowLayout& layout,
      torch::TensorList local_projections) const;

  ProcessGroup* process_group() const { return cp_group_; }
  CpProjectionGatherMode gather_mode() const { return gather_mode_; }

 private:
  ProcessGroup* cp_group_ = nullptr;
  CpProjectionGatherMode gather_mode_ = CpProjectionGatherMode::BUNDLED;
};

}  // namespace layer
}  // namespace xllm
