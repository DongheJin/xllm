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

#include <string>

#include "framework/parallel_state/parallel_args.h"

namespace xllm {

enum class RowParallelReduceMode : int8_t {
  NONE = 0,
  ALL_REDUCE = 1,
  REDUCE_SCATTER = 2,
  MATMUL_REDUCE_SCATTER = 3,
};

struct FlashComm1Context {
  bool enabled = false;
  int32_t tp_rank = 0;
  int32_t tp_world_size = 1;
  int32_t original_num_tokens = 0;
  int32_t padded_num_tokens = 0;
  int32_t padded_local_num_tokens = 0;
  int32_t pad_size = 0;
  bool enable_mmrs_fusion = false;
  std::string mmrs_comm_mode = "aiv";
  ProcessGroup* tp_group = nullptr;
};

struct FlashComm1Options {
  bool enable_flashcomm1 = false;
  int32_t min_prefill_tokens = 8192;
  bool enable_mmrs_fusion = false;
  std::string mmrs_comm_mode = "aiv";
};

// Token geometry for composing an outer model-side CP shard with the inner
// FlashComm1 TP sequence shard.
struct FlashComm1TokenGeometry {
  int32_t global_num_tokens = 0;
  int32_t local_num_tokens = 0;
  bool cp_has_empty_rank = false;
};

FlashComm1TokenGeometry flash_comm1_token_geometry_without_cp(
    int32_t num_tokens);

class FlashComm1ContextScope {
 public:
  explicit FlashComm1ContextScope(const FlashComm1Context* ctx);
  ~FlashComm1ContextScope();

  FlashComm1ContextScope(const FlashComm1ContextScope&) = delete;
  FlashComm1ContextScope& operator=(const FlashComm1ContextScope&) = delete;

 private:
  const FlashComm1Context* previous_;
};

const FlashComm1Context* get_current_flash_comm1_context();

bool is_sequence_sharded(const FlashComm1Context& ctx);

torch::Tensor pad_rows_by_copy(const torch::Tensor& input, int64_t padded_rows);

// Topology/config gate for FC1, independent of the process group and platform.
// FC1 shards the sequence over the TP group. CP-aware callers must apply the CP
// shard first and provide both the pre-CP threshold count and post-CP local row
// count. This keeps CP as the outer token shard and FlashComm1 as the inner one.
bool is_flash_comm1_eligible(const FlashComm1TokenGeometry& geometry,
                             bool is_prefill,
                             const ParallelArgs& parallel_args,
                             const FlashComm1Options& options);

// Legacy interface for callers that do not provide CP-local geometry. It keeps
// rejecting cp_size > 1 to prevent accidental double sharding.
bool is_flash_comm1_eligible(int32_t num_tokens,
                             bool is_prefill,
                             const ParallelArgs& parallel_args,
                             const FlashComm1Options& options);

FlashComm1Context build_flash_comm1_context(
    const FlashComm1TokenGeometry& geometry,
    bool is_prefill,
    const ParallelArgs& parallel_args,
    const FlashComm1Options& options);

FlashComm1Context build_flash_comm1_context(int32_t num_tokens,
                                            bool is_prefill,
                                            const ParallelArgs& parallel_args,
                                            const FlashComm1Options& options);

torch::Tensor shard_sequence(const torch::Tensor& input,
                             const FlashComm1Context& ctx);

torch::Tensor gather_sequence(const torch::Tensor& input,
                              const FlashComm1Context& ctx);

torch::Tensor maybe_pad_and_reduce(torch::Tensor input,
                                   const FlashComm1Context& ctx,
                                   RowParallelReduceMode mode);

RowParallelReduceMode row_parallel_reduce_mode_for_fc1(
    const FlashComm1Context& ctx);

torch::Tensor maybe_shard_residual(const torch::Tensor& residual,
                                   const FlashComm1Context& ctx);

}  // namespace xllm
