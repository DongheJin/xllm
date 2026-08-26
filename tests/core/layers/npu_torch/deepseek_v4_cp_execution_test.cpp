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

#include <gtest/gtest.h>

#include <limits>

#include "framework/model/model_args.h"
#include "framework/parallel_state/process_group.h"
#include "layers/npu_torch/deepseek_v4_eplb_utils.h"

namespace xllm::layer {
namespace {

Dsv4CpMemoryBudgetConfig base_config() {
  Dsv4CpMemoryBudgetConfig config;
  config.persistent_cache_bytes = 1000;
  config.global_real_token_count = 32;
  config.local_padded_token_count = 8;
  config.cp_size = 4;
  config.projection_widths = {2048, 512};
  config.swa_width = 512;
  config.hidden_size = 7168;
  config.model_dtype_size = 2;
  config.collective_workspace_bytes = 4096;
  config.moe_operator_workspace_bytes = 8192;
  config.requires_moe_bridge = true;
  return config;
}

class CountingAllGatherProcessGroup final : public ProcessGroup {
 public:
  CountingAllGatherProcessGroup()
      : ProcessGroup(/*rank=*/0,
                     /*world_size=*/2,
                     torch::Device(torch::kCPU)) {}

  torch::Tensor allgather_base_sync(const torch::Tensor& input) override {
    ++allgather_calls_;
    return torch::stack({input.clone(), input.clone()}, /*dim=*/0);
  }

  int32_t allgather_calls() const { return allgather_calls_; }

 private:
  int32_t allgather_calls_ = 0;
};

CpRowLayout make_cp2_rank0_layout() {
  CpPlanInput input;
  input.q_seq_lens = {4};
  input.position_ids = torch::arange(4, torch::kInt32);
  return CpRowLayout::build(
      input, /*cp_size=*/2, /*cp_rank=*/0, torch::Device(torch::kCPU));
}

TEST(Dsv4CpMemoryBudgetEstimatorTest, AccountsBundledProjectionPeak) {
  Dsv4CpMemoryBudgetConfig config = base_config();
  config.gather_mode = CpProjectionGatherMode::BUNDLED;

  const Dsv4CpMemoryBudget budget =
      Dsv4CpMemoryBudgetEstimator::estimate(config);

  EXPECT_EQ(budget.local_projection_bytes, 8 * (2048 + 512) * 2);
  EXPECT_EQ(budget.global_projection_bytes, 32 * (2048 + 512) * 2);
  EXPECT_GT(budget.projection_gather_bytes, budget.global_projection_bytes);
  EXPECT_GT(budget.swa_gather_bytes, 32 * 512 * 2);
  EXPECT_GT(budget.moe_bridge_bytes, 2 * 32 * 7168 * 2);
  EXPECT_EQ(budget.moe_operator_workspace_bytes, 8192);
  EXPECT_EQ(budget.required_bytes,
            budget.persistent_cache_bytes + budget.peak_transient_bytes);
}

TEST(Dsv4CpMemoryBudgetEstimatorTest, SequentialModeUsesLargestProjection) {
  Dsv4CpMemoryBudgetConfig config = base_config();
  config.gather_mode = CpProjectionGatherMode::SEQUENTIAL;

  const Dsv4CpMemoryBudget budget =
      Dsv4CpMemoryBudgetEstimator::estimate(config);

  EXPECT_EQ(budget.local_projection_bytes, 8 * 2048 * 2);
  EXPECT_EQ(budget.global_projection_bytes, 32 * 2048 * 2);

  Dsv4CpMemoryBudgetConfig bundled_config = base_config();
  bundled_config.gather_mode = CpProjectionGatherMode::BUNDLED;
  const Dsv4CpMemoryBudget bundled_budget =
      Dsv4CpMemoryBudgetEstimator::estimate(bundled_config);
  EXPECT_LT(budget.peak_transient_bytes, bundled_budget.peak_transient_bytes);
}

TEST(Dsv4CpMemoryBudgetEstimatorTest, RejectsInsufficientCapacity) {
#if GTEST_HAS_DEATH_TEST
  const Dsv4CpMemoryBudget budget =
      Dsv4CpMemoryBudgetEstimator::estimate(base_config());
  EXPECT_DEATH(Dsv4CpMemoryBudgetEstimator::require_fits(
                   budget, budget.required_bytes - 1),
               "does not fit");
#endif
}

TEST(Dsv4CpMemoryBudgetEstimatorTest, AcceptsReservedCapacityBoundary) {
  const Dsv4CpMemoryBudget budget =
      Dsv4CpMemoryBudgetEstimator::estimate(base_config());
  EXPECT_NO_FATAL_FAILURE(
      Dsv4CpMemoryBudgetEstimator::require_fits(budget,
                                                budget.required_bytes + 4096,
                                                /*reserved_bytes=*/4096));
}

TEST(Dsv4CpMemoryBudgetEstimatorTest, SelectsSequentialAtBundledBoundary) {
  Dsv4CpMemoryBudgetConfig bundled_config = base_config();
  bundled_config.gather_mode = CpProjectionGatherMode::BUNDLED;
  const Dsv4CpMemoryBudget bundled_budget =
      Dsv4CpMemoryBudgetEstimator::estimate(bundled_config);

  Dsv4CpMemoryBudgetConfig sequential_config = base_config();
  sequential_config.gather_mode = CpProjectionGatherMode::SEQUENTIAL;
  const Dsv4CpMemoryBudget sequential_budget =
      Dsv4CpMemoryBudgetEstimator::estimate(sequential_config);
  ASSERT_LT(sequential_budget.required_bytes, bundled_budget.required_bytes);

  const Dsv4CpMemoryBudgetDecision decision =
      Dsv4CpMemoryBudgetEstimator::select_gather_mode(
          base_config(), bundled_budget.required_bytes - 1);
  EXPECT_EQ(decision.gather_mode, CpProjectionGatherMode::SEQUENTIAL);
  EXPECT_EQ(decision.budget.required_bytes, sequential_budget.required_bytes);
}

TEST(Dsv4CpMemoryBudgetEstimatorTest, RejectsWhenSequentialDoesNotFit) {
#if GTEST_HAS_DEATH_TEST
  Dsv4CpMemoryBudgetConfig config = base_config();
  config.gather_mode = CpProjectionGatherMode::SEQUENTIAL;
  const Dsv4CpMemoryBudget sequential_budget =
      Dsv4CpMemoryBudgetEstimator::estimate(config);
  EXPECT_DEATH(Dsv4CpMemoryBudgetEstimator::select_gather_mode(
                   base_config(), sequential_budget.required_bytes - 1),
               "does not fit");
#endif
}

TEST(Dsv4CpMemoryBudgetEstimatorTest,
     BoundsPerSequenceZigzagPaddingAtMaxBatchShape) {
  EXPECT_EQ(Dsv4CpMemoryBudgetEstimator::max_local_padded_token_count(
                /*max_tokens_per_batch=*/32,
                /*max_seqs_per_batch=*/1,
                /*cp_size=*/8),
            4);
  EXPECT_EQ(Dsv4CpMemoryBudgetEstimator::max_local_padded_token_count(
                /*max_tokens_per_batch=*/32,
                /*max_seqs_per_batch=*/2,
                /*cp_size=*/8),
            6);
  EXPECT_EQ(Dsv4CpMemoryBudgetEstimator::max_local_padded_token_count(
                /*max_tokens_per_batch=*/8,
                /*max_seqs_per_batch=*/256,
                /*cp_size=*/8),
            16);
}

TEST(Dsv4CpMemoryBudgetEstimatorTest,
     BuildsWorstCaseConfigFromLargestLayerProjectionBundle) {
  ModelArgs model_args;
  model_args.head_dim(512);
  model_args.index_head_dim(128);
  model_args.hidden_size(4096);
  model_args.compress_ratios({1, 4, 128});

  const Dsv4CpMemoryBudgetConfig config =
      Dsv4CpMemoryBudgetEstimator::make_worst_case_config(
          model_args,
          /*max_tokens_per_batch=*/32768,
          /*max_seqs_per_batch=*/256,
          /*cp_size=*/8,
          /*model_dtype_size=*/2,
          /*requires_moe_bridge=*/true);

  EXPECT_EQ(config.global_real_token_count, 32768);
  EXPECT_EQ(config.local_padded_token_count, 4576);
  EXPECT_EQ(config.projection_widths, (std::vector<int64_t>{2048, 512}));
  EXPECT_EQ(config.swa_width, 512);
  EXPECT_TRUE(config.requires_moe_bridge);
  EXPECT_EQ(config.moe_operator_workspace_bytes, 0);
}

TEST(Dsv4CpMemoryBudgetEstimatorTest,
     ReservesDispatchFfnCombineWorkspaceFromCpLocalCapacity) {
  constexpr int64_t kMiB = 1024 * 1024;
  EXPECT_EQ(dsv4_eplb::dispatch_ffn_max_output_size(
                /*local_tokens=*/3072,
                /*topk=*/6,
                /*ep_world_size=*/8),
            147456);
  EXPECT_EQ(dsv4_eplb::dispatch_ffn_workspace_reservation_bytes(
                /*local_tokens=*/530,
                /*topk=*/6,
                /*ep_world_size=*/8,
                /*hidden_size=*/4096,
                /*local_experts=*/32),
            316 * kMiB);
}

TEST(Dsv4CpMemoryBudgetEstimatorTest,
     RejectsInvalidDispatchFfnCombineWorkspaceShape) {
  EXPECT_EQ(dsv4_eplb::dispatch_ffn_workspace_reservation_bytes(
                /*local_tokens=*/0,
                /*topk=*/6,
                /*ep_world_size=*/8,
                /*hidden_size=*/4096,
                /*local_experts=*/32),
            0);
  EXPECT_EQ(dsv4_eplb::dispatch_ffn_workspace_reservation_bytes(
                std::numeric_limits<int64_t>::max(),
                /*topk=*/6,
                /*ep_world_size=*/8,
                /*hidden_size=*/4096,
                /*local_experts=*/32),
            0);
}

TEST(Dsv4CpMemoryBudgetEstimatorTest,
     IdentifiesW8A8DynamicQuantizationCaseInsensitively) {
  EXPECT_TRUE(
      dsv4_eplb::is_w8a8_dynamic_quantize_type("w8a8_dynamic"));
  EXPECT_TRUE(
      dsv4_eplb::is_w8a8_dynamic_quantize_type("W8A8_DYNAMIC"));
  EXPECT_FALSE(
      dsv4_eplb::is_w8a8_dynamic_quantize_type("w4a8_dynamic"));
  EXPECT_FALSE(dsv4_eplb::is_w8a8_dynamic_quantize_type(""));
}

TEST(Dsv4CpExecutionContextTest, BundledProjectionUsesOneCollective) {
  const CpRowLayout layout = make_cp2_rank0_layout();
  CountingAllGatherProcessGroup process_group;
  Dsv4CpExecutionContext context(&process_group,
                                 CpProjectionGatherMode::BUNDLED);
  const torch::Tensor first =
      torch::ones({layout.local_padded_token_count(), 2}, torch::kBFloat16);
  const torch::Tensor second =
      torch::ones({layout.local_padded_token_count(), 3}, torch::kBFloat16);

  const std::vector<torch::Tensor> outputs =
      context.gather_projection_bundle(layout, {first, second});

  ASSERT_EQ(outputs.size(), 2);
  EXPECT_EQ(outputs[0].sizes(), torch::IntArrayRef({4, 2}));
  EXPECT_EQ(outputs[1].sizes(), torch::IntArrayRef({4, 3}));
  EXPECT_TRUE(outputs[0].is_contiguous());
  EXPECT_TRUE(outputs[1].is_contiguous());
  EXPECT_EQ(outputs[0].scalar_type(), torch::kBFloat16);
  EXPECT_EQ(outputs[1].scalar_type(), torch::kBFloat16);
  EXPECT_EQ(process_group.allgather_calls(), 1);
}

TEST(Dsv4CpExecutionContextTest, SequentialProjectionUsesOrderedCollectives) {
  const CpRowLayout layout = make_cp2_rank0_layout();
  CountingAllGatherProcessGroup process_group;
  Dsv4CpExecutionContext context(&process_group,
                                 CpProjectionGatherMode::SEQUENTIAL);
  const torch::Tensor first =
      torch::ones({layout.local_padded_token_count(), 2}, torch::kBFloat16);
  const torch::Tensor second =
      torch::ones({layout.local_padded_token_count(), 3}, torch::kBFloat16);

  const std::vector<torch::Tensor> outputs =
      context.gather_projection_bundle(layout, {first, second});

  ASSERT_EQ(outputs.size(), 2);
  EXPECT_EQ(outputs[0].sizes(), torch::IntArrayRef({4, 2}));
  EXPECT_EQ(outputs[1].sizes(), torch::IntArrayRef({4, 3}));
  EXPECT_EQ(outputs[0].scalar_type(), torch::kBFloat16);
  EXPECT_EQ(outputs[1].scalar_type(), torch::kBFloat16);
  EXPECT_EQ(process_group.allgather_calls(), 2);
}

}  // namespace
}  // namespace xllm::layer
