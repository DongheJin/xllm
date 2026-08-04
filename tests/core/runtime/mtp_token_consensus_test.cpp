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

#include "runtime/mtp_token_consensus.h"

#include <gtest/gtest.h>

#include "framework/parallel_state/parallel_args.h"

namespace xllm {
namespace {

class RecordingBroadcastGroup final : public ProcessGroup {
 public:
  RecordingBroadcastGroup(int32_t world_size, int32_t source_token)
      : ProcessGroup(/*rank=*/0, world_size, torch::Device(torch::kCPU)),
        source_token_(source_token) {}

  void broadcast(torch::Tensor& input, int32_t root_rank) override {
    ++broadcast_calls_;
    last_root_rank_ = root_rank;
    input.fill_(source_token_);
  }

  int32_t broadcast_calls() const { return broadcast_calls_; }
  int32_t last_root_rank() const { return last_root_rank_; }

 private:
  int32_t source_token_;
  int32_t broadcast_calls_ = 0;
  int32_t last_root_rank_ = -1;
};

TEST(MtpTokenConsensusTest, BroadcastsGreedyPrefillTokenAcrossCpRanks) {
  RecordingBroadcastGroup world_group(/*world_size=*/8,
                                      /*source_token=*/270);
  RecordingBroadcastGroup cp_group(/*world_size=*/8, /*source_token=*/270);
  RecordingBroadcastGroup tp_group(/*world_size=*/1, /*source_token=*/270);
  ParallelArgs parallel_args(/*rank=*/1,
                             /*world_size=*/8,
                             /*dp_size=*/1,
                             /*cp_size=*/8,
                             &world_group,
                             /*ep_size=*/8);
  parallel_args.cp_group_ = &cp_group;
  parallel_args.tp_group_ = &tp_group;
  torch::Tensor tokens = torch::tensor({778}, torch::kInt32);

  broadcast_mtp_tokens(tokens, parallel_args);

  EXPECT_EQ(tokens.item<int32_t>(), 270);
  EXPECT_EQ(cp_group.broadcast_calls(), 1);
  EXPECT_EQ(cp_group.last_root_rank(), 0);
  EXPECT_EQ(tp_group.broadcast_calls(), 0);
  EXPECT_EQ(world_group.broadcast_calls(), 0);
}

TEST(MtpTokenConsensusTest, SkipsCommunicationWithoutCpParallelism) {
  RecordingBroadcastGroup world_group(/*world_size=*/1,
                                      /*source_token=*/270);
  RecordingBroadcastGroup cp_group(/*world_size=*/1, /*source_token=*/270);
  ParallelArgs parallel_args(/*rank=*/0,
                             /*world_size=*/1,
                             /*dp_size=*/1,
                             /*cp_size=*/1,
                             &world_group,
                             /*ep_size=*/1);
  parallel_args.cp_group_ = &cp_group;
  torch::Tensor tokens = torch::tensor({778}, torch::kInt32);

  broadcast_mtp_tokens(tokens, parallel_args);

  EXPECT_EQ(tokens.item<int32_t>(), 778);
  EXPECT_EQ(cp_group.broadcast_calls(), 0);
  EXPECT_EQ(world_group.broadcast_calls(), 0);
}

TEST(MtpTokenConsensusTest, BroadcastsAcrossTpWhenCpSizeIsOne) {
  RecordingBroadcastGroup world_group(/*world_size=*/4,
                                      /*source_token=*/270);
  RecordingBroadcastGroup tp_group(/*world_size=*/4, /*source_token=*/270);
  RecordingBroadcastGroup cp_group(/*world_size=*/1, /*source_token=*/270);
  ParallelArgs parallel_args(/*rank=*/1,
                             /*world_size=*/4,
                             /*dp_size=*/1,
                             /*cp_size=*/1,
                             &world_group,
                             /*ep_size=*/1);
  parallel_args.tp_group_ = &tp_group;
  parallel_args.cp_group_ = &cp_group;
  torch::Tensor tokens = torch::tensor({778}, torch::kInt32);

  broadcast_mtp_tokens(tokens, parallel_args);

  EXPECT_EQ(tokens.item<int32_t>(), 270);
  EXPECT_EQ(tp_group.broadcast_calls(), 1);
  EXPECT_EQ(tp_group.last_root_rank(), 0);
  EXPECT_EQ(cp_group.broadcast_calls(), 0);
  EXPECT_EQ(world_group.broadcast_calls(), 0);
}

TEST(MtpTokenConsensusTest, PropagatesRootThroughOrthogonalTpAndCpGroups) {
  RecordingBroadcastGroup world_group(/*world_size=*/8,
                                      /*source_token=*/270);
  RecordingBroadcastGroup tp_group(/*world_size=*/2, /*source_token=*/270);
  RecordingBroadcastGroup cp_group(/*world_size=*/4, /*source_token=*/270);
  ParallelArgs parallel_args(/*rank=*/3,
                             /*world_size=*/8,
                             /*dp_size=*/1,
                             /*cp_size=*/4,
                             &world_group,
                             /*ep_size=*/8);
  parallel_args.tp_group_ = &tp_group;
  parallel_args.cp_group_ = &cp_group;
  torch::Tensor tokens = torch::tensor({778}, torch::kInt32);

  broadcast_mtp_tokens(tokens, parallel_args, /*root_rank=*/0);

  EXPECT_EQ(tokens.item<int32_t>(), 270);
  EXPECT_EQ(tp_group.broadcast_calls(), 1);
  EXPECT_EQ(cp_group.broadcast_calls(), 1);
  EXPECT_EQ(world_group.broadcast_calls(), 0);
}

}  // namespace
}  // namespace xllm
