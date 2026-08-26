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

#include "layers/npu_torch/deepseek_v4_cp_attention_exchange.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "framework/parallel_state/process_group.h"

namespace xllm::layer {
namespace {

class FixedAllgatherProcessGroup final : public ProcessGroup {
 public:
  explicit FixedAllgatherProcessGroup(std::vector<torch::Tensor> outputs)
      : ProcessGroup(/*rank=*/0,
                     /*world_size=*/2,
                     torch::Device(torch::kCPU)),
        outputs_(std::move(outputs)) {}

  torch::Tensor allgather_base_sync(const torch::Tensor& input) override {
    CHECK_LT(call_count_, static_cast<int64_t>(outputs_.size()));
    const torch::Tensor& output = outputs_[call_count_++];
    CHECK_EQ(output.scalar_type(), input.scalar_type());
    CHECK_EQ(output.size(0), world_size());
    CHECK_EQ(output.sizes().slice(1), input.sizes());
    return output.clone();
  }

  int64_t call_count() const { return call_count_; }

 private:
  std::vector<torch::Tensor> outputs_;
  int64_t call_count_ = 0;
};

class FixedAttentionProcessGroup final : public ProcessGroup {
 public:
  FixedAttentionProcessGroup(torch::Tensor gathered_lse,
                             torch::Tensor expected_local_output,
                             torch::Tensor reduced_output)
      : ProcessGroup(/*rank=*/0,
                     /*world_size=*/2,
                     torch::Device(torch::kCPU)),
        gathered_lse_(std::move(gathered_lse)),
        expected_local_output_(std::move(expected_local_output)),
        reduced_output_(std::move(reduced_output)) {}

  torch::Tensor allgather_base_sync(const torch::Tensor& input) override {
    ++allgather_count_;
    CHECK_EQ(gathered_lse_.scalar_type(), input.scalar_type());
    CHECK_EQ(gathered_lse_.size(0), world_size());
    CHECK_EQ(gathered_lse_.sizes().slice(1), input.sizes());
    return gathered_lse_.clone();
  }

  void allreduce(torch::Tensor& input) override {
    ++allreduce_count_;
    CHECK(torch::allclose(input, expected_local_output_));
    input.copy_(reduced_output_);
  }

  int64_t allgather_count() const { return allgather_count_; }

  int64_t allreduce_count() const { return allreduce_count_; }

 private:
  torch::Tensor gathered_lse_;
  torch::Tensor expected_local_output_;
  torch::Tensor reduced_output_;
  int64_t allgather_count_ = 0;
  int64_t allreduce_count_ = 0;
};

CpPlanInput make_input(int32_t query_length) {
  CpPlanInput input;
  input.q_seq_lens = {query_length};
  input.position_ids = torch::arange(query_length, torch::kInt32);
  return input;
}

std::vector<Dsv4CpOwnershipPlan> make_cp4_plans() {
  constexpr int32_t kCpSize = 4;
  constexpr int32_t kQueryLength = 19;
  const CpPlanInput input = make_input(kQueryLength);
  const torch::Tensor state_block_table =
      torch::tensor({{2, 3}}, torch::TensorOptions().dtype(torch::kInt64));
  const torch::Tensor compressed_block_table =
      torch::tensor({{1}}, torch::TensorOptions().dtype(torch::kInt64));
  Dsv4CpOwnershipPlanner planner;
  std::vector<Dsv4CpOwnershipPlan> plans;
  plans.reserve(kCpSize);
  for (int32_t cp_rank = 0; cp_rank < kCpSize; ++cp_rank) {
    const CpRowLayout layout =
        CpRowLayout::build(input, kCpSize, cp_rank, torch::Device(torch::kCPU));
    plans.emplace_back(planner.build(layout,
                                     /*global_q_seq_lens=*/{kQueryLength},
                                     /*global_kv_seq_lens=*/{kQueryLength},
                                     state_block_table,
                                     compressed_block_table,
                                     /*compress_ratio=*/4,
                                     /*state_block_size=*/16,
                                     /*compressed_block_size=*/8,
                                     torch::Device(torch::kCPU)));
  }
  return plans;
}

TEST(Dsv4CpAttentionExchangeTest, PacksAndCanonicallyUnpacksAllRanks) {
  constexpr int32_t kCpSize = 4;
  constexpr int64_t kWidth = 2;
  const std::vector<Dsv4CpOwnershipPlan> plans = make_cp4_plans();
  std::vector<torch::Tensor> packed_sends;
  packed_sends.reserve(kCpSize);
  for (int32_t source_rank = 0; source_rank < kCpSize; ++source_rank) {
    const int64_t local_row_count = 6;
    torch::Tensor local_rows = torch::empty(
        {local_row_count, kWidth}, torch::TensorOptions().dtype(torch::kInt64));
    for (int64_t row = 0; row < local_row_count; ++row) {
      local_rows[row][0] = source_rank;
      local_rows[row][1] = row;
    }
    packed_sends.emplace_back(Dsv4CpAttentionExchange::pack_send_rows(
        local_rows, plans[source_rank].main_route()));
  }

  int64_t total_received_rows = 0;
  for (int32_t destination_rank = 0; destination_rank < kCpSize;
       ++destination_rank) {
    const Dsv4CpRouteDescriptor& route = plans[destination_rank].main_route();
    std::vector<torch::Tensor> source_segments;
    source_segments.reserve(kCpSize);
    for (int32_t source_rank = 0; source_rank < kCpSize; ++source_rank) {
      const int64_t segment_begin =
          destination_rank * route.padded_rows_per_peer;
      const int64_t segment_end = segment_begin + route.padded_rows_per_peer;
      source_segments.emplace_back(packed_sends[source_rank].slice(
          /*dim=*/0, segment_begin, segment_end));
    }
    torch::Tensor received_fixed =
        torch::cat(source_segments, /*dim=*/0).contiguous();
    torch::Tensor canonical =
        Dsv4CpAttentionExchange::unpack_received_rows(received_fixed, route);
    const Dsv4CpOwnerMetadata& metadata =
        plans[destination_rank].owner_metadata();
    ASSERT_EQ(canonical.size(0), metadata.real_row_count);
    ASSERT_EQ(canonical.size(1), kWidth);
    total_received_rows += canonical.size(0);

    const auto canonical_accessor = canonical.accessor<int64_t, 2>();
    const int64_t* canonical_slots =
        metadata.recv_canonical_indices.data_ptr<int64_t>();
    const int32_t* expected_source_ranks =
        route.recv_source_ranks.data_ptr<int32_t>();
    for (int64_t row = 0; row < canonical.size(0); ++row) {
      const int64_t fixed_slot = canonical_slots[row];
      EXPECT_EQ(canonical_accessor[row][0], expected_source_ranks[fixed_slot]);
      EXPECT_GE(canonical_accessor[row][1], 0);
    }
  }
  EXPECT_EQ(total_received_rows, 19);
}

TEST(Dsv4CpAttentionExchangeTest, KeepsEmptyOwnerTensorShapeStable) {
  const std::vector<Dsv4CpOwnershipPlan> plans = make_cp4_plans();
  const Dsv4CpRouteDescriptor& empty_route = plans[0].main_route();
  ASSERT_EQ(empty_route.recv_real_row_count, 0);
  torch::Tensor received = torch::zeros(
      {empty_route.recv_source_ranks.numel(), 7}, torch::kBFloat16);

  torch::Tensor unpacked =
      Dsv4CpAttentionExchange::unpack_received_rows(received, empty_route);

  EXPECT_EQ(unpacked.dim(), 2);
  EXPECT_EQ(unpacked.size(0), 0);
  EXPECT_EQ(unpacked.size(1), 7);
  EXPECT_EQ(unpacked.scalar_type(), torch::kBFloat16);
}

TEST(Dsv4CpAttentionExchangeTest, SelectsOwnerCompressedRopeRows) {
  const torch::Tensor global_rows =
      torch::arange(20, torch::kFloat32).view({5, 4});
  Dsv4CpOwnerMetadata metadata;
  metadata.output_rope_indices = torch::tensor({3, 0, 4}, torch::kInt64);
  metadata.output_row_count = 3;

  const torch::Tensor owner_rows =
      Dsv4CpAttentionExchange::select_owner_output_rows(global_rows, metadata);

  EXPECT_TRUE(torch::equal(
      owner_rows,
      torch::stack({global_rows[3], global_rows[0], global_rows[4]})));
}

TEST(Dsv4CpAttentionExchangeTest, WritesOnlyCanonicalOwnerCacheSlots) {
  torch::Tensor cache = torch::zeros({3, 4, 1, 2}, torch::kFloat32);
  const torch::Tensor received_rows =
      torch::tensor({{1.0f, 2.0f}, {3.0f, 4.0f}}, torch::kFloat32);
  Dsv4CpRouteDescriptor route;
  route.recv_canonical_local_slots = torch::tensor({5, 8}, torch::kInt64);
  route.recv_real_row_count = 2;

  Dsv4CpAttentionExchange::write_received_cache_rows(
      cache, received_rows, route);

  const torch::Tensor cache_rows = cache.view({12, 2});
  EXPECT_TRUE(torch::equal(cache_rows[5], received_rows[0]));
  EXPECT_TRUE(torch::equal(cache_rows[8], received_rows[1]));
  EXPECT_EQ(cache.count_nonzero().item<int64_t>(), 4);
}

TEST(Dsv4CpAttentionExchangeTest, KeepsEmptyOwnerCacheUnmodified) {
  torch::Tensor cache = torch::ones({2, 4, 1, 2}, torch::kFloat32);
  const torch::Tensor before = cache.clone();
  const torch::Tensor received_rows = torch::empty({0, 2}, torch::kFloat32);
  Dsv4CpRouteDescriptor route;
  route.recv_canonical_local_slots = torch::empty({0}, torch::kInt64);

  Dsv4CpAttentionExchange::write_received_cache_rows(
      cache, received_rows, route);

  EXPECT_TRUE(torch::equal(cache, before));
}

TEST(Dsv4CpAttentionExchangeTest, MergesAttentionPartialsWithFp32Lse) {
  torch::Tensor scores =
      torch::tensor({{{1000.0f, 999.0f, 998.0f}, {1.0f, 2.0f, 3.0f}},
                     {{997.0f, 996.0f, 995.0f}, {-3.0f, -2.0f, -1.0f}}},
                    torch::kFloat32);
  torch::Tensor values =
      torch::tensor({{{1.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 1.0f}},
                     {{2.0f, 0.0f}, {0.0f, 2.0f}, {2.0f, 2.0f}}},
                    torch::kFloat32);
  std::vector<torch::Tensor> local_outputs;
  std::vector<torch::Tensor> local_lse;
  local_outputs.reserve(2);
  local_lse.reserve(2);
  for (int64_t shard = 0; shard < 2; ++shard) {
    torch::Tensor shard_scores = scores[shard];
    local_outputs.emplace_back(
        torch::softmax(shard_scores, /*dim=*/-1).matmul(values[shard]));
    local_lse.emplace_back(
        torch::logsumexp(shard_scores, /*dim=*/-1, /*keepdim=*/true));
  }
  torch::Tensor partial_outputs = torch::stack(local_outputs, /*dim=*/0);
  torch::Tensor partial_lse = torch::stack(local_lse, /*dim=*/0);

  const Dsv4CpAttentionMergeResult merged =
      Dsv4CpAttentionExchange::merge_gathered_attention_partials(
          partial_outputs, partial_lse);

  torch::Tensor global_scores = scores.permute({1, 0, 2}).reshape({2, 6});
  torch::Tensor global_values =
      values.unsqueeze(/*dim=*/0).expand({2, 2, 3, 2}).reshape({2, 6, 2});
  torch::Tensor expected_output =
      torch::bmm(torch::softmax(global_scores, /*dim=*/-1).unsqueeze(1),
                 global_values)
          .squeeze(1);
  torch::Tensor expected_lse =
      torch::logsumexp(global_scores, /*dim=*/-1, /*keepdim=*/true);
  EXPECT_TRUE(torch::allclose(merged.output,
                              expected_output,
                              /*rtol=*/1e-5,
                              /*atol=*/1e-6));
  EXPECT_TRUE(torch::allclose(merged.lse,
                              expected_lse,
                              /*rtol=*/1e-5,
                              /*atol=*/1e-6));
}

TEST(Dsv4CpAttentionExchangeTest, HandlesAllEmptyAttentionOwners) {
  torch::Tensor partial_outputs = torch::ones({4, 3, 2}, torch::kBFloat16);
  torch::Tensor partial_lse =
      torch::full({4, 3, 1},
                  -std::numeric_limits<float>::infinity(),
                  torch::TensorOptions().dtype(torch::kFloat32));

  const Dsv4CpAttentionMergeResult merged =
      Dsv4CpAttentionExchange::merge_gathered_attention_partials(
          partial_outputs, partial_lse);

  EXPECT_EQ(merged.output.scalar_type(), torch::kBFloat16);
  EXPECT_EQ(merged.output.count_nonzero().item<int64_t>(), 0);
  EXPECT_TRUE(torch::isneginf(merged.lse).all().item<bool>());
}

TEST(Dsv4CpAttentionExchangeTest,
     IgnoresNonFiniteOutputFromEmptyAttentionOwner) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float negative_infinity =
      -std::numeric_limits<float>::infinity();
  const torch::Tensor partial_outputs =
      torch::tensor({{{nan, nan}}, {{2.0f, 3.0f}}}, torch::kFloat32);
  const torch::Tensor partial_lse =
      torch::tensor({{{negative_infinity}}, {{0.0f}}}, torch::kFloat32);

  const Dsv4CpAttentionMergeResult merged =
      Dsv4CpAttentionExchange::merge_gathered_attention_partials(
          partial_outputs, partial_lse);

  EXPECT_TRUE(torch::isfinite(merged.output).all().item<bool>());
  EXPECT_TRUE(torch::equal(
      merged.output, torch::tensor({{2.0f, 3.0f}}, torch::kFloat32)));
  EXPECT_TRUE(torch::equal(merged.lse, torch::tensor({{0.0f}})));
}

TEST(Dsv4CpAttentionExchangeTest, MergesQliCandidatesDeterministically) {
  torch::Tensor candidate_indices =
      torch::tensor({{{9, 3, 7}, {8, 2, 6}}, {{5, 1, 4}, {7, 3, -1}}},
                    torch::TensorOptions().dtype(torch::kInt64));
  torch::Tensor candidate_scores =
      torch::tensor({{{2.0f, 4.0f, 4.0f}, {1.0f, 3.0f, 3.0f}},
                     {{4.0f, 4.0f, 2.0f}, {3.0f, 3.0f, 100.0f}}},
                    torch::TensorOptions().dtype(torch::kFloat32));

  const Dsv4CpQliMergeResult merged =
      Dsv4CpAttentionExchange::merge_gathered_qli_candidates(
          candidate_indices, candidate_scores, /*global_topk=*/4);

  EXPECT_TRUE(
      torch::equal(merged.indices,
                   torch::tensor({{1, 3, 5, 7}, {2, 3, 6, 7}}, torch::kInt64)));
  EXPECT_TRUE(torch::equal(
      merged.scores,
      torch::tensor({{4.0f, 4.0f, 4.0f, 4.0f}, {3.0f, 3.0f, 3.0f, 3.0f}},
                    torch::kFloat32)));
}

TEST(Dsv4CpAttentionExchangeTest, PreservesQliHeadDimensionDuringMerge) {
  const torch::Tensor candidate_indices =
      torch::tensor({{{{1, 3}, {8, 6}}}, {{{2, 4}, {7, 5}}}}, torch::kInt64);
  const torch::Tensor candidate_scores = torch::tensor(
      {{{{1.0f, 3.0f}, {8.0f, 6.0f}}}, {{{2.0f, 4.0f}, {7.0f, 5.0f}}}},
      torch::kFloat32);

  const Dsv4CpQliMergeResult merged =
      Dsv4CpAttentionExchange::merge_gathered_qli_candidates(
          candidate_indices, candidate_scores, /*global_topk=*/2);

  EXPECT_EQ(merged.indices.sizes(), torch::IntArrayRef({1, 2, 2}));
  EXPECT_TRUE(
      torch::equal(merged.indices,
                   torch::tensor({{{4, 3}, {8, 7}}},
                                 torch::TensorOptions().dtype(torch::kInt64))));
}

TEST(Dsv4CpAttentionExchangeTest, RunsFixedAttentionCollectiveSequence) {
  torch::Tensor gathered_lse =
      torch::tensor({{{0.0f}}, {{0.0f}}}, torch::kFloat32);
  FixedAttentionProcessGroup process_group(
      gathered_lse,
      /*expected_local_output=*/torch::tensor({{0.5f, 1.0f}}, torch::kFloat32),
      /*reduced_output=*/torch::tensor({{2.0f, 3.0f}}, torch::kFloat32));
  Dsv4CpAttentionExchange exchange(&process_group);

  const Dsv4CpAttentionMergeResult merged = exchange.merge_attention_partials(
      torch::tensor({{1.0f, 2.0f}}, torch::kFloat32), gathered_lse[0]);

  EXPECT_EQ(process_group.allgather_count(), 1);
  EXPECT_EQ(process_group.allreduce_count(), 1);
  EXPECT_TRUE(torch::allclose(merged.output,
                              torch::tensor({{2.0f, 3.0f}}, torch::kFloat32)));
  EXPECT_TRUE(torch::allclose(
      merged.lse, torch::tensor({{std::log(2.0f)}}, torch::kFloat32)));
}

TEST(Dsv4CpAttentionExchangeTest,
     DistributedMergeIgnoresNonFiniteOutputFromEmptyOwner) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float negative_infinity =
      -std::numeric_limits<float>::infinity();
  const torch::Tensor gathered_lse =
      torch::tensor({{{negative_infinity}}, {{0.0f}}}, torch::kFloat32);
  FixedAttentionProcessGroup process_group(
      gathered_lse,
      /*expected_local_output=*/torch::zeros({1, 2}, torch::kFloat32),
      /*reduced_output=*/torch::tensor({{2.0f, 3.0f}}, torch::kFloat32));
  Dsv4CpAttentionExchange exchange(&process_group);

  const Dsv4CpAttentionMergeResult merged = exchange.merge_attention_partials(
      torch::tensor({{nan, nan}}, torch::kFloat32), gathered_lse[0]);

  EXPECT_TRUE(torch::isfinite(merged.output).all().item<bool>());
  EXPECT_TRUE(torch::equal(
      merged.output, torch::tensor({{2.0f, 3.0f}}, torch::kFloat32)));
  EXPECT_TRUE(torch::equal(merged.lse, torch::tensor({{0.0f}})));
}

TEST(Dsv4CpAttentionExchangeTest, ReplicatesAndSelectsQueryRowsBySourceRank) {
  torch::Tensor gathered =
      torch::tensor({{{10.0f, 11.0f}, {12.0f, 13.0f}, {14.0f, 15.0f}},
                     {{20.0f, 21.0f}, {22.0f, 23.0f}, {24.0f, 25.0f}}},
                    torch::kFloat32);
  FixedAllgatherProcessGroup process_group({gathered});
  Dsv4CpAttentionExchange exchange(&process_group);

  torch::Tensor local_rows = gathered[1].contiguous();
  torch::Tensor replicated = exchange.replicate_query_rows(local_rows);

  EXPECT_EQ(process_group.call_count(), 1);
  EXPECT_TRUE(torch::equal(replicated, gathered));
  EXPECT_TRUE(torch::equal(
      Dsv4CpAttentionExchange::select_query_rank_rows(replicated, 0),
      gathered[0]));
  EXPECT_TRUE(torch::equal(
      Dsv4CpAttentionExchange::select_query_rank_rows(replicated, 1),
      gathered[1]));
}

TEST(Dsv4CpAttentionExchangeTest, BuildsPaddedQuerySequenceIndices) {
  const torch::Tensor sequence_indices =
      Dsv4CpAttentionExchange::build_query_sequence_indices(
          /*local_padded_seq_lens=*/{2, 0, 3}, torch::Device(torch::kCPU));
  EXPECT_TRUE(torch::equal(sequence_indices,
                           torch::tensor({0, 0, 2, 2, 2}, torch::kInt64)));
}

TEST(Dsv4CpAttentionExchangeTest,
     BuildsOwnerIndexMapAndTranslatesMultiSequenceIndices) {
  const torch::Tensor block_table =
      torch::tensor({{0, 1, 2, 3}, {8, 9, 10, 11}}, torch::kInt64);
  const Dsv4CpOwnerIndexMap index_map =
      Dsv4CpAttentionExchange::build_owner_index_map(
          block_table,
          /*global_seq_lens=*/{7, 5},
          /*block_size=*/2,
          /*cp_size=*/4,
          /*cp_rank=*/2,
          torch::Device(torch::kCPU));

  EXPECT_TRUE(torch::equal(index_map.local_block_table,
                           torch::tensor({{0}, {2}}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(index_map.local_key_seq_lens,
                           torch::tensor({2, 1}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(index_map.local_to_global_indices,
                           torch::tensor({{4, 5}, {4, -1}}, torch::kInt64)));
  EXPECT_TRUE(torch::equal(
      index_map.global_to_local_indices,
      torch::tensor({{-1, -1, -1, -1, 0, 1, -1}, {-1, -1, -1, -1, 0, -1, -1}},
                    torch::kInt64)));

  const torch::Tensor query_sequence_indices =
      torch::tensor({0, 1}, torch::kInt64);
  const torch::Tensor local_indices =
      torch::tensor({{0, 1}, {0, 1}}, torch::kInt64);
  EXPECT_TRUE(
      torch::equal(Dsv4CpAttentionExchange::map_owner_local_indices_to_global(
                       local_indices, query_sequence_indices, index_map),
                   torch::tensor({{4, 5}, {4, -1}}, torch::kInt64)));

  const torch::Tensor global_indices =
      torch::tensor({{4, 6}, {4, 1}}, torch::kInt64);
  EXPECT_TRUE(
      torch::equal(Dsv4CpAttentionExchange::map_global_indices_to_owner_local(
                       global_indices, query_sequence_indices, index_map),
                   torch::tensor({{0, -1}, {0, -1}}, torch::kInt64)));
}

TEST(Dsv4CpAttentionExchangeTest, BuildsReplicatedOwnerIndexBlockTable) {
  const torch::Tensor block_table =
      torch::tensor({{0, 1, 2, 3}, {8, 9, 10, 11}}, torch::kInt64);

  const Dsv4CpOwnerIndexMap index_map =
      Dsv4CpAttentionExchange::build_owner_index_map(
          block_table,
          /*global_seq_lens=*/{7, 5},
          /*block_size=*/2,
          /*cp_size=*/4,
          /*cp_rank=*/2,
          torch::Device(torch::kCPU),
          Dsv4CpCacheAddressing::REPLICATED);

  EXPECT_TRUE(torch::equal(index_map.local_block_table,
                           torch::tensor({{2}, {10}}, torch::kInt32)));
}

TEST(Dsv4CpAttentionExchangeTest, BuildsCausalOwnerQliVirtualSequences) {
  const torch::Tensor block_table =
      torch::tensor({{0, 1}, {8, 9}}, torch::kInt64);

  const Dsv4CpOwnerQliMetadata metadata =
      Dsv4CpAttentionExchange::build_owner_qli_metadata(
          block_table,
          /*global_q_seq_lens=*/{4, 2},
          /*global_kv_seq_lens=*/{12, 6},
          /*compressed_block_size=*/2,
          /*compress_ratio=*/4,
          /*cp_size=*/4,
          /*cp_rank=*/0,
          torch::Device(torch::kCPU));

  EXPECT_TRUE(torch::equal(metadata.query_sequence_indices,
                           torch::tensor({0, 0, 0, 0, 1, 1}, torch::kInt64)));
  EXPECT_TRUE(torch::equal(metadata.query_seq_endpoints,
                           torch::tensor({1, 2, 3, 4, 5, 6}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(metadata.local_key_seq_lens,
                           torch::tensor({8, 8, 8, 8, 4, 4}, torch::kInt32)));
  EXPECT_EQ(metadata.valid_query_row_count, 6);
  EXPECT_EQ(metadata.max_local_key_seq_len, 8);
  EXPECT_TRUE(metadata.valid_query_rows.all().item<bool>());
  EXPECT_TRUE(torch::equal(metadata.index_map.local_block_table,
                           torch::tensor({{0}, {2}}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(metadata.query_block_table,
                           torch::tensor({{0}, {0}, {0}, {0}, {2}, {2}},
                                         torch::kInt32)));
}

TEST(Dsv4CpAttentionExchangeTest, MarksQliRowsWithoutOwnerKeysInvalid) {
  const Dsv4CpOwnerQliMetadata metadata =
      Dsv4CpAttentionExchange::build_owner_qli_metadata(
          torch::tensor({{0}}, torch::kInt64),
          /*global_q_seq_lens=*/{4},
          /*global_kv_seq_lens=*/{4},
          /*compressed_block_size=*/128,
          /*compress_ratio=*/4,
          /*cp_size=*/4,
          /*cp_rank=*/3,
          torch::Device(torch::kCPU));

  EXPECT_EQ(metadata.index_map.local_block_table.sizes(),
            torch::IntArrayRef({1, 0}));
  EXPECT_EQ(metadata.query_block_table.sizes(), torch::IntArrayRef({4, 0}));
  EXPECT_EQ(metadata.local_key_seq_lens.count_nonzero().item<int64_t>(), 0);
  EXPECT_EQ(metadata.valid_query_rows.count_nonzero().item<int64_t>(), 0);
  EXPECT_EQ(metadata.valid_query_row_count, 0);
  EXPECT_EQ(metadata.max_local_key_seq_len, 0);
}

TEST(Dsv4CpAttentionExchangeTest, KeepsEmptyOwnerIndexMapAddressable) {
  const torch::Tensor block_table =
      torch::tensor({{0, 1}, {4, 5}}, torch::kInt64);
  const Dsv4CpOwnerIndexMap index_map =
      Dsv4CpAttentionExchange::build_owner_index_map(
          block_table,
          /*global_seq_lens=*/{3, 3},
          /*block_size=*/2,
          /*cp_size=*/4,
          /*cp_rank=*/3,
          torch::Device(torch::kCPU));

  EXPECT_EQ(index_map.local_block_table.sizes(), torch::IntArrayRef({2, 0}));
  EXPECT_EQ(index_map.local_to_global_indices.sizes(),
            torch::IntArrayRef({2, 0}));
  EXPECT_EQ(index_map.local_key_seq_lens.numel(), 2);
  const torch::Tensor query_sequence_indices =
      torch::tensor({0, 1}, torch::kInt64);
  const torch::Tensor local_indices =
      torch::tensor({{0, -1}, {1, 2}}, torch::kInt64);
  EXPECT_TRUE(
      torch::equal(Dsv4CpAttentionExchange::map_owner_local_indices_to_global(
                       local_indices, query_sequence_indices, index_map),
                   torch::full_like(local_indices, -1)));
}

TEST(Dsv4CpAttentionExchangeTest,
     CompactsInterleavedOwnerCandidatesBeforeSparseAttention) {
  const torch::Tensor owner_local_indices =
      torch::tensor({{{8, -1, 3, -1, 5}, {-1, 7, -1, 2, -1}}},
                    torch::kInt32);

  const torch::Tensor compacted =
      Dsv4CpAttentionExchange::compact_valid_indices(owner_local_indices);

  EXPECT_TRUE(torch::equal(
      compacted,
      torch::tensor({{{8, 3, 5, -1, -1}, {7, 2, -1, -1, -1}}},
                    torch::kInt32)));
}

TEST(Dsv4CpAttentionExchangeTest, RunsFixedQliCollectiveSequence) {
  torch::Tensor gathered_indices =
      torch::tensor({{{9, 3, 7}}, {{5, 1, 4}}}, torch::kInt64);
  torch::Tensor gathered_scores = torch::tensor(
      {{{2.0f, 4.0f, 4.0f}}, {{4.0f, 4.0f, 2.0f}}}, torch::kFloat32);
  FixedAllgatherProcessGroup process_group({gathered_indices, gathered_scores});
  Dsv4CpAttentionExchange exchange(&process_group);

  const Dsv4CpQliMergeResult merged = exchange.global_topk(
      gathered_indices[0], gathered_scores[0], /*global_topk=*/4);

  EXPECT_EQ(process_group.call_count(), 2);
  EXPECT_TRUE(torch::equal(merged.indices,
                           torch::tensor({{1, 3, 5, 7}}, torch::kInt64)));
  EXPECT_TRUE(
      torch::equal(merged.scores,
                   torch::tensor({{4.0f, 4.0f, 4.0f, 4.0f}}, torch::kFloat32)));
}

#if GTEST_HAS_DEATH_TEST
TEST(Dsv4CpAttentionExchangeTest, RejectsOutOfRangeSourceRows) {
  const std::vector<Dsv4CpOwnershipPlan> plans = make_cp4_plans();
  const Dsv4CpRouteDescriptor& route = plans[0].main_route();
  torch::Tensor too_short = torch::zeros({1, 4}, torch::kFloat32);
  EXPECT_DEATH(Dsv4CpAttentionExchange::pack_send_rows(too_short, route),
               "source row exceeds");
}
#endif

}  // namespace
}  // namespace xllm::layer
