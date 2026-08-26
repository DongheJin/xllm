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

#include "layers/npu_torch/deepseek_v4_cp_owner_attention.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace xllm::layer {
namespace {

Dsv4CpOwnerKeyShard make_key_shard(const torch::Tensor& values,
                                   const std::vector<int64_t>& sequences,
                                   const std::vector<int64_t>& logical_indices,
                                   const std::vector<int64_t>& positions,
                                   const std::vector<int64_t>& windowed,
                                   const std::vector<int64_t>& sparse_selected) {
  Dsv4CpOwnerKeyShard shard;
  shard.values = values;
  shard.sequence_indices = torch::tensor(sequences, torch::kInt64);
  shard.logical_indices = torch::tensor(logical_indices, torch::kInt64);
  shard.absolute_positions = torch::tensor(positions, torch::kInt64);
  shard.windowed = torch::tensor(windowed, torch::kInt64).to(torch::kBool);
  shard.sparse_selected =
      torch::tensor(sparse_selected, torch::kInt64).to(torch::kBool);
  return shard;
}

Dsv4CpAttentionMergeResult run_reference(
    const torch::Tensor& query,
    const Dsv4CpOwnerKeyShard& keys,
    const std::optional<torch::Tensor>& candidates = std::nullopt,
    int64_t window_left = -1,
    bool include_sinks = false) {
  Dsv4CpOwnerAttentionInput input;
  input.query = query;
  input.query_sequence_indices = torch::tensor({0, 1}, torch::kInt64);
  input.query_absolute_positions = torch::tensor({5, 3}, torch::kInt64);
  input.keys = keys;
  input.sparse_global_indices = candidates;
  input.window_left = window_left;
  input.softmax_scale = 1.0;
  input.include_sinks = include_sinks;
  if (include_sinks) {
    input.sinks = torch::tensor({0.25f, -0.5f}, torch::kFloat32);
  }
  return Dsv4CpOwnerAttentionReference::run(input);
}

TEST(Dsv4CpOwnerAttentionReferenceTest, BuildsCompactRowsFromOwnerPagedCache) {
  torch::Tensor cache = torch::arange(6 * 2 * 1 * 2, torch::kFloat32)
                            .view({6, 2, 1, 2});
  Dsv4CpOwnerIndexMap index_map;
  index_map.local_block_table =
      torch::tensor({{1, 3}, {4, 0}}, torch::kInt32);
  index_map.local_to_global_indices =
      torch::tensor({{0, 1, 4, 5}, {2, 3, -1, -1}}, torch::kInt64);

  const Dsv4CpOwnerKeyShard shard =
      Dsv4CpOwnerAttentionReference::build_paged_key_shard(
          cache,
          index_map,
          /*block_size=*/2,
          /*absolute_position_stride=*/4,
          /*absolute_position_offset=*/0,
          /*windowed=*/false,
          /*sparse_selected=*/true);

  EXPECT_EQ(shard.values.sizes(), torch::IntArrayRef({6, 1, 2}));
  EXPECT_TRUE(torch::equal(
      shard.values.squeeze(1),
      torch::tensor({{4.0f, 5.0f},
                     {6.0f, 7.0f},
                     {12.0f, 13.0f},
                     {14.0f, 15.0f},
                     {16.0f, 17.0f},
                     {18.0f, 19.0f}})));
  EXPECT_TRUE(torch::equal(shard.sequence_indices,
                           torch::tensor({0, 0, 0, 0, 1, 1}, torch::kInt64)));
  EXPECT_TRUE(torch::equal(shard.logical_indices,
                           torch::tensor({0, 1, 4, 5, 2, 3}, torch::kInt64)));
  EXPECT_TRUE(torch::equal(
      shard.absolute_positions,
      torch::tensor({0, 4, 16, 20, 8, 12}, torch::kInt64)));
  EXPECT_TRUE(shard.sparse_selected.all().item<bool>());
}

TEST(Dsv4CpOwnerPaCacheTest, SkipsEvictedSlidingWindowBlocks) {
  const torch::Tensor cache =
      torch::arange(8 * 2, torch::kFloat32).view({8, 2, 1, 1});
  const torch::Tensor block_table =
      torch::tensor({{-1, -1, 4, 5}}, torch::kInt32);

  const Dsv4CpOwnerPaCache owner = build_dsv4_cp_owner_pa_cache(
      cache,
      block_table,
      /*global_seq_lens=*/{8},
      /*cp_size=*/2,
      /*cp_rank=*/0,
      Dsv4CpBlockTableHolePolicy::SKIP_EVICTED);

  EXPECT_EQ(owner.cache.sizes(), torch::IntArrayRef({1, 2, 1, 1}));
  EXPECT_TRUE(torch::equal(owner.cache.flatten(),
                           torch::tensor({8.0f, 9.0f})));
  EXPECT_TRUE(torch::equal(owner.block_table,
                           torch::tensor({{0}}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(owner.local_seq_lens,
                           torch::tensor({2}, torch::kInt32)));
  ASSERT_EQ(owner.global_to_local.size(), 1u);
  EXPECT_EQ(owner.global_to_local[0],
            std::vector<int32_t>({-1, -1, -1, -1, 0, 1, -1, -1}));
}

#if GTEST_HAS_DEATH_TEST
TEST(Dsv4CpOwnerPaCacheTest, RejectsHolesInFullHistoryCache) {
  const torch::Tensor cache = torch::zeros({8, 2, 1, 1}, torch::kFloat32);
  const torch::Tensor block_table =
      torch::tensor({{-1, 1}}, torch::kInt32);

  EXPECT_DEATH(build_dsv4_cp_owner_pa_cache(
                   cache,
                   block_table,
                   /*global_seq_lens=*/{4},
                   /*cp_size=*/2,
                   /*cp_rank=*/0,
                   Dsv4CpBlockTableHolePolicy::REJECT),
               "full-history DSV4 block table entry cannot be negative");
}
#endif

TEST(Dsv4CpOwnerAttentionReferenceTest,
     C1OwnerPartialsMergeToGlobalMultiSequenceReference) {
  torch::Tensor query =
      torch::tensor({{{1.0f, 0.5f}, {0.5f, 1.0f}},
                     {{0.25f, 1.0f}, {1.0f, 0.25f}}},
                    torch::kFloat32);
  const Dsv4CpOwnerKeyShard owner0 = make_key_shard(
      torch::tensor({{{1.0f, 0.0f}},
                     {{0.0f, 1.0f}},
                     {{1.0f, 1.0f}},
                     {{0.5f, 1.5f}}}),
      {0, 0, 1, 1},
      {2, 4, 0, 2},
      {2, 4, 0, 2},
      {true, true, true, true},
      {false, false, false, false});
  const Dsv4CpOwnerKeyShard owner1 = make_key_shard(
      torch::tensor({{{1.5f, 0.5f}},
                     {{0.5f, 0.5f}},
                     {{1.5f, 1.0f}}}),
      {0, 0, 1},
      {3, 5, 1},
      {3, 5, 1},
      {true, true, true},
      {false, false, false});

  const Dsv4CpAttentionMergeResult partial0 =
      run_reference(query, owner0, std::nullopt, /*window_left=*/3);
  const Dsv4CpAttentionMergeResult partial1 =
      run_reference(query, owner1, std::nullopt, /*window_left=*/3);
  const Dsv4CpAttentionMergeResult merged =
      Dsv4CpAttentionExchange::merge_gathered_attention_partials(
          torch::stack({partial0.output, partial1.output}),
          torch::stack({partial0.lse, partial1.lse}));
  const Dsv4CpOwnerKeyShard global =
      Dsv4CpOwnerAttentionReference::concatenate_key_shards({owner0, owner1});
  const Dsv4CpAttentionMergeResult expected =
      run_reference(query, global, std::nullopt, /*window_left=*/3);

  EXPECT_TRUE(torch::allclose(merged.output, expected.output, 1e-5, 1e-6));
  EXPECT_TRUE(torch::allclose(merged.lse, expected.lse, 1e-5, 1e-6));
}

TEST(Dsv4CpOwnerAttentionReferenceTest,
     C4QliFilteringIsHeadSpecificAndSinkIsCountedOnce) {
  torch::Tensor query =
      torch::tensor({{{1.0f, 0.5f}, {0.5f, 1.0f}},
                     {{0.25f, 1.0f}, {1.0f, 0.25f}}},
                    torch::kFloat32);
  const Dsv4CpOwnerKeyShard owner0 = make_key_shard(
      torch::tensor({{{1.0f, 0.0f}},
                     {{0.0f, 1.0f}},
                     {{0.5f, 0.5f}},
                     {{1.0f, 1.0f}}}),
      {0, 1, 0, 1},
      {4, 2, 0, 0},
      {4, 2, 0, 0},
      {true, true, false, false},
      {false, false, true, true});
  const Dsv4CpOwnerKeyShard owner1 = make_key_shard(
      torch::tensor({{{1.5f, 0.5f}},
                     {{0.25f, 1.5f}},
                     {{1.5f, 1.0f}}}),
      {0, 0, 1},
      {5, 1, 1},
      {5, 4, 4},
      {true, false, false},
      {false, true, true});
  const torch::Tensor candidates =
      torch::tensor({{{0, -1}, {1, -1}}, {{0, -1}, {1, -1}}},
                    torch::kInt64);

  const Dsv4CpAttentionMergeResult partial0 = run_reference(
      query, owner0, candidates, /*window_left=*/1, /*include_sinks=*/true);
  const Dsv4CpAttentionMergeResult partial1 = run_reference(
      query, owner1, candidates, /*window_left=*/1, /*include_sinks=*/false);
  const Dsv4CpAttentionMergeResult merged =
      Dsv4CpAttentionExchange::merge_gathered_attention_partials(
          torch::stack({partial0.output, partial1.output}),
          torch::stack({partial0.lse, partial1.lse}));
  const Dsv4CpOwnerKeyShard global =
      Dsv4CpOwnerAttentionReference::concatenate_key_shards({owner0, owner1});
  const Dsv4CpAttentionMergeResult expected = run_reference(
      query, global, candidates, /*window_left=*/1, /*include_sinks=*/true);

  EXPECT_TRUE(torch::allclose(merged.output, expected.output, 1e-5, 1e-6));
  EXPECT_TRUE(torch::allclose(merged.lse, expected.lse, 1e-5, 1e-6));
}

TEST(Dsv4CpOwnerAttentionReferenceTest,
     C128SupportsEmptyOwnerAndExtremeScores) {
  torch::Tensor query =
      torch::tensor({{{1000.0f, 0.0f}, {0.0f, 1000.0f}},
                     {{0.0f, 1000.0f}, {1000.0f, 0.0f}}},
                    torch::kFloat32);
  const Dsv4CpOwnerKeyShard owner = make_key_shard(
      torch::tensor({{{1.0f, 0.0f}}, {{0.0f, 1.0f}}}),
      {0, 1},
      {0, 0},
      {0, 0},
      {false, false},
      {false, false});
  const Dsv4CpOwnerKeyShard empty = make_key_shard(
      torch::empty({0, 1, 2}, torch::kFloat32), {}, {}, {}, {}, {});

  const Dsv4CpAttentionMergeResult partial0 = run_reference(query, owner);
  const Dsv4CpAttentionMergeResult partial1 = run_reference(query, empty);
  EXPECT_TRUE(torch::isneginf(partial1.lse).all().item<bool>());
  EXPECT_EQ(partial1.output.count_nonzero().item<int64_t>(), 0);

  const Dsv4CpAttentionMergeResult merged =
      Dsv4CpAttentionExchange::merge_gathered_attention_partials(
          torch::stack({partial0.output, partial1.output}),
          torch::stack({partial0.lse, partial1.lse}));
  EXPECT_TRUE(torch::isfinite(merged.output).all().item<bool>());
  EXPECT_TRUE(torch::isfinite(merged.lse).all().item<bool>());
  EXPECT_TRUE(torch::allclose(merged.output, partial0.output, 1e-5, 1e-6));
}

TEST(Dsv4CpOwnerAttentionReferenceTest,
     HeadWithoutLocalQliCandidateStaysZeroAndFinite) {
  torch::Tensor query =
      torch::tensor({{{1.0f, 0.0f}, {0.0f, 1.0f}},
                     {{1.0f, 0.0f}, {0.0f, 1.0f}}},
                    torch::kFloat32);
  const Dsv4CpOwnerKeyShard owner = make_key_shard(
      torch::tensor({{{1.0f, 1.0f}}}),
      {0},
      {7},
      {4},
      {false},
      {true});
  const torch::Tensor candidates =
      torch::tensor({{{7}, {9}}, {{-1}, {-1}}}, torch::kInt64);

  const Dsv4CpAttentionMergeResult partial =
      run_reference(query, owner, candidates);

  EXPECT_TRUE(torch::isfinite(partial.output).all().item<bool>());
  EXPECT_EQ(partial.output[0][1].count_nonzero().item<int64_t>(), 0);
  EXPECT_TRUE(torch::isneginf(partial.lse[0][1]).all().item<bool>());
  EXPECT_EQ(partial.output[1].count_nonzero().item<int64_t>(), 0);
  EXPECT_TRUE(torch::isneginf(partial.lse[1]).all().item<bool>());
}

#if GTEST_HAS_DEATH_TEST
TEST(Dsv4CpOwnerAttentionReferenceTest, RejectsMismatchedHeadDimensions) {
  Dsv4CpOwnerAttentionInput input;
  input.query = torch::zeros({1, 2, 4}, torch::kFloat32);
  input.query_sequence_indices = torch::zeros({1}, torch::kInt64);
  input.query_absolute_positions = torch::zeros({1}, torch::kInt64);
  input.keys = make_key_shard(torch::zeros({1, 3, 4}, torch::kFloat32),
                              {0},
                              {0},
                              {0},
                              {false},
                              {false});
  EXPECT_DEATH(Dsv4CpOwnerAttentionReference::run(input),
               "heads must be shared or match");
}
#endif

}  // namespace
}  // namespace xllm::layer
