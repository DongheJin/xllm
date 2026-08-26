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

#include "deepseek_v4_cp_cache_layout.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>

namespace xllm {
namespace {

TEST(Dsv4CpCacheLayoutTest, MapsEveryBlockToOneOwner) {
  for (int32_t cp_size : {2, 4, 8}) {
    Dsv4CpCacheLayout layout(cp_size, 0);
    for (int64_t block = 0; block < 37; ++block) {
      EXPECT_EQ(layout.owner_rank(block), block % cp_size);
      EXPECT_EQ(layout.local_block_id(block), block / cp_size);
    }
  }
}

TEST(Dsv4CpCacheLayoutTest, LocalCountsHandleNonDivisibleGlobalCount) {
  for (int32_t cp_size : {2, 4, 8}) {
    std::array<int64_t, 8> counts{};
    for (int32_t rank = 0; rank < cp_size; ++rank) {
      counts[rank] = Dsv4CpCacheLayout(cp_size, rank).local_block_count(37);
    }
    int64_t sum = 0;
    for (int32_t rank = 0; rank < cp_size; ++rank) {
      EXPECT_EQ(counts[rank], 37 / cp_size + (rank < 37 % cp_size ? 1 : 0));
      sum += counts[rank];
    }
    EXPECT_EQ(sum, 37);
  }
}

TEST(Dsv4CpCacheLayoutTest, ResolvesReplicatedAndOwnerLocalPhysicalIds) {
  const Dsv4CpCacheLayout layout(/*cp_size=*/4, /*local_rank=*/2);
  EXPECT_EQ(layout.physical_block_id(
                10, Dsv4CpCacheAddressing::REPLICATED),
            10);
  EXPECT_EQ(layout.physical_block_id(
                10, Dsv4CpCacheAddressing::OWNER_LOCAL),
            2);
  EXPECT_EQ(layout.physical_slot(10,
                                 /*block_offset=*/7,
                                 /*block_size=*/16,
                                 Dsv4CpCacheAddressing::REPLICATED),
            167);
  EXPECT_EQ(layout.physical_slot(10,
                                 /*block_offset=*/7,
                                 /*block_size=*/16,
                                 Dsv4CpCacheAddressing::OWNER_LOCAL),
            39);
}

TEST(Dsv4CpCacheLayoutTest, MapsNonOwnerEntriesToInvalid) {
  const auto global = torch::tensor({0, 1, 2, 3, 4, 5, 6, 7, 8},
                                    torch::dtype(torch::kInt));
  const auto local = Dsv4CpCacheLayout(4, 2).map_global_block_table(global);
  const auto expected = torch::tensor({-1, -1, 0, -1, -1, -1, 1, -1, -1},
                                      torch::dtype(torch::kInt));
  EXPECT_TRUE(torch::equal(local, expected));
}

TEST(Dsv4CpCacheLayoutTest, PreservesShapeDtypeAndDevice) {
  const auto global =
      torch::tensor({0, 4, 1, 5}, torch::dtype(torch::kLong)).reshape({2, 2});
  const auto local = Dsv4CpCacheLayout(4, 1).map_global_block_table(global);
  EXPECT_EQ(local.sizes(), global.sizes());
  EXPECT_EQ(local.scalar_type(), global.scalar_type());
  EXPECT_EQ(local.device(), global.device());
  const auto expected =
      torch::tensor({-1, -1, 0, 1}, torch::dtype(torch::kLong))
          .reshape({2, 2});
  EXPECT_TRUE(torch::equal(local, expected));
}

TEST(Dsv4CpCacheLayoutTest, RejectsInvalidLayoutAndBlockIds) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_DEATH(Dsv4CpCacheLayout(0, 0), "cp_size > 0");
  EXPECT_DEATH(Dsv4CpCacheLayout(4, 4), "smaller than cp_size");

  Dsv4CpCacheLayout layout(4, 0);
  EXPECT_DEATH(layout.owner_rank(-1), "global_block_id");
  EXPECT_DEATH(layout.local_block_id(-1), "global_block_id");
  EXPECT_DEATH(layout.local_block_count(-1), "global_block_count");
  EXPECT_DEATH(layout.physical_block_id(
                   1, Dsv4CpCacheAddressing::OWNER_LOCAL),
               "non-owner block");
  EXPECT_DEATH(
      layout.map_global_block_table(torch::tensor({0, -1})),
      "cannot contain negative block ids");
}

}  // namespace
}  // namespace xllm
