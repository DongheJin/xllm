/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "core/distributed_runtime/overlap_result_collection.h"

#include <gtest/gtest.h>

namespace xllm {
namespace {

TEST(OverlapResultCollectionTest, SelectsOnlyCp8Tp1Driver) {
  EXPECT_EQ(overlap_result_collection_stride(/*dp_local_size=*/8,
                                             /*enable_eplb=*/false),
            8);
}

TEST(OverlapResultCollectionTest, SelectsOneDriverPerDpGroup) {
  const uint32_t stride = overlap_result_collection_stride(
      /*dp_local_size=*/4, /*enable_eplb=*/false);

  EXPECT_EQ(stride, 4);
  EXPECT_EQ(overlap_driver_result_index(/*worker_rank=*/0, stride), 0);
  EXPECT_EQ(overlap_driver_result_index(/*worker_rank=*/4, stride), 1);
}

TEST(OverlapResultCollectionTest, IncludesTpAndCpRanksInDriverStride) {
  // TP2 * CP4 workers belong to one DP-local group.
  EXPECT_EQ(overlap_result_collection_stride(/*dp_local_size=*/8,
                                             /*enable_eplb=*/false),
            8);
}

TEST(OverlapResultCollectionTest, CollectsEveryRankForEplb) {
  const uint32_t stride = overlap_result_collection_stride(
      /*dp_local_size=*/8, /*enable_eplb=*/true);

  EXPECT_EQ(stride, 1);
  EXPECT_EQ(overlap_driver_result_index(/*worker_rank=*/8, stride), 8);
}

}  // namespace
}  // namespace xllm
