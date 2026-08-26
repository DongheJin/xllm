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

#include "layers/npu_torch/deepseek_v4_cp_wavefront.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <utility>
#include <vector>

namespace xllm::layer {
namespace {

TEST(Dsv4CpWavefrontTest, EmitsDiagonalOrderAndStableDoubleBufferSlots) {
  Dsv4CpWavefrontScheduler scheduler;
  const std::vector<Dsv4CpWavefrontNode> nodes =
      scheduler.build(/*layer_count=*/3,
                      /*microchunk_count=*/2,
                      /*compression_alignment=*/4);

  ASSERT_EQ(nodes.size(), 6u);
  const std::vector<std::pair<int32_t, int32_t>> expected = {
      {0, 0}, {0, 1}, {1, 0}, {1, 1}, {2, 0}, {2, 1}};
  for (size_t index = 0; index < nodes.size(); ++index) {
    EXPECT_EQ(std::make_pair(nodes[index].layer_id, nodes[index].microchunk_id),
              expected[index]);
    EXPECT_EQ(nodes[index].buffer_slot,
              (nodes[index].layer_id + nodes[index].microchunk_id) % 2);
    EXPECT_EQ(nodes[index].collective_sequence_base,
              static_cast<int64_t>(index) * 4);
  }
}

TEST(Dsv4CpWavefrontTest, EmitsFullTwoDimensionalWavefront) {
  Dsv4CpWavefrontScheduler scheduler;
  const std::vector<Dsv4CpWavefrontNode> nodes =
      scheduler.build(/*layer_count=*/2,
                      /*microchunk_count=*/3,
                      /*compression_alignment=*/128);

  ASSERT_EQ(nodes.size(), 6u);
  const std::vector<std::pair<int32_t, int32_t>> expected = {
      {0, 0}, {0, 1}, {1, 0}, {0, 2}, {1, 1}, {1, 2}};
  for (size_t index = 0; index < nodes.size(); ++index) {
    EXPECT_EQ(std::make_pair(nodes[index].layer_id, nodes[index].microchunk_id),
              expected[index]);
  }
}

#if GTEST_HAS_DEATH_TEST
TEST(Dsv4CpWavefrontTest, RejectsInvalidDimensions) {
  EXPECT_DEATH(Dsv4CpWavefrontScheduler().build(/*layer_count=*/0,
                                                /*microchunk_count=*/2,
                                                /*compression_alignment=*/4),
               "layer_count");
  EXPECT_DEATH(Dsv4CpWavefrontScheduler().build(/*layer_count=*/2,
                                                /*microchunk_count=*/0,
                                                /*compression_alignment=*/4),
               "microchunk_count");
  EXPECT_DEATH(Dsv4CpWavefrontScheduler().build(/*layer_count=*/2,
                                                /*microchunk_count=*/2,
                                                /*compression_alignment=*/0),
               "compression_alignment");
}
#endif

}  // namespace
}  // namespace xllm::layer
