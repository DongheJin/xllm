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

#include "layers/npu_torch/deepseek_v4_cp_buffer_pool.h"

#include <gtest/gtest.h>

#include <cstdint>

namespace xllm::layer {
namespace {

Dsv4CpBufferPoolConfig make_config() {
  Dsv4CpBufferPoolConfig config;
  config.slot_count = 2;
  config.cp_size = 8;
  config.cp_rank = 3;
  config.layout_version = 2;
  config.phase = 1;
  config.topology_signature = 12345;
  config.device = torch::Device(torch::kCPU);
  config.buffers = {
      {Dsv4CpBufferKind::PROJECTION, {16, 32}, torch::kBFloat16},
      {Dsv4CpBufferKind::OWNER_RECEIVE, {64, 32}, torch::kBFloat16},
      {Dsv4CpBufferKind::QLI_CANDIDATES, {8, 16, 4}, torch::kFloat32},
      {Dsv4CpBufferKind::PARTIAL_OUTPUT, {8, 16, 2, 32}, torch::kBFloat16},
      {Dsv4CpBufferKind::PARTIAL_LSE, {8, 16, 2, 1}, torch::kFloat32},
  };
  return config;
}

void finish_slot(Dsv4CpBufferSlot& slot, int64_t generation) {
  slot.transition(Dsv4CpBufferState::PROJECTION_READY,
                  Dsv4CpBufferState::ROUTE_IN_FLIGHT,
                  generation,
                  generation);
  slot.transition(Dsv4CpBufferState::ROUTE_IN_FLIGHT,
                  Dsv4CpBufferState::OWNER_READY,
                  generation,
                  generation);
  slot.transition(Dsv4CpBufferState::OWNER_READY,
                  Dsv4CpBufferState::CORE_DONE,
                  generation,
                  generation);
  slot.transition(Dsv4CpBufferState::CORE_DONE,
                  Dsv4CpBufferState::ATTENTION_DONE,
                  generation,
                  generation);
}

TEST(Dsv4CpBufferPoolTest, KeepsTensorAddressesStableAcrossGenerations) {
  Dsv4CpBufferPool pool(make_config());
  Dsv4CpBufferSlot& first = pool.acquire(/*slot_id=*/0, /*generation=*/7);
  const void* projection_address =
      first.tensor(Dsv4CpBufferKind::PROJECTION).data_ptr();
  const void* partial_lse_address =
      first.tensor(Dsv4CpBufferKind::PARTIAL_LSE).data_ptr();

  finish_slot(first, /*generation=*/7);
  pool.release(/*slot_id=*/0,
               /*generation=*/7,
               /*collective_sequence_id=*/7);
  Dsv4CpBufferSlot& second = pool.acquire(/*slot_id=*/0, /*generation=*/8);

  EXPECT_EQ(second.tensor(Dsv4CpBufferKind::PROJECTION).data_ptr(),
            projection_address);
  EXPECT_EQ(second.tensor(Dsv4CpBufferKind::PARTIAL_LSE).data_ptr(),
            partial_lse_address);
  EXPECT_EQ(second.generation(), 8);
  EXPECT_EQ(second.state(), Dsv4CpBufferState::PROJECTION_READY);
}

TEST(Dsv4CpBufferPoolTest, CapacitySignatureIncludesGraphIdentityFields) {
  const Dsv4CpBufferPoolConfig config = make_config();
  Dsv4CpBufferPool first(config);
  Dsv4CpBufferPool second(config);
  EXPECT_EQ(first.capacity_signature(), second.capacity_signature());

  Dsv4CpBufferPoolConfig different_phase = config;
  different_phase.phase = 2;
  Dsv4CpBufferPool phase_pool(different_phase);
  EXPECT_NE(first.capacity_signature(), phase_pool.capacity_signature());

  Dsv4CpBufferPoolConfig different_shape = config;
  different_shape.buffers[0].shape[0] = 32;
  Dsv4CpBufferPool shape_pool(different_shape);
  EXPECT_NE(first.capacity_signature(), shape_pool.capacity_signature());
}

TEST(Dsv4CpBufferPoolTest, KeepsSlotsIndependent) {
  Dsv4CpBufferPool pool(make_config());
  Dsv4CpBufferSlot& first = pool.acquire(/*slot_id=*/0, /*generation=*/1);
  Dsv4CpBufferSlot& second = pool.acquire(/*slot_id=*/1, /*generation=*/2);

  EXPECT_EQ(first.generation(), 1);
  EXPECT_EQ(second.generation(), 2);
  EXPECT_NE(first.tensor(Dsv4CpBufferKind::PROJECTION).data_ptr(),
            second.tensor(Dsv4CpBufferKind::PROJECTION).data_ptr());
}

#if GTEST_HAS_DEATH_TEST
TEST(Dsv4CpBufferPoolTest, RejectsEarlyReuseAndInvalidStateEdges) {
  EXPECT_DEATH(
      {
        Dsv4CpBufferPool pool(make_config());
        pool.acquire(/*slot_id=*/0, /*generation=*/1);
        pool.acquire(/*slot_id=*/0, /*generation=*/2);
      },
      "invalid DSV4 CP buffer state transition");
  EXPECT_DEATH(
      {
        Dsv4CpBufferPool pool(make_config());
        Dsv4CpBufferSlot& slot = pool.acquire(/*slot_id=*/0, /*generation=*/1);
        slot.transition(Dsv4CpBufferState::PROJECTION_READY,
                        Dsv4CpBufferState::CORE_DONE,
                        /*generation=*/1,
                        /*collective_sequence_id=*/1);
      },
      "illegal DSV4 CP buffer state edge");
  EXPECT_DEATH(
      {
        Dsv4CpBufferPool pool(make_config());
        Dsv4CpBufferSlot& slot = pool.acquire(/*slot_id=*/0, /*generation=*/1);
        slot.transition(Dsv4CpBufferState::PROJECTION_READY,
                        Dsv4CpBufferState::ROUTE_IN_FLIGHT,
                        /*generation=*/2,
                        /*collective_sequence_id=*/1);
      },
      "generation changed while in flight");
}
#endif

}  // namespace
}  // namespace xllm::layer
