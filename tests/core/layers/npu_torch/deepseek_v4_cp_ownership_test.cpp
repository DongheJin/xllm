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

#include "layers/npu_torch/deepseek_v4_cp_ownership.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <tuple>
#include <utility>
#include <vector>

namespace xllm::layer {
namespace {

struct RouteEventView {
  int32_t source_rank = -1;
  int32_t destination_rank = -1;
  int32_t sequence_id = -1;
  int64_t window_id = -1;
  int32_t window_offset = -1;
  int64_t source_row = -1;
  bool cache_write = false;
};

CpPlanInput make_plan_input(const std::vector<int32_t>& q_seq_lens,
                            const std::vector<int32_t>& kv_seq_lens) {
  CHECK_EQ(q_seq_lens.size(), kv_seq_lens.size());
  std::vector<int32_t> position_ids;
  for (size_t sequence_id = 0; sequence_id < q_seq_lens.size();
       ++sequence_id) {
    CHECK_GE(q_seq_lens[sequence_id], 0);
    CHECK_GE(kv_seq_lens[sequence_id], q_seq_lens[sequence_id]);
    const int32_t start_position =
        kv_seq_lens[sequence_id] - q_seq_lens[sequence_id];
    for (int32_t query_offset = 0;
         query_offset < q_seq_lens[sequence_id];
         ++query_offset) {
      position_ids.emplace_back(start_position + query_offset);
    }
  }

  CpPlanInput input;
  input.q_seq_lens = q_seq_lens;
  input.position_ids = torch::tensor(
      position_ids, torch::TensorOptions().dtype(torch::kInt32));
  return input;
}

torch::Tensor make_block_table(const std::vector<int32_t>& logical_lengths,
                               int64_t block_size,
                               int64_t first_global_block_id) {
  CHECK_GT(block_size, 0);
  int64_t max_block_count = 1;
  for (int32_t logical_length : logical_lengths) {
    CHECK_GE(logical_length, 0);
    const int64_t block_count =
        (logical_length + block_size - 1) / block_size;
    max_block_count = std::max(max_block_count, block_count);
  }

  torch::Tensor block_table =
      torch::empty({static_cast<int64_t>(logical_lengths.size()),
                    max_block_count},
                   torch::TensorOptions().dtype(torch::kInt64));
  auto accessor = block_table.accessor<int64_t, 2>();
  int64_t global_block_id = first_global_block_id;
  for (int64_t sequence_id = 0;
       sequence_id < static_cast<int64_t>(logical_lengths.size());
       ++sequence_id) {
    for (int64_t block_index = 0; block_index < max_block_count;
         ++block_index) {
      accessor[sequence_id][block_index] = global_block_id++;
    }
  }
  return block_table;
}

std::vector<int32_t> compressed_lengths(
    const std::vector<int32_t>& kv_seq_lens,
    int64_t compress_ratio) {
  std::vector<int32_t> result;
  result.reserve(kv_seq_lens.size());
  for (int32_t kv_length : kv_seq_lens) {
    result.emplace_back(static_cast<int32_t>(
        (kv_length + compress_ratio - 1) / compress_ratio));
  }
  return result;
}

std::vector<Dsv4CpOwnershipPlan> build_plans(
    const std::vector<int32_t>& q_seq_lens,
    const std::vector<int32_t>& kv_seq_lens,
    const torch::Tensor& state_block_table,
    const torch::Tensor& compressed_block_table,
    int32_t cp_size,
    int64_t compress_ratio,
    int64_t state_block_size,
    int64_t compressed_block_size) {
  const CpPlanInput input = make_plan_input(q_seq_lens, kv_seq_lens);
  Dsv4CpOwnershipPlanner planner;
  std::vector<Dsv4CpOwnershipPlan> plans;
  plans.reserve(cp_size);
  for (int32_t cp_rank = 0; cp_rank < cp_size; ++cp_rank) {
    const CpRowLayout row_layout = CpRowLayout::build(
        input, cp_size, cp_rank, torch::Device(torch::kCPU));
    plans.emplace_back(planner.build(row_layout,
                                     q_seq_lens,
                                     kv_seq_lens,
                                     state_block_table,
                                     compressed_block_table,
                                     compress_ratio,
                                     state_block_size,
                                     compressed_block_size,
                                     torch::Device(torch::kCPU)));
  }
  return plans;
}

std::vector<RouteEventView> collect_send_events(
    const std::vector<Dsv4CpOwnershipPlan>& plans,
    const Dsv4CpRouteDescriptor& (Dsv4CpOwnershipPlan::*route)() const) {
  std::vector<RouteEventView> events;
  for (int32_t source_rank = 0;
       source_rank < static_cast<int32_t>(plans.size());
       ++source_rank) {
    const Dsv4CpRouteDescriptor& descriptor =
        (plans[source_rank].*route)();
    const int64_t* source_rows = descriptor.send_row_indices.data_ptr<int64_t>();
    const int32_t* owner_ranks =
        descriptor.send_owner_ranks.data_ptr<int32_t>();
    const int32_t* sequence_ids =
        descriptor.send_sequence_ids.data_ptr<int32_t>();
    const int64_t* window_ids =
        descriptor.send_window_ids.data_ptr<int64_t>();
    const int32_t* window_offsets =
        descriptor.send_window_offsets.data_ptr<int32_t>();
    const uint8_t* cache_write_flags =
        descriptor.send_cache_write_flags.data_ptr<uint8_t>();
    for (int64_t slot = 0; slot < descriptor.send_row_indices.numel();
         ++slot) {
      if (sequence_ids[slot] < 0) {
        continue;
      }
      events.emplace_back(RouteEventView{source_rank,
                                         owner_ranks[slot],
                                         sequence_ids[slot],
                                         window_ids[slot],
                                         window_offsets[slot],
                                         source_rows[slot],
                                         cache_write_flags[slot] != 0});
    }
  }
  return events;
}

void expect_fixed_capacity(const std::vector<Dsv4CpOwnershipPlan>& plans,
                           const Dsv4CpRouteDescriptor& (
                               Dsv4CpOwnershipPlan::*route)() const) {
  ASSERT_FALSE(plans.empty());
  const int64_t cp_size = static_cast<int64_t>(plans.size());
  const int64_t padded_rows_per_peer =
      (plans.front().*route)().padded_rows_per_peer;
  for (const Dsv4CpOwnershipPlan& plan : plans) {
    const Dsv4CpRouteDescriptor& descriptor = (plan.*route)();
    EXPECT_EQ(descriptor.padded_rows_per_peer, padded_rows_per_peer);
    EXPECT_EQ(descriptor.send_row_indices.numel(),
              cp_size * padded_rows_per_peer);
    EXPECT_EQ(descriptor.recv_source_ranks.numel(),
              cp_size * padded_rows_per_peer);
    EXPECT_EQ(descriptor.send_real_counts.numel(), cp_size);
    EXPECT_EQ(descriptor.recv_real_counts.numel(), cp_size);
    EXPECT_EQ(descriptor.send_real_counts.sum().item<int64_t>(),
              descriptor.send_real_row_count);
    EXPECT_EQ(descriptor.recv_real_counts.sum().item<int64_t>(),
              descriptor.recv_real_row_count);
    EXPECT_EQ(descriptor.recv_canonical_local_slots.numel(),
              descriptor.recv_real_row_count);
  }
}

void expect_canonical_receive_order(
    const std::vector<Dsv4CpOwnershipPlan>& plans,
    const Dsv4CpRouteDescriptor& (Dsv4CpOwnershipPlan::*route)() const) {
  for (const Dsv4CpOwnershipPlan& plan : plans) {
    const Dsv4CpRouteDescriptor& descriptor = (plan.*route)();
    const int64_t* canonical_slots =
        descriptor.recv_canonical_indices.data_ptr<int64_t>();
    const int32_t* source_ranks =
        descriptor.recv_source_ranks.data_ptr<int32_t>();
    const int32_t* sequence_ids =
        descriptor.recv_sequence_ids.data_ptr<int32_t>();
    const int64_t* window_ids =
        descriptor.recv_window_ids.data_ptr<int64_t>();
    const int32_t* window_offsets =
        descriptor.recv_window_offsets.data_ptr<int32_t>();
    std::tuple<int32_t, int64_t, int32_t, int32_t> previous;
    bool has_previous = false;
    for (int64_t index = 0; index < descriptor.recv_real_row_count;
         ++index) {
      const int64_t slot = canonical_slots[index];
      ASSERT_GE(slot, 0);
      ASSERT_LT(slot, descriptor.recv_sequence_ids.numel());
      const auto current = std::make_tuple(sequence_ids[slot],
                                           window_ids[slot],
                                           window_offsets[slot],
                                           source_ranks[slot]);
      if (has_previous) {
        EXPECT_LT(previous, current);
      }
      previous = current;
      has_previous = true;
    }
  }
}

TEST(Dsv4CpOwnershipPlannerTest, RoutesC4RowsOnceWithUniqueWindowWrites) {
  constexpr int32_t kCpSize = 4;
  constexpr int64_t kCompressRatio = 4;
  const std::vector<int32_t> q_seq_lens = {19, 7};
  const std::vector<int32_t> kv_seq_lens = q_seq_lens;
  const torch::Tensor state_block_table =
      make_block_table(kv_seq_lens, /*block_size=*/16, /*first_id=*/0);
  const torch::Tensor compressed_block_table = make_block_table(
      compressed_lengths(kv_seq_lens, kCompressRatio),
      /*block_size=*/8,
      /*first_id=*/20);

  const std::vector<Dsv4CpOwnershipPlan> plans = build_plans(
      q_seq_lens,
      kv_seq_lens,
      state_block_table,
      compressed_block_table,
      kCpSize,
      kCompressRatio,
      /*state_block_size=*/16,
      /*compressed_block_size=*/8);
  const std::vector<RouteEventView> events =
      collect_send_events(plans, &Dsv4CpOwnershipPlan::main_route);

  EXPECT_EQ(events.size(), 26);
  std::map<std::pair<int32_t, int64_t>, int32_t> row_counts;
  std::map<std::pair<int32_t, int64_t>, int32_t> cache_write_counts;
  for (const RouteEventView& event : events) {
    EXPECT_GE(event.source_row, 0);
    EXPECT_GE(event.destination_rank, 0);
    EXPECT_LT(event.destination_rank, kCpSize);
    EXPECT_GE(event.window_offset, 0);
    EXPECT_LT(event.window_offset, kCompressRatio);
    const std::pair<int32_t, int64_t> window_key{event.sequence_id,
                                                 event.window_id};
    ++row_counts[window_key];
    if (event.cache_write) {
      ++cache_write_counts[window_key];
    }
  }
  for (const auto& [window_key, row_count] : row_counts) {
    EXPECT_LE(row_count - 1, kCompressRatio - 1);
    int32_t max_window_offset = -1;
    for (const RouteEventView& event : events) {
      if (event.sequence_id == window_key.first &&
          event.window_id == window_key.second) {
        max_window_offset = std::max(max_window_offset, event.window_offset);
      }
    }
    const int32_t expected_writes =
        max_window_offset == kCompressRatio - 1 ? 1 : 0;
    EXPECT_EQ(cache_write_counts[window_key], expected_writes);
  }
  const std::vector<RouteEventView> compressed_events = collect_send_events(
      plans, &Dsv4CpOwnershipPlan::compressed_cache_route);
  EXPECT_EQ(compressed_events.size(), 5);
  expect_fixed_capacity(plans, &Dsv4CpOwnershipPlan::main_route);
  expect_canonical_receive_order(plans,
                                 &Dsv4CpOwnershipPlan::main_route);
  for (const Dsv4CpOwnershipPlan& plan : plans) {
    EXPECT_TRUE(torch::equal(plan.main_route().send_row_indices,
                             plan.index_route().send_row_indices));
    EXPECT_TRUE(torch::equal(plan.main_route().recv_canonical_indices,
                             plan.index_route().recv_canonical_indices));
  }
}

TEST(Dsv4CpOwnershipPlannerTest,
     SupportsC128ContinuationAndEmptySourceRanks) {
  constexpr int32_t kCpSize = 8;
  const std::vector<int32_t> q_seq_lens = {3};
  const std::vector<int32_t> kv_seq_lens = {128};
  const torch::Tensor state_block_table = torch::tensor(
      {{7}}, torch::TensorOptions().dtype(torch::kInt64));
  const torch::Tensor compressed_block_table = torch::tensor(
      {{0}}, torch::TensorOptions().dtype(torch::kInt64));

  const std::vector<Dsv4CpOwnershipPlan> plans = build_plans(
      q_seq_lens,
      kv_seq_lens,
      state_block_table,
      compressed_block_table,
      kCpSize,
      /*compress_ratio=*/128,
      /*state_block_size=*/128,
      /*compressed_block_size=*/128);

  for (int32_t cp_rank = 3; cp_rank < kCpSize; ++cp_rank) {
    EXPECT_EQ(plans[cp_rank].main_route().send_real_row_count, 0);
    EXPECT_EQ(plans[cp_rank].swa_route().send_real_row_count, 0);
  }
  EXPECT_EQ(plans[7].main_route().recv_real_row_count, 3);
  EXPECT_EQ(plans[7].swa_route().recv_real_row_count, 3);
  EXPECT_EQ(plans[7].compressed_cache_route().send_real_row_count, 1);
  EXPECT_EQ(plans[0].compressed_cache_route().recv_real_row_count, 1);

  const std::vector<RouteEventView> main_events =
      collect_send_events(plans, &Dsv4CpOwnershipPlan::main_route);
  ASSERT_EQ(main_events.size(), 3);
  EXPECT_EQ(main_events[0].window_id, 0);
  EXPECT_EQ(main_events[0].window_offset, 125);
  EXPECT_EQ(main_events[1].window_offset, 126);
  EXPECT_EQ(main_events[2].window_offset, 127);
  EXPECT_FALSE(main_events[0].cache_write);
  EXPECT_FALSE(main_events[1].cache_write);
  EXPECT_TRUE(main_events[2].cache_write);
  expect_fixed_capacity(plans, &Dsv4CpOwnershipPlan::main_route);
  expect_fixed_capacity(plans,
                        &Dsv4CpOwnershipPlan::compressed_cache_route);
}

TEST(Dsv4CpOwnershipPlannerTest, SupportsCp2Cp4AndCp8Topologies) {
  for (int32_t cp_size : {2, 4, 8}) {
    const std::vector<int32_t> q_seq_lens = {2 * cp_size + 5,
                                             cp_size + 1};
    const std::vector<int32_t> kv_seq_lens = q_seq_lens;
    const torch::Tensor state_block_table =
        make_block_table(kv_seq_lens, /*block_size=*/128, /*first_id=*/3);
    const torch::Tensor compressed_block_table = make_block_table(
        compressed_lengths(kv_seq_lens, /*compress_ratio=*/4),
        /*block_size=*/128,
        /*first_id=*/11);
    const std::vector<Dsv4CpOwnershipPlan> plans = build_plans(
        q_seq_lens,
        kv_seq_lens,
        state_block_table,
        compressed_block_table,
        cp_size,
        /*compress_ratio=*/4,
        /*state_block_size=*/128,
        /*compressed_block_size=*/128);

    const std::vector<RouteEventView> events =
        collect_send_events(plans, &Dsv4CpOwnershipPlan::main_route);
    EXPECT_EQ(events.size(),
              static_cast<size_t>(3 * cp_size + 6));
    expect_fixed_capacity(plans, &Dsv4CpOwnershipPlan::main_route);
    expect_canonical_receive_order(plans,
                                   &Dsv4CpOwnershipPlan::main_route);
  }
}

TEST(Dsv4CpOwnershipPlannerTest,
     RoutesCoreOutputWhenStateAndCompressedOwnersDiffer) {
  constexpr int32_t kCpSize = 2;
  const std::vector<int32_t> q_seq_lens = {8};
  const std::vector<int32_t> kv_seq_lens = q_seq_lens;
  const torch::Tensor state_block_table = torch::tensor(
      {{1}}, torch::TensorOptions().dtype(torch::kInt64));
  const torch::Tensor compressed_block_table = torch::tensor(
      {{0}}, torch::TensorOptions().dtype(torch::kInt64));

  const std::vector<Dsv4CpOwnershipPlan> plans = build_plans(
      q_seq_lens,
      kv_seq_lens,
      state_block_table,
      compressed_block_table,
      kCpSize,
      /*compress_ratio=*/4,
      /*state_block_size=*/128,
      /*compressed_block_size=*/128);
  const std::vector<RouteEventView> output_events = collect_send_events(
      plans, &Dsv4CpOwnershipPlan::compressed_cache_route);

  ASSERT_EQ(output_events.size(), 2);
  for (const RouteEventView& event : output_events) {
    EXPECT_EQ(event.source_rank, 1);
    EXPECT_EQ(event.destination_rank, 0);
    EXPECT_TRUE(event.cache_write);
  }
  EXPECT_EQ(output_events[0].source_row, 0);
  EXPECT_EQ(output_events[1].source_row, 1);
}

TEST(Dsv4CpOwnershipPlannerTest, SupportsReplicatedPhysicalAddressing) {
  constexpr int32_t kCpSize = 2;
  const std::vector<int32_t> q_seq_lens = {8};
  const std::vector<int32_t> kv_seq_lens = q_seq_lens;
  const torch::Tensor state_block_table = torch::tensor(
      {{3}}, torch::TensorOptions().dtype(torch::kInt64));
  const torch::Tensor compressed_block_table = torch::tensor(
      {{2}}, torch::TensorOptions().dtype(torch::kInt64));
  const CpPlanInput input = make_plan_input(q_seq_lens, kv_seq_lens);
  const CpRowLayout rank0 = CpRowLayout::build(
      input, kCpSize, /*cp_rank=*/0, torch::kCPU);
  const CpRowLayout rank1 = CpRowLayout::build(
      input, kCpSize, /*cp_rank=*/1, torch::kCPU);
  Dsv4CpOwnershipPlanner planner;

  const Dsv4CpOwnershipPlan rank0_plan = planner.build(
      rank0,
      q_seq_lens,
      kv_seq_lens,
      state_block_table,
      compressed_block_table,
      /*compress_ratio=*/4,
      /*state_block_size=*/16,
      /*compressed_block_size=*/8,
      torch::Device(torch::kCPU),
      Dsv4CpCacheAddressing::REPLICATED);
  const Dsv4CpOwnershipPlan rank1_plan = planner.build(
      rank1,
      q_seq_lens,
      kv_seq_lens,
      state_block_table,
      compressed_block_table,
      /*compress_ratio=*/4,
      /*state_block_size=*/16,
      /*compressed_block_size=*/8,
      torch::Device(torch::kCPU),
      Dsv4CpCacheAddressing::REPLICATED);

  EXPECT_EQ(rank1_plan.owner_metadata()
                .local_state_block_table.item<int32_t>(),
            3);
  EXPECT_TRUE(torch::equal(
      rank1_plan.swa_route().recv_canonical_local_slots,
      torch::arange(3 * 16, 3 * 16 + 8, torch::kInt64)));
  const Dsv4CpRouteDescriptor& compressed_send =
      rank1_plan.compressed_cache_route();
  EXPECT_TRUE(torch::equal(
      compressed_send.send_row_indices.index_select(
          /*dim=*/0, compressed_send.send_valid_indices),
      torch::tensor({0, 1}, torch::kInt64)));
  EXPECT_TRUE(torch::equal(
      rank0_plan.compressed_cache_route().recv_canonical_local_slots,
      torch::tensor({2 * 8, 2 * 8 + 1}, torch::kInt64)));
}

TEST(Dsv4CpOwnershipPlannerTest, BuildsCanonicalOwnerCoreMetadata) {
  constexpr int32_t kCpSize = 4;
  constexpr int64_t kCompressRatio = 4;
  const std::vector<int32_t> q_seq_lens = {19, 7};
  const std::vector<int32_t> kv_seq_lens = q_seq_lens;
  const torch::Tensor state_block_table =
      make_block_table(kv_seq_lens, /*block_size=*/16, /*first_id=*/0);
  const torch::Tensor compressed_block_table = make_block_table(
      compressed_lengths(kv_seq_lens, kCompressRatio),
      /*block_size=*/8,
      /*first_id=*/20);
  const std::vector<Dsv4CpOwnershipPlan> plans = build_plans(
      q_seq_lens,
      kv_seq_lens,
      state_block_table,
      compressed_block_table,
      kCpSize,
      kCompressRatio,
      /*state_block_size=*/16,
      /*compressed_block_size=*/8);

  int64_t total_rows = 0;
  int64_t total_segments = 0;
  int64_t total_outputs = 0;
  int64_t total_committed_outputs = 0;
  for (int32_t cp_rank = 0; cp_rank < kCpSize; ++cp_rank) {
    const Dsv4CpOwnerMetadata& metadata = plans[cp_rank].owner_metadata();
    total_rows += metadata.real_row_count;
    total_segments += metadata.segment_count;
    total_outputs += metadata.output_row_count;
    total_committed_outputs += metadata.committed_output_row_count;
    EXPECT_EQ(metadata.row_capacity,
              plans[cp_rank].main_route().recv_source_ranks.numel());
    EXPECT_EQ(metadata.recv_canonical_indices.numel(),
              metadata.real_row_count);
    EXPECT_EQ(metadata.row_sequence_ids.numel(), metadata.real_row_count);
    EXPECT_EQ(metadata.row_window_ids.numel(), metadata.real_row_count);
    EXPECT_EQ(metadata.row_window_offsets.numel(), metadata.real_row_count);
    EXPECT_EQ(metadata.row_absolute_positions.numel(),
              metadata.real_row_count);
    EXPECT_EQ(metadata.q_cu_seq_lens.numel(),
              metadata.segment_count + 1);
    EXPECT_EQ(metadata.start_positions.numel(), metadata.segment_count);
    EXPECT_EQ(metadata.output_rope_indices.numel(), metadata.output_row_count);
    EXPECT_EQ(metadata.local_state_block_table.size(0),
              metadata.segment_count);
    EXPECT_EQ(metadata.local_state_block_table.scalar_type(), torch::kInt32);

    const int32_t* q_cu_seq_lens =
        metadata.q_cu_seq_lens.data_ptr<int32_t>();
    const int32_t* window_offsets =
        metadata.row_window_offsets.data_ptr<int32_t>();
    const uint8_t* cache_write_flags =
        metadata.output_cache_write_flags.data_ptr<uint8_t>();
    const int64_t* cache_write_rows =
        metadata.output_cache_write_row_indices.data_ptr<int64_t>();
    const int32_t* local_state_blocks =
        metadata.local_state_block_table.data_ptr<int32_t>();
    const int64_t state_block_columns =
        metadata.local_state_block_table.size(1);
    const int32_t* row_sequence_ids =
        metadata.row_sequence_ids.data_ptr<int32_t>();
    for (int64_t segment = 0; segment < metadata.segment_count; ++segment) {
      const int32_t row_begin = q_cu_seq_lens[segment];
      const int32_t row_end = q_cu_seq_lens[segment + 1];
      ASSERT_GT(row_end, row_begin);
      const int32_t sequence_id = row_sequence_ids[row_begin];
      for (int64_t block_index = 0; block_index < state_block_columns;
           ++block_index) {
        const int64_t global_block_id =
            state_block_table[sequence_id][block_index].item<int64_t>();
        const int32_t expected_local_id =
            global_block_id % kCpSize == cp_rank
                ? static_cast<int32_t>(global_block_id / kCpSize)
                : -1;
        EXPECT_EQ(local_state_blocks[segment * state_block_columns +
                                     block_index],
                  expected_local_id);
      }
    }
    for (int64_t output_row = 0; output_row < metadata.output_row_count;
         ++output_row) {
      const int64_t cache_write_row = cache_write_rows[output_row];
      const bool expected_cache_write = cache_write_row >= 0;
      EXPECT_EQ(cache_write_flags[output_row] != 0, expected_cache_write);
      if (expected_cache_write) {
        ASSERT_LT(cache_write_row, metadata.real_row_count);
        EXPECT_EQ(window_offsets[cache_write_row], kCompressRatio - 1);
      }
    }
  }

  EXPECT_EQ(total_rows, 26);
  EXPECT_EQ(total_segments, 3);
  EXPECT_EQ(total_outputs, 7);
  EXPECT_EQ(total_committed_outputs, 5);
}

TEST(Dsv4CpOwnershipPlannerTest,
     KeepsMultipleWindowsInOneStateBlockSequential) {
  constexpr int32_t kCpSize = 2;
  const torch::Tensor state_block_table = torch::tensor(
      {{1}}, torch::TensorOptions().dtype(torch::kInt64));
  const torch::Tensor compressed_block_table = torch::tensor(
      {{0}}, torch::TensorOptions().dtype(torch::kInt64));
  const std::vector<Dsv4CpOwnershipPlan> plans = build_plans(
      /*q_seq_lens=*/{8},
      /*kv_seq_lens=*/{8},
      state_block_table,
      compressed_block_table,
      kCpSize,
      /*compress_ratio=*/4,
      /*state_block_size=*/128,
      /*compressed_block_size=*/128);

  const Dsv4CpOwnerMetadata& owner = plans[1].owner_metadata();
  ASSERT_EQ(owner.real_row_count, 8);
  ASSERT_EQ(owner.segment_count, 1);
  ASSERT_EQ(owner.output_row_count, 2);
  EXPECT_TRUE(torch::equal(
      owner.q_cu_seq_lens, torch::tensor({0, 8}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(
      owner.start_positions, torch::tensor({0}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(
      owner.window_ids, torch::tensor({0, 1}, torch::kInt64)));
  EXPECT_TRUE(torch::equal(
      owner.output_cache_write_row_indices,
      torch::tensor({3, 7}, torch::kInt64)));
  EXPECT_TRUE(torch::equal(
      owner.output_rope_indices, torch::tensor({0, 1}, torch::kInt64)));
  EXPECT_TRUE(torch::equal(
      plans[1].swa_route().recv_canonical_local_slots,
      torch::arange(8, torch::kInt64)));
  EXPECT_TRUE(torch::equal(
      plans[0].compressed_cache_route().recv_canonical_local_slots,
      torch::tensor({0, 1}, torch::kInt64)));
}

TEST(Dsv4CpOwnershipPlannerTest, OwnerMetadataPreservesDecodeBoundary) {
  constexpr int32_t kCpSize = 2;
  const torch::Tensor state_block_table = torch::tensor(
      {{1}}, torch::TensorOptions().dtype(torch::kInt64));
  const torch::Tensor compressed_block_table = torch::tensor(
      {{0}}, torch::TensorOptions().dtype(torch::kInt64));
  const std::vector<Dsv4CpOwnershipPlan> completing = build_plans(
      /*q_seq_lens=*/{1},
      /*kv_seq_lens=*/{4},
      state_block_table,
      compressed_block_table,
      kCpSize,
      /*compress_ratio=*/4,
      /*state_block_size=*/128,
      /*compressed_block_size=*/128);
  const std::vector<Dsv4CpOwnershipPlan> partial = build_plans(
      /*q_seq_lens=*/{1},
      /*kv_seq_lens=*/{5},
      state_block_table,
      compressed_block_table,
      kCpSize,
      /*compress_ratio=*/4,
      /*state_block_size=*/128,
      /*compressed_block_size=*/128);

  const Dsv4CpOwnerMetadata& completing_owner =
      completing[1].owner_metadata();
  ASSERT_EQ(completing_owner.real_row_count, 1);
  ASSERT_EQ(completing_owner.output_row_count, 1);
  EXPECT_EQ(completing_owner.start_positions.item<int32_t>(), 3);
  EXPECT_EQ(completing_owner.output_rope_indices.item<int64_t>(), 0);
  EXPECT_EQ(completing_owner.committed_output_row_count, 1);
  EXPECT_EQ(completing[1].compressed_cache_route().send_real_row_count, 1);

  const Dsv4CpOwnerMetadata& partial_owner = partial[1].owner_metadata();
  ASSERT_EQ(partial_owner.real_row_count, 1);
  ASSERT_EQ(partial_owner.output_row_count, 1);
  EXPECT_EQ(partial_owner.start_positions.item<int32_t>(), 4);
  EXPECT_EQ(partial_owner.output_rope_indices.item<int64_t>(), 0);
  EXPECT_EQ(partial_owner.committed_output_row_count, 0);
  EXPECT_EQ(partial[1].compressed_cache_route().send_real_row_count, 0);
}

TEST(Dsv4CpOwnershipPlannerTest, RoutesC1SwaRowsWithoutCompressorState) {
  const std::vector<int32_t> q_seq_lens = {3, 1};
  const std::vector<int32_t> kv_seq_lens = {130, 1};
  const CpPlanInput input = make_plan_input(q_seq_lens, kv_seq_lens);
  const torch::Tensor swa_block_table = torch::tensor(
      {{0, 1}, {2, 3}}, torch::TensorOptions().dtype(torch::kInt64));
  Dsv4CpOwnershipPlanner planner;
  std::vector<Dsv4CpRouteDescriptor> routes;
  for (int32_t cp_rank = 0; cp_rank < 2; ++cp_rank) {
    const CpRowLayout layout = CpRowLayout::build(
        input, /*cp_size=*/2, cp_rank, torch::kCPU);
    routes.emplace_back(planner.build_swa_route(
        layout,
        q_seq_lens,
        kv_seq_lens,
        swa_block_table,
        /*swa_block_size=*/128,
        torch::kCPU,
        Dsv4CpCacheAddressing::REPLICATED));
  }

  ASSERT_EQ(routes.size(), 2u);
  EXPECT_EQ(routes[0].recv_real_row_count, 2);
  EXPECT_EQ(routes[1].recv_real_row_count, 2);
  EXPECT_TRUE(torch::equal(routes[0].recv_canonical_local_slots,
                           torch::tensor({127, 256}, torch::kInt64)));
  EXPECT_TRUE(torch::equal(routes[1].recv_canonical_local_slots,
                           torch::tensor({128, 129}, torch::kInt64)));
  EXPECT_EQ(routes[0].send_real_row_count + routes[1].send_real_row_count, 4);
}

TEST(Dsv4CpOwnershipPlannerTest, SignatureUsesTopologyAndBlockTableValues) {
  const std::vector<int32_t> q_seq_lens = {8};
  const std::vector<int32_t> kv_seq_lens = q_seq_lens;
  const CpPlanInput input = make_plan_input(q_seq_lens, kv_seq_lens);
  const CpRowLayout rank0 = CpRowLayout::build(
      input, /*cp_size=*/2, /*cp_rank=*/0, torch::kCPU);
  const CpRowLayout rank1 = CpRowLayout::build(
      input, /*cp_size=*/2, /*cp_rank=*/1, torch::kCPU);
  const torch::Tensor state_table = torch::tensor(
      {{0}}, torch::TensorOptions().dtype(torch::kInt64));
  const torch::Tensor changed_state_table = torch::tensor(
      {{2}}, torch::TensorOptions().dtype(torch::kInt64));
  const torch::Tensor compressed_table = torch::tensor(
      {{1}}, torch::TensorOptions().dtype(torch::kInt64));
  Dsv4CpOwnershipPlanner planner;

  const Dsv4CpOwnershipPlan first = planner.build(rank0,
                                                   q_seq_lens,
                                                   kv_seq_lens,
                                                   state_table,
                                                   compressed_table,
                                                   /*compress_ratio=*/4,
                                                   /*state_block_size=*/128,
                                                   /*compressed_block_size=*/128,
                                                   torch::kCPU);
  const Dsv4CpOwnershipPlan same = planner.build(rank0,
                                                  q_seq_lens,
                                                  kv_seq_lens,
                                                  state_table.clone(),
                                                  compressed_table.clone(),
                                                  /*compress_ratio=*/4,
                                                  /*state_block_size=*/128,
                                                  /*compressed_block_size=*/128,
                                                  torch::kCPU);
  const Dsv4CpOwnershipPlan other_rank = planner.build(
      rank1,
      q_seq_lens,
      kv_seq_lens,
      state_table,
      compressed_table,
      /*compress_ratio=*/4,
      /*state_block_size=*/128,
      /*compressed_block_size=*/128,
      torch::kCPU);
  const Dsv4CpOwnershipPlan other_table = planner.build(
      rank0,
      q_seq_lens,
      kv_seq_lens,
      changed_state_table,
      compressed_table,
      /*compress_ratio=*/4,
      /*state_block_size=*/128,
      /*compressed_block_size=*/128,
      torch::kCPU);

  EXPECT_EQ(first.signature(), same.signature());
  EXPECT_NE(first.signature(), other_rank.signature());
  EXPECT_NE(first.signature(), other_table.signature());
}

TEST(Dsv4CpOwnershipPlannerTest, RejectsInvalidOwnershipMetadata) {
#if GTEST_HAS_DEATH_TEST
  const std::vector<int32_t> q_seq_lens = {8};
  const std::vector<int32_t> kv_seq_lens = q_seq_lens;
  const CpPlanInput input = make_plan_input(q_seq_lens, kv_seq_lens);
  const CpRowLayout layout = CpRowLayout::build(
      input, /*cp_size=*/2, /*cp_rank=*/0, torch::kCPU);
  const torch::Tensor table = torch::tensor(
      {{0}}, torch::TensorOptions().dtype(torch::kInt64));
  Dsv4CpOwnershipPlanner planner;

  EXPECT_DEATH(planner.build(layout,
                             q_seq_lens,
                             kv_seq_lens,
                             table,
                             table,
                             /*compress_ratio=*/8,
                             /*state_block_size=*/128,
                             /*compressed_block_size=*/128,
                             torch::kCPU),
               "supports C4 or C128");
  EXPECT_DEATH(planner.build(layout,
                             q_seq_lens,
                             kv_seq_lens,
                             table,
                             table,
                             /*compress_ratio=*/4,
                             /*state_block_size=*/6,
                             /*compressed_block_size=*/128,
                             torch::kCPU),
               "must not cross state blocks");
  EXPECT_DEATH(planner.build(layout,
                             q_seq_lens,
                             kv_seq_lens,
                             torch::empty({1, 0}, torch::kInt64),
                             table,
                             /*compress_ratio=*/4,
                             /*state_block_size=*/4,
                             /*compressed_block_size=*/128,
                             torch::kCPU),
               "does not cover block index");
  EXPECT_DEATH(planner.build(layout,
                             q_seq_lens,
                             kv_seq_lens,
                             table,
                             torch::tensor(
                                 {{-1}},
                                 torch::TensorOptions().dtype(torch::kInt64)),
                             /*compress_ratio=*/4,
                             /*state_block_size=*/128,
                             /*compressed_block_size=*/128,
                             torch::kCPU),
               "invalid block id");
#endif
}

}  // namespace
}  // namespace xllm::layer
