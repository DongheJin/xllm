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

#include "layers/npu_torch/deepseek_v4_cp_metadata.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "framework/parallel_state/npu_cp_plan.h"

namespace xllm::layer {
namespace {

torch::Tensor positions_for(const std::vector<int32_t>& q_seq_lens) {
  std::vector<int32_t> positions;
  for (int32_t length : q_seq_lens) {
    for (int32_t position = 0; position < length; ++position) {
      positions.push_back(position);
    }
  }
  return torch::tensor(positions, torch::dtype(torch::kInt32));
}

CpRowLayout make_layout(const std::vector<int32_t>& q_seq_lens,
                        int32_t cp_size,
                        int32_t cp_rank) {
  CpPlanInput input;
  input.q_seq_lens = q_seq_lens;
  input.position_ids = positions_for(q_seq_lens);
  return CpRowLayout::build(
      input, cp_size, cp_rank, torch::Device(torch::kCPU));
}

template <typename T>
std::vector<T> tensor_values(const torch::Tensor& tensor) {
  const torch::Tensor cpu = tensor.to(torch::kCPU).contiguous();
  const T* data = cpu.data_ptr<T>();
  return std::vector<T>(data, data + cpu.numel());
}

TEST(Dsv4CpMetadataBuilderTest, BuildsFrontBackRowsAndPrefixEndpoints) {
  const std::vector<int32_t> q_seq_lens = {5, 7};
  const std::vector<int32_t> kv_seq_lens = {15, 7};
  const CpRowLayout layout =
      make_layout(q_seq_lens, /*cp_size=*/2, /*cp_rank=*/0);

  const Dsv4CpMetadata metadata = Dsv4CpMetadataBuilder::build(
      layout, q_seq_lens, kv_seq_lens, torch::Device(torch::kCPU));

  EXPECT_FALSE(layout.has_empty_rank());
  EXPECT_EQ(metadata.layout_signature, layout.signature());
  EXPECT_EQ(metadata.local_real_row_count, 5);
  EXPECT_EQ(metadata.front.real_row_count, 4);
  EXPECT_EQ(metadata.back.real_row_count, 1);
  EXPECT_EQ(metadata.front.host_q_seq_lens, std::vector<int32_t>({2, 2}));
  EXPECT_EQ(metadata.front.host_kv_seq_lens, std::vector<int32_t>({12, 2}));
  EXPECT_EQ(metadata.back.host_q_seq_lens, std::vector<int32_t>({1}));
  EXPECT_EQ(metadata.back.host_kv_seq_lens, std::vector<int32_t>({7}));
  EXPECT_EQ(tensor_values<int64_t>(metadata.front.pack_indices),
            std::vector<int64_t>({0, 1, 4, 5}));
  EXPECT_EQ(tensor_values<int64_t>(metadata.back.pack_indices),
            std::vector<int64_t>({6}));
  EXPECT_EQ(tensor_values<int64_t>(metadata.front.active_sequence_indices),
            std::vector<int64_t>({0, 1}));
  EXPECT_EQ(tensor_values<int64_t>(metadata.back.active_sequence_indices),
            std::vector<int64_t>({1}));
  EXPECT_EQ(tensor_values<int32_t>(metadata.front.q_cu_seq_lens),
            std::vector<int32_t>({0, 2, 4}));
  EXPECT_EQ(tensor_values<int32_t>(metadata.front.q_seq_endpoints),
            std::vector<int32_t>({2, 4}));
  EXPECT_EQ(tensor_values<int32_t>(metadata.back.q_seq_endpoints),
            std::vector<int32_t>({1}));
}

TEST(Dsv4CpMetadataBuilderTest, KeepsBothHalvesDisjointOnMiddleRank) {
  const std::vector<int32_t> q_seq_lens = {5, 7};
  const CpRowLayout layout =
      make_layout(q_seq_lens, /*cp_size=*/2, /*cp_rank=*/1);
  const Dsv4CpMetadata metadata =
      Dsv4CpMetadataBuilder::build(layout,
                                   q_seq_lens,
                                   /*global_kv_seq_lens=*/q_seq_lens,
                                   torch::Device(torch::kCPU));

  EXPECT_FALSE(layout.has_empty_rank());
  EXPECT_EQ(tensor_values<int64_t>(metadata.front.pack_indices),
            std::vector<int64_t>({0, 1, 4, 5}));
  EXPECT_EQ(tensor_values<int64_t>(metadata.back.pack_indices),
            std::vector<int64_t>({2, 6, 7}));
  EXPECT_EQ(metadata.front.host_kv_seq_lens, std::vector<int32_t>({4, 4}));
  EXPECT_EQ(metadata.back.host_kv_seq_lens, std::vector<int32_t>({5, 6}));
}

TEST(Dsv4CpMetadataBuilderTest, AllowsBothHalvesToBeEmpty) {
  const std::vector<int32_t> q_seq_lens = {1, 2};
  const CpRowLayout layout =
      make_layout(q_seq_lens, /*cp_size=*/4, /*cp_rank=*/3);
  const Dsv4CpMetadata metadata =
      Dsv4CpMetadataBuilder::build(layout,
                                   q_seq_lens,
                                   /*global_kv_seq_lens=*/q_seq_lens,
                                   torch::Device(torch::kCPU));

  EXPECT_EQ(metadata.local_real_row_count, 0);
  EXPECT_EQ(metadata.front.real_row_count, 0);
  EXPECT_EQ(metadata.back.real_row_count, 0);
  EXPECT_EQ(metadata.front.pack_indices.numel(), 0);
  EXPECT_EQ(metadata.back.pack_indices.numel(), 0);
  EXPECT_EQ(tensor_values<int32_t>(metadata.front.q_cu_seq_lens),
            std::vector<int32_t>({0}));
  EXPECT_EQ(tensor_values<int32_t>(metadata.back.q_cu_seq_lens),
            std::vector<int32_t>({0}));
  EXPECT_EQ(metadata.front.q_seq_endpoints.numel(), 0);
  EXPECT_EQ(metadata.back.q_seq_endpoints.numel(), 0);
}

TEST(Dsv4CpMetadataBuilderTest, RejectsKvLengthShorterThanQueryLength) {
  const std::vector<int32_t> q_seq_lens = {8};
  const CpRowLayout layout =
      make_layout(q_seq_lens, /*cp_size=*/2, /*cp_rank=*/0);
  EXPECT_DEATH(Dsv4CpMetadataBuilder::build(layout,
                                            q_seq_lens,
                                            /*global_kv_seq_lens=*/{7},
                                            torch::Device(torch::kCPU)),
               "kv_seq_len >= q_seq_len");
}

}  // namespace
}  // namespace xllm::layer
