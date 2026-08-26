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

#pragma once

#include <torch/torch.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace xllm::layer {

enum class Dsv4CpBufferKind : int8_t {
  PROJECTION = 0,
  OWNER_RECEIVE = 1,
  QLI_CANDIDATES = 2,
  PARTIAL_OUTPUT = 3,
  PARTIAL_LSE = 4,
};

enum class Dsv4CpBufferState : int8_t {
  FREE = 0,
  PROJECTION_READY = 1,
  ROUTE_IN_FLIGHT = 2,
  OWNER_READY = 3,
  CORE_DONE = 4,
  ATTENTION_DONE = 5,
};

struct Dsv4CpBufferSpec {
  Dsv4CpBufferKind kind = Dsv4CpBufferKind::PROJECTION;
  std::vector<int64_t> shape;
  torch::ScalarType dtype = torch::kBFloat16;
};

struct Dsv4CpBufferPoolConfig {
  int32_t slot_count = 2;
  int32_t cp_size = 1;
  int32_t cp_rank = 0;
  int32_t layout_version = 1;
  int32_t phase = 0;
  int64_t topology_signature = 0;
  torch::Device device = torch::Device(torch::kCPU);
  std::vector<Dsv4CpBufferSpec> buffers;
};

class Dsv4CpBufferSlot final {
 public:
  Dsv4CpBufferSlot() = default;

  const torch::Tensor& tensor(Dsv4CpBufferKind kind) const;
  Dsv4CpBufferState state() const { return state_; }
  int64_t generation() const { return generation_; }
  int64_t collective_sequence_id() const { return collective_sequence_id_; }

  void transition(Dsv4CpBufferState expected,
                  Dsv4CpBufferState next,
                  int64_t generation,
                  int64_t collective_sequence_id);

 private:
  friend class Dsv4CpBufferPool;

  void initialize(const std::vector<Dsv4CpBufferSpec>& specs,
                  const torch::Device& device);

  std::array<torch::Tensor, 5> tensors_;
  Dsv4CpBufferState state_ = Dsv4CpBufferState::FREE;
  int64_t generation_ = -1;
  int64_t collective_sequence_id_ = -1;
};

class Dsv4CpBufferPool final {
 public:
  explicit Dsv4CpBufferPool(const Dsv4CpBufferPoolConfig& config);

  Dsv4CpBufferSlot& acquire(int32_t slot_id, int64_t generation);
  void release(int32_t slot_id,
               int64_t generation,
               int64_t collective_sequence_id);
  uint64_t capacity_signature() const { return capacity_signature_; }
  int32_t slot_count() const { return static_cast<int32_t>(slots_.size()); }

 private:
  static uint64_t compute_signature(const Dsv4CpBufferPoolConfig& config);

  std::vector<Dsv4CpBufferSlot> slots_;
  uint64_t capacity_signature_ = 0;
};

}  // namespace xllm::layer
