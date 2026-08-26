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

#include <glog/logging.h>

#include <cstddef>

namespace xllm::layer {
namespace {

constexpr uint64_t kFnvOffsetBasis = 1469598103934665603ULL;
constexpr uint64_t kFnvPrime = 1099511628211ULL;
constexpr size_t kBufferKindCount = 5;

size_t buffer_index(Dsv4CpBufferKind kind) {
  const size_t index = static_cast<size_t>(kind);
  CHECK_LT(index, kBufferKindCount) << "invalid DSV4 CP buffer kind";
  return index;
}

bool is_valid_transition(Dsv4CpBufferState expected, Dsv4CpBufferState next) {
  switch (expected) {
    case Dsv4CpBufferState::FREE:
      return next == Dsv4CpBufferState::PROJECTION_READY;
    case Dsv4CpBufferState::PROJECTION_READY:
      return next == Dsv4CpBufferState::ROUTE_IN_FLIGHT;
    case Dsv4CpBufferState::ROUTE_IN_FLIGHT:
      return next == Dsv4CpBufferState::OWNER_READY;
    case Dsv4CpBufferState::OWNER_READY:
      return next == Dsv4CpBufferState::CORE_DONE;
    case Dsv4CpBufferState::CORE_DONE:
      return next == Dsv4CpBufferState::ATTENTION_DONE;
    case Dsv4CpBufferState::ATTENTION_DONE:
      return next == Dsv4CpBufferState::FREE;
  }
  return false;
}

void hash_bytes(uint64_t* hash, const void* data, size_t size) {
  CHECK(hash != nullptr);
  const uint8_t* bytes = static_cast<const uint8_t*>(data);
  for (size_t index = 0; index < size; ++index) {
    *hash ^= bytes[index];
    *hash *= kFnvPrime;
  }
}

template <typename T>
void hash_value(uint64_t* hash, const T& value) {
  hash_bytes(hash, &value, sizeof(T));
}

}  // namespace

const torch::Tensor& Dsv4CpBufferSlot::tensor(Dsv4CpBufferKind kind) const {
  const size_t index = buffer_index(kind);
  CHECK(tensors_[index].defined()) << "requested DSV4 CP buffer is undefined";
  return tensors_[index];
}

void Dsv4CpBufferSlot::initialize(const std::vector<Dsv4CpBufferSpec>& specs,
                                  const torch::Device& device) {
  for (const Dsv4CpBufferSpec& spec : specs) {
    const size_t index = buffer_index(spec.kind);
    CHECK(!spec.shape.empty()) << "DSV4 CP buffer shape must not be empty";
    for (int64_t dimension : spec.shape) {
      CHECK_GE(dimension, 0) << "DSV4 CP buffer shape contains a negative dim";
    }
    CHECK(!tensors_[index].defined())
        << "duplicate DSV4 CP buffer kind in pool configuration";
    tensors_[index] = torch::empty(
        spec.shape, torch::TensorOptions().dtype(spec.dtype).device(device));
  }
  for (const torch::Tensor& tensor : tensors_) {
    CHECK(tensor.defined()) << "DSV4 CP buffer configuration is incomplete";
  }
}

void Dsv4CpBufferSlot::transition(Dsv4CpBufferState expected,
                                  Dsv4CpBufferState next,
                                  int64_t generation,
                                  int64_t collective_sequence_id) {
  CHECK_EQ(static_cast<int8_t>(state_), static_cast<int8_t>(expected))
      << "invalid DSV4 CP buffer state transition";
  CHECK(is_valid_transition(expected, next))
      << "illegal DSV4 CP buffer state edge";
  CHECK_GE(generation, 0);
  CHECK_GE(collective_sequence_id, 0);
  if (expected == Dsv4CpBufferState::FREE) {
    CHECK_GT(generation, generation_)
        << "DSV4 CP buffer generation must increase on acquire";
  } else {
    CHECK_EQ(generation, generation_)
        << "DSV4 CP buffer generation changed while in flight";
    CHECK_EQ(collective_sequence_id, collective_sequence_id_)
        << "DSV4 CP collective sequence changed while in flight";
  }
  if (next == Dsv4CpBufferState::FREE) {
    CHECK_EQ(static_cast<int8_t>(expected),
             static_cast<int8_t>(Dsv4CpBufferState::ATTENTION_DONE))
        << "only completed attention buffers can be released";
  }
  state_ = next;
  generation_ = generation;
  collective_sequence_id_ = collective_sequence_id;
}

uint64_t Dsv4CpBufferPool::compute_signature(
    const Dsv4CpBufferPoolConfig& config) {
  CHECK_GT(config.slot_count, 0);
  CHECK_GT(config.cp_size, 0);
  CHECK_GE(config.cp_rank, 0);
  CHECK_LT(config.cp_rank, config.cp_size);
  CHECK_GE(config.layout_version, 0);
  CHECK_GE(config.phase, 0);
  uint64_t hash = kFnvOffsetBasis;
  hash_value(&hash, config.slot_count);
  hash_value(&hash, config.cp_size);
  hash_value(&hash, config.cp_rank);
  hash_value(&hash, config.layout_version);
  hash_value(&hash, config.phase);
  hash_value(&hash, config.topology_signature);
  const int32_t device_type = static_cast<int32_t>(config.device.type());
  const int32_t device_index =
      config.device.has_index() ? config.device.index() : -1;
  hash_value(&hash, device_type);
  hash_value(&hash, device_index);
  for (const Dsv4CpBufferSpec& spec : config.buffers) {
    const int8_t kind = static_cast<int8_t>(spec.kind);
    const int32_t dtype = static_cast<int32_t>(spec.dtype);
    hash_value(&hash, kind);
    hash_value(&hash, dtype);
    const uint64_t rank = static_cast<uint64_t>(spec.shape.size());
    hash_value(&hash, rank);
    for (int64_t dimension : spec.shape) {
      hash_value(&hash, dimension);
    }
  }
  return hash;
}

Dsv4CpBufferPool::Dsv4CpBufferPool(const Dsv4CpBufferPoolConfig& config)
    : capacity_signature_(compute_signature(config)) {
  CHECK_GT(config.slot_count, 0);
  CHECK_EQ(config.buffers.size(), 5u)
      << "DSV4 CP buffer pool requires one spec for every buffer kind";
  slots_.reserve(static_cast<size_t>(config.slot_count));
  for (int32_t slot_id = 0; slot_id < config.slot_count; ++slot_id) {
    slots_.emplace_back();
    slots_.back().initialize(config.buffers, config.device);
  }
}

Dsv4CpBufferSlot& Dsv4CpBufferPool::acquire(int32_t slot_id,
                                            int64_t generation) {
  CHECK_GE(slot_id, 0);
  CHECK_LT(slot_id, static_cast<int32_t>(slots_.size()));
  Dsv4CpBufferSlot& slot = slots_[static_cast<size_t>(slot_id)];
  slot.transition(Dsv4CpBufferState::FREE,
                  Dsv4CpBufferState::PROJECTION_READY,
                  generation,
                  /*collective_sequence_id=*/generation);
  return slot;
}

void Dsv4CpBufferPool::release(int32_t slot_id,
                               int64_t generation,
                               int64_t collective_sequence_id) {
  CHECK_GE(slot_id, 0);
  CHECK_LT(slot_id, static_cast<int32_t>(slots_.size()));
  slots_[static_cast<size_t>(slot_id)].transition(
      Dsv4CpBufferState::ATTENTION_DONE,
      Dsv4CpBufferState::FREE,
      generation,
      collective_sequence_id);
}

}  // namespace xllm::layer
