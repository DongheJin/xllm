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

#include <cstdint>

namespace xllm {

enum class Dsv4CpCacheAddressing : int8_t {
  REPLICATED = 0,
  OWNER_LOCAL = 1,
};

// Maps the global block namespace to one owner-local physical namespace. The
// class has no cache or communication ownership; callers can use it from
// planning, allocation, swap, and transfer code without sharing mutable state.
class Dsv4CpCacheLayout final {
 public:
  Dsv4CpCacheLayout(int32_t cp_size, int32_t local_rank);

  int64_t owner_rank(int64_t global_block_id) const;
  int64_t local_block_id(int64_t global_block_id) const;
  int64_t physical_block_id(int64_t global_block_id,
                            Dsv4CpCacheAddressing addressing) const;
  int64_t physical_slot(int64_t global_block_id,
                        int64_t block_offset,
                        int64_t block_size,
                        Dsv4CpCacheAddressing addressing) const;
  int64_t local_block_count(int64_t global_block_count) const;

  // Returns the local block-table view. Entries owned by another CP rank are
  // -1; owner entries contain their dense local block id.
  torch::Tensor map_global_block_table(
      const torch::Tensor& global_block_table) const;

  int32_t cp_size() const { return cp_size_; }
  int32_t local_rank() const { return local_rank_; }

 private:
  int32_t cp_size_;
  int32_t local_rank_;
};

}  // namespace xllm
