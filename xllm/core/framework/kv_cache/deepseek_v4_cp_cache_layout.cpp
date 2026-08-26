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

#include <glog/logging.h>

namespace xllm {

Dsv4CpCacheLayout::Dsv4CpCacheLayout(int32_t cp_size, int32_t local_rank)
    : cp_size_(cp_size), local_rank_(local_rank) {
  CHECK_GT(cp_size_, 0) << "DSV4 CP cache layout requires cp_size > 0";
  CHECK_GE(local_rank_, 0);
  CHECK_LT(local_rank_, cp_size_)
      << "local CP rank must be smaller than cp_size";
}

int64_t Dsv4CpCacheLayout::owner_rank(int64_t global_block_id) const {
  CHECK_GE(global_block_id, 0);
  return global_block_id % cp_size_;
}

int64_t Dsv4CpCacheLayout::local_block_id(int64_t global_block_id) const {
  CHECK_GE(global_block_id, 0);
  return global_block_id / cp_size_;
}

int64_t Dsv4CpCacheLayout::physical_block_id(
    int64_t global_block_id,
    Dsv4CpCacheAddressing addressing) const {
  CHECK_GE(global_block_id, 0);
  CHECK_EQ(owner_rank(global_block_id), local_rank_)
      << "DSV4 CP rank cannot address a non-owner block";
  switch (addressing) {
    case Dsv4CpCacheAddressing::REPLICATED:
      return global_block_id;
    case Dsv4CpCacheAddressing::OWNER_LOCAL:
      return local_block_id(global_block_id);
  }
  LOG(FATAL) << "Unsupported DSV4 CP cache addressing mode";
}

int64_t Dsv4CpCacheLayout::physical_slot(
    int64_t global_block_id,
    int64_t block_offset,
    int64_t block_size,
    Dsv4CpCacheAddressing addressing) const {
  CHECK_GT(block_size, 0);
  CHECK_GE(block_offset, 0);
  CHECK_LT(block_offset, block_size);
  return physical_block_id(global_block_id, addressing) * block_size +
         block_offset;
}

int64_t Dsv4CpCacheLayout::local_block_count(
    int64_t global_block_count) const {
  CHECK_GE(global_block_count, 0);
  if (global_block_count <= local_rank_) {
    return 0;
  }
  return (global_block_count - 1 - local_rank_) / cp_size_ + 1;
}

torch::Tensor Dsv4CpCacheLayout::map_global_block_table(
    const torch::Tensor& global_block_table) const {
  CHECK(global_block_table.defined());
  CHECK(global_block_table.scalar_type() == torch::kInt ||
        global_block_table.scalar_type() == torch::kLong)
      << "DSV4 CP block tables must use int32 or int64";
  CHECK(global_block_table.ge(0).all().item<bool>())
      << "global block table cannot contain negative block ids";

  const torch::Tensor local_ids =
      torch::div(global_block_table, cp_size_, "trunc");
  const torch::Tensor owned =
      torch::remainder(global_block_table, cp_size_).eq(local_rank_);
  return torch::where(owned, local_ids, torch::full_like(global_block_table, -1));
}

}  // namespace xllm
