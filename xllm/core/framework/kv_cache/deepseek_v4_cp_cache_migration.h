/* Copyright 2025-2026 The xLLM Authors.

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

#include <cstdint>
#include <vector>

namespace xllm {

// A cache-group-qualified global block migration. Prefix reuse is represented
// by equal source and destination ids and therefore requires no physical copy.
struct Dsv4CpCacheMigrationRequest {
  int32_t cache_group_id = 0;
  int64_t source_global_block_id = -1;
  int64_t destination_global_block_id = -1;
};

enum class Dsv4CpCacheMigrationKind : int8_t {
  NO_COPY = 0,
  LOCAL_COPY = 1,
  CROSS_OWNER_COPY = 2,
};

struct Dsv4CpCacheMigrationRoute {
  int32_t cache_group_id = 0;
  int32_t source_owner_rank = 0;
  int32_t destination_owner_rank = 0;
  int64_t source_global_block_id = -1;
  int64_t destination_global_block_id = -1;
  int64_t source_local_block_id = -1;
  int64_t destination_local_block_id = -1;
  Dsv4CpCacheMigrationKind kind = Dsv4CpCacheMigrationKind::NO_COPY;
};

// Immutable route plan shared by swap, host offload, and PD transfer adapters.
// It contains only value semantics and never owns tensors or a ProcessGroup.
class Dsv4CpCacheMigrationPlan final {
 public:
  const std::vector<Dsv4CpCacheMigrationRoute>& routes() const;

  std::vector<Dsv4CpCacheMigrationRoute> routes_for_owner_pair(
      int32_t source_owner_rank,
      int32_t destination_owner_rank) const;

  bool requires_copy() const;
  int64_t source_generation() const;
  int64_t destination_generation() const;

 private:
  friend class Dsv4CpCacheMigrationPlanner;

  std::vector<Dsv4CpCacheMigrationRoute> routes_;
  int64_t source_generation_ = 0;
  int64_t destination_generation_ = 1;
};

class Dsv4CpCacheMigrationPlanner final {
 public:
  explicit Dsv4CpCacheMigrationPlanner(int32_t cp_size);

  Dsv4CpCacheMigrationPlan build(
      const std::vector<Dsv4CpCacheMigrationRequest>& requests,
      int64_t source_generation) const;

 private:
  int32_t cp_size_ = 1;
};

enum class Dsv4CpCacheMigrationState : int8_t {
  PREPARED = 0,
  COPY_IN_FLIGHT = 1,
  COPY_COMPLETE = 2,
  COMMITTED = 3,
  ABORTED = 4,
};

// Metadata visibility is deliberately separate from physical copy execution.
// The caller may publish destination block tables only after commit succeeds.
class Dsv4CpCacheMigrationTransaction final {
 public:
  explicit Dsv4CpCacheMigrationTransaction(
      const Dsv4CpCacheMigrationPlan& plan);

  void begin_copy();
  void mark_copy_complete();
  int64_t commit(int64_t observed_generation);
  void abort();

  Dsv4CpCacheMigrationState state() const;
  int64_t visible_generation() const;

 private:
  bool requires_copy_ = false;
  int64_t source_generation_ = 0;
  int64_t destination_generation_ = 1;
  Dsv4CpCacheMigrationState state_ = Dsv4CpCacheMigrationState::PREPARED;
};

}  // namespace xllm
