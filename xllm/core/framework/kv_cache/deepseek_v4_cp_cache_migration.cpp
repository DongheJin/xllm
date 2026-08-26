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

#include "framework/kv_cache/deepseek_v4_cp_cache_migration.h"

#include <glog/logging.h>

#include <limits>
#include <unordered_map>
#include <unordered_set>

#include "framework/kv_cache/deepseek_v4_cp_cache_layout.h"

namespace xllm {

const std::vector<Dsv4CpCacheMigrationRoute>& Dsv4CpCacheMigrationPlan::routes()
    const {
  return routes_;
}

std::vector<Dsv4CpCacheMigrationRoute>
Dsv4CpCacheMigrationPlan::routes_for_owner_pair(
    int32_t source_owner_rank,
    int32_t destination_owner_rank) const {
  std::vector<Dsv4CpCacheMigrationRoute> selected;
  for (const Dsv4CpCacheMigrationRoute& route : routes_) {
    if (route.source_owner_rank != source_owner_rank ||
        route.destination_owner_rank != destination_owner_rank) {
      continue;
    }
    selected.emplace_back(route);
  }
  return selected;
}

bool Dsv4CpCacheMigrationPlan::requires_copy() const {
  for (const Dsv4CpCacheMigrationRoute& route : routes_) {
    if (route.kind != Dsv4CpCacheMigrationKind::NO_COPY) {
      return true;
    }
  }
  return false;
}

int64_t Dsv4CpCacheMigrationPlan::source_generation() const {
  return source_generation_;
}

int64_t Dsv4CpCacheMigrationPlan::destination_generation() const {
  return destination_generation_;
}

Dsv4CpCacheMigrationPlanner::Dsv4CpCacheMigrationPlanner(int32_t cp_size)
    : cp_size_(cp_size) {
  CHECK_GT(cp_size_, 0) << "DSV4 CP migration requires cp_size > 0";
}

Dsv4CpCacheMigrationPlan Dsv4CpCacheMigrationPlanner::build(
    const std::vector<Dsv4CpCacheMigrationRequest>& requests,
    int64_t source_generation) const {
  CHECK_GE(source_generation, 0)
      << "DSV4 CP cache generation must be non-negative";
  CHECK_LT(source_generation, std::numeric_limits<int64_t>::max())
      << "DSV4 CP cache generation overflow";

  Dsv4CpCacheMigrationPlan plan;
  plan.source_generation_ = source_generation;
  plan.destination_generation_ = source_generation + 1;
  plan.routes_.reserve(requests.size());

  std::unordered_map<int32_t, std::unordered_set<int64_t>>
      destinations_by_group;
  for (const Dsv4CpCacheMigrationRequest& request : requests) {
    CHECK_GE(request.cache_group_id, 0)
        << "DSV4 CP cache group id must be non-negative";
    CHECK_GE(request.source_global_block_id, 0)
        << "DSV4 CP source global block id must be non-negative";
    CHECK_GE(request.destination_global_block_id, 0)
        << "DSV4 CP destination global block id must be non-negative";

    bool inserted = destinations_by_group[request.cache_group_id]
                        .insert(request.destination_global_block_id)
                        .second;
    CHECK(inserted) << "duplicate DSV4 CP migration destination: group="
                    << request.cache_group_id << ", global_block_id="
                    << request.destination_global_block_id;

    const int32_t source_owner_rank =
        static_cast<int32_t>(request.source_global_block_id % cp_size_);
    const int32_t destination_owner_rank =
        static_cast<int32_t>(request.destination_global_block_id % cp_size_);
    const Dsv4CpCacheLayout source_layout(cp_size_, source_owner_rank);
    const Dsv4CpCacheLayout destination_layout(cp_size_,
                                               destination_owner_rank);

    Dsv4CpCacheMigrationKind kind = Dsv4CpCacheMigrationKind::CROSS_OWNER_COPY;
    if (request.source_global_block_id == request.destination_global_block_id) {
      kind = Dsv4CpCacheMigrationKind::NO_COPY;
    } else if (source_owner_rank == destination_owner_rank) {
      kind = Dsv4CpCacheMigrationKind::LOCAL_COPY;
    }

    plan.routes_.emplace_back(Dsv4CpCacheMigrationRoute{
        request.cache_group_id,
        source_owner_rank,
        destination_owner_rank,
        request.source_global_block_id,
        request.destination_global_block_id,
        source_layout.local_block_id(request.source_global_block_id),
        destination_layout.local_block_id(request.destination_global_block_id),
        kind});
  }
  return plan;
}

Dsv4CpCacheMigrationTransaction::Dsv4CpCacheMigrationTransaction(
    const Dsv4CpCacheMigrationPlan& plan)
    : requires_copy_(plan.requires_copy()),
      source_generation_(plan.source_generation()),
      destination_generation_(plan.destination_generation()) {}

void Dsv4CpCacheMigrationTransaction::begin_copy() {
  CHECK(state_ == Dsv4CpCacheMigrationState::PREPARED)
      << "DSV4 CP cache migration copy can only start from PREPARED";
  state_ = requires_copy_ ? Dsv4CpCacheMigrationState::COPY_IN_FLIGHT
                          : Dsv4CpCacheMigrationState::COPY_COMPLETE;
}

void Dsv4CpCacheMigrationTransaction::mark_copy_complete() {
  CHECK(state_ == Dsv4CpCacheMigrationState::COPY_IN_FLIGHT)
      << "DSV4 CP cache migration completion requires an in-flight copy";
  state_ = Dsv4CpCacheMigrationState::COPY_COMPLETE;
}

int64_t Dsv4CpCacheMigrationTransaction::commit(int64_t observed_generation) {
  CHECK_EQ(observed_generation, source_generation_)
      << "stale DSV4 CP cache migration generation";
  CHECK(state_ == Dsv4CpCacheMigrationState::COPY_COMPLETE)
      << "DSV4 CP cache migration cannot commit before copy completion";
  state_ = Dsv4CpCacheMigrationState::COMMITTED;
  return destination_generation_;
}

void Dsv4CpCacheMigrationTransaction::abort() {
  CHECK(state_ != Dsv4CpCacheMigrationState::COMMITTED)
      << "committed DSV4 CP cache migration cannot be aborted";
  state_ = Dsv4CpCacheMigrationState::ABORTED;
}

Dsv4CpCacheMigrationState Dsv4CpCacheMigrationTransaction::state() const {
  return state_;
}

int64_t Dsv4CpCacheMigrationTransaction::visible_generation() const {
  return state_ == Dsv4CpCacheMigrationState::COMMITTED
             ? destination_generation_
             : source_generation_;
}

}  // namespace xllm
