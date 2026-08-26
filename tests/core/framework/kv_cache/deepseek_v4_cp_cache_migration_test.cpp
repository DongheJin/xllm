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

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

namespace xllm {
namespace {

TEST(Dsv4CpCacheMigrationTest, PrefixReuseKeepsStableOwnerWithoutCopy) {
  for (int32_t cp_size : {2, 4, 8}) {
    const Dsv4CpCacheMigrationPlan plan =
        Dsv4CpCacheMigrationPlanner(cp_size).build(
            {{/*cache_group_id=*/1,
              /*source_global_block_id=*/19,
              /*destination_global_block_id=*/19}},
            /*source_generation=*/7);

    ASSERT_EQ(plan.routes().size(), 1u);
    const Dsv4CpCacheMigrationRoute& route = plan.routes().front();
    EXPECT_EQ(route.source_owner_rank, 19 % cp_size);
    EXPECT_EQ(route.destination_owner_rank, route.source_owner_rank);
    EXPECT_EQ(route.source_local_block_id, 19 / cp_size);
    EXPECT_EQ(route.destination_local_block_id, route.source_local_block_id);
    EXPECT_EQ(route.kind, Dsv4CpCacheMigrationKind::NO_COPY);
    EXPECT_FALSE(plan.requires_copy());
  }
}

TEST(Dsv4CpCacheMigrationTest, ClassifiesLocalAndCrossOwnerCopies) {
  const Dsv4CpCacheMigrationPlan plan =
      Dsv4CpCacheMigrationPlanner(/*cp_size=*/4)
          .build({{/*cache_group_id=*/1, 0, 4},
                  {/*cache_group_id=*/1, 1, 2},
                  {/*cache_group_id=*/2, 5, 9},
                  {/*cache_group_id=*/3, 7, 8}},
                 /*source_generation=*/3);

  ASSERT_EQ(plan.routes().size(), 4u);
  EXPECT_EQ(plan.routes()[0].kind, Dsv4CpCacheMigrationKind::LOCAL_COPY);
  EXPECT_EQ(plan.routes()[0].source_local_block_id, 0);
  EXPECT_EQ(plan.routes()[0].destination_local_block_id, 1);
  EXPECT_EQ(plan.routes()[1].kind, Dsv4CpCacheMigrationKind::CROSS_OWNER_COPY);
  EXPECT_EQ(plan.routes()[2].kind, Dsv4CpCacheMigrationKind::LOCAL_COPY);
  EXPECT_EQ(plan.routes()[3].kind, Dsv4CpCacheMigrationKind::CROSS_OWNER_COPY);

  EXPECT_EQ(plan.routes_for_owner_pair(/*source_owner_rank=*/1,
                                       /*destination_owner_rank=*/1)
                .size(),
            1u);
  EXPECT_EQ(plan.routes_for_owner_pair(/*source_owner_rank=*/3,
                                       /*destination_owner_rank=*/0)
                .size(),
            1u);
  EXPECT_TRUE(plan.requires_copy());
  EXPECT_EQ(plan.source_generation(), 3);
  EXPECT_EQ(plan.destination_generation(), 4);
}

TEST(Dsv4CpCacheMigrationTest, EveryRouteUsesRequestedTopology) {
  for (int32_t cp_size : {2, 4, 8}) {
    std::vector<Dsv4CpCacheMigrationRequest> requests;
    for (int64_t source = 0; source < 37; ++source) {
      requests.emplace_back(Dsv4CpCacheMigrationRequest{
          /*cache_group_id=*/2,
          source,
          /*destination_global_block_id=*/source + 64});
    }
    const Dsv4CpCacheMigrationPlan plan =
        Dsv4CpCacheMigrationPlanner(cp_size).build(requests,
                                                   /*source_generation=*/0);

    for (const Dsv4CpCacheMigrationRoute& route : plan.routes()) {
      EXPECT_EQ(route.source_owner_rank,
                route.source_global_block_id % cp_size);
      EXPECT_EQ(route.destination_owner_rank,
                route.destination_global_block_id % cp_size);
      EXPECT_EQ(route.source_local_block_id,
                route.source_global_block_id / cp_size);
      EXPECT_EQ(route.destination_local_block_id,
                route.destination_global_block_id / cp_size);
    }
  }
}

TEST(Dsv4CpCacheMigrationTest, RejectsDuplicateDestinationWithinCacheGroup) {
  EXPECT_DEATH(
      Dsv4CpCacheMigrationPlanner(/*cp_size=*/4)
          .build({{/*cache_group_id=*/1, 0, 8}, {/*cache_group_id=*/1, 1, 8}},
                 /*source_generation=*/0),
      "duplicate DSV4 CP migration destination");

  const Dsv4CpCacheMigrationPlan plan =
      Dsv4CpCacheMigrationPlanner(/*cp_size=*/4)
          .build({{/*cache_group_id=*/1, 0, 8}, {/*cache_group_id=*/2, 1, 8}},
                 /*source_generation=*/0);
  EXPECT_EQ(plan.routes().size(), 2u);
}

TEST(Dsv4CpCacheMigrationTest, CommitRequiresCopyCompletion) {
  const Dsv4CpCacheMigrationPlan plan =
      Dsv4CpCacheMigrationPlanner(/*cp_size=*/4)
          .build({{/*cache_group_id=*/1, 1, 2}},
                 /*source_generation=*/11);
  Dsv4CpCacheMigrationTransaction transaction(plan);

  EXPECT_EQ(transaction.visible_generation(), 11);
  EXPECT_DEATH(transaction.commit(/*observed_generation=*/11),
               "cannot commit before copy completion");
  transaction.begin_copy();
  EXPECT_EQ(transaction.state(), Dsv4CpCacheMigrationState::COPY_IN_FLIGHT);
  EXPECT_DEATH(transaction.commit(/*observed_generation=*/11),
               "cannot commit before copy completion");
  transaction.mark_copy_complete();
  EXPECT_EQ(transaction.commit(/*observed_generation=*/11), 12);
  EXPECT_EQ(transaction.state(), Dsv4CpCacheMigrationState::COMMITTED);
  EXPECT_EQ(transaction.visible_generation(), 12);
}

TEST(Dsv4CpCacheMigrationTest, AbortAndRetryNeverPublishPartialGeneration) {
  const Dsv4CpCacheMigrationPlan plan =
      Dsv4CpCacheMigrationPlanner(/*cp_size=*/8)
          .build({{/*cache_group_id=*/3, 7, 8}},
                 /*source_generation=*/23);
  Dsv4CpCacheMigrationTransaction interrupted(plan);
  interrupted.begin_copy();
  interrupted.abort();

  EXPECT_EQ(interrupted.state(), Dsv4CpCacheMigrationState::ABORTED);
  EXPECT_EQ(interrupted.visible_generation(), 23);
  EXPECT_DEATH(interrupted.commit(/*observed_generation=*/23),
               "cannot commit before copy completion");

  Dsv4CpCacheMigrationTransaction retry(plan);
  retry.begin_copy();
  retry.mark_copy_complete();
  EXPECT_DEATH(retry.commit(/*observed_generation=*/22),
               "stale DSV4 CP cache migration generation");
  EXPECT_EQ(retry.visible_generation(), 23);
  EXPECT_EQ(retry.commit(/*observed_generation=*/23), 24);
}

TEST(Dsv4CpCacheMigrationTest, NoCopyTransactionCanCommitAfterPrepareStep) {
  const Dsv4CpCacheMigrationPlan plan =
      Dsv4CpCacheMigrationPlanner(/*cp_size=*/8)
          .build({{/*cache_group_id=*/1, 9, 9}},
                 /*source_generation=*/5);
  Dsv4CpCacheMigrationTransaction transaction(plan);

  transaction.begin_copy();
  EXPECT_EQ(transaction.state(), Dsv4CpCacheMigrationState::COPY_COMPLETE);
  EXPECT_EQ(transaction.commit(/*observed_generation=*/5), 6);
}

}  // namespace
}  // namespace xllm
