# vlm_master_test is a standalone manual test binary and has no registered
# CTest target yet.

include(cc_test)

cc_test(
  NAME
    spawn_worker_protocol_test
  SRCS
    spawn_worker_protocol_test.cpp
  DEPS
    GTest::gtest_main
)

cc_test(
  NAME
    overlap_result_collection_test
  SRCS
    overlap_result_collection_test.cpp
  DEPS
    GTest::gtest_main
  NO_NPU_RUNTIME
)
