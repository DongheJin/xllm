include(cc_test)

cc_test(
  NAME
    deepseek_v4_cp_wavefront_test
  NO_NPU_RUNTIME
  SRCS
    deepseek_v4_cp_wavefront_test.cpp
  DEPS
    :deepseek_v4_cp_wavefront
    glog::glog
    GTest::gtest_main
)

cc_test(
  NAME
    deepseek_v4_cp_buffer_pool_test
  NO_NPU_RUNTIME
  SRCS
    deepseek_v4_cp_buffer_pool_test.cpp
  DEPS
    :deepseek_v4_cp_buffer_pool
    glog::glog
    torch
    GTest::gtest_main
)

cc_test(
  NAME
    deepseek_v4_cp_attention_exchange_test
  NO_NPU_RUNTIME
  SRCS
    deepseek_v4_cp_attention_exchange_test.cpp
  DEPS
    :deepseek_v4_cp_attention_exchange
    :deepseek_v4_cp_ownership
    :parallel_state
    glog::glog
    torch
    GTest::gtest_main
)

cc_test(
  NAME
    deepseek_v4_cp_owner_attention_test
  NO_NPU_RUNTIME
  SRCS
    deepseek_v4_cp_owner_attention_test.cpp
  DEPS
    :deepseek_v4_cp_attention_exchange
    :deepseek_v4_cp_owner_attention
    glog::glog
    torch
    GTest::gtest_main
)

cc_test(
  NAME
    deepseek_v4_cp_ownership_test
  NO_NPU_RUNTIME
  SRCS
    deepseek_v4_cp_ownership_test.cpp
  DEPS
    :deepseek_v4_cp_ownership
    :parallel_state
    glog::glog
    torch
    GTest::gtest_main
)

cc_test(
  NAME
    deepseek_v4_cp_execution_test
  SRCS
    deepseek_v4_cp_execution_test.cpp
  DEPS
    :npu_torch_layers
    :parallel_state
    glog::glog
    torch
    GTest::gtest_main
)

cc_test(
  NAME
    deepseek_v4_cp_metadata_test
  NO_NPU_RUNTIME
  SRCS
    deepseek_v4_cp_metadata_test.cpp
  DEPS
    :npu_torch_layers
    :parallel_state
    glog::glog
    torch
    GTest::gtest_main
)

cc_test(
  NAME
    npu_deepseek_v4_indexer_test
  SRCS
    deepseek_v4_indexer_tests.cpp
  DEPS
    :npu_torch_layers
    :state_dict
    :xllm_ops
    glog::glog
    torch
    GTest::gtest_main
)

target_link_libraries(npu_deepseek_v4_indexer_test
                      PRIVATE
                      ascendcl
                      hccl
                      c_sec
                      nnopbase
                      atb)

target_link_options(npu_deepseek_v4_indexer_test PRIVATE
  "-Wl,--whole-archive"
  "${CMAKE_BINARY_DIR}/third_party/spdlog/libspdlog.a"
  "-Wl,--no-whole-archive")

cc_test(
  NAME
    npu_linear_w8a8_dynamic_test
  SRCS
    linear_w8a8_dynamic_tests.cpp
    tests_utils.cpp
  DEPS
    :npu_torch_layers
    :common_layers
    :parallel_state
    :model
    :state_dict
    :xllm_ops
    glog::glog
    torch
    GTest::gtest_main
)

target_link_libraries(npu_linear_w8a8_dynamic_test
                      PRIVATE
                      ascendcl
                      hccl
                      c_sec
                      nnopbase
                      atb)

target_link_options(npu_linear_w8a8_dynamic_test PRIVATE
  "-Wl,--whole-archive"
  "${CMAKE_BINARY_DIR}/third_party/spdlog/libspdlog.a"
  "-Wl,--no-whole-archive")
