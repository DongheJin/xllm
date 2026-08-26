/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <cstdint>

namespace xllm {

// Schedule-overlap outputs are published by the driver at the start of each
// DP-local worker group. EPLB additionally publishes per-rank expert load.
inline uint32_t overlap_result_collection_stride(uint32_t dp_local_size,
                                                 bool enable_eplb) {
  return enable_eplb ? 1 : dp_local_size;
}

inline uint32_t overlap_driver_result_index(uint32_t worker_rank,
                                            uint32_t collection_stride) {
  return worker_rank / collection_stride;
}

}  // namespace xllm
