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

#include <cstdint>
#include <vector>

namespace xllm::layer {

struct Dsv4CpWavefrontNode {
  int32_t layer_id = 0;
  int32_t microchunk_id = 0;
  int32_t buffer_slot = 0;
  int64_t collective_sequence_base = 0;
};

class Dsv4CpWavefrontScheduler final {
 public:
  std::vector<Dsv4CpWavefrontNode> build(int32_t layer_count,
                                         int32_t microchunk_count,
                                         int32_t compression_alignment) const;
};

}  // namespace xllm::layer
