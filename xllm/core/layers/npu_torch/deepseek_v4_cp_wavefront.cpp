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

#include "layers/npu_torch/deepseek_v4_cp_wavefront.h"

#include <glog/logging.h>

namespace xllm::layer {

std::vector<Dsv4CpWavefrontNode> Dsv4CpWavefrontScheduler::build(
    int32_t layer_count,
    int32_t microchunk_count,
    int32_t compression_alignment) const {
  CHECK_GT(layer_count, 0);
  CHECK_GT(microchunk_count, 0);
  CHECK_GT(compression_alignment, 0);

  const int64_t node_count = static_cast<int64_t>(layer_count) *
                             static_cast<int64_t>(microchunk_count);
  std::vector<Dsv4CpWavefrontNode> nodes;
  nodes.reserve(static_cast<size_t>(node_count));
  int64_t sequence_id = 0;
  for (int32_t diagonal = 0; diagonal < layer_count + microchunk_count - 1;
       ++diagonal) {
    const int32_t first_layer =
        std::max<int32_t>(0, diagonal - microchunk_count + 1);
    const int32_t last_layer = std::min<int32_t>(layer_count - 1, diagonal);
    for (int32_t layer_id = first_layer; layer_id <= last_layer; ++layer_id) {
      const int32_t microchunk_id = diagonal - layer_id;
      CHECK_GE(microchunk_id, 0);
      CHECK_LT(microchunk_id, microchunk_count);
      nodes.push_back({layer_id,
                       microchunk_id,
                       (layer_id + microchunk_id) % 2,
                       sequence_id * compression_alignment});
      ++sequence_id;
    }
  }
  CHECK_EQ(nodes.size(), static_cast<size_t>(node_count));
  return nodes;
}

}  // namespace xllm::layer
