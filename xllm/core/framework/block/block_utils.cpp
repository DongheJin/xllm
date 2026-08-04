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

#include "framework/block/block_utils.h"

#include <glog/logging.h>

namespace xllm {

int64_t get_swa_blocks_per_seq(int64_t window_size, int64_t block_size) {
  CHECK_GT(window_size, 0) << "sliding_window_size must be positive";
  CHECK_GT(block_size, 0) << "block_size must be positive";
  // Align with vLLM/vllm-ascend sliding-window semantics: keep enough
  // contiguous KV blocks to cover `sliding_window - 1` history tokens plus
  // the current block being written.
  return (window_size - 1) / block_size + 1;
}

int64_t get_swa_pool_num_blocks(int64_t swa_blocks_per_seq,
                                int64_t max_seqs,
                                int64_t max_tokens_per_batch,
                                int64_t block_size) {
  CHECK_GT(swa_blocks_per_seq, 0) << "swa_blocks_per_seq must be positive";
  CHECK_GT(max_seqs, 0) << "max_seqs must be positive";
  CHECK_GT(max_tokens_per_batch, 0) << "max_tokens_per_batch must be positive";
  CHECK_GT(block_size, 0) << "block_size must be positive";

  const int64_t burst_blocks = (max_tokens_per_batch - 1) / block_size + 1;
  // Chunked prefill allocates the next chunk before the previous chunk's
  // slid-out blocks can be released. Keep capacity for the live windows and
  // both adjacent chunk bursts while preserving transactional allocation.
  return swa_blocks_per_seq * max_seqs + 2 * burst_blocks + max_seqs + 2;
}

}  // namespace xllm
