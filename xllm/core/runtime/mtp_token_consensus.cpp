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

#include "mtp_token_consensus.h"

#include "framework/parallel_state/parallel_args.h"

namespace xllm {

void broadcast_mtp_tokens(torch::Tensor& tokens,
                          const ParallelArgs& parallel_args,
                          int32_t root_rank) {
  if (!tokens.defined()) {
    return;
  }

  tokens = tokens.contiguous();
  ProcessGroup* tp_group = parallel_args.tp_group_ != nullptr
                               ? parallel_args.tp_group_
                               : parallel_args.process_group_;
  if (tp_group != nullptr && tp_group->world_size() > 1) {
    tp_group->broadcast(tokens, root_rank);
  }

  ProcessGroup* cp_group = parallel_args.cp_group_;
  if (cp_group != nullptr && cp_group != tp_group &&
      cp_group->world_size() > 1) {
    cp_group->broadcast(tokens, root_rank);
  }
}

}  // namespace xllm
