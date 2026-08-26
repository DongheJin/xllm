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

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "core/kernels/npu/xllm_ops/xllm_ops_api.h"

namespace xllm::kernel::npu {
namespace {

void check_int32_metadata(const torch::Tensor& tensor, const char* name) {
  CHECK(tensor.defined()) << name << " must be defined";
  CHECK(tensor.is_contiguous()) << name << " must be contiguous";
  CHECK_EQ(tensor.scalar_type(), torch::kInt32)
      << name << " must have int32 dtype";
}

bool has_op_api(const char* workspace_api, const char* launch_api) {
  return aclnn::detail::get_op_api_func_addr(workspace_api) != nullptr &&
         aclnn::detail::get_op_api_func_addr(launch_api) != nullptr;
}

}  // namespace

bool has_split_compressor() {
  static const bool is_available =
      has_op_api("aclnnCompressorProjectionGetWorkspaceSize",
                 "aclnnCompressorProjection") &&
      has_op_api("aclnnCompressorCoreGetWorkspaceSize", "aclnnCompressorCore");
  return is_available;
}

torch::Tensor compressor_core(const torch::Tensor& packed_projection,
                              torch::Tensor& kv_state,
                              torch::Tensor& score_state,
                              const torch::Tensor& ape,
                              const torch::Tensor& kv_block_table,
                              const torch::Tensor& score_block_table,
                              const torch::Tensor& cu_seqlens,
                              const torch::Tensor& seqused,
                              const torch::Tensor& start_pos,
                              int64_t output_row_count,
                              int64_t cmp_ratio,
                              int64_t coff) {
  CHECK(has_split_compressor())
      << "CompressorProjection/CompressorCore custom operators are missing";
  CHECK_EQ(packed_projection.dim(), 2)
      << "CompressorCore expects 2-D packed_projection";
  CHECK(packed_projection.is_contiguous())
      << "CompressorCore expects contiguous packed_projection";
  CHECK(packed_projection.scalar_type() == torch::kFloat16 ||
        packed_projection.scalar_type() == torch::kBFloat16)
      << "CompressorCore expects FP16 or BF16 packed_projection";
  CHECK(kv_state.scalar_type() == torch::kFloat32 ||
        kv_state.scalar_type() == torch::kBFloat16)
      << "CompressorCore expects FP32 or BF16 kv_state";
  CHECK_EQ(score_state.scalar_type(), kv_state.scalar_type())
      << "CompressorCore state tensors must use the same dtype";
  CHECK_EQ(ape.scalar_type(), torch::kFloat32);
  CHECK_GE(kv_state.dim(), 1);
  CHECK_GE(score_state.dim(), 1);
  CHECK_EQ(ape.dim(), 2);
  CHECK(kv_state.is_contiguous());
  CHECK(score_state.is_contiguous());
  CHECK(ape.is_contiguous());
  CHECK_EQ(kv_state.device(), packed_projection.device());
  CHECK_EQ(score_state.device(), packed_projection.device());
  CHECK_EQ(ape.device(), packed_projection.device());
  check_int32_metadata(kv_block_table, "kv_block_table");
  check_int32_metadata(score_block_table, "score_block_table");
  check_int32_metadata(cu_seqlens, "cu_seqlens");
  check_int32_metadata(seqused, "seqused");
  check_int32_metadata(start_pos, "start_pos");
  CHECK_EQ(kv_block_table.device(), packed_projection.device());
  CHECK_EQ(score_block_table.device(), packed_projection.device());
  CHECK_EQ(cu_seqlens.device(), packed_projection.device());
  CHECK_EQ(seqused.device(), packed_projection.device());
  CHECK_EQ(start_pos.device(), packed_projection.device());
  CHECK_GE(output_row_count, 0);
  CHECK_GT(cmp_ratio, 0);
  CHECK(coff == 1 || coff == 2) << "CompressorCore expects coff to be 1 or 2";
  CHECK_EQ(packed_projection.size(1) % (2 * coff), 0)
      << "CompressorCore packed width is incompatible with coff";
  const int64_t head_dim = packed_projection.size(1) / (2 * coff);
  CHECK_EQ(ape.size(0), cmp_ratio);
  CHECK_EQ(ape.size(1), coff * head_dim);
  CHECK_EQ(kv_state.size(kv_state.dim() - 1), coff * head_dim);
  CHECK_EQ(score_state.size(score_state.dim() - 1), coff * head_dim);
  CHECK_EQ(cu_seqlens.numel(), seqused.numel() + 1);
  CHECK_EQ(start_pos.numel(), seqused.numel());

  torch::Tensor pre_norm =
      torch::empty({output_row_count, head_dim},
                   packed_projection.options().dtype(torch::kFloat32));
  EXEC_NPU_CMD(aclnnCompressorCore,
               packed_projection,
               kv_state,
               score_state,
               ape,
               kv_block_table,
               score_block_table,
               cu_seqlens,
               seqused,
               start_pos,
               output_row_count,
               cmp_ratio,
               coff,
               pre_norm);
  return pre_norm;
}

}  // namespace xllm::kernel::npu
