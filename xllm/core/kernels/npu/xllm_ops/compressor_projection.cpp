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

torch::Tensor compressor_projection(const torch::Tensor& x,
                                    const torch::Tensor& wkv,
                                    const torch::Tensor& wgate,
                                    int64_t coff) {
  CHECK(x.dim() == 2 || x.dim() == 3)
      << "CompressorProjection expects x to be 2-D or 3-D, got " << x.dim();
  CHECK_EQ(wkv.dim(), 2) << "CompressorProjection expects 2-D wkv";
  CHECK_EQ(wgate.dim(), 2) << "CompressorProjection expects 2-D wgate";
  CHECK(x.is_contiguous()) << "CompressorProjection expects contiguous x";
  CHECK(wkv.is_contiguous()) << "CompressorProjection expects contiguous wkv";
  CHECK(wgate.is_contiguous())
      << "CompressorProjection expects contiguous wgate";
  CHECK(x.scalar_type() == torch::kFloat16 ||
        x.scalar_type() == torch::kBFloat16)
      << "CompressorProjection expects FP16 or BF16 x";
  CHECK_EQ(wkv.scalar_type(), x.scalar_type());
  CHECK_EQ(wgate.scalar_type(), x.scalar_type());
  CHECK_EQ(wkv.device(), x.device());
  CHECK_EQ(wgate.device(), x.device());
  CHECK_EQ(wkv.sizes(), wgate.sizes());
  CHECK_EQ(wkv.size(1), x.size(x.dim() - 1));
  CHECK(coff == 1 || coff == 2)
      << "CompressorProjection expects coff to be 1 or 2";
  CHECK_EQ(wkv.size(0) % coff, 0)
      << "CompressorProjection weight rows must be divisible by coff";

  const int64_t token_count = x.numel() / x.size(x.dim() - 1);
  torch::Tensor packed_projection = torch::empty(
      {token_count, 2 * wkv.size(0)}, x.options());
  EXEC_NPU_CMD(
      aclnnCompressorProjection, x, wkv, wgate, coff, packed_projection);
  return packed_projection;
}

}  // namespace xllm::kernel::npu
