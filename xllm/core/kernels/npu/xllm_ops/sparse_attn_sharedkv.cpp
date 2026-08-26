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

#include <torch/library.h>

#include <limits>
#include <string>

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "xllm_ops_api.h"

namespace xllm::kernel::npu {
namespace {

bool has_op_api(const char* workspace_api, const char* launch_api) {
  return aclnn::detail::get_op_api_func_addr(workspace_api) != nullptr &&
         aclnn::detail::get_op_api_func_addr(launch_api) != nullptr;
}

void check_sparse_attn_sharedkv_shape_and_dtype(const at::Tensor& q,
                                                c10::string_view layout_q,
                                                c10::string_view layout_kv) {
  TORCH_CHECK(q.dim() >= 1,
              "Input tensor q's dim num should be at least 1, actual ",
              q.dim(),
              ".");
  TORCH_CHECK(q.dtype() == at::kHalf || q.dtype() == at::kBFloat16,
              "q should be FLOAT16 or BFLOAT16.");
  TORCH_CHECK(!layout_q.empty(), "layout_q should not be empty.");
  TORCH_CHECK(!layout_kv.empty(), "layout_kv should not be empty.");
}

at::Tensor construct_sparse_attn_sharedkv_attn_out_tensor(const at::Tensor& q) {
  return at::empty(q.sizes(), q.options().dtype(q.dtype()));
}

at::Tensor construct_sparse_attn_sharedkv_softmax_lse_tensor(
    const at::Tensor& q,
    bool return_softmax_lse) {
  if (!return_softmax_lse) {
    return at::empty({0}, q.options().dtype(at::kFloat));
  }
  auto softmax_lse_shape = q.sizes().vec();
  softmax_lse_shape.back() = 1;
  return at::empty(softmax_lse_shape, q.options().dtype(at::kFloat));
}

int64_t get_sparse_attn_sharedkv_kv_stride(
    const c10::optional<at::Tensor>& kv) {
  if (!kv.has_value() || !kv.value().defined() || kv.value().dim() == 0) {
    return 0;
  }
  return kv.value().stride(0);
}

int64_t max_sequence_length(const c10::optional<at::Tensor>& lengths) {
  if (!lengths.has_value() || !lengths.value().defined() ||
      lengths.value().numel() == 0) {
    return 0;
  }
  return lengths.value().to(torch::kCPU).to(torch::kInt64).max().item<int64_t>();
}

}  // namespace

bool has_owner_sparse_attention() {
  static const bool is_available =
      has_op_api("aclnnSparseAttnSharedkvMetadataGetWorkspaceSize",
                 "aclnnSparseAttnSharedkvMetadata") &&
      has_op_api("aclnnSparseAttnSharedkvGetWorkspaceSize",
                 "aclnnSparseAttnSharedkv");
  return is_available;
}

std::tuple<at::Tensor, at::Tensor> sparse_attn_sharedkv(
    const at::Tensor& q,
    const c10::optional<at::Tensor>& ori_kv,
    const c10::optional<at::Tensor>& cmp_kv,
    const c10::optional<at::Tensor>& ori_sparse_indices,
    const c10::optional<at::Tensor>& cmp_sparse_indices,
    const c10::optional<at::Tensor>& ori_block_table,
    const c10::optional<at::Tensor>& cmp_block_table,
    const c10::optional<at::Tensor>& cu_seqlens_q,
    const c10::optional<at::Tensor>& cu_seqlens_ori_kv,
    const c10::optional<at::Tensor>& cu_seqlens_cmp_kv,
    const c10::optional<at::Tensor>& seqused_q,
    const c10::optional<at::Tensor>& seqused_kv,
    const c10::optional<at::Tensor>& sinks,
    const c10::optional<at::Tensor>& metadata,
    double softmax_scale,
    int64_t cmp_ratio,
    int64_t ori_mask_mode,
    int64_t cmp_mask_mode,
    int64_t ori_win_left,
    int64_t ori_win_right,
    c10::string_view layout_q,
    c10::string_view layout_kv,
    bool return_softmax_lse) {
  check_sparse_attn_sharedkv_shape_and_dtype(q, layout_q, layout_kv);
  at::Tensor attn_out = construct_sparse_attn_sharedkv_attn_out_tensor(q);
  at::Tensor softmax_lse =
      construct_sparse_attn_sharedkv_softmax_lse_tensor(q, return_softmax_lse);

  std::string layout_q_str = std::string(layout_q);
  std::string layout_kv_str = std::string(layout_kv);
  auto layout_q_arg = const_cast<char*>(layout_q_str.c_str());
  auto layout_kv_arg = const_cast<char*>(layout_kv_str.c_str());
  const int64_t ori_kv_stride = get_sparse_attn_sharedkv_kv_stride(ori_kv);
  const int64_t cmp_kv_stride = get_sparse_attn_sharedkv_kv_stride(cmp_kv);

  EXEC_NPU_CMD(aclnnSparseAttnSharedkv,
               q,
               ori_kv,
               cmp_kv,
               ori_sparse_indices,
               cmp_sparse_indices,
               ori_block_table,
               cmp_block_table,
               cu_seqlens_q,
               cu_seqlens_ori_kv,
               cu_seqlens_cmp_kv,
               seqused_q,
               seqused_kv,
               sinks,
               metadata,
               softmax_scale,
               cmp_ratio,
               ori_mask_mode,
               cmp_mask_mode,
               ori_kv_stride,
               cmp_kv_stride,
               ori_win_left,
               ori_win_right,
               layout_q_arg,
               layout_kv_arg,
               return_softmax_lse,
               attn_out,
               softmax_lse);
  return std::make_tuple(attn_out, softmax_lse);
}

std::tuple<at::Tensor, at::Tensor> sparse_attn_sharedkv_owner(
    const at::Tensor& q,
    const c10::optional<at::Tensor>& ori_kv,
    const c10::optional<at::Tensor>& cmp_kv,
    const c10::optional<at::Tensor>& ori_sparse_indices,
    const c10::optional<at::Tensor>& cmp_sparse_indices,
    const c10::optional<at::Tensor>& ori_block_table,
    const c10::optional<at::Tensor>& cmp_block_table,
    const c10::optional<at::Tensor>& cu_seqlens_q,
    const c10::optional<at::Tensor>& seqused_kv,
    const c10::optional<at::Tensor>& sinks,
    int64_t ori_topk,
    int64_t cmp_topk,
    int64_t cmp_ratio,
    double softmax_scale,
    int64_t ori_win_left) {
  TORCH_CHECK(has_owner_sparse_attention(),
              "SparseAttnSharedkv owner custom operators are missing");
  check_sparse_attn_sharedkv_shape_and_dtype(q, "TND", "PA_ND");
  TORCH_CHECK(q.dim() == 3, "owner sparse attention expects TND query");
  TORCH_CHECK(cu_seqlens_q.has_value() && cu_seqlens_q.value().defined(),
              "owner sparse attention requires cu_seqlens_q");
  TORCH_CHECK(seqused_kv.has_value() && seqused_kv.value().defined(),
              "owner sparse attention requires seqused_kv");
  TORCH_CHECK(ori_kv.has_value() && ori_kv.value().defined(),
              "owner sparse attention requires ori_kv");
  TORCH_CHECK(ori_sparse_indices.has_value() &&
                  ori_sparse_indices.value().defined(),
              "owner sparse attention requires ori_sparse_indices");
  TORCH_CHECK_EQ(cu_seqlens_q.value().numel(), seqused_kv.value().numel() + 1);
  TORCH_CHECK(seqused_kv.value().numel() > 0,
              "owner sparse attention requires a non-empty batch");
  TORCH_CHECK(q.size(0) > 0,
              "owner sparse attention requires non-empty query rows");
  TORCH_CHECK_EQ(ori_sparse_indices.value().size(0), q.size(0));
  TORCH_CHECK_EQ(ori_sparse_indices.value().dim(), 3);
  TORCH_CHECK_EQ(ori_sparse_indices.value().size(1), 1);
  TORCH_CHECK_EQ(ori_sparse_indices.value().size(2), ori_topk);
  TORCH_CHECK_EQ(ori_sparse_indices.value().scalar_type(), at::kInt);
  TORCH_CHECK(ori_block_table.has_value() &&
                  ori_block_table.value().defined(),
              "owner sparse attention requires ori_block_table");
  TORCH_CHECK_EQ(ori_block_table.value().scalar_type(), at::kInt);
  TORCH_CHECK(sinks.has_value() && sinks.value().defined(),
              "owner sparse attention requires sinks");
  TORCH_CHECK_EQ(sinks.value().scalar_type(), at::kFloat);
  TORCH_CHECK_EQ(sinks.value().dim(), 1);
  TORCH_CHECK_EQ(sinks.value().size(0), q.size(1));
  TORCH_CHECK(ori_win_left >= 0,
              "owner sparse attention requires a non-negative SWA window");

  const bool has_cmp_kv = cmp_kv.has_value() && cmp_kv.value().defined();
  TORCH_CHECK_EQ(has_cmp_kv,
                 cmp_sparse_indices.has_value() &&
                     cmp_sparse_indices.value().defined());
  TORCH_CHECK_EQ(has_cmp_kv,
                 cmp_block_table.has_value() && cmp_block_table.value().defined());
  if (has_cmp_kv) {
    TORCH_CHECK_EQ(cmp_sparse_indices.value().dim(), 3);
    TORCH_CHECK_EQ(cmp_sparse_indices.value().size(0), q.size(0));
    TORCH_CHECK_EQ(cmp_sparse_indices.value().size(1), 1);
    TORCH_CHECK_GE(cmp_topk, 0);
    if (cmp_topk > 0) {
      TORCH_CHECK_EQ(cmp_sparse_indices.value().size(2), cmp_topk);
    } else {
      // A zero cmp_topk selects the C128 CFA path. Its owner indices are
      // still required for rank-local cache addressing, but their width is a
      // physical capacity rather than a QLI top-k value.
      TORCH_CHECK(cmp_ratio == 128,
                  "cmp_topk=0 is only valid for C128 owner attention");
      TORCH_CHECK_GT(cmp_sparse_indices.value().size(2), 0);
    }
    TORCH_CHECK_EQ(cmp_sparse_indices.value().scalar_type(), at::kInt);
    TORCH_CHECK_EQ(cmp_block_table.value().scalar_type(), at::kInt);
  }

  const int64_t batch_size = seqused_kv.value().numel();
  const int64_t max_seqlen_q =
      (cu_seqlens_q.value()
           .slice(/*dim=*/0, /*start=*/1, /*end=*/cu_seqlens_q.value().numel()) -
       cu_seqlens_q.value().slice(
           /*dim=*/0, /*start=*/0, /*end=*/cu_seqlens_q.value().numel() - 1))
          .to(torch::kCPU)
          .to(torch::kInt64)
          .max()
          .item<int64_t>();
  const int64_t max_seqlen_kv = max_sequence_length(seqused_kv);
  const int64_t num_heads_q = q.size(1);
  const int64_t head_dim = q.size(2);

  const c10::optional<at::Tensor> empty;
  at::Tensor metadata = sparse_attn_sharedkv_metadata(
      num_heads_q,
      /*num_heads_kv=*/1,
      head_dim,
      cu_seqlens_q,
      empty,
      empty,
      empty,
      seqused_kv,
      batch_size,
      max_seqlen_q,
      max_seqlen_kv,
      ori_topk,
      cmp_topk,
      cmp_ratio,
      /*ori_mask_mode=*/4,
      /*cmp_mask_mode=*/3,
      ori_win_left,
      /*ori_win_right=*/0,
      "TND",
      "PA_ND",
      /*has_ori_kv=*/true,
      /*has_cmp_kv=*/has_cmp_kv);

  // Owner-local sparse attention may leave entries untouched when a tiling
  // partition has no local contribution for them. Keep those entries as the
  // additive identity instead of exposing uninitialized device memory.
  at::Tensor attn_out = torch::zeros_like(q);
  // A tiling partition may not write every LSE entry when its local owner
  // has no valid key for a query row. -inf is the identity for the later
  // cross-CP logsumexp merge; exposing at::empty() here can inject NaNs into
  // otherwise valid rows.
  auto softmax_lse_shape = q.sizes().vec();
  softmax_lse_shape.back() = 1;
  at::Tensor softmax_lse = torch::full(
      softmax_lse_shape,
      -std::numeric_limits<float>::infinity(),
      q.options().dtype(at::kFloat));
  const int64_t ori_kv_stride = get_sparse_attn_sharedkv_kv_stride(ori_kv);
  const int64_t cmp_kv_stride = get_sparse_attn_sharedkv_kv_stride(cmp_kv);
  int64_t ori_mask_mode = 4;
  int64_t cmp_mask_mode = 3;
  int64_t ori_win_right = 0;
  bool return_softmax_lse = true;
  std::string layout_q = "TND";
  std::string layout_kv = "PA_ND";
  auto layout_q_arg = const_cast<char*>(layout_q.c_str());
  auto layout_kv_arg = const_cast<char*>(layout_kv.c_str());
  EXEC_NPU_CMD(aclnnSparseAttnSharedkv,
               q,
               ori_kv,
               cmp_kv,
               ori_sparse_indices,
               cmp_sparse_indices,
               ori_block_table,
               cmp_block_table,
               cu_seqlens_q,
               empty,
               empty,
               empty,
               seqused_kv,
               sinks,
               metadata,
               softmax_scale,
               cmp_ratio,
               ori_mask_mode,
               cmp_mask_mode,
               ori_kv_stride,
               cmp_kv_stride,
               ori_win_left,
               ori_win_right,
               layout_q_arg,
               layout_kv_arg,
               return_softmax_lse,
               attn_out,
               softmax_lse);
  torch::Tensor valid_rows =
      ori_sparse_indices.value().ge(0).any({1, 2});
  if (has_cmp_kv) {
    valid_rows = valid_rows.logical_or(
        cmp_sparse_indices.value().ge(0).any({1, 2}));
  }
  attn_out = torch::where(valid_rows.view({-1, 1, 1}),
                          attn_out,
                          torch::zeros_like(attn_out));
  softmax_lse = torch::where(
      valid_rows.view({-1, 1, 1}),
      softmax_lse,
      sinks.value().view({1, -1, 1}).expand(softmax_lse.sizes()));
  return std::make_tuple(attn_out, softmax_lse);
}

}  // namespace xllm::kernel::npu
