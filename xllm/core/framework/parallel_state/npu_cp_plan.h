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

#include <torch/torch.h>

#include <cstdint>
#include <vector>

namespace xllm {

class ProcessGroup;

enum class CpProjectionGatherMode : int8_t {
  BUNDLED = 0,
  SEQUENTIAL = 1,
};
struct ModelInputParams;
struct ParallelInput;
struct ForwardInput;
enum class KvSlotLayout : int8_t;
namespace npu {
class GraphPersistentParam;
}

// Global-real inputs from which one leaf worker derives its CP execution plan.
struct CpPlanInput {
  std::vector<int32_t> q_seq_lens;
  torch::Tensor position_ids;
  std::vector<int32_t> prefix_token_counts;
  torch::Tensor block_tables;
  bool has_prefix_slots = false;
  bool q_seq_lens_are_cumulative = false;
  bool kv_seq_lens_are_cumulative = false;
};

// Typed parallel configuration used by CP planning. JSON and global config
// reads are resolved by the worker before entering the planner.
struct CpPlanConfig {
  int32_t cp_size = 1;
  int32_t cp_rank = 0;
  int32_t kv_split_size = 1;
  int32_t kv_split_rank = 0;
  int32_t block_size = 0;
  int32_t attention_tp_size = 1;
  int32_t attention_tp_rank = 0;
  int32_t attention_cp_size = 1;
  int32_t attention_cp_group_size = 1;
  int32_t moe_ep_size = 1;
  int32_t expert_parallel_degree = 1;
  int32_t num_experts_per_token = 1;
  // Dynamic EP degree 3 only applies to prefill.
  bool is_prefill = true;
  CpProjectionGatherMode projection_gather_mode =
      CpProjectionGatherMode::BUNDLED;
  torch::Device device = torch::kCPU;
  torch::ScalarType dtype = torch::kFloat;
};

// Per-worker static inputs for NpuCpPlan::prepare().
struct CpPlanRuntimeConfig {
  bool enabled = false;
  CpPlanConfig plan_config;
  ProcessGroup* cp_group = nullptr;
  bool has_prefix_slots = false;
  // DSV4 owns global cache and attention metadata. ATB model-side CP keeps the
  // legacy remap behavior when this flag is false.
  bool model_managed_global_cache = false;
};

// Pre-model mapping from global-real rows to this rank's local-padded rows.
struct CpInputShardMeta {
  int64_t global_real_token_count = 0;
  int64_t local_real_token_count = 0;
  int64_t local_padded_token_count = 0;
  std::vector<int32_t> local_real_seq_lens;
  std::vector<int32_t> local_padded_seq_lens;
  torch::Tensor input_source_indices;
  torch::Tensor input_destination_indices;
  torch::Tensor local_position_ids;
};

// All local-padded metadata consumed by CP attention and its ATB graph nodes.
struct CpAttentionMeta {
  std::vector<int32_t> host_q_seq_lens;
  std::vector<int32_t> host_kv_seq_lens;
  std::vector<int32_t> host_q_cu_seq_lens;
  int32_t q_max_seq_len = 0;
  int32_t kv_max_seq_len = 0;

  torch::Tensor q_seq_lens;
  torch::Tensor kv_seq_lens;
  torch::Tensor q_cu_seq_lens;
  torch::Tensor query_balance_indices;
  torch::Tensor attention_output_reorder_indices;
  torch::Tensor kv_reorder_indices;
  torch::Tensor prev_kv_gather_indices;
  torch::Tensor next_kv_gather_indices;
  torch::Tensor prev_query_cu_seq_lens;
  torch::Tensor next_query_cu_seq_lens;
  torch::Tensor prev_key_cu_seq_lens;
  torch::Tensor next_key_cu_seq_lens;
  torch::Tensor prefix_cache_slots;
};

// CP-to-EP bridge metadata, named by the graph input each tensor serves.
struct CpEpMeta {
  torch::Tensor attention_tp_padding_indices;
  torch::Tensor attention_tp_unpadding_indices;
  torch::Tensor ffn_padding_indices;
  torch::Tensor ffn_unpadding_indices;
  torch::Tensor lm_head_skip_padding_indices;
  torch::Tensor prenorm_gather_indices;
  torch::Tensor attention_padding_indices;
  torch::Tensor attention_unpadding_indices;
  torch::Tensor dynamic_ep_indices;
  torch::Tensor moe_indices;
  torch::Tensor expert_array;
};

// Post-model mapping from rank-major gathered rows back to global-real order.
struct CpOutputMergeMeta {
  int64_t global_padded_token_count = 0;
  torch::Tensor output_restore_indices;
};

// Backend-neutral zigzag row ownership. It only transforms dim 0 and does not
// contain attention, cache, compressor, or MoE semantics.
class CpRowLayout final {
 public:
  CpRowLayout() = default;

  static CpRowLayout build(const CpPlanInput& input,
                           int32_t cp_size,
                           int32_t cp_rank,
                           const torch::Device& device);

  torch::Tensor shard_rows(const torch::Tensor& global_rows,
                           const c10::Scalar& pad_value) const;
  torch::Tensor pack_local_real_rows(
      const torch::Tensor& local_padded_rows) const;
  torch::Tensor scatter_local_real_rows(const torch::Tensor& local_real_rows,
                                        const c10::Scalar& pad_value) const;
  torch::Tensor gather_global_rows(const torch::Tensor& local_padded_rows,
                                   ProcessGroup* cp_group) const;
  std::vector<torch::Tensor> gather_global_rows_bundle(
      torch::TensorList local_padded_tensors,
      ProcessGroup* cp_group) const;

  int32_t cp_size() const { return cp_size_; }
  int32_t cp_rank() const { return cp_rank_; }
  int64_t global_real_token_count() const {
    return input_shard_meta_.global_real_token_count;
  }
  int64_t local_real_token_count() const {
    return input_shard_meta_.local_real_token_count;
  }
  int64_t local_padded_token_count() const {
    return input_shard_meta_.local_padded_token_count;
  }
  int64_t gathered_padded_token_count() const {
    return static_cast<int64_t>(cp_size_) * local_padded_token_count();
  }
  bool has_empty_rank() const { return has_empty_rank_; }
  uint64_t signature() const { return signature_; }

  const CpInputShardMeta& input_shard_meta() const { return input_shard_meta_; }
  const CpOutputMergeMeta& output_merge_meta() const {
    return output_merge_meta_;
  }

 private:
  friend class NpuCpPlan;

  CpRowLayout to(const torch::Device& device) const;

  int32_t cp_size_ = 1;
  int32_t cp_rank_ = 0;
  bool has_empty_rank_ = false;
  uint64_t signature_ = 0;
  CpInputShardMeta input_shard_meta_;
  CpOutputMergeMeta output_merge_meta_;
};

// Complete execution plan for one model-side NPU CP forward.
class NpuCpPlan final {
 public:
  NpuCpPlan() = default;

  static NpuCpPlan build(const CpPlanInput& input, const CpPlanConfig& config);

  bool enabled() const { return enabled_; }
  int32_t size() const { return cp_size_; }
  int32_t rank() const { return cp_rank_; }
  int64_t global_real_token_count() const {
    return row_layout_.global_real_token_count();
  }
  int64_t local_padded_token_count() const {
    return row_layout_.local_padded_token_count();
  }
  int64_t recovered_token_count() const {
    return static_cast<int64_t>(size()) * local_padded_token_count();
  }

  const CpRowLayout& row_layout() const { return row_layout_; }
  const CpInputShardMeta& input_shard_meta() const {
    return row_layout_.input_shard_meta();
  }
  const CpAttentionMeta& attention_meta() const { return attention_meta_; }
  const CpEpMeta& cp_ep_meta() const { return cp_ep_meta_; }
  const CpOutputMergeMeta& output_merge_meta() const {
    return row_layout_.output_merge_meta();
  }
  const std::vector<int32_t>& global_q_seq_lens() const {
    return global_q_seq_lens_;
  }
  const std::vector<int32_t>& global_kv_seq_lens() const {
    return global_kv_seq_lens_;
  }
  ProcessGroup* process_group() const { return cp_group_; }
  CpProjectionGatherMode projection_gather_mode() const {
    return projection_gather_mode_;
  }

  bool has_same_row_layout(const NpuCpPlan& other) const {
    return enabled_ == other.enabled_ &&
           (!enabled_ ||
            row_layout_.signature() == other.row_layout_.signature());
  }

  // Build CP plan and localize attention meta after global-meta consumers.
  void prepare(ForwardInput& processed_input,
               const CpPlanRuntimeConfig& runtime_config);

  // Rewrite hidden/positions to the rank-local padded layout in place.
  void shard_model_input(torch::Tensor& hidden_states,
                         torch::Tensor& position_ids) const;
  // Expand LOGICAL_REAL cache slots to recovered-physical layout.
  torch::Tensor prepare_cache_slots(
      const torch::Tensor& global_logical_slots) const;
  // AllGather then restore global-real order via the prepare()-bound cp_group.
  torch::Tensor merge_model_output(
      const torch::Tensor& local_hidden_states) const;

  // Binds the CP process group used by merge_model_output(). Workers set this
  // through prepare(); tests and direct builders use this instead.
  void set_process_group(ProcessGroup* process_group);

  // Rewrites the model attention metadata to the per-rank local-padded view.
  // Called from prepare(); public for unit tests that build a plan directly.
  void apply_attention_meta(ModelInputParams& params) const;

 private:
  friend class npu::GraphPersistentParam;
  friend struct ParallelInput;

  NpuCpPlan to(const torch::Device& device) const;

  // ACL graph capture replaces only storage while preserving plan semantics.
  void replace_cp_ep_meta_storage(CpEpMeta meta);

  bool enabled_ = false;
  int32_t cp_size_ = 1;
  int32_t cp_rank_ = 0;
  int32_t kv_split_size_ = 1;
  int32_t kv_split_rank_ = 0;
  int32_t block_size_ = 0;
  CpProjectionGatherMode projection_gather_mode_ =
      CpProjectionGatherMode::BUNDLED;
  // Non-owning CP process group bound at prepare()/set_process_group() time
  // and consumed by merge_model_output().
  ProcessGroup* cp_group_ = nullptr;
  std::vector<int32_t> global_q_seq_lens_;
  std::vector<int32_t> global_kv_seq_lens_;
  CpRowLayout row_layout_;
  CpAttentionMeta attention_meta_;
  CpEpMeta cp_ep_meta_;
};

}  // namespace xllm
