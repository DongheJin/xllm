#pragma once

#include <atb/atb_infer.h>
#include <c10/core/ScalarType.h>
#include <torch/torch.h>

#include <unordered_map>

#include "atb_layers/core/include/atb_speed/log.h"
#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/model_input_params.h"
#include "core/layers/ascend/llm_head.h"
#include "core/layers/ascend/qwen2_decoder_layer.h"
#include "core/layers/ascend/rms_norm.h"
#include "model_registry.h"
#include "processors/input_processor.h"
#include "processors/qwen2_vl_image_processor.h"
#include "qwen2.h"

namespace xllm::hf {

#define PrintTensor(tensor) print_tensor(tensor, #tensor, 10, true, false);

class Qwen2_5_VLInputProcessor : public InputProcessor {
  enum class TokenType {
    INVALID,
    IMAGE,
    VIDEO,
  };

 public:
  Qwen2_5_VLInputProcessor(const ModelArgs& args) {
    merge_size_ = args.mm_image_merge_size();
  }

  void process(std::string& prompt, const MMData& mm_data) override {
    torch::Tensor image_grid_thw;
    if (auto res = mm_data.get<torch::Tensor>("image_grid_thw"))
      image_grid_thw = res.value();

    torch::Tensor video_grid_thw;
    if (auto res = mm_data.get<torch::Tensor>("video_grid_thw"))
      video_grid_thw = res.value();

    if (!image_grid_thw.defined() && !video_grid_thw.defined()) return;

    auto merge_length = merge_size_ * merge_size_;
    int total_image_token = 0;
    if (image_grid_thw.defined()) {
      auto count = image_grid_thw.sizes()[0];
      for (int idx = 0; idx < count; ++idx)
        total_image_token +=
            image_grid_thw[idx].prod().item<int>() / merge_length;
    }

    int total_video_token = 0;
    if (video_grid_thw.defined()) {
      auto count = video_grid_thw.sizes()[0];
      for (int idx = 0; idx < count; ++idx)
        total_video_token +=
            video_grid_thw[idx].prod().item<int>() / merge_length;
    }

    size_t total_token_len = total_image_token * image_token_.size() +
                             total_video_token * video_token_.size();
    std::string data;
    data.reserve(prompt.size() + total_token_len);

    int image_index = 0;
    int video_index = 0;

    const torch::Tensor* grid_thw = nullptr;
    const std::string* token = nullptr;
    int* index = 0;

    size_t begin = 0;
    auto pair = find_vision_token(prompt, begin);

    while (pair.second != std::string::npos) {
      data.append(prompt, begin, pair.second - begin);

      if (pair.first == TokenType::IMAGE) {
        grid_thw = &image_grid_thw;
        token = &image_token_;
        index = &image_index;
      } else if (pair.first == TokenType::VIDEO) {
        grid_thw = &video_grid_thw;
        token = &video_token_;
        index = &video_index;
      } else {
        assert(false);
      }

      auto token_num = (*grid_thw)[(*index)].prod().item<int>() / merge_length;
      while (token_num--) data.append(*token);

      ++(*index);
      begin = pair.second + token->size();
      pair = find_vision_token(prompt, begin);
    }

    if (begin < prompt.size()) data.append(prompt, begin, std::string::npos);

    prompt = std::move(data);
  }

 private:
  std::pair<TokenType, size_t> find_vision_token(const std::string& prompt,
                                                 size_t begin) {
    auto img_pos = prompt.find(image_token_, begin);
    auto vid_pos = prompt.find(video_token_, begin);

    if (img_pos == std::string::npos && vid_pos == std::string::npos)
      return {TokenType::INVALID, std::string::npos};
    else if (vid_pos == std::string::npos)
      return {TokenType::IMAGE, img_pos};
    else if (img_pos == std::string::npos)
      return {TokenType::VIDEO, vid_pos};
    else
      return img_pos < vid_pos ? std::make_pair(TokenType::IMAGE, img_pos)
                               : std::make_pair(TokenType::VIDEO, vid_pos);
  }

 private:
  const std::string image_token_ = "<|image_pad|>";
  const std::string video_token_ = "<|video_pad|>";

  int merge_size_ = 0;
};

class Qwen2_5_VisionMLPImpl : public torch::nn::Module {
  using ActFunc = torch::Tensor (*)(const torch::Tensor&);

 public:
  Qwen2_5_VisionMLPImpl(const Context& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    auto quant_args = context.get_quant_args();
    auto parallel_args = context.get_parallel_args();

    auto in_features = model_args.mm_hidden_size();
    auto hidden_features = model_args.mm_intermediate_size();

    gate_proj_ = register_module("gate_proj",
                                 ColumnParallelLinear(in_features,
                                                      hidden_features,
                                                      /*bias=*/true,
                                                      /*gather_output*/ false,
                                                      quant_args,
                                                      parallel_args,
                                                      options));

    up_proj_ = register_module("up_proj",
                               ColumnParallelLinear(in_features,
                                                    hidden_features,
                                                    /*bias=*/true,
                                                    /*gather_output*/ false,
                                                    quant_args,
                                                    parallel_args,
                                                    options));

    down_proj_ = register_module("down_proj",
                                 RowParallelLinear(hidden_features,
                                                   in_features,
                                                   /*bias=*/true,
                                                   /*input_is_parallel*/ true,
                                                   quant_args,
                                                   parallel_args,
                                                   options));
    namespace F = torch::nn::functional;
    act_func_ = F::silu;  // Activation::get_act_func(args.mm_hidden_act(),
                          // options.device());
  }

  torch::Tensor forward(torch::Tensor x) {
    auto x_gate = gate_proj_(x);
    x_gate = act_func_(x_gate);

    auto x_up = up_proj_(x);
    auto x_down = down_proj_(x_gate * x_up);
    return x_down;
  }

  void load_state_dict(const StateDict& state_dict) {
    gate_proj_->load_state_dict(state_dict.get_dict_with_prefix("gate_proj."));
    up_proj_->load_state_dict(state_dict.get_dict_with_prefix("up_proj."));
    down_proj_->load_state_dict(state_dict.get_dict_with_prefix("down_proj."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    gate_proj_->verify_loaded_weights(prefix + "gate_proj.");
    up_proj_->verify_loaded_weights(prefix + "up_proj.");
    down_proj_->verify_loaded_weights(prefix + "down_proj.");
  }

 private:
  ColumnParallelLinear gate_proj_{nullptr};
  ColumnParallelLinear up_proj_{nullptr};
  RowParallelLinear down_proj_{nullptr};
  ActFunc act_func_{nullptr};
};
TORCH_MODULE(Qwen2_5_VisionMLP);

class Qwen2_5_VisionAttentionImpl : public torch::nn::Module {
 public:
  Qwen2_5_VisionAttentionImpl(const Context& context)
      : parallel_args_(context.get_parallel_args()) {
    auto quant_args = context.get_quant_args();
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();

    auto embed_dim = model_args.mm_hidden_size();
    auto num_heads = model_args.mm_num_attention_heads();
    auto projection_size = model_args.mm_hidden_size();

    tp_size_ = parallel_args_.world_size();
    tp_rank_ = parallel_args_.rank();

    hidden_size_per_attention_head_ = projection_size / num_heads;
    num_attention_heads_per_partition_ = num_heads / tp_size_;

    qkv_ = register_module("qkv",
                           ColumnParallelLinear(embed_dim,
                                                3 * projection_size,
                                                /*bias=*/true,
                                                /*gather_output*/ false,
                                                quant_args,
                                                parallel_args_,
                                                options));
    proj_ = register_module("proj",
                            RowParallelLinear(projection_size,
                                              embed_dim,
                                              /*bias=*/true,
                                              /*input_is_parallel*/ true,
                                              quant_args,
                                              parallel_args_,
                                              options));
  }

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> split_qkv(
      torch::Tensor qkv) {
    auto shape = qkv.sizes();
    auto seq_len = shape[0];
    auto bs = shape[1];

    if (tp_size_ > 1) qkv = parallel_state::gather(qkv, parallel_args_);

    auto chunks = qkv.chunk(3, 2);
    auto q = chunks[0];
    auto k = chunks[1];
    auto v = chunks[2];

    if (tp_size_ > 1) {
      auto splitter = [this](torch::Tensor t) {
        int last_dim = t.sizes().back();
        CHECK(last_dim % tp_size_ == 0)
            << last_dim << "is not divisible by " << tp_size_;
        return torch::split(t, last_dim / tp_size_, -1);
      };

      q = splitter(q)[tp_rank_];
      k = splitter(k)[tp_rank_];
      v = splitter(v)[tp_rank_];
    }

    auto new_shape = {seq_len,
                      bs,
                      num_attention_heads_per_partition_,
                      hidden_size_per_attention_head_};
    q = q.view(new_shape);
    k = k.view(new_shape);
    v = v.view(new_shape);

    return std::make_tuple(q, k, v);
  }

  torch::Tensor rotate_half(torch::Tensor x, bool interleaved = false) {
    if (!interleaved) {
      auto chunks = x.chunk(2, -1);
      return torch::cat({-chunks[1], chunks[0]}, -1);
    } else {
      auto x1 =
          x.index({"...", torch::indexing::Slice(0, torch::indexing::None, 2)});
      auto x2 =
          x.index({"...", torch::indexing::Slice(1, torch::indexing::None, 2)});

      auto stacked = torch::stack({-x2, x1}, -1);
      return stacked.flatten(-2, -1);
    }
  }

  torch::Tensor apply_rotary_emb_torch(torch::Tensor x,
                                       torch::Tensor cos,
                                       torch::Tensor sin,
                                       bool interleaved = false) {
    int64_t ro_dim = cos.sizes().back() * 2;
    assert(ro_dim <= x.sizes().back());

    auto expand = [interleaved](const torch::Tensor& t) {
      auto expanded = t.unsqueeze(-2);
      if (interleaved) {  // "... d -> ... 1 (d 2)"
        return torch::repeat_interleave(expanded, 2, -1);
      } else {  // "... d -> ... 1 (2 d)"
        std::vector<int64_t> dims(expanded.dim(), 1);
        dims.back() = 2;
        return expanded.repeat(dims);
      }
    };
    auto x_rot = x.narrow(-1, 0, ro_dim);
    auto x_rotated =
        x_rot * expand(cos) + rotate_half(x_rot, interleaved) * expand(sin);

    return torch::cat(
        {x_rotated, x.narrow(-1, ro_dim, x.sizes().back() - ro_dim)}, -1);
  }

  torch::Tensor apply_rotary_pos_emb_vision(torch::Tensor t,
                                            torch::Tensor freqs) {
    auto t_ = t.to(torch::kFloat32);
    auto cos = freqs.cos();
    auto sin = freqs.sin();

    auto output = apply_rotary_emb_torch(t_, cos, sin).type_as(t);
    return output;
  }

  torch::Tensor forward(torch::Tensor x,
                        torch::Tensor cu_seqlens,
                        torch::Tensor rotary_pos_emb) {
    x = qkv_(x);
    auto [q, k, v] = split_qkv(x);

    auto shape = q.sizes();
    auto seq_len = shape[0];
    auto batch_size = shape[1];
    auto head = shape[2];
    auto head_dim = shape[3];

    q = q.permute({1, 0, 2, 3}).contiguous();
    k = k.permute({1, 0, 2, 3}).contiguous();
    v = v.permute({1, 0, 2, 3}).contiguous();

    q = apply_rotary_pos_emb_vision(q, rotary_pos_emb);
    k = apply_rotary_pos_emb_vision(k, rotary_pos_emb);
    auto count = cu_seqlens.sizes()[0];

    auto options =
        torch::TensorOptions().dtype(torch::kBool).device(q.device());

    auto attn_mask = torch::zeros({1, seq_len, seq_len}, options);
    for (int idx = 1; idx < count; ++idx) {
      auto start = cu_seqlens[idx - 1].item<int>();
      auto end = cu_seqlens[idx].item<int>();

      attn_mask.index_put_({torch::indexing::Slice(),
                            torch::indexing::Slice(start, end),
                            torch::indexing::Slice(start, end)},
                           true);
    }

    q = q.permute({0, 2, 1, 3});
    k = k.permute({0, 2, 1, 3});
    v = v.permute({0, 2, 1, 3});

    auto attn_output = torch::scaled_dot_product_attention(q, k, v, attn_mask);
    attn_output = attn_output.permute({2, 0, 1, 3})
                      .reshape({seq_len, batch_size, -1})
                      .contiguous();

    auto output = proj_(attn_output);
    return output;
  }

  void load_state_dict(const StateDict& state_dict) {
    qkv_->load_state_dict(state_dict.get_dict_with_prefix("qkv."));
    proj_->load_state_dict(state_dict.get_dict_with_prefix("proj."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    qkv_->verify_loaded_weights(prefix + "qkv.");
    proj_->verify_loaded_weights(prefix + "proj.");
  }

 private:
  int tp_size_ = 0;
  int tp_rank_ = 0;
  ParallelArgs parallel_args_;

  int64_t hidden_size_per_attention_head_ = 0;
  int64_t num_attention_heads_per_partition_ = 0;

  ColumnParallelLinear qkv_{nullptr};
  RowParallelLinear proj_{nullptr};
};
TORCH_MODULE(Qwen2_5_VisionAttention);

class Qwen2_5_VisionBlockImpl : public torch::nn::Module {
 public:
  Qwen2_5_VisionBlockImpl(const Context& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();

    auto dim = model_args.mm_hidden_size();

    norm1_ = register_module("norm1",
                             RMSNorm(dim, model_args.rms_norm_eps(), options));
    norm2_ = register_module("norm2",
                             RMSNorm(dim, model_args.rms_norm_eps(), options));

    attn_ = register_module("attn", Qwen2_5_VisionAttention(context));
    mlp_ = register_module("mlp", Qwen2_5_VisionMLP(context));
  }

  torch::Tensor forward(torch::Tensor x,
                        torch::Tensor cu_seqlens,
                        torch::Tensor rotary_pos_emb) {
    x = x + attn_(norm1_(x), cu_seqlens, rotary_pos_emb);
    x = x + mlp_(norm2_(x));
    return x;
  }

  void load_state_dict(const StateDict& state_dict) {
    norm1_->load_state_dict(state_dict.get_dict_with_prefix("norm1."));
    norm2_->load_state_dict(state_dict.get_dict_with_prefix("norm2."));

    attn_->load_state_dict(state_dict.get_dict_with_prefix("attn."));
    mlp_->load_state_dict(state_dict.get_dict_with_prefix("mlp."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    norm1_->verify_loaded_weights(prefix + "norm1.");
    norm2_->verify_loaded_weights(prefix + "norm2.");

    attn_->verify_loaded_weights(prefix + "attn.");
    mlp_->verify_loaded_weights(prefix + "mlp.");
  }

 private:
  RMSNorm norm1_{nullptr};
  RMSNorm norm2_{nullptr};

  Qwen2_5_VisionAttention attn_{nullptr};
  Qwen2_5_VisionMLP mlp_{nullptr};
};
TORCH_MODULE(Qwen2_5_VisionBlock);

class Qwen2_5_VisionPatchEmbedImpl : public torch::nn::Module {
 public:
  Qwen2_5_VisionPatchEmbedImpl(const Context& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();

    auto in_features = model_args.mm_num_channels() *
                       model_args.mm_temporal_patch_size() *
                       model_args.mm_patch_size() * model_args.mm_patch_size();

    auto out_features = model_args.mm_hidden_size();

    proj_ = register_module(
        "proj",
        torch::nn::Linear(
            torch::nn::LinearOptions(in_features, out_features).bias(false)));

    proj_->weight.set_data(proj_->weight.to(options));
  }

  torch::Tensor forward(torch::Tensor x) { return proj_(x); }

  void load_state_dict(const StateDict& state_dict) {
    auto weight = state_dict.get_tensor("proj.weight");
    if (weight.defined()) {
      weight = weight.reshape({weight.size(0), -1});
      DCHECK_EQ(proj_->weight.sizes(), weight.sizes())
          << "proj weight size mismatch for " << name();
      proj_->weight.data().copy_(weight);
      proj_weight_loaded_ = true;
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(proj_weight_loaded_)
        << "weight is not loaded for " << prefix + "proj.weight";
  }

 private:
  bool proj_weight_loaded_ = false;
  torch::nn::Linear proj_{nullptr};
};
TORCH_MODULE(Qwen2_5_VisionPatchEmbed);

class Qwen2_5_VisionRotaryEmbeddingImpl : public torch::nn::Module {
 public:
  Qwen2_5_VisionRotaryEmbeddingImpl(const Context& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();

    dim_ = model_args.mm_head_dim() / 2;
    theta_ = 10000.0;

    auto opts = options.dtype(torch::kFloat32);
    auto inv_freq =
        1.0 / torch::pow(theta_, torch::arange(0, dim_, 2, opts) / dim_);
    inv_freq_ = register_buffer("inv_freq", inv_freq);
  }

  void update_freqs_cache(int64_t seqlen) {
    if (seqlen <= seq_len_cached_) return;

    seqlen *= 2;
    seq_len_cached_ = seqlen;

    auto options = torch::TensorOptions()
                       .dtype(torch::kFloat32)
                       .device(inv_freq_.device());
    inv_freq_ =
        1.0 / torch::pow(theta_, torch::arange(0, dim_, 2, options) / dim_);
    auto seq = torch::arange(seqlen, options);
    freqs_cached_ = torch::outer(seq, inv_freq_);
  }

  torch::Tensor forward(int seqlen) {
    update_freqs_cache(seqlen);
    return freqs_cached_.slice(0, 0, seqlen);
  }

 private:
  int dim_ = 0;
  double theta_ = 0.0;

  int64_t seq_len_cached_ = 0;
  torch::Tensor inv_freq_;
  torch::Tensor freqs_cached_;
};
TORCH_MODULE(Qwen2_5_VisionRotaryEmbedding);

class Qwen2_5_VisionPatchMergerImpl : public torch::nn::Module {
 public:
  Qwen2_5_VisionPatchMergerImpl(const Context& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    auto quant_args = context.get_quant_args();
    auto parallel_args = context.get_parallel_args();

    int64_t d_model = model_args.mm_projection_dim();  // out_hidden_size
    int context_dim = model_args.mm_hidden_size();
    int spatial_merge_size = model_args.mm_spatial_merge_size();

    hidden_size_ =
        context_dim * static_cast<int>(std::pow(spatial_merge_size, 2));
    ln_q_ = register_module(
        "ln_q", RMSNorm(context_dim, model_args.rms_norm_eps(), options));

    auto cpl = ColumnParallelLinear(hidden_size_,
                                    hidden_size_,
                                    /*bias=*/true,
                                    /*gather_output*/ false,
                                    quant_args,
                                    parallel_args,
                                    options);
    auto act = torch::nn::GELU();
    auto rpl = RowParallelLinear(hidden_size_,
                                 d_model,
                                 /*bias=*/true,
                                 /*input_is_parallel*/ true,
                                 quant_args,
                                 parallel_args,
                                 options);

    mlp_ = register_module("mlp", torch::nn::Sequential(cpl, act, rpl));
    layers_ = std::make_tuple(cpl, act, rpl);
  }

  torch::Tensor forward(torch::Tensor x) {
    x = ln_q_(x);
    x = x.view({-1, hidden_size_});

    return mlp_->forward(x);
  }

  void load_state_dict(const StateDict& state_dict) {
    ln_q_->load_state_dict(state_dict.get_dict_with_prefix("ln_q."));
    std::get<0>(layers_)->load_state_dict(
        state_dict.get_dict_with_prefix("mlp.0."));
    std::get<2>(layers_)->load_state_dict(
        state_dict.get_dict_with_prefix("mlp.2."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    ln_q_->verify_loaded_weights(prefix + "ln_q.");
    std::get<0>(layers_)->verify_loaded_weights(prefix + "mlp.0.");
    std::get<2>(layers_)->verify_loaded_weights(prefix + "mlp.2.");
  }

 private:
  int64_t hidden_size_;

  RMSNorm ln_q_{nullptr};
  torch::nn::Sequential mlp_{nullptr};
  std::tuple<ColumnParallelLinear, torch::nn::GELU, RowParallelLinear> layers_ =
      {nullptr, nullptr, nullptr};
};
TORCH_MODULE(Qwen2_5_VisionPatchMerger);

class Qwen2_5_VisionTransformerImpl : public torch::nn::Module {
 public:
  Qwen2_5_VisionTransformerImpl(const Context& context) {
    auto model_args = context.get_model_args();

    hidden_size_ = model_args.mm_hidden_size();
    num_heads_ = model_args.mm_num_attention_heads();

    window_size_ = model_args.mm_window_size();
    patch_size_ = model_args.mm_patch_size();
    spatial_merge_size_ = model_args.mm_spatial_merge_size();
    const auto& block_indexes = model_args.mm_fullatt_block_indexes();
    fullatt_block_indexes_.insert(block_indexes.begin(), block_indexes.end());
    spatial_merge_unit_ = static_cast<int>(std::pow(spatial_merge_size_, 2));

    patch_embed_ =
        register_module("patch_embed", Qwen2_5_VisionPatchEmbed(context));
    rotary_pos_emb_ = register_module("rotary_pos_emb",
                                      Qwen2_5_VisionRotaryEmbedding(context));
    blocks_ = register_module("blocks", torch::nn::ModuleList());

    for (int32_t idx = 0; idx < model_args.mm_num_hidden_layers(); idx++) {
      auto block = Qwen2_5_VisionBlock(context);
      blocks_->push_back(block);
      layers_.push_back(block);
    }
    merger_ = register_module("merger", Qwen2_5_VisionPatchMerger(context));
  }

  torch::Tensor rot_pos_emb(torch::Tensor grid_thw) {
    std::vector<torch::Tensor> pos_ids_vec;
    auto count = grid_thw.sizes()[0];
    pos_ids_vec.reserve(count);

    auto grid_thw_cpu = grid_thw.cpu();
    auto options =
        torch::TensorOptions().dtype(torch::kLong).device(grid_thw.device());

    for (int idx = 0; idx < count; ++idx) {
      auto t = grid_thw_cpu[idx][0].item<int64_t>();
      auto h = grid_thw_cpu[idx][1].item<int64_t>();
      auto w = grid_thw_cpu[idx][2].item<int64_t>();

      auto hpos_ids = torch::arange(h, options).unsqueeze(1).expand({-1, w});
      hpos_ids = hpos_ids
                     .reshape({h / spatial_merge_size_,
                               spatial_merge_size_,
                               w / spatial_merge_size_,
                               spatial_merge_size_})
                     .permute({0, 2, 1, 3})
                     .flatten();

      auto wpos_ids = torch::arange(w, options).unsqueeze(0).expand({h, -1});
      wpos_ids = wpos_ids
                     .reshape({h / spatial_merge_size_,
                               spatial_merge_size_,
                               w / spatial_merge_size_,
                               spatial_merge_size_})
                     .permute({0, 2, 1, 3})
                     .flatten();

      pos_ids_vec.push_back(
          torch::stack({hpos_ids, wpos_ids}, -1).repeat({t, 1}));
    }

    auto pos_ids = torch::cat(pos_ids_vec, 0);
    auto max_grid_size =
        grid_thw
            .index({torch::indexing::Slice(),
                    torch::indexing::Slice(1, torch::indexing::None)})
            .max();

    auto rotary_pos_emb_full = rotary_pos_emb_(max_grid_size.item<int64_t>());
    auto rotary_pos_emb = rotary_pos_emb_full.index({pos_ids}).flatten(1);

    return rotary_pos_emb;
  }

  torch::Tensor get_window_index(torch::Tensor grid_thw,
                                 std::vector<int>& cu_window_seqlens) {
    auto count = grid_thw.sizes()[0];
    std::vector<torch::Tensor> window_index;
    window_index.reserve(count);
    cu_window_seqlens.reserve(count * 128);
    cu_window_seqlens.emplace_back(0);

    int window_index_id = 0;
    int vit_merger_window_size =
        window_size_ / spatial_merge_size_ / patch_size_;

    auto grid_thw_cpu = grid_thw.cpu();
    auto options =
        torch::TensorOptions().dtype(torch::kLong).device(grid_thw.device());

    for (int idx = 0; idx < count; ++idx) {
      auto grid_t = grid_thw_cpu[idx][0].item<int64_t>();
      auto grid_h = grid_thw_cpu[idx][1].item<int64_t>();
      auto grid_w = grid_thw_cpu[idx][2].item<int64_t>();

      auto llm_grid_h = grid_h / spatial_merge_size_;
      auto llm_grid_w = grid_w / spatial_merge_size_;

      auto index = torch::arange(grid_t * llm_grid_h * llm_grid_w, options)
                       .reshape({grid_t, llm_grid_h, llm_grid_w});
      auto pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size;
      auto pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size;

      auto num_windows_h = (llm_grid_h + pad_h) / vit_merger_window_size;
      auto num_windows_w = (llm_grid_w + pad_w) / vit_merger_window_size;

      namespace F = torch::nn::functional;
      auto index_padded = F::pad(index,
                                 F::PadFuncOptions({0, pad_w, 0, pad_h})
                                     .mode(torch::kConstant)
                                     .value(-100));
      index_padded = index_padded.reshape({grid_t,
                                           num_windows_h,
                                           vit_merger_window_size,
                                           num_windows_w,
                                           vit_merger_window_size});

      index_padded = index_padded.permute({0, 1, 3, 2, 4})
                         .reshape({grid_t,
                                   num_windows_h * num_windows_w,
                                   vit_merger_window_size,
                                   vit_merger_window_size});

      auto index_padded_ne = torch::ne(index_padded, -100);
      auto seqlens = index_padded_ne.sum({2, 3}).reshape({-1});
      index_padded = index_padded.reshape({-1});
      auto index_new =
          index_padded.masked_select(index_padded_ne.reshape({-1}));

      window_index.push_back(index_new + window_index_id);
      auto cu_seqlens_tmp =
          (seqlens.cumsum(0, torch::kInt32) * spatial_merge_unit_ +
           cu_window_seqlens.back())
              .cpu();
      cu_window_seqlens.insert(
          cu_window_seqlens.end(),
          cu_seqlens_tmp.data_ptr<int>(),
          cu_seqlens_tmp.data_ptr<int>() + cu_seqlens_tmp.numel());
      window_index_id += grid_t * llm_grid_h * llm_grid_w;
    }

    return torch::cat(window_index, 0);
  }

  torch::Tensor forward(torch::Tensor hidden_states, torch::Tensor grid_thw) {
    // patchify
    // hidden_states = x.to(device=self.device, dtype=self.dtype);
    hidden_states = patch_embed_(hidden_states);

    // compute position embedding
    auto rotary_pos_emb = rot_pos_emb(grid_thw);

    // windows attention
    std::vector<int> cu_window_seqlens_vec;
    auto window_index = get_window_index(grid_thw, cu_window_seqlens_vec);
    torch::TensorOptions options = torch::TensorOptions()
                                       .dtype(torch::kInt32)
                                       .device(hidden_states.device());
    auto cu_window_seqlens = torch::tensor(cu_window_seqlens_vec, options);
    cu_window_seqlens =
        std::get<0>(torch::unique_consecutive(cu_window_seqlens));
    auto seq_len = hidden_states.sizes()[0];
    hidden_states = hidden_states.reshape(
        {seq_len / spatial_merge_unit_, spatial_merge_unit_, -1});
    hidden_states = hidden_states.index(
        {window_index, torch::indexing::Slice(), torch::indexing::Slice()});
    hidden_states = hidden_states.reshape({seq_len, -1});

    rotary_pos_emb = rotary_pos_emb.reshape(
        {seq_len / spatial_merge_unit_, spatial_merge_unit_, -1});
    rotary_pos_emb = rotary_pos_emb.index(
        {window_index, torch::indexing::Slice(), torch::indexing::Slice()});
    rotary_pos_emb = rotary_pos_emb.reshape({seq_len, -1});

    // compute cu_seqlens
    auto cu_seqlens = torch::repeat_interleave(
                          grid_thw.index({torch::indexing::Slice(), 1}) *
                              grid_thw.index({torch::indexing::Slice(), 2}),
                          grid_thw.index({torch::indexing::Slice(), 0}))
                          .cumsum(0, torch::kInt32);
    namespace F = torch::nn::functional;
    cu_seqlens = F::pad(
        cu_seqlens, F::PadFuncOptions({1, 0}).mode(torch::kConstant).value(0));

    cu_window_seqlens = cu_window_seqlens.cpu();
    cu_seqlens = cu_seqlens.cpu();

    // transformers
    hidden_states = hidden_states.unsqueeze(1);
    for (int idx = 0; idx < blocks_->size(); ++idx) {
      torch::Tensor cu_seqlens_now;
      if (fullatt_block_indexes_.find(idx) != fullatt_block_indexes_.end())
        cu_seqlens_now = cu_seqlens;
      else
        cu_seqlens_now = cu_window_seqlens;

      hidden_states =
          layers_[idx](hidden_states, cu_seqlens_now, rotary_pos_emb);
    }
    // adapter
    hidden_states = merger_(hidden_states);
    auto reverse_indices = torch::argsort(window_index);
    hidden_states =
        hidden_states.index({reverse_indices, torch::indexing::Slice()});
    return hidden_states;
  }

  void load_state_dict(const StateDict& state_dict) {
    patch_embed_->load_state_dict(
        state_dict.get_dict_with_prefix("patch_embed."));
    for (int idx = 0; idx < blocks_->size(); ++idx) {
      layers_[idx]->load_state_dict(state_dict.get_dict_with_prefix(
          "blocks." + std::to_string(idx) + "."));
    }

    merger_->load_state_dict(state_dict.get_dict_with_prefix("merger."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    patch_embed_->verify_loaded_weights(prefix + "patch_embed.");
    for (int idx = 0; idx < blocks_->size(); ++idx) {
      layers_[idx]->verify_loaded_weights(prefix + "blocks." +
                                          std::to_string(idx) + ".");
    }
    merger_->verify_loaded_weights(prefix + "merger.");
  }

 private:
  int hidden_size_ = 0;
  int num_heads_ = 0;
  int window_size_ = 0;
  int patch_size_ = 0;
  int spatial_merge_size_ = 0;
  std::set<int> fullatt_block_indexes_;
  int spatial_merge_unit_ = 0;

  Qwen2_5_VisionPatchEmbed patch_embed_{nullptr};
  Qwen2_5_VisionRotaryEmbedding rotary_pos_emb_{nullptr};
  torch::nn::ModuleList blocks_{nullptr};
  std::vector<Qwen2_5_VisionBlock> layers_;
  Qwen2_5_VisionPatchMerger merger_{nullptr};
};
TORCH_MODULE(Qwen2_5_VisionTransformer);

struct Qwen2_5_VLImageInputs {
  torch::Tensor pixel_values;
  torch::Tensor image_grid_thw;
};

struct Qwen2_5_VLVideoInputs {
  torch::Tensor pixel_values_videos;
  torch::Tensor video_grid_thw;
  torch::Tensor second_per_grid_ts;
};

class Qwen2_5_VLForConditionalGenerationImpl : public torch::nn::Module {
 public:
  Qwen2_5_VLForConditionalGenerationImpl(const Context& context)
      : model_args_(context.get_model_args()),
        options_(context.get_tensor_options()) {
    Context vision_context(ParallelArgs(0, 1, nullptr));
    vision_context.set_model_args(model_args_);
    vision_context.set_quant_args(context.get_quant_args());
    vision_context.set_tensor_options(options_);
    visual_ =
        register_module("visual", Qwen2_5_VisionTransformer(vision_context));

    language_model_ =
        register_module("language_model", QWen2ForCausalLM(context));
  }

  torch::Tensor get_input_embeddings(
      torch::Tensor input_ids,
      const std::optional<Qwen2_5_VLImageInputs>& image_input,
      const std::optional<Qwen2_5_VLVideoInputs>& video_input) {
    auto inputs_embeds = language_model_->get_input_embeddings(input_ids);
    if (image_input) {
      // visual
      auto image_embeds = visual_(image_input->pixel_values.to(options_),
                                  image_input->image_grid_thw);
      // merge
      auto is_multimodal = torch::isin(input_ids, model_args_.image_token_id());
      inputs_embeds.index_put_({is_multimodal}, image_embeds);
    }
    return inputs_embeds;
  }

  torch::Tensor forward(const torch::Tensor& tokens,
                        const torch::Tensor& positions,
                        std::vector<KVCache>& kv_caches,
                        const ModelInputParams& input_params) {
    torch::NoGradGuard no_grad;
    const auto& mm_data = input_params.mm_data;

    torch::Tensor pixel_values;
    if (const auto& res = mm_data.get<torch::Tensor>("pixel_values"))
      pixel_values = res.value();

    torch::Tensor image_grid_thw;
    if (const auto& res = mm_data.get<torch::Tensor>("image_grid_thw"))
      image_grid_thw = res.value();

    std::optional<Qwen2_5_VLImageInputs> image_inputs;
    std::optional<Qwen2_5_VLVideoInputs> video_inputs;

    if (pixel_values.defined() && image_grid_thw.defined())
      image_inputs = Qwen2_5_VLImageInputs{pixel_values, image_grid_thw};

    auto inputs_embeds =
        get_input_embeddings(tokens, image_inputs, video_inputs);
    input_params.input_embedding = inputs_embeds;

    return language_model_(tokens, positions, kv_caches, input_params);
  }

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) {
    return language_model_->logits(hidden_states, seleted_idxes);
  }

  void load_model(std::unique_ptr<ModelLoader> loader) {
    for (const auto& state_dict : loader->get_state_dicts()) {
      visual_->load_state_dict(state_dict->get_dict_with_prefix("visual."));
    }
    // verify
    visual_->verify_loaded_weights("visual.");

    if (!model_args_.image_embedding_mode()) {
      language_model_->load_model(std::move(loader));
    }
  }

  LlmHead get_lm_head() { return language_model_->get_lm_head(); }
  void set_lm_head(LlmHead& head) { language_model_->set_lm_head(head); }

  AtbWordEmbedding get_word_embedding() {
    return language_model_->get_word_embedding();
  }

  void set_word_embedding(AtbWordEmbedding& word_embedding) {
    language_model_->set_word_embedding(word_embedding);
  }

 private:
  ModelArgs model_args_;
  torch::TensorOptions options_;

  Qwen2_5_VisionTransformer visual_{nullptr};
  QWen2ForCausalLM language_model_{nullptr};
};
TORCH_MODULE(Qwen2_5_VLForConditionalGeneration);

REGISTER_INPUT_PROCESSOR(qwen2_5_vl, Qwen2_5_VLInputProcessor);
REGISTER_CAUSAL_VLM_MODEL(qwen2_5_vl, Qwen2_5_VLForConditionalGeneration);
REGISTER_IMAGE_PROCESSOR(qwen2_5_vl, Qwen2VLImageProcessor);

REGISTER_MODEL_ARGS(qwen2_5_vl, [&] {
  // text config
  // LOAD_ARG_OR(attention_dropout, "attention_dropout", 0.0);
  LOAD_ARG_OR(bos_token_id, "bos_token_id", 151643);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 151645);
  LOAD_ARG_OR(vision_start_token_id, "vision_start_token_id", 151652);
  LOAD_ARG_OR(vision_end_token_id, "vision_end_token_id", 151653);
  LOAD_ARG_OR(vision_token_id, "vision_token_id", 151654);
  LOAD_ARG_OR(image_token_id, "image_token_id", 151655);
  LOAD_ARG_OR(video_token_id, "video_token_id", 151656);
  LOAD_ARG_OR(hidden_act, "hidden_act", "silu");
  LOAD_ARG_OR(hidden_size, "hidden_size", 3584);
  // LOAD_ARG_OR(initializer_range, "initializer_range", 0.02);
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 18944);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 128000);
  LOAD_ARG_OR(max_window_layers, "max_window_layers", 28);
  LOAD_ARG_OR(model_type, "model_type", "qwen2_5_vl");
  LOAD_ARG_OR(n_heads, "num_attention_heads", 28);
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 28);
  LOAD_ARG_OR(n_kv_heads, "num_key_value_heads", 4);
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-06);
  LOAD_ARG_OR(rope_theta, "rope_theta", 1000000.0f);
  LOAD_ARG_OR(sliding_window, "sliding_window", 32768);
  LOAD_ARG_OR(tie_word_embeddings, "tie_word_embeddings", false);
  LOAD_ARG_OR(dtype, "torch_dtype", "");
  // LOAD_ARG_OR(transformers_version, "transformers_version", "4.41.2");
  // LOAD_ARG_OR(use_cache, "use_cache", true);
  LOAD_ARG_OR(use_sliding_window, "use_sliding_window", false);
  LOAD_ARG_OR_FUNC(head_dim, "head_dim", [&] {
    return args->hidden_size() / args->n_heads();
  });

  // vision_config
  LOAD_ARG_OR(mm_num_hidden_layers, "vision_config.depth", 32);
  LOAD_ARG_OR(mm_hidden_act, "vision_config.hidden_act", "silu");
  LOAD_ARG_OR(mm_hidden_size, "vision_config.hidden_size", 1280);
  LOAD_ARG_OR(mm_intermediate_size, "vision_config.intermediate_size", 3420);
  LOAD_ARG_OR(mm_num_attention_heads, "vision_config.num_heads", 16);
  LOAD_ARG_OR(mm_num_channels, "vision_config.in_chans", 3);
  LOAD_ARG_OR(mm_projection_dim, "vision_config.out_hidden_size", 3584);
  LOAD_ARG_OR(mm_patch_size, "vision_config.patch_size", 14);
  LOAD_ARG_OR(mm_spatial_merge_size, "vision_config.spatial_merge_size", 2);
  LOAD_ARG_OR(mm_spatial_patch_size, "vision_config.spatial_patch_size", 14);
  LOAD_ARG_OR(mm_window_size, "vision_config.window_size", 112);
  LOAD_ARG(mm_fullatt_block_indexes, "vision_config.fullatt_block_indexes");
  LOAD_ARG_OR(mm_tokens_per_second, "vision_config.tokens_per_second", 2);
  LOAD_ARG_OR(mm_temporal_patch_size, "vision_config.temporal_patch_size", 2);
  LOAD_ARG_OR_FUNC(mm_head_dim, "head_dim", [&] {
    return args->mm_hidden_size() / args->mm_num_attention_heads();
  });

  LOAD_ARG_OR(
      rope_scaling_rope_type, "vision_config.rope_scaling.type", "mrope");
  LOAD_ARG(rope_scaling_mrope_section, "rope_scaling.mrope_section");
  LOAD_ARG_OR(vocab_size, "vocab_size", 152064);
});
}  // namespace xllm::hf
