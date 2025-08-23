#pragma once

#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#endif

#include <torch_npu/csrc/libs/init_npu.h>

#include <functional>

#include "atb/atb_infer.h"
#include "atb_base.h"
#include "framework/context.h"
#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/state_dict/state_dict.h"
#include "nlohmann/json.hpp"
#include "pytorch/adapter/utils/utils.h"
#include "xllm_kernels/core/include/atb_speed/base/hosttensor_binder.h"
#include "xllm_kernels/core/include/atb_speed/base/model.h"
#include "xllm_kernels/core/include/atb_speed/log.h"
#include "xllm_kernels/core/include/atb_speed/utils/model_factory.h"
#include "xllm_kernels/operations/fusion/embedding/positional_embedding.h"

namespace xllm::hf {

class AtbRotaryEmbeddingImpl : public torch::nn::Module, public ATBBase {
 public:
  using Task = std::function<int()>;
  using RunTaskFunc =
      std::function<void(const std::string& taskName, Task task)>;

  explicit AtbRotaryEmbeddingImpl(const Context& context);

  ~AtbRotaryEmbeddingImpl() {};

  int64_t init_layer();

  torch::Tensor forward(const torch::Tensor& cos_sin_pos,
                        const torch::Tensor& position,
                        atb::Context* context,
                        AtbWorkspace& workspace,
                        int nodeId);

  void build_node_variant_pack(atb_speed::Model::Node& node,
                               const torch::Tensor& cos_sin_pos,
                               const torch::Tensor& position);

 private:
  int64_t init_node(atb_speed::Model::Node& node);
  atb_speed::Model::Node embedding_node_;
  std::string modelName_;
  std::vector<at::Tensor> atOutTensors_;
  torch::Tensor inv_freq;
  atb::Tensor internal_cos_sin_pos;
  atb::Tensor internal_position;
};

class AtbRotaryEmbedding
    : public torch::nn::ModuleHolder<AtbRotaryEmbeddingImpl> {
 public:
  using torch::nn::ModuleHolder<AtbRotaryEmbeddingImpl>::ModuleHolder;
  using Impl __attribute__((__unused__)) = AtbRotaryEmbeddingImpl;

  AtbRotaryEmbedding(const Context& context);
};

std::shared_ptr<AtbRotaryEmbeddingImpl> create_pos_embedding_layer(
    const Context& context);

}  // namespace xllm::hf
