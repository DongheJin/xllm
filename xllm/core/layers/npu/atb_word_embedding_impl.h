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
#include "framework/model/model_input_params.h"
#include "layers/npu/word_embedding.h"
#include "nlohmann/json.hpp"
#include "pytorch/adapter/utils/utils.h"
#include "xllm_kernels/core/include/atb_speed/base/hosttensor_binder.h"
#include "xllm_kernels/core/include/atb_speed/base/model.h"
#include "xllm_kernels/core/include/atb_speed/log.h"
#include "xllm_kernels/core/include/atb_speed/utils/model_factory.h"
#include "xllm_kernels/operations/fusion/embedding/word_embedding.h"

namespace xllm::hf {
class AtbWordEmbeddingImpl : public AtbEmbeddingImpl, public ATBBase {
 public:
  using Task = std::function<int()>;
  using RunTaskFunc =
      std::function<void(const std::string& taskName, Task task)>;

  explicit AtbWordEmbeddingImpl(const Context& context);

  ~AtbWordEmbeddingImpl() {};

  void load_state_dict(const StateDict& state_dict) override;

  void verify_loaded_weights(const std::string weight_str) const override;

  void merge_loaded_weights() override;

  void param_from_args(atb_speed::common::WordEmbeddingParam& param,
                       const xllm::ModelArgs& args,
                       const xllm::ParallelArgs& parallel_args);

  int64_t init_layer();

  torch::Tensor forward(const torch::Tensor& x,
                        atb::Context* context,
                        AtbWorkspace& workspace,
                        int nodeId) override;

  void build_node_variant_pack(atb_speed::Model::Node& node,
                               const torch::Tensor& x);

 private:
  int64_t init_node(atb_speed::Model::Node& node,
                    atb_speed::common::WordEmbeddingParam& param);

  atb_speed::Model::Node embedding_node_;
  std::string modelName_;
  std::vector<at::Tensor> atOutTensors_;
  // std::string name_;
  atb_speed::common::WordEmbeddingParam embedding_param_;
  atb::Tensor internalTensors;
};
}  // namespace xllm::hf
