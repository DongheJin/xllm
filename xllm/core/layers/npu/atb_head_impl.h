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
#include "framework/model/model_input_params.h"
#include "layers/npu/llm_head.h"
#include "nlohmann/json.hpp"
#include "pytorch/adapter/utils/utils.h"
#include "xllm_kernels/core/include/atb_speed/base/hosttensor_binder.h"
#include "xllm_kernels/core/include/atb_speed/base/model.h"
#include "xllm_kernels/core/include/atb_speed/log.h"
#include "xllm_kernels/core/include/atb_speed/utils/model_factory.h"
#include "xllm_kernels/operations/fusion/lmhead/lmhead.h"

namespace xllm::hf {

class AtbLmHeadImpl : public LlmHeadImpl, public ATBBase {
 public:
  using Task = std::function<int()>;
  using RunTaskFunc =
      std::function<void(const std::string& taskName, Task task)>;

  explicit AtbLmHeadImpl(const Context& context);

  ~AtbLmHeadImpl() {};

  void load_state_dict(const StateDict& state_dict) override;

  void verify_loaded_weights(const std::string weight_str) const override;

  void merge_loaded_weights() override;

  void param_from_args(atb_speed::common::LmHeadParam& param,
                       const ModelArgs& args,
                       const ParallelArgs& parallel_args,
                       bool isPrefill);

  int64_t init_layer();

  torch::Tensor forward(const torch::Tensor& hidden_states,
                        const torch::Tensor& seleted_idxes,
                        atb::Context* context,
                        AtbWorkspace& workspace,
                        int nodeId) override;

  // void build_node_variant_pack(atb_speed::Model::Node& node, torch::Tensor&
  // hidden_states,torch::Tensor&
  // seleted_idxes,std::vector<std::shared_ptr<at::Tensor>>& tensor_storage);
  void build_node_variant_pack(atb_speed::Model::Node& node,
                               const torch::Tensor& hidden_states,
                               const torch::Tensor& seleted_idxes);

 private:
  int64_t init_node(atb_speed::Model::Node& node,
                    atb_speed::common::LmHeadParam& param);

  atb_speed::Model::Node llm_head_node_prefill_;
  atb_speed::Model::Node llm_head_node_decode_;

  std::string model_name_;
  torch::Tensor torch_placeholder_;
  atb::Tensor placeholder_;
  std::vector<at::Tensor> atOutTensors_;

  atb_speed::common::LmHeadParam llm_head_param_prefill_;
  atb_speed::common::LmHeadParam llm_head_param_decode_;

  std::vector<std::shared_ptr<at::Tensor>> prefill_tensor_storage_;
  std::vector<std::shared_ptr<at::Tensor>> decode_tensor_storage_;
  atb::Tensor hidden_states_atb_;
  atb::Tensor seleted_idxes_atb_;
};

}  // namespace xllm::hf
