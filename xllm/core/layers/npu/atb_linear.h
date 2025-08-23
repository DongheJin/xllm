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
#include "framework/state_dict/state_dict.h"
#include "nlohmann/json.hpp"
#include "pytorch/adapter/utils/utils.h"
#include "xllm_kernels/core/include/atb_speed/base/hosttensor_binder.h"
#include "xllm_kernels/core/include/atb_speed/base/model.h"
#include "xllm_kernels/core/include/atb_speed/log.h"
#include "xllm_kernels/core/include/atb_speed/utils/model_factory.h"

namespace xllm::hf {

class AtbLinearImpl : public torch::nn::Module, public ATBBase {
 public:
  using Task = std::function<int()>;
  using RunTaskFunc =
      std::function<void(const std::string& taskName, Task task)>;

  explicit AtbLinearImpl(const Context& context);

  ~AtbLinearImpl() {};

  void load_state_dict(const StateDict& state_dict);

  void verify_loaded_weights(const std::string weight_str) const;

  void merge_loaded_weights();

  int64_t init_layer();

  torch::Tensor forward(const torch::Tensor& input,
                        atb::Context* context,
                        AtbWorkspace& workspace,
                        int nodeId);

  void build_node_variant_pack(atb_speed::Model::Node& node,
                               const torch::Tensor& input);

 private:
  int64_t init_node(atb_speed::Model::Node& node);

  atb_speed::Model::Node linear_node_;
  std::string model_name_;

  std::vector<at::Tensor> at_out_tensors_;
  atb::Tensor internal_input;
  torch::Tensor tensor_placeholder_;
};

class AtbLinear : public torch::nn::ModuleHolder<AtbLinearImpl> {
 public:
  using torch::nn::ModuleHolder<AtbLinearImpl>::ModuleHolder;
  using Impl __attribute__((__unused__)) = AtbLinearImpl;

  AtbLinear(const Context& context);
};

std::shared_ptr<AtbLinearImpl> create_atb_linear_layer(const Context& context);

}  // namespace xllm::hf
