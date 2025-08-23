#pragma once
#include <torch/torch.h>

#include "framework/context.h"
#include "framework/model/model_input_params.h"
#include "framework/state_dict/state_dict.h"
#include "xllm_kernels/pytorch/atb_torch/core/include/base_operation.h"
#include "xllm_kernels/pytorch/atb_torch/core/include/graph_operation.h"

namespace xllm::hf {

class SiglipEncoderLayerUpImpl : public torch::nn::Module {
 public:
  SiglipEncoderLayerUpImpl(const Context& context,
                           const std::string& prefix = "");

  ~SiglipEncoderLayerUpImpl() {};

  void load_state_dict(const StateDict& state_dict);

  void verify_loaded_weights(const std::string& weight_str) const {};

  torch::Tensor forward(torch::Tensor& x);

 private:
  void build_graph(const std::string& prefix = "");

  atb_torch::GraphOperation graph_;
  std::vector<std::shared_ptr<atb_torch::BaseOperation>> ops_;
  std::vector<torch::Tensor> weights_;

  ModelArgs model_args_;
  torch::TensorOptions options_;

  std::string prefix_;
};

class SiglipEncoderLayerUp
    : public torch::nn::ModuleHolder<SiglipEncoderLayerUpImpl> {
 public:
  using torch::nn::ModuleHolder<SiglipEncoderLayerUpImpl>::ModuleHolder;
  using Impl __attribute__((__unused__)) = SiglipEncoderLayerUpImpl;

  SiglipEncoderLayerUp(const Context& context, const std::string& prefix = "");
};

class SiglipEncoderLayerDownImpl : public torch::nn::Module {
 public:
  SiglipEncoderLayerDownImpl(const Context& context,
                             const std::string& prefix = "");

  ~SiglipEncoderLayerDownImpl() {};

  void load_state_dict(const StateDict& state_dict);

  void verify_loaded_weights(const std::string& weight_str) const {};

  torch::Tensor forward(torch::Tensor& x, torch::Tensor& y);

 private:
  void build_graph(const std::string& prefix = "");

  std::string prefix_;

  atb_torch::GraphOperation graph_;
  std::vector<std::shared_ptr<atb_torch::BaseOperation>> ops_;
  std::vector<torch::Tensor> weights_;

  ModelArgs model_args_;
  torch::TensorOptions options_;
};

class SiglipEncoderLayerDown
    : public torch::nn::ModuleHolder<SiglipEncoderLayerDownImpl> {
 public:
  using torch::nn::ModuleHolder<SiglipEncoderLayerDownImpl>::ModuleHolder;
  using Impl __attribute__((__unused__)) = SiglipEncoderLayerDownImpl;

  SiglipEncoderLayerDown(const Context& context,
                         const std::string& prefix = "");
};

class SiglipEncoderLayerImpl : public torch::nn::Module {
 public:
  SiglipEncoderLayerImpl(const Context& context,
                         const std::string& prefix = "");

  ~SiglipEncoderLayerImpl() {};

  void load_state_dict(const StateDict& state_dict);

  void verify_loaded_weights(const std::string& weight_str) const {};

  torch::Tensor forward(torch::Tensor& x);

 private:
  std::string prefix_;

  ModelArgs model_args_;
  torch::TensorOptions options_;

  SiglipEncoderLayerUp up_{nullptr};
  SiglipEncoderLayerDown down_{nullptr};
};

class SiglipEncoderLayer
    : public torch::nn::ModuleHolder<SiglipEncoderLayerImpl> {
 public:
  using torch::nn::ModuleHolder<SiglipEncoderLayerImpl>::ModuleHolder;
  using Impl __attribute__((__unused__)) = SiglipEncoderLayerImpl;

  SiglipEncoderLayer(const Context& context, const std::string& prefix = "");
};

}  // namespace xllm::hf
