/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef ATB_SPEED_MODELS_MOE_DECODER_MODEL_H
#define ATB_SPEED_MODELS_MOE_DECODER_MODEL_H

#include <vector>
#include <models/base/model/decoder_model.h>
#include "atb_speed/base/model.h"
#include "models/base/param/model_param.h"
#include "models/moe/layer/decoder_layer.h"

namespace atb_speed {
namespace moe {
/// A class that defines the parameters of base MoE models.
class MoeModelParam : public atb_speed::base::ModelParam {
public:
    /// A function that prints parameters.
    void PrintParam() override;
    /// A function that check the validity of parameters.
    void CheckParam() override;

    /// Deprecated.
    bool normTopkProb = false;
    /// A flag that indicates whether the norm in layers has bias.
    bool normHasBias = false;
    /// A flag that indicates whether there are shared experts.
    bool hasSharedExpert = true;
    /// A flag that indicates whether there is routing mechanism for shared experts.
    bool hasSharedExpertGate = false;

    // moe相关成员变量

    /// The number of experts loaded to the device.
    uint32_t numOfDeviceExperts = 64;
    /// The total number of experts utilized by the model.
    uint32_t numOfExperts = 64;
    /// The size of the expert parallelism communication domain.
    int expertParallelDegree = 0;
    /// The number of layers to replace MoE by dense MLP at the beginning of the MoE model.
    int firstKDenseReplace = 1;
    /// The total number of shared experts utilized by the model.
    int numOfSharedExperts = 2;
    /// Deprecated.
    int maskStartIdx = 0;
    /// A flag indicating whether or not to use integrated routing operators.
    bool enableFusedRouting = false;
    /// A flag indicating whether to use the integrated routing-quant operator
    bool enableInitQuant = false;
    /// A flag indicating whether to use the integrated swiglu-quant operator
    bool enableSwigluQuant = false;
    /// A flag indicating whether or not to use fused atb GMM+Swiglu+quant operators instead of aclnn.
    bool enableAtlasGMMFused = false;
    /// The list of experts loaded on the device.
    std::vector<int32_t> deviceExpert = {};

    // ==== 并行策略参数 ====

    /// A flag indicating whether or not to use tensor parallelism in attention.
    bool hasAttnTp = false;
    /// The rank of this device in the tensor parallelism communication domain in attention.
    int attnTpRank = 0;
    /// The size of the tensor parallelism communication domain in attention.
    int attnTpSize = 1;
    /// The communication domain of tensor parallelism in attention.
    std::string attnTpDomain = "";
    /// The rankTableFile for the device in the attnTp communication domain.
    std::string attnTpRankTableFile = "";
    std::string attnTpBackend = "";
    /// A flag indicating whether or not to use tensor parallelism in attention.
    bool hasAttnOprojTp = false;
    /// The rank of this device in the tensor parallelism communication domain in attention.
    int attnOprojTpRank = 0;
    /// The size of the tensor parallelism communication domain in attention.
    int attnOprojTpSize = 1;
    /// The communication domain of tensor parallelism in attention.
    std::string attnOprojTpDomain = "";
    /// The rankTableFile for the device in the attnTp communication domain.
    std::string attnOprojTpRankTableFile = "";
    std::string attnOprojTpBackend = "";
    bool attnOprojPrefetch = false;
    /// A flag indicating whether or not to use data parallelism in attention.
    bool hasAttnDp = false;
    /// The rank of this device in the data parallelism communication domain in attention.
    int attnDpRank = 0;
    /// The size of the data parallelism communication domain in attention.
    int attnDpSize = 1;
    /// The communication domain of data parallelism in attention.
    std::string attnDpDomain = "";
    /// The rankTableFile for the device in the attnDp communication domain.
    std::string attnDpRankTableFile = "";
    std::string attnDpBackend = "";
    /// A flag indicating whether the model utilizes expert parallelism.
    bool hasMoeEp = false;
    /// The rank of this device in the expert parallelism communication domain.
    int moeEpRank = 0;
    /// The size of the expert parallelism communication domain.
    int moeEpSize = 1;
    int maxDecodeDpTokenSize = 0;
    /// The communication domain of expert parallelism.
    std::string moeEpDomain = "";
    /// The rankTableFile for the device in the communication domain.
    std::string moeEpRankTableFile = "";
    std::string moeEpBackend = "";

    bool hasMoeTp = false;
    /// The rank of this device in the expert parallelism communication domain.
    int moeTpRank = 0;
    /// The size of the expert parallelism communication domain.
    int moeTpSize = 1;
    /// The communication domain of expert parallelism.
    std::string moeTpDomain = "";
    /// The rankTableFile for the device in the communication domain.
    std::string moeTpRankTableFile = "";
    std::string moeTpBackend = "";

    /// A flag indicating whether or not to use integrated GMM+Swiglu+quant operators.
    bool enableGMMSwigluQuant = false;
    /// A flag indicating whether FFN utilizes tensor parallelism.
    bool hasMlpTp = false;
    /// The rank of this device in the tensor parallelism communication domain of FFN.
    int mlpTpRank = 0;
    /// The size of the tensor parallelism communication domain of FFN.
    int mlpTpSize = 1;
    /// The communication domain of FFN tensor parallelism of FFN.
    std::string mlpTpDomain = "";
    /// The rankTableFile for the device in the mlpTp communication domain.
    std::string mlpTpRankTableFile = "";
    std::string mlpTpBackend = "";

    int lmHeadTpRank = 0;
    int lmHeadTpSize = 1;
    std::string lmHeadTpDomain = "";
    bool enableDpOut = false;
    bool lmHeadLocalTp = false;

    /// The way in which the top k experts are selected.
    std::string routingMethod = "softMaxTopK";
    /// The way in which expert scores are further processed.
    std::string processLogits = "normalization";
    /// A matrix that defines the quantization types of linears in MoE.
    /// It is automatically generated by MoE weight wrapper.
    std::vector<std::vector<int>> moeLinearQuantType = {};
    /// A matrix that defines the quantization types of linears in MLP.
    /// It is automatically generated by MoE weight wrapper.
    std::vector<std::vector<int>> mlpLinearQuantType = {};
    /// A matrix that defines the transpose types of linears in MoE.
    /// It is automatically generated by MoE weight wrapper.
    std::vector<std::vector<int>> moeLinearTransposeType = {};
    /// A matrix that defines the transpose types of linears in MLP.
    /// It is automatically generated by MoE weight wrapper.
    std::vector<std::vector<int>> mlpLinearTransposeType = {};
    /// The number of experts selected for each token.
    atb::SVector<int32_t> numOfSelectedExperts = {};

protected:
    /// A function that reads and parses parameters from json-like objects.
    /// \param paramJson Json-like objects (e.g. dictionaries in Python) containing parameters that defines a MoE model.
    void ParseParam(const nlohmann::json &paramJson) override;
    /// A function that reads and parses parameters from json-like objects, specially used for quantization.
    /// \param paramJson Json-like objects (e.g. dictionaries in Python) containing parameters that defines a MoE model.
    virtual void ParseQuantParams(const nlohmann::json &paramJson);
    /// A function that reads and parses parameters from json-like objects, specially used for parallelism.
    /// \param paramJson Json-like objects (e.g. dictionaries in Python) containing parameters that defines a MoE model.
    virtual void ParseAttnParallelParams(const nlohmann::json &paramJson);
    virtual void ParseParallelParams(const nlohmann::json &paramJson);
    /// A function that reads and parses parameters from json-like objects, specially used for integrated operators.
    /// \param paramJson Json-like objects (e.g. dictionaries in Python) containing parameters that defines a MoE model.
    virtual void ParseInteOpParams(const nlohmann::json &paramJson);
    /// A function that check the validity of parallelism params.
    /// \throw std::runtime_error If the input parallelism params is not valid.
    virtual void CheckParallelParamValid();
    /// A function that check the validity of routingMethod.
    /// \throw std::runtime_error If the input routingMethod is not valid.
    virtual void CheckRoutingMethodValid();
    /// A function that check the validity of processLogits.
    /// \throw std::runtime_error If the input processLogits is not valid.
    virtual void CheckProcessLogitsValid();
};

/// A class that defines base MoE models, inherited from the `DecoderModel` class.
class MoeDecoderModel : public atb_speed::base::DecoderModel {
public:
    explicit MoeDecoderModel(const std::string &param);

protected:
    /// A function that constructs a map from in tensor names to in tensor ids.
    /// The order of the input tensors should align with the sequence passed from the Python side.
    void ConstructInTensorMap() override;
    /// A function that update parameters of a MoE layer based on parsed MoE model parameters.
    /// \param layerParam An `MoeLayerParam` object that needs to be updated.
    /// \param layerId The index of the current layer.
    void SetLayerParam(MoeLayerParam &layerParam, uint32_t layerId);
    /// A function that defines default inputs for MoE layers.
    /// \param layerNode A `atb_speed::Model::Node` object representing a MoE layer in the computation graph.
    /// \param layerId The index of the current layer.
    void SetLayerNodeInput(atb_speed::Model::Node &layerNode, uint32_t layerId) override;
    /// A function that builds a layer opeartion, call `SetLayerParam` and
    /// call `MoeDecoderLayer`'s `buildGraph` function to create an operation.
    /// \param op The address of a pointer to a default operation.
    /// \param layerId The index of the current layer.
    /// \return A flag that indicates whether a layer operation is created successfully.
    atb::Status CreateLayerOperation(atb::Operation **op, uint32_t layerId) override;

    MoeModelParam param;
};

}  // namespace moe
}  // namespace atb_speed
#endif  // ATB_SPEED_MODELS_MOE_DECODER_MODEL_H