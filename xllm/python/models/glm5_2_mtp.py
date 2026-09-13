# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GLM-5.2 MTP draft model for the Python executor.

The native GLM MTP implementation uses the regular DeepSeek-V3.2 decoder
layer for its draft body.  GLM-5.2 and DeepSeek-V3.2 use the same absorbed MLA
and MoE tensor contract for that body, while the target GLM model keeps its
GLM-specific indexer sharing in the target worker.  This adapter translates
the GLM config into the existing DeepSeek MTP graph and leaves embedding and
LM-head sharing to ``PyCausalLM::share_weights_from``.
"""

from __future__ import annotations

from dataclasses import fields

from xllm.python.models.deepseek_v32 import DeepseekV3Config
from xllm.python.models.deepseek_v32_mtp import DeepseekV32MtpForCausalLM
from xllm.python.models.glm5_2 import Glm52Config


def _adapt_glm_mtp_config(config: dict) -> dict:
    """Return the common decoder config expected by the DeepSeek MTP graph.

    C++ passes ``n_layers`` from the MTP model registration (normally the
    number of next-token prediction layers).  For callers that only provide
    the target GLM config, prefer ``num_nextn_predict_layers`` so we do not
    accidentally instantiate all target layers as the draft body.
    """
    glm_cfg = Glm52Config.from_dict(config)
    adapted = dict(config)
    for field in fields(DeepseekV3Config):
        name = field.name
        if hasattr(glm_cfg, name):
            adapted[name] = getattr(glm_cfg, name)

    configured_layers = int(config.get("n_layers", config.get("num_hidden_layers", 0)))
    mtp_layers = int(config.get("num_nextn_predict_layers", 0))
    if mtp_layers > 0:
        configured_layers = mtp_layers
    adapted["n_layers"] = max(configured_layers, 1)
    adapted["model_type"] = "glm_moe_dsa_mtp"
    return adapted


class Glm52MtpForCausalLM(DeepseekV32MtpForCausalLM):
    """GLM-5.2 MTP graph using the shared DeepSeek MTP body contract."""

    def __init__(self, config: dict) -> None:
        super().__init__(_adapt_glm_mtp_config(config))


__all__ = ["Glm52MtpForCausalLM", "_adapt_glm_mtp_config"]
