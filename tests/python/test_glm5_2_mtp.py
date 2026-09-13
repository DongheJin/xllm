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

from unittest.mock import patch

import pytest

pytest.importorskip("torch_npu")

from xllm.python.models.glm5_2_mtp import (  # noqa: E402
    Glm52MtpForCausalLM,
    _adapt_glm_mtp_config,
)
from xllm.python.registry import get_model_class  # noqa: E402


def test_glm_mtp_registry_resolves_python_draft() -> None:
    assert get_model_class("glm_moe_dsa_mtp") is Glm52MtpForCausalLM


def test_glm_mtp_config_uses_nextn_layers_and_glm_dimensions() -> None:
    adapted = _adapt_glm_mtp_config(
        {
            "model_type": "glm_moe_dsa",
            "hidden_size": 6144,
            "num_hidden_layers": 78,
            "num_nextn_predict_layers": 2,
            "num_attention_heads": 64,
            "q_lora_rank": 2048,
            "kv_lora_rank": 512,
            "qk_nope_head_dim": 192,
            "qk_rope_head_dim": 64,
            "v_head_dim": 256,
        }
    )

    assert adapted["model_type"] == "glm_moe_dsa_mtp"
    assert adapted["n_layers"] == 2
    assert adapted["hidden_size"] == 6144
    assert adapted["qk_nope_head_dim"] == 192
    assert adapted["v_head_dim"] == 256


def test_glm_mtp_constructor_passes_adapted_config() -> None:
    config = {
        "model_type": "glm_moe_dsa_mtp",
        "n_layers": 1,
        "num_nextn_predict_layers": 2,
    }
    with patch(
        "xllm.python.models.glm5_2_mtp.DeepseekV32MtpForCausalLM.__init__",
        return_value=None,
    ) as init:
        Glm52MtpForCausalLM(config)

    init.assert_called_once()
    adapted = init.call_args.args[0]
    assert adapted["model_type"] == "glm_moe_dsa_mtp"
    assert adapted["n_layers"] == 2
