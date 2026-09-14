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

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

pytest.importorskip("torch_npu")

from xllm.python.models.glm5_2 import Glm52DecoderLayer, Glm52MoE  # noqa: E402
from xllm.python.models.glm5_2_mtp import (  # noqa: E402
    Glm52MtpForCausalLM,
    Glm52MtpModel,
    _adapt_glm_mtp_config,
)
from xllm.python.registry import get_model_class  # noqa: E402


def test_glm_mtp_registry_resolves_python_draft() -> None:
    assert get_model_class("glm_moe_dsa_mtp") is Glm52MtpForCausalLM
    # export_mtp.py writes this HF architecture name into the generated
    # config; keep the architecture and model_type entry points equivalent.
    assert get_model_class("GlmMoeDsaMtpForCausalLM") is Glm52MtpForCausalLM


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


def test_glm_mtp_constructor_builds_glm_sparse_indexer_body() -> None:
    config = {
        "model_type": "glm_moe_dsa_mtp",
        "n_layers": 1,
        "num_nextn_predict_layers": 2,
        "hidden_size": 16,
        "num_attention_heads": 4,
        "intermediate_size": 32,
        "vocab_size": 32,
        "max_position_embeddings": 16,
        "q_lora_rank": 8,
        "kv_lora_rank": 4,
        "qk_nope_head_dim": 4,
        "qk_rope_head_dim": 4,
        "v_head_dim": 4,
        "index_n_heads": 2,
        "index_head_dim": 8,
        "index_topk": 4,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 8,
        "dtype": "float32",
        "device": "cpu",
    }
    model = Glm52MtpForCausalLM(config)

    assert model.cfg.n_layers == len(model.model.layers) == 2
    assert model.model.embed_tokens is None
    assert model.lm_head is None
    for layer in model.model.layers:
        assert isinstance(layer, Glm52DecoderLayer)
        assert isinstance(layer.mlp, Glm52MoE)
        assert layer.self_attn.indexer is not None
        assert layer.self_attn.indexer.wq_b.weight.shape == (16, 8)


def test_glm_mtp_target_config_selects_sparse_full_indexer_body() -> None:
    adapted = _adapt_glm_mtp_config(
        {
            "num_hidden_layers": 78,
            "num_nextn_predict_layers": 1,
            "first_k_dense_replace": 3,
            "mlp_layer_types": ["dense"] * 3 + ["sparse"] * 75,
            "indexer_types": ["shared"] * 78,
        }
    )
    assert adapted["first_k_dense_replace"] == 0
    assert adapted["mlp_layer_types"] == ["sparse"]
    assert adapted["indexer_types"] == ["full"]


def test_glm_mtp_logits_normalize_without_mutating_recurrent_hidden() -> None:
    model = Glm52MtpForCausalLM.__new__(Glm52MtpForCausalLM)
    nn.Module.__init__(model)
    model.model = nn.Module()
    model.model.norm = nn.LayerNorm(4, elementwise_affine=False)
    model.lm_head = nn.Linear(4, 3, bias=False)
    hidden = torch.tensor([[1.0, 4.0, 2.0, -3.0], [2.0, 3.0, 8.0, -1.0]])
    original = hidden.clone()
    selected = torch.tensor([1])

    logits = model.compute_logits(hidden, selected)

    torch.testing.assert_close(logits, model.lm_head(model.model.norm(hidden[selected])))
    torch.testing.assert_close(hidden, original)


def test_glm_mtp_forward_returns_raw_hidden_with_residual() -> None:
    class Decoder(nn.Module):
        def forward(self, hidden, residual, positions, cache, topk):
            assert residual is None
            return hidden * 2, hidden + 1, None

    model = Glm52MtpModel.__new__(Glm52MtpModel)
    nn.Module.__init__(model)
    model.embed_tokens = nn.Embedding.from_pretrained(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    model.enorm = nn.Identity()
    model.hnorm = nn.Identity()
    model.eh_proj = nn.Linear(4, 2, bias=False)
    model.eh_proj.weight.data.copy_(torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]]))
    model.norm = nn.LayerNorm(2, elementwise_affine=False)
    model.layers = nn.ModuleList([Decoder()])
    model.rotary = SimpleNamespace(cos_sin_cache=torch.zeros(2, 4))
    model.enable_rot = False
    hidden = torch.tensor([[5.0, 7.0]])

    output = model(torch.tensor([1]), torch.tensor([0]), hidden)

    expected = (torch.tensor([[3.0, 4.0]]) + hidden) * 3 + 1
    torch.testing.assert_close(output, expected)
    assert not torch.allclose(output, model.norm(expected))
