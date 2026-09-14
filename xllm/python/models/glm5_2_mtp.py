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

"""GLM MTP body with GLM indexer/quantization and recurrent hidden states."""

from __future__ import annotations

import torch
import torch.nn as nn

from xllm.python.layers import ColumnParallelLinear, RMSNorm
from xllm.python.model_executor.forward_context import record_layer_event
from xllm.python.models.glm5_2 import (
    Glm52Config,
    Glm52DecoderLayer,
    Glm52ForCausalLM,
    Glm52YarnRotaryEmbedding,
)
from xllm.python.models.weight_utils import W8A8WeightLoader


def _adapt_glm_mtp_config(config: dict) -> dict:
    """Select the exported MTP layers without inheriting target layer types.

    C++ passes ``n_layers`` from the MTP model registration (normally the
    number of next-token prediction layers).  For callers that only provide
    the target GLM config, prefer ``num_nextn_predict_layers`` so we do not
    accidentally instantiate all target layers as the draft body.
    """
    adapted = dict(config)
    configured_layers = int(config.get("n_layers", config.get("num_hidden_layers", 0)))
    mtp_layers = int(config.get("num_nextn_predict_layers", 0))
    if mtp_layers > 0:
        configured_layers = mtp_layers
    adapted["n_layers"] = max(configured_layers, 1)
    adapted["model_type"] = "glm_moe_dsa_mtp"
    adapted["first_k_dense_replace"] = 0
    adapted["mlp_layer_types"] = ["sparse"] * adapted["n_layers"]
    adapted["indexer_types"] = ["full"] * adapted["n_layers"]
    return adapted


class Glm52MtpModel(nn.Module):
    """Draft decoder; the final head norm is applied only to logits."""

    def __init__(self, cfg: Glm52Config, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.embed_tokens: nn.Module | None = None
        self.eh_proj = ColumnParallelLinear(
            2 * cfg.hidden_size,
            cfg.hidden_size // cfg.tp_size,
            cfg.tp_size,
            gather_output=True,
            dtype=dtype,
            device=device,
        )
        self.rot = ColumnParallelLinear(
            cfg.hidden_size,
            cfg.hidden_size // cfg.tp_size,
            cfg.tp_size,
            gather_output=True,
            dtype=dtype,
            device=device,
        )
        self.enorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype, device)
        self.hnorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype, device)
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype, device)
        self.layers = nn.ModuleList([Glm52DecoderLayer(cfg, i, dtype, device) for i in range(cfg.n_layers)])
        self.rotary = Glm52YarnRotaryEmbedding(
            cfg.qk_rope_head_dim,
            cfg.original_max_position_embeddings,
            cfg.rope_scaling_factor,
            cfg.rope_theta,
            cfg.rope_beta_fast,
            cfg.rope_beta_slow,
            cfg.rope_mscale,
            cfg.rope_mscale_all_dim,
            dtype=dtype,
            device=device,
        )
        self.enable_rot = False

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert self.embed_tokens is not None
        token_hidden = self.embed_tokens(input_ids)
        hidden = token_hidden if input_embedding is None else input_embedding
        if self.enable_rot:
            hidden = self.rot(hidden)
        hidden = self.eh_proj(torch.cat((self.enorm(token_hidden), self.hnorm(hidden)), dim=-1))
        positions = positions.to(torch.int64).contiguous()
        residual = None
        topk = None
        for layer_id, layer in enumerate(self.layers):
            hidden, residual, topk = layer(hidden, residual, positions, self.rotary.cos_sin_cache, topk)
            record_layer_event(layer_id)
        # Native MtpModelImplBase feeds the unnormalized decoder result back
        # into the next draft step. shared_head.norm belongs to logits only.
        return hidden if residual is None else hidden + residual


class Glm52MtpForCausalLM(Glm52ForCausalLM):
    """GLM draft body; C++ shares the target embedding and output head."""

    def __init__(self, config: dict) -> None:
        super().__init__(_adapt_glm_mtp_config(config), build_model=False)
        self.model = Glm52MtpModel(self.cfg, self.dtype, self.device)
        self.lm_head = None

    def load_weights(self, state_dicts: list, tp_rank: int, tp_size: int) -> None:
        loader = W8A8WeightLoader(
            self,
            state_dicts,
            self.cfg.tp_size,
            self.cfg.tp_rank,
            src_prefixes=("", "model."),
            name_aliases={
                "model.norm.weight": (
                    "model.norm.weight",
                    "model.final_norm.weight",
                    "model.shared_head.norm.weight",
                ),
            },
        )
        super().load_weights(
            state_dicts,
            tp_rank,
            tp_size,
            load_lm_head=False,
            load_embedding=False,
            loader=loader,
        )
        for name in ("eh_proj", "enorm", "hnorm"):
            key = f"model.{name}.weight"
            if name == "eh_proj":
                loader.copy_shard(key, dim=0)
            else:
                loader.copy_replicated(key)
        self.model.enable_rot = loader.has("model.rot.weight")
        if self.model.enable_rot:
            loader.copy_shard("model.rot.weight", dim=0)

    def compute_logits(self, hidden: torch.Tensor, selected_idxes: torch.Tensor | None) -> torch.Tensor:
        if selected_idxes is not None and selected_idxes.numel() > 0:
            hidden = hidden.index_select(0, selected_idxes)
        return self.lm_head(self.model.norm(hidden))


__all__ = ["Glm52MtpForCausalLM", "_adapt_glm_mtp_config"]
