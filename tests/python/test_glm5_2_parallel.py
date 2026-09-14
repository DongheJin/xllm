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

"""Parallel-layout tests for the GLM-5.2 Python NPU model."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from xllm.python.models import glm5_2
from xllm.python.models.deepseek_v32 import W8A8DynamicLinear, W8A8StaticLinear
from xllm.python.models.glm5_2 import Glm52Config, Glm52ForCausalLM
from xllm.python.models.weight_utils import W8A8WeightLoader


def _config(**overrides) -> dict:
    values = {
        "model_type": "glm_moe_dsa",
        "hidden_size": 16,
        "num_hidden_layers": 1,
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
        "first_k_dense_replace": 0,
        "n_routed_experts": 8,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 8,
        "tp_size": 2,
        "tp_rank": 0,
        "dp_size": 2,
        "dp_rank": 0,
        "cp_size": 1,
        "cp_rank": 0,
        "world_size": 4,
        "moe_tp_size": 1,
        "moe_tp_rank": 0,
        "ep_size": 4,
        "ep_rank": 0,
        "dtype": "float32",
        "device": "cpu",
    }
    values.update(overrides)
    return values


def test_full_world_ep_partitions_glm_experts() -> None:
    cfg = Glm52Config.from_dict(_config(ep_rank=3))
    cfg.validate()

    model = Glm52ForCausalLM(_config(ep_rank=3))
    moe = model.model.layers[0].mlp

    assert moe.local_expert_start == 6
    assert moe.local_expert_end == 8
    assert moe.num_local_experts == 2

    assert moe.experts_w13.numel() == 0
    assert moe.experts_w2.numel() == 0
    moe.allocate_experts_w13_for_loading()
    assert moe.experts_w13.shape == (2, 16, 16)
    assert moe.experts_w2.numel() == 0
    moe.allocate_experts_w2_for_loading()
    assert moe.experts_w2.shape == (2, 16, 8)


def test_glm_parallel_world_size_defaults_to_tp_dp_product() -> None:
    values = _config()
    values.pop("world_size")

    cfg = Glm52Config.from_dict(values)

    assert cfg.world_size == cfg.tp_size * cfg.dp_size * cfg.cp_size == 4


def test_glm_parallel_world_size_includes_context_parallel() -> None:
    cfg = Glm52Config.from_dict(_config(cp_size=2, cp_rank=1, world_size=8, ep_size=8))

    cfg.validate()

    assert cfg.world_size == cfg.tp_size * cfg.dp_size * cfg.cp_size == 8
    assert cfg.cp_rank == 1


def test_glm_layerwise_split_rank_is_validated() -> None:
    cfg = Glm52Config.from_dict(_config(layerwise_split_size=2, layerwise_split_rank=1))
    cfg.validate()
    assert cfg.layerwise_split_rank == 1

    invalid = Glm52Config.from_dict(_config(layerwise_split_size=2, layerwise_split_rank=2))
    with pytest.raises(ValueError, match="layerwise_split_rank"):
        invalid.validate()


def test_glm_layerwise_split_cannot_overlap_context_parallel() -> None:
    cfg = Glm52Config.from_dict(
        _config(
            cp_size=2,
            cp_rank=0,
            world_size=8,
            ep_size=8,
            layerwise_split_size=2,
        )
    )
    with pytest.raises(ValueError, match="CP and layerwise"):
        cfg.validate()


def test_glm_dynamic_checkpoint_switches_attention_projections() -> None:
    model = Glm52ForCausalLM(_config(first_k_dense_replace=1, indexer_types=["full"]))
    probe = MagicMock()
    probe.has.side_effect = lambda name: name.endswith("weight_scale")

    assert model._configure_attention_quantization(probe) is True
    attn = model.model.layers[0].self_attn
    for name in ("q_a_proj", "kv_a_proj_with_mqa", "q_b_proj", "o_proj"):
        assert isinstance(getattr(attn, name), W8A8DynamicLinear)
        assert not isinstance(getattr(attn, name), W8A8StaticLinear)
    assert attn.indexer is not None
    assert isinstance(attn.indexer.wq_b, W8A8DynamicLinear)


class _TensorStateDict:
    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self.tensors = tensors

    def has(self, name: str) -> bool:
        return name in self.tensors

    def get_tensor(self, name: str) -> torch.Tensor:
        return self.tensors[name]


def _quantized_checkpoint(dynamic_projections: set[str]) -> _TensorStateDict:
    """Build a full, unsharded checkpoint with distinct channel values."""
    model = Glm52ForCausalLM(
        _config(
            tp_size=1,
            dp_size=1,
            world_size=1,
            ep_size=1,
            num_hidden_layers=2,
            first_k_dense_replace=2,
            indexer_types=["full", "full"],
        )
    )
    tensors = {}
    for name, value in model.state_dict().items():
        tensor = (torch.arange(value.numel()).reshape(value.shape) % 97 + 1).to(value.dtype)
        if name.endswith("weight_offset"):
            tensor.zero_()
        if ".gate_up_proj." in name:
            gate, up = tensor.chunk(2, dim=0)
            tensors[name.replace("gate_up_proj", "gate_proj")] = gate.clone()
            tensors[name.replace("gate_up_proj", "up_proj")] = up.clone()
        else:
            tensors[name] = tensor
    for prefix in dynamic_projections:
        out_features = tensors[prefix + ".weight"].shape[0]
        for suffix in ("deq_scale", "quant_bias", "input_scale", "input_offset"):
            tensors.pop(prefix + "." + suffix)
        tensors[prefix + ".weight_scale"] = torch.arange(1, out_features + 1, dtype=torch.float32).view(-1, 1) / 128
        tensors[prefix + ".weight_offset"] = torch.zeros(out_features, 1)
    return _TensorStateDict(tensors)


@pytest.mark.parametrize("tp_rank", [0, 1])
@pytest.mark.parametrize("mixed", [False, True])
def test_glm_dynamic_projection_shards_match_checkpoint(monkeypatch, tp_rank: int, mixed: bool) -> None:
    projections = ("q_a_proj", "kv_a_proj_with_mqa", "q_b_proj", "o_proj", "indexer.wq_b")
    dynamic = {
        f"model.layers.{layer}.self_attn.{proj}"
        for layer in range(2)
        for proj in projections
        if not mixed or (layer == 1 and proj in ("q_b_proj", "o_proj", "indexer.wq_b"))
    }
    checkpoint = _quantized_checkpoint(dynamic)
    model = Glm52ForCausalLM(
        _config(
            tp_rank=tp_rank,
            dp_size=1,
            world_size=2,
            ep_size=1,
            num_hidden_layers=2,
            first_k_dense_replace=2,
            indexer_types=["full", "full"],
        )
    )
    # Keep the actual loader, shape checks and post-load processing; only
    # replace the device-specific NZ conversion with its logical transpose.
    monkeypatch.setattr(glm5_2.kernels, "prepare_quant_weight", lambda w: w.t().contiguous(), raising=False)

    model.load_weights([checkpoint], tp_rank=tp_rank, tp_size=2)

    for prefix in dynamic:
        proj = model.get_submodule(prefix)
        assert isinstance(proj, W8A8DynamicLinear)
        weight = checkpoint.get_tensor(prefix + ".weight")
        scale = checkpoint.get_tensor(prefix + ".weight_scale").flatten()
        offset = checkpoint.get_tensor(prefix + ".weight_offset").flatten()
        if prefix.endswith(".o_proj"):
            weight = weight.chunk(2, dim=1)[tp_rank]
        elif prefix.endswith(".q_b_proj"):
            weight = weight.chunk(2, dim=0)[tp_rank]
            scale = scale.chunk(2)[tp_rank]
            offset = offset.chunk(2)[tp_rank]
        torch.testing.assert_close(proj.weight, weight.t().contiguous())
        torch.testing.assert_close(proj.weight_scale, scale)
        torch.testing.assert_close(proj.weight_offset, offset)
    if mixed:
        assert isinstance(model.model.layers[0].self_attn.q_a_proj, W8A8StaticLinear)
        assert isinstance(model.model.layers[1].self_attn.kv_a_proj_with_mqa, W8A8StaticLinear)


def test_glm_static_checkpoint_with_weight_scale_remains_static() -> None:
    model = Glm52ForCausalLM(_config(first_k_dense_replace=1, indexer_types=["full"]))
    prefix = "model.layers.0.self_attn.q_a_proj."
    checkpoint = _TensorStateDict({prefix + "deq_scale": torch.ones(8), prefix + "weight_scale": torch.ones(8, 1)})
    loader = W8A8WeightLoader(model, [checkpoint], 2, 0)

    assert model._configure_attention_quantization(loader) is False
    assert isinstance(model.model.layers[0].self_attn.q_a_proj, W8A8StaticLinear)


def test_glm_dynamic_projection_rejects_nonzero_weight_offset() -> None:
    proj = W8A8DynamicLinear(8, 4, torch.device("cpu"))
    proj.weight_offset.fill_(1)
    with pytest.raises(ValueError, match="zero weight_offset"):
        proj.process_weights_after_loading()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"ep_size": 2}, "ep_size must be 1 or world_size"),
        ({"world_size": 8}, r"world_size must equal tp_size \* dp_size \* cp_size"),
        ({"cp_rank": 2}, "cp_rank must be in"),
        ({"n_routed_experts": 10}, "n_routed_experts must be divisible by ep_size"),
        ({"moe_tp_size": 2}, r"moe_tp_size \* ep_size"),
        ({"ep_rank": 4}, "ep_rank must be in"),
    ],
)
def test_invalid_glm_parallel_topology_is_rejected(overrides: dict, message: str) -> None:
    cfg = Glm52Config.from_dict(_config(**overrides))

    with pytest.raises(ValueError, match=message):
        cfg.validate()


class _RecordingLoader(W8A8WeightLoader):
    latest: _RecordingLoader | None = None

    def __init__(self, model, state_dicts, tp_size: int, tp_rank: int) -> None:
        super().__init__(model, state_dicts, tp_size, tp_rank)
        self.loaded: list[str] = []
        self.shared_shards: list[tuple[str, int, int]] = []
        type(self).latest = self

    def load_tensor(self, name: str) -> torch.Tensor:
        self.loaded.append(name)
        if ".mlp.experts." not in name:
            return torch.zeros(32, 32)
        expert_id = int(name.split(".experts.")[1].split(".")[0])
        if name.endswith(("gate_proj.weight", "up_proj.weight")):
            value = expert_id + (11 if name.endswith("up_proj.weight") else 1)
            return torch.full((8, 16), value, dtype=torch.int8)
        if name.endswith(("gate_proj.weight_scale", "up_proj.weight_scale")):
            return torch.zeros(8, 1)
        if name.endswith(("gate_proj.weight_offset", "up_proj.weight_offset")):
            return torch.zeros(8, 1)
        if name.endswith("down_proj.weight"):
            return torch.full((16, 8), expert_id + 21, dtype=torch.int8)
        if name.endswith(("down_proj.weight_scale", "down_proj.weight_offset")):
            return torch.zeros(16, 1)
        raise AssertionError(f"unexpected expert tensor: {name}")

    def copy_in(self, name: str, tensor: torch.Tensor) -> None:
        self.loaded.append(name)
        assert tensor.is_contiguous()

    def load_w8a8_projection(self, prefix: str, proj: str, _shard_dims: dict | None = None) -> None:
        self.loaded.append(prefix + proj)

    def load_w8a8_mlp(
        self,
        prefix: str,
        world: int | None = None,
        rank: int | None = None,
    ) -> None:
        self.loaded.append(prefix)
        if ".shared_experts." in prefix:
            self.shared_shards.append((prefix, world, rank))


@pytest.mark.parametrize("ep_rank", [0, 2, 3])
def test_glm_weight_loader_reads_only_local_ep_experts(monkeypatch: pytest.MonkeyPatch, ep_rank: int) -> None:
    model = Glm52ForCausalLM(_config(ep_rank=ep_rank))
    moe = model.model.layers[0].mlp
    model.model.layers[0].self_attn.process_weights_after_loading = MagicMock()
    moe.shared_experts.process_weights_after_loading = MagicMock()

    monkeypatch.setattr(glm5_2, "W8A8WeightLoader", _RecordingLoader)
    formatted_shapes: list[tuple[int, ...]] = []

    def _format_cast_nz(weight: torch.Tensor) -> torch.Tensor:
        assert weight.is_contiguous()
        if not formatted_shapes:
            assert moe.experts_w2.numel() == 0
        formatted_shapes.append(tuple(weight.shape))
        return weight

    monkeypatch.setattr(glm5_2.kernels, "format_cast_nz", _format_cast_nz, raising=False)

    model.load_weights([], tp_rank=0, tp_size=2)

    loader = _RecordingLoader.latest
    assert loader is not None
    expert_names = [name for name in loader.loaded if ".mlp.experts." in name]
    assert expert_names
    expert_ids = {int(name.split(".experts.")[1].split(".")[0]) for name in expert_names}
    assert expert_ids == {2 * ep_rank, 2 * ep_rank + 1}
    assert formatted_shapes == [(2, 16, 16), (2, 8, 16)]
    for local_idx, expert_id in enumerate(sorted(expert_ids)):
        expected_w13 = (
            torch.cat(
                [
                    torch.full((8, 16), expert_id + 1, dtype=torch.int8),
                    torch.full((8, 16), expert_id + 11, dtype=torch.int8),
                ]
            )
            .t()
            .contiguous()
        )
        torch.testing.assert_close(moe.experts_w13[local_idx], expected_w13)
        torch.testing.assert_close(moe.experts_w2[local_idx], torch.full((8, 16), expert_id + 21, dtype=torch.int8))
    moe.shared_experts.process_weights_after_loading.assert_called_once_with()
    assert loader.tp_size == 2
    assert loader.tp_rank == 0
    assert loader.shared_shards == [("model.layers.0.mlp.shared_experts.", 1, 0)]
