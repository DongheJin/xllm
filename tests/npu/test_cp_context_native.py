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

"""Opt-in native CP planner probe for a single-node prefill contract."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch


def test_native_cp_plan_and_shard_contract() -> None:
    """Exercise the production C++ planner and Python shard path together.

    This is intentionally CPU-only: ``build_cp_context`` is a host index
    planner, so it gives us a real native prefill contract check even when a
    full GLM52 checkpoint is unavailable on the NPU host.
    """
    library = os.environ.get("XLLM_TEST_NATIVE_LIBRARY")
    if not library:
        pytest.skip("set XLLM_TEST_NATIVE_LIBRARY for the native CP probe")
    assert Path(library).is_file(), f"native operator library does not exist: {library}"
    torch.ops.load_library(library)

    from xllm.python.model_executor.cp_utils import build_cp_context, cp_shard_rows

    source = torch.arange(11, dtype=torch.float32).view(-1, 1)
    source[2] = float("nan")
    contexts = [
        build_cp_context([5, 6], [7, 8], cp_size=2, cp_rank=rank, device=torch.device("cpu")) for rank in (0, 1)
    ]
    assert contexts[0].total_local == contexts[1].total_local

    for context in contexts:
        local = cp_shard_rows(source, context)
        valid = context.shard_valid_mask
        torch.testing.assert_close(local[~valid], torch.zeros_like(local[~valid]))
        expected = source.index_select(0, context.shard_gather_index)
        assert torch.equal(torch.isnan(local[valid]), torch.isnan(expected[valid]))
        assert context.q_cu_seqlens
        assert context.kv_cu_seqlens
        assert context.segment_kv_seq_lens
