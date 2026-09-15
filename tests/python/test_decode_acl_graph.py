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

"""Tests for the NPU ACL decode-graph runner."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from xllm.python.model_executor.runners.decode_acl_graph import (
    DecodeAclGraphRunner,
)


def _runner() -> DecodeAclGraphRunner:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    return DecodeAclGraphRunner(
        nn.Identity(),
        attention_backend,
        torch.device("cpu"),
        max_batch=8,
        max_model_len=8,
    )


def _metadata(linear_state_indices: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(
        slot_mapping=torch.arange(4, dtype=torch.int32),
        paged_kv_indptr=torch.arange(5, dtype=torch.int32),
        paged_kv_indices=torch.tensor([10, 20, 30, 40], dtype=torch.int32),
        paged_kv_last_page_len=torch.arange(1, 5, dtype=torch.int32),
        block_table=torch.tensor(
            [[10, 0], [20, 0], [30, 0], [40, 0]],
            dtype=torch.int32,
        ),
        kv_seq_lens=torch.arange(1, 5, dtype=torch.int32),
        kv_seq_lens_host_values=[1, 2, 3, 4],
        kv_cu_seq_lens=torch.tensor([0, 1, 3, 6, 10], dtype=torch.int32),
        linear_state_indices=linear_state_indices,
        expanded_decode_metadata=None,
    )


def test_linear_state_indices_use_stable_graph_buffer() -> None:
    runner = _runner()
    input_ids = torch.arange(4, dtype=torch.int32)
    positions = torch.arange(4, dtype=torch.int32)
    metadata = _metadata(torch.tensor([3, 7, 11, 15], dtype=torch.int32))
    entry = runner._allocate_entry(
        padded_batch_size=8,
        input_ids=input_ids,
        positions=positions,
        metadata=metadata,
    )
    static_indices = entry.static_metadata.linear_state_indices
    data_ptr = static_indices.data_ptr()

    with patch(
        "xllm.python.model_executor.runners.decode_acl_graph.kernels.update_decode_graph_metadata",
        create=True,
    ):
        runner._fill_entry(
            entry,
            input_ids,
            positions,
            metadata,
            batch_size=4,
            input_embedding=None,
        )
        assert static_indices.tolist() == [3, 7, 11, 15, 0, 0, 0, 0]

        metadata.linear_state_indices = torch.tensor(
            [4, 8, 12, 16],
            dtype=torch.int32,
        )
        runner._fill_entry(
            entry,
            input_ids,
            positions,
            metadata,
            batch_size=4,
            input_embedding=None,
        )

    assert static_indices.data_ptr() == data_ptr
    assert static_indices.tolist() == [4, 8, 12, 16, 0, 0, 0, 0]


@pytest.mark.parametrize("expanded", [False, True])
@pytest.mark.parametrize("decoding_tokens", [1, 2])
def test_dcp_graph_metadata_uses_logical_pages(expanded: bool, decoding_tokens: int) -> None:
    runner = _runner()
    runner.attention_backend.logical_page_size = 8
    runner.max_model_len = 16
    runner.num_decoding_tokens = decoding_tokens
    metadata = _metadata(torch.tensor([3, 7], dtype=torch.int32))
    metadata.slot_mapping = torch.tensor([87, 88], dtype=torch.int32)
    metadata.block_table = torch.tensor([[10, 11], [10, 11]], dtype=torch.int32)
    metadata.kv_seq_lens = torch.tensor([8, 9], dtype=torch.int32)
    metadata.kv_seq_lens_host_values = [8, 9]
    # The unexpanded input still has sequence-scoped paged metadata. The
    # runner must rebuild it for the two token rows using logical pages.
    metadata.paged_kv_indptr = torch.tensor([0, 2], dtype=torch.int32)
    metadata.paged_kv_indices = torch.tensor([10, 11], dtype=torch.int32)
    metadata.paged_kv_last_page_len = torch.tensor([1], dtype=torch.int32)
    if expanded:
        metadata.expanded_decode_metadata = SimpleNamespace(
            enabled=True,
            block_table=metadata.block_table,
            kv_seq_lens=metadata.kv_seq_lens,
            kv_seq_lens_host_values=[8, 9],
            kv_seq_lens_host=None,
            paged_kv_indptr=torch.tensor([0, 1, 3], dtype=torch.int32),
            paged_kv_indices=torch.tensor([10, 10, 11], dtype=torch.int32),
            paged_kv_last_page_len=torch.tensor([8, 1], dtype=torch.int32),
            paged_attention_tiling_data=None,
        )

    _, _, _, indptr, indices, last_page_lens = runner._decode_metadata(metadata)
    assert indptr.tolist() == [0, 1, 3]
    assert indices.tolist() == [10, 10, 11]
    assert last_page_lens.tolist() == [8, 1]

    entry = runner._allocate_entry(
        padded_batch_size=2,
        input_ids=torch.tensor([42, 43], dtype=torch.int32),
        positions=torch.tensor([7, 8], dtype=torch.int32),
        metadata=metadata,
    )
    assert entry.static_metadata.block_table.shape == (2, 3)


def test_mtp_linear_state_indices_repeat_for_expanded_rows() -> None:
    runner = _runner()
    input_ids = torch.arange(4, dtype=torch.int32)
    positions = torch.arange(4, dtype=torch.int32)
    metadata = _metadata(torch.tensor([3, 7], dtype=torch.int32))
    metadata.slot_mapping = torch.arange(4, dtype=torch.int32)
    metadata.block_table = torch.tensor([[10, 0], [20, 0], [30, 0], [40, 0]], dtype=torch.int32)
    metadata.kv_seq_lens = torch.arange(1, 5, dtype=torch.int32)
    metadata.kv_seq_lens_host_values = [1, 2, 3, 4]
    metadata.paged_kv_indptr = torch.arange(5, dtype=torch.int32)
    metadata.paged_kv_last_page_len = torch.arange(1, 5, dtype=torch.int32)
    metadata.expanded_decode_metadata = SimpleNamespace(
        enabled=True,
        kv_seq_lens=metadata.kv_seq_lens,
        block_table=metadata.block_table,
        paged_kv_indptr=metadata.paged_kv_indptr,
        paged_kv_indices=metadata.paged_kv_indices,
        paged_kv_last_page_len=metadata.paged_kv_last_page_len,
        paged_attention_tiling_data=None,
        kv_seq_lens_host=None,
        kv_seq_lens_host_values=metadata.kv_seq_lens_host_values,
    )
    entry = runner._allocate_entry(
        padded_batch_size=4,
        input_ids=input_ids,
        positions=positions,
        metadata=metadata,
    )

    with patch(
        "xllm.python.model_executor.runners.decode_acl_graph.kernels.update_decode_graph_metadata",
        create=True,
    ):
        runner._fill_entry(
            entry,
            input_ids,
            positions,
            metadata,
            batch_size=4,
            input_embedding=None,
        )

    assert entry.static_metadata.linear_state_indices.tolist() == [3, 3, 7, 7]


def _dp_metadata(
    token_counts: tuple[int, int],
    dp_is_decode: tuple[int, int] = (1, 1),
) -> SimpleNamespace:
    return SimpleNamespace(
        is_prefill=False,
        is_chunked_prefill=False,
        dp_execution_token_counts=tuple(1 if count == 0 else count for count in token_counts),
        dp_is_decode=dp_is_decode,
    )


def test_dp_empty_rank_uses_group_wide_acl_graph_bucket() -> None:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        attention_backend,
        torch.device("cpu"),
        max_batch=16,
        max_model_len=8,
        dp_size=2,
        dp_rank=1,
    )

    with patch.object(
        runner,
        "_has_compatible_decode_metadata",
        return_value=True,
    ):
        assert runner.can_execute(
            torch.zeros(1, dtype=torch.int32),
            _dp_metadata((5, 0)),
        )


def test_dp_mixed_step_does_not_enter_acl_decode_graph() -> None:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        attention_backend,
        torch.device("cpu"),
        max_batch=16,
        max_model_len=8,
        dp_size=2,
        dp_rank=0,
    )

    with patch.object(
        runner,
        "_has_compatible_decode_metadata",
        return_value=True,
    ):
        assert not runner.can_execute(
            torch.zeros(3, dtype=torch.int32),
            _dp_metadata((3, 2), dp_is_decode=(0, 1)),
        )


def test_dp_acl_graph_requires_group_wide_token_counts() -> None:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        attention_backend,
        torch.device("cpu"),
        max_batch=16,
        max_model_len=8,
        dp_size=2,
        dp_rank=0,
    )
    metadata = _dp_metadata((3, 2))
    metadata.dp_execution_token_counts = (3,)

    with (
        patch.object(
            runner,
            "_has_compatible_decode_metadata",
            return_value=True,
        ),
        pytest.raises(RuntimeError, match="valid dp_execution_token_counts"),
    ):
        runner.can_execute(torch.zeros(3, dtype=torch.int32), metadata)
