# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for MiMo BF16 vectorized-5D PA decode."""

from __future__ import annotations

import math

import pytest
import torch

from flydsl.runtime.device import get_rocm_arch
from kernels.attention.pa_decode_tile import (
    BF16_KV_SUPPORTED_ARCHS,
    pa_decode_tile,
)

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

if not torch.cuda.is_available():
    pytest.skip("requires a ROCm GPU", allow_module_level=True)

ARCH = str(get_rocm_arch()).split(":", 1)[0]


def _torch_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_tables: torch.Tensor,
    context_length: int,
    query_length: int,
) -> torch.Tensor:
    outputs = []
    head_dim = query.shape[-1]
    value_head_dim = value.shape[3]
    num_q_heads = query.shape[1]
    token_positions = torch.arange(context_length, device=query.device)
    causal_limit = context_length - query_length + torch.arange(query_length, device=query.device)
    causal_mask = token_positions[None, :] <= causal_limit[:, None]
    for sequence in range(block_tables.shape[0]):
        physical_pages = block_tables[sequence].long()
        key_seq = (
            key[physical_pages]
            .permute(0, 3, 1, 2, 4)
            .reshape(-1, key.shape[1], head_dim)[:context_length]
            .float()
            .expand(-1, num_q_heads, -1)
        )
        value_seq = (
            value[physical_pages]
            .permute(0, 2, 4, 1, 3)
            .reshape(-1, value.shape[1], value_head_dim)[:context_length]
            .float()
            .expand(-1, num_q_heads, -1)
        )
        query_seq = query[sequence * query_length : (sequence + 1) * query_length].float()
        logits = torch.einsum("qhd,khd->hqk", query_seq, key_seq) * (head_dim**-0.5)
        logits.masked_fill_(~causal_mask[None, :, :], float("-inf"))
        probabilities = torch.softmax(logits, dim=-1)
        outputs.append(torch.einsum("hqk,khd->qhd", probabilities, value_seq).to(torch.bfloat16))
    return torch.cat(outputs)


@pytest.mark.skipif(
    ARCH not in BF16_KV_SUPPORTED_ARCHS,
    reason=f"the MiMo BF16 vectorized-5D specialization requires gfx942/gfx950, got {ARCH}",
)
@pytest.mark.parametrize("query_length", [1, 4])
@pytest.mark.parametrize("value_head_dim", [128, 192])
def test_mimo_bf16_vectorized_5d_matches_torch(
    query_length: int, value_head_dim: int
) -> None:
    batch = 2
    num_q_heads = 16
    num_kv_heads = 1
    head_dim = 192
    page_size = 64
    context_length = 1027
    blocks_per_sequence = math.ceil(context_length / page_size)
    num_blocks = batch * blocks_per_sequence
    generator = torch.Generator(device="cuda").manual_seed(20260815)

    query = torch.empty(
        (batch * query_length, num_q_heads, head_dim),
        dtype=torch.bfloat16,
        device="cuda",
    ).uniform_(-1.0, 1.0, generator=generator)
    key = torch.empty(
        (num_blocks, num_kv_heads, head_dim // 8, page_size, 8),
        dtype=torch.bfloat16,
        device="cuda",
    ).uniform_(-1.0, 1.0, generator=generator)
    value = torch.empty(
        (num_blocks, num_kv_heads, page_size // 8, value_head_dim, 8),
        dtype=torch.bfloat16,
        device="cuda",
    ).uniform_(-1.0, 1.0, generator=generator)
    block_tables = torch.arange(num_blocks - 1, -1, -1, dtype=torch.int32, device="cuda").reshape(
        batch, blocks_per_sequence
    )
    context_lengths = torch.full((batch,), context_length, dtype=torch.int32, device="cuda")

    expected = _torch_reference(
        query,
        key,
        value,
        block_tables,
        context_length,
        query_length,
    )

    equivalent_group = query_length * num_q_heads // num_kv_heads
    flydsl_partitions = 4
    flydsl_shape = (batch, num_kv_heads, flydsl_partitions, equivalent_group)
    actual = torch.full(
        (batch * query_length, num_q_heads, value_head_dim),
        float("nan"),
        dtype=query.dtype,
        device=query.device,
    )
    pa_decode_tile(
        output=actual,
        query=query,
        key_cache=key,
        value_cache=value,
        block_tables=block_tables,
        context_lengths=context_lengths,
        key_scale=None,
        value_scale=None,
        softmax_scale=head_dim**-0.5,
        num_partitions=flydsl_partitions,
        pmax=torch.empty(flydsl_shape, dtype=torch.float32, device="cuda"),
        psum=torch.empty(flydsl_shape, dtype=torch.float32, device="cuda"),
        pout=torch.empty(
            (*flydsl_shape, value_head_dim),
            dtype=torch.bfloat16,
            device="cuda",
        ),
    )
    torch.cuda.synchronize()

    assert bool(torch.isfinite(actual).all().item())
    torch.testing.assert_close(actual, expected, rtol=5.0e-3, atol=5.0e-3)


@pytest.mark.large_shape
def test_mimo_bf16_vectorized_5d_supports_64_bit_page_offsets() -> None:
    num_q_heads = 16
    num_kv_heads = 1
    head_dim = 192
    page_size = 64
    page_bytes = num_kv_heads * head_dim * page_size * torch.bfloat16.itemsize
    high_page = math.ceil(2**31 / page_bytes)
    num_blocks = high_page + 1
    generator = torch.Generator(device="cuda").manual_seed(20260815)

    query = torch.empty((1, num_q_heads, head_dim), dtype=torch.bfloat16, device="cuda").uniform_(
        -1.0, 1.0, generator=generator
    )
    key = torch.empty(
        (num_blocks, num_kv_heads, head_dim // 8, page_size, 8), dtype=torch.bfloat16, device="cuda"
    )
    value = torch.empty(
        (num_blocks, num_kv_heads, page_size // 8, head_dim, 8), dtype=torch.bfloat16, device="cuda"
    )
    key[high_page].uniform_(-1.0, 1.0, generator=generator)
    value[high_page].uniform_(-1.0, 1.0, generator=generator)
    block_tables = torch.tensor([[high_page]], dtype=torch.int32, device="cuda")
    context_lengths = torch.ones(1, dtype=torch.int32, device="cuda")

    expected = _torch_reference(query, key, value, block_tables, context_length=1, query_length=1)
    actual = torch.full_like(query, float("nan"))
    pa_decode_tile(
        output=actual,
        query=query,
        key_cache=key,
        value_cache=value,
        block_tables=block_tables,
        context_lengths=context_lengths,
        key_scale=None,
        value_scale=None,
        softmax_scale=head_dim**-0.5,
        num_partitions=1,
    )
    torch.cuda.synchronize()

    assert bool(torch.isfinite(actual).all().item())
    torch.testing.assert_close(actual, expected, rtol=5.0e-3, atol=5.0e-3)
