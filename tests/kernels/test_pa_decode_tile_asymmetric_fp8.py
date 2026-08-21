# SPDX-License-Identifier: Apache-2.0

"""MiMo QK192/V128 coverage for FP8 vectorized-5D target verification."""

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
FP8_DTYPE = torch.float8_e4m3fn if "gfx95" in ARCH else torch.float8_e4m3fnuz
FP8_MAX = float(torch.finfo(FP8_DTYPE).max)


def _quantize_per_tensor(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = tensor.float().abs().max() / FP8_MAX
    scale = scale.clamp_min(torch.finfo(torch.float32).tiny)
    return (tensor.float() / scale).to(FP8_DTYPE), scale.reshape(1)


@pytest.mark.skipif(
    ARCH not in BF16_KV_SUPPORTED_ARCHS,
    reason=f"the MiMo vectorized-5D specialization requires gfx942/gfx950, got {ARCH}",
)
def test_mimo_fp8_vectorized_5d_qk192_v128_matches_torch() -> None:
    batch = 2
    query_length = 4
    num_q_heads = 16
    num_kv_heads = 1
    qk_head_dim = 192
    value_head_dim = 128
    page_size = 64
    context_length = 1027
    blocks_per_sequence = math.ceil(context_length / page_size)
    num_blocks = batch * blocks_per_sequence
    generator = torch.Generator(device="cuda").manual_seed(20260817)

    query = torch.empty(
        (batch * query_length, num_q_heads, qk_head_dim),
        dtype=torch.bfloat16,
        device="cuda",
    ).uniform_(-0.5, 0.5, generator=generator)
    key_source = torch.empty(
        (num_blocks, num_kv_heads, qk_head_dim // 16, page_size, 16),
        dtype=torch.bfloat16,
        device="cuda",
    ).uniform_(-0.5, 0.5, generator=generator)
    value_source = torch.empty(
        (num_blocks, num_kv_heads, page_size // 16, value_head_dim, 16),
        dtype=torch.bfloat16,
        device="cuda",
    ).uniform_(-0.5, 0.5, generator=generator)
    key, key_scale = _quantize_per_tensor(key_source)
    value, value_scale = _quantize_per_tensor(value_source)

    block_tables = torch.arange(
        num_blocks - 1, -1, -1, dtype=torch.int32, device="cuda"
    ).reshape(batch, blocks_per_sequence)
    context_lengths = torch.full(
        (batch,), context_length, dtype=torch.int32, device="cuda"
    )

    token_positions = torch.arange(context_length, device="cuda")
    causal_limit = context_length - query_length + torch.arange(
        query_length, device="cuda"
    )
    causal_mask = token_positions[None, :] <= causal_limit[:, None]
    expected = []
    for sequence in range(batch):
        physical_pages = block_tables[sequence].long()
        key_seq = (
            key[physical_pages]
            .permute(0, 3, 1, 2, 4)
            .reshape(-1, num_kv_heads, qk_head_dim)[:context_length]
            .float()
            * key_scale
        ).expand(-1, num_q_heads, -1)
        value_seq = (
            value[physical_pages]
            .permute(0, 2, 4, 1, 3)
            .reshape(-1, num_kv_heads, value_head_dim)[:context_length]
            .float()
            * value_scale
        ).expand(-1, num_q_heads, -1)
        query_seq = query[
            sequence * query_length : (sequence + 1) * query_length
        ].float()
        logits = torch.einsum("qhd,khd->hqk", query_seq, key_seq) * (
            qk_head_dim**-0.5
        )
        logits.masked_fill_(~causal_mask[None, :, :], float("-inf"))
        probabilities = torch.softmax(logits, dim=-1)
        expected.append(
            torch.einsum("hqk,khd->qhd", probabilities, value_seq).to(
                torch.bfloat16
            )
        )
    expected = torch.cat(expected)

    partitions = 4
    equivalent_group = query_length * num_q_heads // num_kv_heads
    partial_shape = (batch, num_kv_heads, partitions, equivalent_group)
    actual = torch.full(
        (batch * query_length, num_q_heads, value_head_dim),
        float("nan"),
        dtype=torch.bfloat16,
        device="cuda",
    )
    pa_decode_tile(
        output=actual,
        query=query,
        key_cache=key,
        value_cache=value,
        block_tables=block_tables,
        context_lengths=context_lengths,
        key_scale=key_scale,
        value_scale=value_scale,
        softmax_scale=qk_head_dim**-0.5,
        num_partitions=partitions,
        pmax=torch.empty(partial_shape, dtype=torch.float32, device="cuda"),
        psum=torch.empty(partial_shape, dtype=torch.float32, device="cuda"),
        pout=torch.empty(
            (*partial_shape, value_head_dim),
            dtype=torch.bfloat16,
            device="cuda",
        ),
    )
    torch.cuda.synchronize()

    assert bool(torch.isfinite(actual).all().item())
    torch.testing.assert_close(actual, expected, rtol=2.0e-2, atol=2.0e-2)
