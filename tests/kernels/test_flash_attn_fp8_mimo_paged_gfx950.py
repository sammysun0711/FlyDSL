# SPDX-License-Identifier: Apache-2.0

"""MiMo cached-prefill coverage for the gfx950 paged FP8 specialization."""

from __future__ import annotations

import math

import pytest
import torch

from kernels.attention.flash_attn_fp8_mimo_paged_gfx950 import (
    mimo_paged_flash_attn_fp8,
)

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

if not torch.cuda.is_available():
    pytest.skip("requires a ROCm GPU", allow_module_level=True)

ARCH = torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0]
FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = float(torch.finfo(FP8_DTYPE).max)


def _quantize_per_tensor(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = tensor.abs().max() / FP8_MAX
    scale = scale.clamp_min(torch.finfo(torch.float32).tiny)
    return (tensor / scale).to(FP8_DTYPE), scale.reshape(1)


@pytest.mark.skipif(ARCH != "gfx950", reason=f"requires gfx950, got {ARCH}")
@pytest.mark.parametrize("value_head_dim", [128, 192])
def test_mimo_paged_fp8_cached_prefill_native_value_head(
    value_head_dim: int,
) -> None:
    torch.manual_seed(17)
    query_lengths = [64, 32]
    kv_lengths = [128, 96]
    q_indptr = torch.tensor([0, 64, 96], device="cuda", dtype=torch.int32)
    kv_indptr = torch.tensor([0, 128, 224], device="cuda", dtype=torch.int32)
    page_indptr = torch.tensor([0, 2, 4], device="cuda", dtype=torch.int32)
    page_indices = torch.tensor([2, 0, 3, 1], device="cuda", dtype=torch.int32)
    num_pages = 4

    query_source = (
        torch.randn(sum(query_lengths), 16, 192, device="cuda") * 0.2
    )
    key_source = (
        torch.randn(num_pages, 1, 12, 64, 16, device="cuda") * 0.2
    )
    value_source = (
        torch.randn(
            num_pages,
            1,
            4,
            value_head_dim,
            16,
            device="cuda",
        )
        * 0.2
    )
    query, query_descale = _quantize_per_tensor(query_source)
    key, key_descale = _quantize_per_tensor(key_source)
    value, value_descale = _quantize_per_tensor(value_source)

    actual = mimo_paged_flash_attn_fp8(
        query,
        key,
        value,
        q_indptr,
        kv_indptr,
        page_indptr,
        page_indices,
        max_seqlen_q=max(query_lengths),
        max_seqlen_kv=max(kv_lengths),
        q_descale=query_descale,
        k_descale=key_descale,
        v_descale=value_descale,
        value_head_dim=value_head_dim,
    )

    expected = []
    for batch_idx, (query_length, kv_length) in enumerate(
        zip(query_lengths, kv_lengths)
    ):
        query_batch = (
            query[q_indptr[batch_idx] : q_indptr[batch_idx + 1]].float()
            * query_descale
        )
        physical_pages = page_indices[
            page_indptr[batch_idx] : page_indptr[batch_idx + 1]
        ].long()
        key_batch = (
            key[physical_pages]
            .permute(0, 3, 1, 2, 4)
            .contiguous()
            .view(-1, 1, 192)[:kv_length]
            .float()
            * key_descale
        ).expand(-1, 16, -1)
        value_batch = (
            value[physical_pages]
            .permute(0, 2, 4, 1, 3)
            .contiguous()
            .view(-1, 1, value_head_dim)[:kv_length]
            .float()
            * value_descale
        ).expand(-1, 16, -1)

        logits = torch.einsum(
            "qhd,khd->hqk", query_batch, key_batch
        ) / math.sqrt(192)
        query_positions = torch.arange(query_length, device="cuda")
        key_positions = torch.arange(kv_length, device="cuda")
        causal_mask = key_positions[None, :] <= (
            kv_length - query_length + query_positions
        )[:, None]
        logits.masked_fill_(~causal_mask[None, :, :], float("-inf"))
        probabilities = torch.softmax(logits, dim=-1)
        expected.append(
            torch.einsum("hqk,khd->qhd", probabilities, value_batch).to(
                torch.bfloat16
            )
        )
    expected = torch.cat(expected)

    assert actual.shape == (sum(query_lengths), 16, value_head_dim)
    assert bool(torch.isfinite(actual).all().item())
    torch.testing.assert_close(actual, expected, rtol=2.0e-2, atol=2.0e-2)
