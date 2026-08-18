# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""MiMo-V2.5-Pro D192 page-64 FP8 prefill attention for gfx950.

This is the narrow production-facing wrapper around the D192 specialization
of :mod:`flash_attn_fp8_gfx950`.  Q is packed request-major OCP E4M3, K/V use
SGLang/AITER's SHUFFLE 5D page layout, and O is BF16:

* Q: ``[total_q, 16, 192]``
* K: ``[num_pages, 1, 12, 64, 16]``
* V: ``[num_pages, 1, 4, 128, 16]``
* O: ``[total_q, 16, 128]``

``q_indptr`` and ``kv_indptr`` contain cumulative query/KV token lengths;
``page_indptr`` and ``page_indices`` contain cumulative page counts and the
flat physical-page list.  The kernel applies bottom-right causal masking, so a
cached suffix of length Q attends the corresponding prefix+suffix KV stream.
"""

from __future__ import annotations

import functools

import torch

from kernels.attention.flash_attn_fp8_gfx950 import (
    build_flash_attn_dualwave_swp_fp8_module,
)

MIMO_Q_HEADS = 16
MIMO_KV_HEADS = 1
MIMO_HEAD_DIM = 192
MIMO_VALUE_HEAD_DIM = 128
MIMO_PAGE_SIZE = 64
MIMO_FP8 = torch.float8_e4m3fn


@functools.lru_cache(maxsize=8)
def compile_mimo_paged_flash_attn_fp8(
    *,
    waves_per_eu: int = 1,
    setprio: bool = True,
    enable_stagger: bool = True,
    lazy_rescale: bool = True,
    value_head_dim: int = MIMO_VALUE_HEAD_DIM,
):
    """Compile/cache the exact MiMo TP8 paged-prefill specialization."""

    return build_flash_attn_dualwave_swp_fp8_module(
        num_heads=MIMO_Q_HEADS,
        num_kv_heads=MIMO_KV_HEADS,
        head_dim=MIMO_HEAD_DIM,
        value_head_dim=value_head_dim,
        causal=True,
        dtype_str="fp8",
        waves_per_eu=waves_per_eu,
        daz=True,
        dualwave_swp_lazy_rescale=lazy_rescale,
        dualwave_swp_setprio=setprio,
        dualwave_swp_enable_stagger=enable_stagger,
        num_kv_splits=1,
        varlen=True,
        cross_seqlen=True,
        paged=True,
        kv_cache_layout="vectorized",
    )


def _require_i32_indptr(name: str, value: torch.Tensor, expected: int | None = None) -> torch.Tensor:
    if value.dtype != torch.int32 or value.ndim != 1:
        raise ValueError(f"{name} must be contiguous int32 1D, got shape={tuple(value.shape)} dtype={value.dtype}")
    if expected is not None and value.numel() != expected:
        raise ValueError(f"{name} must have {expected} entries, got {value.numel()}")
    return value.contiguous()


def _require_descale(name: str, value: torch.Tensor, device: torch.device) -> torch.Tensor:
    if value.numel() != 1 or value.dtype != torch.float32 or value.device != device:
        raise ValueError(
            f"{name} must be one float32 value on {device}, got shape={tuple(value.shape)} "
            f"dtype={value.dtype} device={value.device}"
        )
    return value.contiguous().view(1)


def mimo_paged_flash_attn_fp8(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    q_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    page_indptr: torch.Tensor,
    page_indices: torch.Tensor,
    *,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    q_descale: torch.Tensor,
    k_descale: torch.Tensor,
    v_descale: torch.Tensor,
    out: torch.Tensor | None = None,
    waves_per_eu: int = 1,
    setprio: bool = True,
    enable_stagger: bool = True,
    lazy_rescale: bool = True,
    value_head_dim: int = MIMO_VALUE_HEAD_DIM,
    stream=None,
) -> torch.Tensor:
    """Run exact MiMo D192 paged causal attention on gfx950."""

    if q.device.type != "cuda":
        raise ValueError(f"q must be on a ROCm device, got {q.device}")
    arch = torch.cuda.get_device_properties(q.device).gcnArchName.split(":", 1)[0]
    if arch != "gfx950":
        raise RuntimeError(f"MiMo paged FlyDSL attention requires gfx950, got {arch!r}")
    if q.dtype != MIMO_FP8 or q.ndim != 3 or tuple(q.shape[1:]) != (MIMO_Q_HEADS, MIMO_HEAD_DIM):
        raise ValueError(
            f"q must be OCP E4M3 [total_q,{MIMO_Q_HEADS},{MIMO_HEAD_DIM}], "
            f"got shape={tuple(q.shape)} dtype={q.dtype}"
        )
    if value_head_dim not in (MIMO_VALUE_HEAD_DIM, MIMO_HEAD_DIM):
        raise ValueError(f"value_head_dim must be 128 or 192, got {value_head_dim}")
    expected_k_tail = (MIMO_KV_HEADS, MIMO_HEAD_DIM // 16, MIMO_PAGE_SIZE, 16)
    expected_v_tail = (
        MIMO_KV_HEADS,
        MIMO_PAGE_SIZE // 16,
        value_head_dim,
        16,
    )
    if k_cache.ndim != 5 or tuple(k_cache.shape[1:]) != expected_k_tail:
        raise ValueError(f"K cache must have tail {expected_k_tail}, got {tuple(k_cache.shape)}")
    if v_cache.ndim != 5 or tuple(v_cache.shape[1:]) != expected_v_tail:
        raise ValueError(f"V cache must have tail {expected_v_tail}, got {tuple(v_cache.shape)}")
    if k_cache.shape[0] != v_cache.shape[0]:
        raise ValueError(f"K/V physical page counts differ: {k_cache.shape[0]} vs {v_cache.shape[0]}")
    if k_cache.dtype == torch.uint8:
        k_cache = k_cache.view(MIMO_FP8)
    if v_cache.dtype == torch.uint8:
        v_cache = v_cache.view(MIMO_FP8)
    if k_cache.dtype != MIMO_FP8 or v_cache.dtype != MIMO_FP8:
        raise ValueError(f"K/V caches must be OCP E4M3 or byte views, got {k_cache.dtype}/{v_cache.dtype}")
    if not q.is_contiguous() or not k_cache.is_contiguous() or not v_cache.is_contiguous():
        raise ValueError("MiMo paged FlyDSL attention requires contiguous Q and physical K/V buffers")

    batch_size = q_indptr.numel() - 1
    if batch_size < 1:
        raise ValueError("q_indptr must describe at least one request")
    q_indptr = _require_i32_indptr("q_indptr", q_indptr, batch_size + 1)
    kv_indptr = _require_i32_indptr("kv_indptr", kv_indptr, batch_size + 1)
    page_indptr = _require_i32_indptr("page_indptr", page_indptr, batch_size + 1)
    page_indices = _require_i32_indptr("page_indices", page_indices)
    for name, tensor in (
        ("q_indptr", q_indptr),
        ("kv_indptr", kv_indptr),
        ("page_indptr", page_indptr),
        ("page_indices", page_indices),
    ):
        if tensor.device != q.device:
            raise ValueError(f"{name} must be on {q.device}, got {tensor.device}")

    q_descale = _require_descale("q_descale", q_descale, q.device)
    k_descale = _require_descale("k_descale", k_descale, q.device)
    v_descale = _require_descale("v_descale", v_descale, q.device)
    expected_out_shape = (q.shape[0], MIMO_Q_HEADS, value_head_dim)
    if out is None:
        out = torch.empty(
            expected_out_shape, dtype=torch.bfloat16, device=q.device
        )
    if (
        out.shape != expected_out_shape
        or out.dtype != torch.bfloat16
        or not out.is_contiguous()
    ):
        raise ValueError(
            f"out must be contiguous BF16 with shape {expected_out_shape}, "
            f"got shape={tuple(out.shape)} dtype={out.dtype}"
        )

    exe = compile_mimo_paged_flash_attn_fp8(
        waves_per_eu=waves_per_eu,
        setprio=setprio,
        enable_stagger=enable_stagger,
        lazy_rescale=lazy_rescale,
        value_head_dim=value_head_dim,
    )
    # FlyDSL's low-level executor interprets ``None`` as the default HIP
    # stream.  Serving may invoke attention on an overlap/speculative stream,
    # so bind to PyTorch's current stream exactly like flash_attn_interface.
    launch_stream = torch.cuda.current_stream(q.device) if stream is None else stream
    exe(
        q.view(-1),
        # Keep the physical caches rank-5 at the Python/JIT ABI boundary.  The
        # paged kernel only consumes their base pointers and constructs a
        # one-page buffer descriptor with explicit 64-bit address arithmetic.
        # Flattening a serving-sized MiMo pool makes the dynamic 1-D shape
        # exceed signed int32 (FlyDSL encodes dynamic memref shapes as i32),
        # even though every physical page and page id is representable.
        k_cache,
        v_cache,
        out.view(-1),
        batch_size,
        int(max_seqlen_q),
        seq_len_kv=int(max_seqlen_kv),
        stride_q_n=MIMO_Q_HEADS * MIMO_HEAD_DIM,
        stride_o_n=MIMO_Q_HEADS * value_head_dim,
        stride_kv_n=MIMO_KV_HEADS * MIMO_HEAD_DIM,
        head_dim_runtime=MIMO_HEAD_DIM,
        cu_seqlens_q=q_indptr,
        cu_seqlens_kv=kv_indptr,
        page_indptr=page_indptr,
        page_indices=page_indices,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        stream=launch_stream,
    )
    return out


__all__ = [
    "compile_mimo_paged_flash_attn_fp8",
    "mimo_paged_flash_attn_fp8",
]
