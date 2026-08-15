#!/usr/bin/env python3
"""Compare Gluon/FlyDSL BF16 and FP8 MiMo paged-attention decode."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

import torch

import aiter
from aiter.ops.attention import pa_decode_gluon

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kernels.attention.pa_decode_tile import pa_decode_tile  # noqa: E402


QUERY_LENGTH = 4
NUM_Q_HEADS = 16
NUM_KV_HEADS = 1
HEAD_DIM = 192
PAGE_SIZE = 64
CONTEXT_PARTITION_SIZE = 256
FP8_VECTOR_WIDTH = 16
BF16_VECTOR_WIDTH = 8
KV_SCALE = 0.01

PATHS = ("gluon_bf16", "gluon_fp8", "flydsl_bf16", "flydsl_fp8")


def parse_int_list(value: str) -> list[int]:
    values = [int(item) for item in value.split(",") if item]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError(
            "expected a comma-separated list of positive integers"
        )
    return values


def make_workspace(batch: int, partitions: int) -> dict[str, torch.Tensor]:
    scalar_shape = (
        batch,
        NUM_KV_HEADS,
        partitions,
        QUERY_LENGTH * NUM_Q_HEADS // NUM_KV_HEADS,
    )
    return {
        "max": torch.empty(scalar_shape, dtype=torch.float32, device="cuda"),
        "sum": torch.empty(scalar_shape, dtype=torch.float32, device="cuda"),
        "out": torch.empty(
            (*scalar_shape, HEAD_DIM), dtype=torch.bfloat16, device="cuda"
        ),
    }


def dequantize_key_cache(key_fp8: torch.Tensor) -> torch.Tensor:
    blocks = key_fp8.shape[0]
    key_bf16 = key_fp8.to(torch.bfloat16).mul_(KV_SCALE)
    return (
        key_bf16.view(
            blocks,
            NUM_KV_HEADS,
            HEAD_DIM // FP8_VECTOR_WIDTH,
            PAGE_SIZE,
            2,
            BF16_VECTOR_WIDTH,
        )
        .permute(0, 1, 2, 4, 3, 5)
        .contiguous()
        .view(
            blocks,
            NUM_KV_HEADS,
            HEAD_DIM // BF16_VECTOR_WIDTH,
            PAGE_SIZE,
            BF16_VECTOR_WIDTH,
        )
    )


def dequantize_value_cache(value_fp8: torch.Tensor) -> torch.Tensor:
    blocks = value_fp8.shape[0]
    value_bf16 = value_fp8.to(torch.bfloat16).mul_(KV_SCALE)
    return (
        value_bf16.view(
            blocks,
            NUM_KV_HEADS,
            PAGE_SIZE // FP8_VECTOR_WIDTH,
            HEAD_DIM,
            2,
            BF16_VECTOR_WIDTH,
        )
        .permute(0, 1, 2, 4, 3, 5)
        .contiguous()
        .view(
            blocks,
            NUM_KV_HEADS,
            PAGE_SIZE // BF16_VECTOR_WIDTH,
            HEAD_DIM,
            BF16_VECTOR_WIDTH,
        )
    )


def make_problem(context_length: int, batch: int, seed: int) -> dict:
    blocks_per_sequence = math.ceil(context_length / PAGE_SIZE)
    num_blocks = batch * blocks_per_sequence
    generator = torch.Generator(device="cuda").manual_seed(seed)

    query_master = torch.empty(
        (batch * QUERY_LENGTH, NUM_Q_HEADS, HEAD_DIM),
        dtype=torch.bfloat16,
        device="cuda",
    ).uniform_(-0.25, 0.25, generator=generator)
    fp8_max = torch.finfo(aiter.dtypes.fp8).max
    query_scale = (
        query_master.float().abs().amax(dim=-1, keepdim=True) / fp8_max
    ).clamp_min_(1.0e-20)
    query_fp8 = (query_master.float() / query_scale).to(aiter.dtypes.fp8)
    # Both precision families consume the same logical quantized query.
    query_bf16 = (query_fp8.float() * query_scale).to(torch.bfloat16)

    key_fp8 = torch.empty(
        (
            num_blocks,
            NUM_KV_HEADS,
            HEAD_DIM // FP8_VECTOR_WIDTH,
            PAGE_SIZE,
            FP8_VECTOR_WIDTH,
        ),
        dtype=aiter.dtypes.fp8,
        device="cuda",
    )
    value_fp8 = torch.empty(
        (
            num_blocks,
            NUM_KV_HEADS,
            PAGE_SIZE // FP8_VECTOR_WIDTH,
            HEAD_DIM,
            FP8_VECTOR_WIDTH,
        ),
        dtype=aiter.dtypes.fp8,
        device="cuda",
    )
    # OCP E4M3 encodings [0, 119] are finite. Dequantizing these exact bytes
    # creates logically identical BF16 caches for the cross-precision check.
    key_fp8.view(torch.uint8).random_(0, 120, generator=generator)
    value_fp8.view(torch.uint8).random_(0, 120, generator=generator)
    key_bf16 = dequantize_key_cache(key_fp8)
    value_bf16 = dequantize_value_cache(value_fp8)

    block_tables = torch.arange(
        num_blocks - 1, -1, -1, dtype=torch.int32, device="cuda"
    ).reshape(batch, blocks_per_sequence)
    context_lengths = torch.full(
        (batch,), context_length, dtype=torch.int32, device="cuda"
    )
    kv_scale = torch.tensor([KV_SCALE], dtype=torch.float32, device="cuda")
    return {
        "query_bf16": query_bf16,
        "query_fp8": query_fp8,
        "query_scale": query_scale,
        "key_bf16": key_bf16,
        "value_bf16": value_bf16,
        "key_fp8": key_fp8,
        "value_fp8": value_fp8,
        "kv_scale": kv_scale,
        "block_tables": block_tables,
        "context_lengths": context_lengths,
        "num_blocks": num_blocks,
    }


def launch_gluon(
    problem: dict,
    output: torch.Tensor,
    partitions: int,
    workspace: dict[str, torch.Tensor],
    *,
    fp8: bool,
) -> None:
    pa_decode_gluon(
        output=output,
        query=problem["query_fp8" if fp8 else "query_bf16"],
        key_cache=problem["key_fp8" if fp8 else "key_bf16"],
        value_cache=problem["value_fp8" if fp8 else "value_bf16"],
        context_lengths=problem["context_lengths"],
        block_tables=problem["block_tables"],
        softmax_scale=HEAD_DIM**-0.5,
        query_length=QUERY_LENGTH,
        max_context_partition_num=partitions,
        context_partition_size=CONTEXT_PARTITION_SIZE,
        compute_type=aiter.dtypes.fp8 if fp8 else torch.bfloat16,
        query_scale=problem["query_scale"] if fp8 else None,
        key_scale=problem["kv_scale"] if fp8 else None,
        value_scale=problem["kv_scale"] if fp8 else None,
        exp_sums=workspace["sum"],
        max_logits=workspace["max"],
        temporary_output=workspace["out"],
        ps=True,
    )


def launch_flydsl(
    problem: dict,
    output: torch.Tensor,
    partitions: int,
    workspace: dict[str, torch.Tensor],
    *,
    fp8: bool,
) -> None:
    pa_decode_tile(
        output=output,
        query=problem["query_bf16"],
        key_cache=problem["key_fp8" if fp8 else "key_bf16"],
        value_cache=problem["value_fp8" if fp8 else "value_bf16"],
        block_tables=problem["block_tables"],
        context_lengths=problem["context_lengths"],
        key_scale=problem["kv_scale"] if fp8 else None,
        value_scale=problem["kv_scale"] if fp8 else None,
        softmax_scale=HEAD_DIM**-0.5,
        num_partitions=partitions,
        pmax=workspace["max"],
        psum=workspace["sum"],
        pout=workspace["out"],
    )


def launch_path(
    path: str,
    problem: dict,
    output: torch.Tensor,
    partitions: int,
    workspace: dict[str, torch.Tensor],
) -> None:
    if path == "gluon_bf16":
        launch_gluon(problem, output, partitions, workspace, fp8=False)
    elif path == "gluon_fp8":
        launch_gluon(problem, output, partitions, workspace, fp8=True)
    elif path == "flydsl_bf16":
        launch_flydsl(problem, output, partitions, workspace, fp8=False)
    elif path == "flydsl_fp8":
        launch_flydsl(problem, output, partitions, workspace, fp8=True)
    else:
        raise ValueError(f"unknown path: {path}")


def event_samples(
    function: Callable[[], None], warmup: int, repeats: int
) -> list[float]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return samples


def summarize(samples: list[float]) -> dict:
    return {
        "samples": samples,
        "median": statistics.median(samples),
        "mean": statistics.fmean(samples),
        "min": min(samples),
        "max": max(samples),
    }


def compare(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    difference = (actual.float() - expected.float()).abs()
    return {
        "max_abs": float(difference.max().item()),
        "mean_abs": float(difference.mean().item()),
        "fraction_abs_gt_5e-3": float(
            (difference > 5.0e-3).float().mean().item()
        ),
    }


def run_case(
    context_length: int,
    batch: int,
    partitions: list[int],
    warmup: int,
    repeats: int,
    seed: int,
) -> dict:
    torch.cuda.empty_cache()
    free_before, total_memory = torch.cuda.mem_get_info()
    problem = make_problem(context_length, batch, seed)
    torch.cuda.synchronize()

    states = {path: [] for path in PATHS}
    for partition_count in partitions:
        for path in PATHS:
            workspace = make_workspace(batch, partition_count)
            output = torch.full_like(problem["query_bf16"], float("nan"))
            launch_path(path, problem, output, partition_count, workspace)
            torch.cuda.synchronize()
            if not bool(torch.isfinite(output).all().item()):
                raise RuntimeError(
                    f"context {context_length}, {path}, partitions "
                    f"{partition_count}: output is not finite"
                )
            states[path].append((partition_count, output, workspace))

    reference = states["gluon_bf16"][0][1]
    results = {}
    for path in PATHS:
        path_results = []
        for partition_count, output, workspace in states[path]:
            accuracy = compare(output, reference)
            torch.testing.assert_close(output, reference, rtol=5.0e-3, atol=5.0e-3)
            timing = summarize(
                event_samples(
                    lambda path=path, output=output, partition_count=partition_count, workspace=workspace: launch_path(
                        path, problem, output, partition_count, workspace
                    ),
                    warmup,
                    repeats,
                )
            )
            path_results.append(
                {
                    "partitions": partition_count,
                    "accuracy_vs_gluon_bf16_reference": accuracy,
                    "timing_ms": timing,
                }
            )
        results[path] = path_results

    best = {
        path: min(results[path], key=lambda item: item["timing_ms"]["median"])
        for path in PATHS
    }
    gluon_bf16_ms = best["gluon_bf16"]["timing_ms"]["median"]
    for path in PATHS:
        best[path]["speedup_over_gluon_bf16"] = (
            gluon_bf16_ms / best[path]["timing_ms"]["median"]
        )

    max_byte_offset_fp8 = problem["num_blocks"] * PAGE_SIZE * HEAD_DIM - 1
    max_byte_offset_bf16 = 2 * (max_byte_offset_fp8 + 1) - 1
    result = {
        "context_length": context_length,
        "batch": batch,
        "query_length": QUERY_LENGTH,
        "num_q_heads": NUM_Q_HEADS,
        "num_kv_heads": NUM_KV_HEADS,
        "head_dim": HEAD_DIM,
        "page_size": PAGE_SIZE,
        "physical_page_order": "globally_reversed",
        "accuracy_reference_partitions": partitions[0],
        "max_physical_byte_offset_fp8_per_cache": max_byte_offset_fp8,
        "max_physical_byte_offset_bf16_per_cache": max_byte_offset_bf16,
        "results": results,
        "best": best,
        "device_free_bytes_before": free_before,
        "device_total_bytes": total_memory,
    }
    del states, results, best, reference, problem
    torch.cuda.empty_cache()
    return result


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contexts",
        type=parse_int_list,
        default=parse_int_list(
            "4096,8192,16384,32768,65536,131072,262144,524288,786432,1048576"
        ),
    )
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument(
        "--partitions", type=parse_int_list, default=parse_int_list("8,16,32")
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("a ROCm GPU is required")
    arch = torch.cuda.get_device_properties(0).gcnArchName
    if not arch.startswith("gfx950"):
        raise RuntimeError(f"expected gfx950, got {arch}")

    payload = {
        "timestamp_unix": time.time(),
        "hostname": os.uname().nodename,
        "torch_version": torch.__version__,
        "aiter_path": str(Path(aiter.__file__).resolve()),
        "device_name": torch.cuda.get_device_name(0),
        "device_arch": arch,
        "shape": {
            "batch": args.batch,
            "query_length": QUERY_LENGTH,
            "num_q_heads": NUM_Q_HEADS,
            "num_kv_heads": NUM_KV_HEADS,
            "head_dim": HEAD_DIM,
            "page_size": PAGE_SIZE,
        },
        "partitions_tested": args.partitions,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "input_matching": (
            "BF16 caches and queries are exact dequantizations of the FP8 "
            "inputs used by the FP8 paths"
        ),
        "paths": {
            "gluon_bf16": "BF16 Q/K/V, BF16 compute, FP32 accumulation",
            "gluon_fp8": "FP8 Q/K/V with scales, FP8 compute, FP32 accumulation",
            "flydsl_bf16": "BF16 Q/K/V, native BF16 MFMA, FP32 accumulation",
            "flydsl_fp8": "BF16 Q quantized in-kernel, FP8 K/V, FP8 MFMA, FP32 accumulation",
        },
        "cases": [],
    }
    for index, context_length in enumerate(args.contexts):
        case = run_case(
            context_length,
            args.batch,
            args.partitions,
            args.warmup,
            args.repeats,
            args.seed + index,
        )
        payload["cases"].append(case)
        write_json(args.output, payload)
        summaries = []
        for path in PATHS:
            item = case["best"][path]
            summaries.append(
                f"{path}={item['timing_ms']['median']:.6f}ms/p{item['partitions']}"
            )
        print(f"context={context_length} " + " ".join(summaries), flush=True)


if __name__ == "__main__":
    main()
