#!/usr/bin/env python3
"""Benchmark MiMo BF16 vectorized-5D FlyDSL PA decode against Gluon."""

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

from aiter.ops.attention import pa_decode_gluon
from aiter.ops.triton.gluon.pa_decode_gluon import get_recommended_splits

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


def parse_int_list(value: str) -> list[int]:
    values = [int(item) for item in value.split(",") if item]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def make_problem(context_length: int, batch: int, seed: int) -> dict:
    blocks_per_sequence = math.ceil(context_length / PAGE_SIZE)
    num_blocks = batch * blocks_per_sequence
    generator = torch.Generator(device="cuda").manual_seed(seed)
    query = torch.empty(
        (batch * QUERY_LENGTH, NUM_Q_HEADS, HEAD_DIM),
        dtype=torch.bfloat16,
        device="cuda",
    ).uniform_(-1.0, 1.0, generator=generator)
    key = torch.empty(
        (num_blocks, NUM_KV_HEADS, HEAD_DIM // 8, PAGE_SIZE, 8),
        dtype=torch.bfloat16,
        device="cuda",
    ).uniform_(-1.0, 1.0, generator=generator)
    value = torch.empty(
        (num_blocks, NUM_KV_HEADS, PAGE_SIZE // 8, HEAD_DIM, 8),
        dtype=torch.bfloat16,
        device="cuda",
    ).uniform_(-1.0, 1.0, generator=generator)
    block_tables = torch.arange(
        num_blocks - 1, -1, -1, dtype=torch.int32, device="cuda"
    ).reshape(batch, blocks_per_sequence)
    context_lengths = torch.full(
        (batch,), context_length, dtype=torch.int32, device="cuda"
    )
    return {
        "query": query,
        "key": key,
        "value": value,
        "block_tables": block_tables,
        "context_lengths": context_lengths,
        "num_blocks": num_blocks,
    }


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


def launch_gluon(
    problem: dict, output: torch.Tensor, workspace: dict[str, torch.Tensor]
) -> None:
    partitions = workspace["max"].shape[2]
    pa_decode_gluon(
        output=output,
        query=problem["query"],
        key_cache=problem["key"],
        value_cache=problem["value"],
        context_lengths=problem["context_lengths"],
        block_tables=problem["block_tables"],
        softmax_scale=HEAD_DIM**-0.5,
        query_length=QUERY_LENGTH,
        max_context_partition_num=partitions,
        context_partition_size=CONTEXT_PARTITION_SIZE,
        compute_type=torch.bfloat16,
        query_scale=None,
        key_scale=None,
        value_scale=None,
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
) -> None:
    pa_decode_tile(
        output=output,
        query=problem["query"],
        key_cache=problem["key"],
        value_cache=problem["value"],
        block_tables=problem["block_tables"],
        context_lengths=problem["context_lengths"],
        key_scale=None,
        value_scale=None,
        softmax_scale=HEAD_DIM**-0.5,
        num_partitions=partitions,
        pmax=workspace["max"],
        psum=workspace["sum"],
        pout=workspace["out"],
    )


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
    diff = (actual.float() - expected.float()).abs()
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "fraction_abs_gt_5e-3": float((diff > 5.0e-3).float().mean().item()),
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

    gluon_partitions = int(get_recommended_splits(batch, NUM_KV_HEADS))
    gluon_workspace = make_workspace(batch, gluon_partitions)
    gluon_output = torch.full_like(problem["query"], float("nan"))
    launch_gluon(problem, gluon_output, gluon_workspace)
    torch.cuda.synchronize()
    if not bool(torch.isfinite(gluon_output).all().item()):
        raise RuntimeError(f"context {context_length}: Gluon output is not finite")

    flydsl_states = []
    for partition_count in partitions:
        workspace = make_workspace(batch, partition_count)
        output = torch.full_like(gluon_output, float("nan"))
        launch_flydsl(problem, output, partition_count, workspace)
        torch.cuda.synchronize()
        if not bool(torch.isfinite(output).all().item()):
            raise RuntimeError(
                f"context {context_length}, partitions {partition_count}: "
                "FlyDSL output is not finite"
            )
        metrics = compare(output, gluon_output)
        torch.testing.assert_close(output, gluon_output, rtol=5.0e-3, atol=5.0e-3)
        flydsl_states.append((partition_count, output, workspace, metrics))

    gluon_timing = summarize(
        event_samples(
            lambda: launch_gluon(problem, gluon_output, gluon_workspace),
            warmup,
            repeats,
        )
    )
    flydsl_results = []
    for partition_count, output, workspace, metrics in flydsl_states:
        timing = summarize(
            event_samples(
                lambda output=output, partition_count=partition_count, workspace=workspace: launch_flydsl(
                    problem, output, partition_count, workspace
                ),
                warmup,
                repeats,
            )
        )
        timing["speedup_over_gluon"] = gluon_timing["median"] / timing["median"]
        flydsl_results.append(
            {
                "partitions": partition_count,
                "accuracy_vs_gluon": metrics,
                "timing_ms": timing,
            }
        )

    logical_kv_bytes = 2 * batch * context_length * NUM_KV_HEADS * HEAD_DIM * 2
    max_byte_offset = problem["num_blocks"] * PAGE_SIZE * HEAD_DIM * 2 - 1
    result = {
        "context_length": context_length,
        "batch": batch,
        "query_length": QUERY_LENGTH,
        "num_q_heads": NUM_Q_HEADS,
        "num_kv_heads": NUM_KV_HEADS,
        "head_dim": HEAD_DIM,
        "page_size": PAGE_SIZE,
        "key_shape": list(problem["key"].shape),
        "value_shape": list(problem["value"].shape),
        "logical_kv_bytes": logical_kv_bytes,
        "max_physical_byte_offset_per_cache": max_byte_offset,
        "crosses_2gib_per_cache": max_byte_offset >= 2**31,
        "crosses_4gib_per_cache": max_byte_offset >= 2**32,
        "gluon": {"partitions": gluon_partitions, "timing_ms": gluon_timing},
        "flydsl": flydsl_results,
        "device_free_bytes_before": free_before,
        "device_total_bytes": total_memory,
    }
    del flydsl_states, flydsl_results, gluon_output, gluon_workspace, problem
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
        "device_name": torch.cuda.get_device_name(0),
        "device_arch": arch,
        "kernel_path": str(Path(sys.modules[pa_decode_tile.__module__].__file__).resolve()),
        "cache_dtype": "bfloat16",
        "cache_layout": "vectorized_5d_inner8",
        "mfma_compute_dtype": "bfloat16",
        "mfma_instruction": "v_mfma_f32_16x16x16_bf16",
        "accumulator_dtype": "float32",
        "cache_value_range": [-1.0, 1.0],
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
        best = min(case["flydsl"], key=lambda item: item["timing_ms"]["median"])
        print(
            f"context={context_length} gluon={case['gluon']['timing_ms']['median']:.6f} ms "
            f"flydsl={best['timing_ms']['median']:.6f} ms p={best['partitions']} "
            f"speedup={best['timing_ms']['speedup_over_gluon']:.3f}x "
            f"max_abs={best['accuracy_vs_gluon']['max_abs']:.6g}",
            flush=True,
        )


if __name__ == "__main__":
    main()
