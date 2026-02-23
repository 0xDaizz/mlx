#!/usr/bin/env python3
# Copyright © 2026 Apple Inc.

"""Phase 5.5 Chunked Expert FFN Benchmark — loop vs chunked vs full-batched.

Measures _run_local_experts() performance with different modes and chunk sizes.
Single-node benchmark — no distributed required.

Run:
    python3 benchmarks/python/moe_ep_phase55_chunked_bench.py [--scale small|medium|large]

Options:
    --scale small|medium|large   Model scale (default: medium)
    --warmup N                   Warmup iterations (default: 5)
    --iters N                    Timing iterations (default: 10)
    --chunk-sizes LIST           Comma-separated chunk sizes (default: 8,16,32,64)
"""

import argparse
import math
import os
import time

import numpy as np

import mlx.core as mx
from mlx.nn.layers.moe import MixtureOfExperts


# ── scale configs ────────────────────────────────────────────────────────────

SCALES = {
    "small": dict(
        E_total=32,
        D=1024,
        expert_dim=4096,
        top_k=4,
        N_values=[1, 8, 64, 128, 256, 512],
    ),
    "medium": dict(
        E_total=64,
        D=2048,
        expert_dim=8192,
        top_k=8,
        N_values=[1, 8, 64, 128, 256, 512, 1024],
    ),
    "large": dict(
        E_total=128,
        D=4096,
        expert_dim=16384,
        top_k=8,
        N_values=[1, 8, 64, 128, 256, 512, 1024],
    ),
}


def geomean(values):
    valid = [v for v in values if v is not None and not math.isnan(v) and v > 0]
    if not valid:
        return float("nan")
    return float(np.exp(np.mean(np.log(valid))))


def bench_experts(model, dispatched, warmup, iters):
    """Time _run_local_experts with current env settings. Returns median ms.

    Clears cached stacked weights once before warmup, then lets the cache
    persist for all warmup + timed iterations. This reflects the realistic
    inference path where weights are stacked once and reused.
    """
    # Clear cache once — first warmup call will rebuild and cache it
    model._cached_stacked = None

    # warmup
    for _ in range(warmup):
        out = model._run_local_experts(dispatched)
        mx.eval(out)
        mx.synchronize()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = model._run_local_experts(dispatched)
        mx.eval(out)
        mx.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)

    return float(np.median(times))


def bench_chunked(model, dispatched, chunk_e, warmup, iters):
    """Time _run_local_experts_batched_chunked directly. Returns median ms.

    Called directly to bypass the E_local <= 64 auto-routing in
    _run_local_experts, allowing chunked benchmarks even for small E_local.
    """
    # warmup
    for _ in range(warmup):
        out = model._run_local_experts_batched_chunked(dispatched, chunk_e)
        mx.eval(out)
        mx.synchronize()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = model._run_local_experts_batched_chunked(dispatched, chunk_e)
        mx.eval(out)
        mx.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)

    return float(np.median(times))


def main():
    parser = argparse.ArgumentParser(
        description="Phase 5.5 Chunked Expert FFN Benchmark"
    )
    parser.add_argument(
        "--scale", choices=["small", "medium", "large"], default="medium",
        help="Model scale (default: medium)",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument(
        "--chunk-sizes", type=str, default="8,16,32,64",
        help="Comma-separated chunk sizes to test (default: 8,16,32,64)",
    )
    args = parser.parse_args()

    chunk_sizes = [int(x) for x in args.chunk_sizes.split(",")]
    cfg = SCALES[args.scale]
    E_total = cfg["E_total"]
    D = cfg["D"]
    expert_dim = cfg["expert_dim"]
    top_k = cfg["top_k"]
    N_values = cfg["N_values"]
    dtype = mx.float16

    # Simulate 2-rank EP: E_local = E_total // 2
    ws = 2
    E_local = E_total // ws
    cf = 1.25

    print()
    print(f"Phase 5.5 Chunked Expert FFN Benchmark")
    print(f"E_total={E_total} E_local={E_local} D={D} expert_dim={expert_dim} "
          f"top_k={top_k} dtype=float16 scale={args.scale}")
    print(f"warmup={args.warmup} iters={args.iters} chunk_sizes={chunk_sizes}")
    print()

    # Create model (no distributed group — just for weight access)
    print("Creating MixtureOfExperts model ...", flush=True)
    model = MixtureOfExperts(
        hidden_dim=D,
        expert_dim=expert_dim,
        num_experts=E_total,
        top_k=top_k,
        capacity_factor=cf,
    )

    # Trim to E_local experts (simulate EP partition)
    model.experts = model.experts[:E_local]
    model.set_dtype(dtype)
    mx.eval(model.parameters())
    mx.synchronize()

    weight_bytes = E_local * D * expert_dim * 2 * 3  # 3 weights (gate, up, down), float16
    print(f"  E_local={E_local}, individual weight memory: {weight_bytes / 1e9:.1f} GB")
    print(f"  Full stack extra: {weight_bytes / 1e9:.1f} GB")
    for cs in chunk_sizes:
        chunk_extra = weight_bytes * cs / E_local
        print(f"  Chunk {cs} extra: {chunk_extra / 1e9:.1f} GB "
              f"({cs}/{E_local} = {cs / E_local:.1%})")
    print()

    # ── modes to test ──
    modes = [("loop", None)]
    for cs in chunk_sizes:
        if cs < E_local:
            modes.append((f"chunked_{cs}", cs))
    if E_local <= 64:
        modes.append(("full_batched", None))

    # ── header ──
    mode_labels = [m[0] for m in modes]
    hdr = f"{'N':>6s}"
    for label in mode_labels:
        hdr += f" | {label:>12s}"
    # Speedup columns: each vs loop
    for label in mode_labels[1:]:
        hdr += f" | {label[:8] + '/loop':>12s}"
    print(hdr)
    print("-" * len(hdr))

    # ── benchmark ──
    decode_speedups = {label: [] for label in mode_labels[1:]}
    prefill_speedups = {label: [] for label in mode_labels[1:]}

    for N in N_values:
        cap = max(1, math.ceil(N * top_k / E_total * cf))
        cap_total = ws * cap
        dispatched = mx.random.normal((E_local, cap_total, D)).astype(dtype)
        mx.eval(dispatched)

        row = f"{N:>6d}"
        results = {}

        for label, chunk_e in modes:
            if "chunked" in label:
                # Call _run_local_experts_batched_chunked directly to bypass
                # the E_local <= 64 auto-routing in _run_local_experts.
                med = bench_chunked(
                    model, dispatched, chunk_e, args.warmup, args.iters
                )
            elif label == "full_batched":
                os.environ["MLX_MOE_EP_LOCAL_FFN"] = "batched"
                os.environ.pop("MLX_MOE_EP_LOCAL_FFN_CHUNK_E", None)
                med = bench_experts(model, dispatched, args.warmup, args.iters)
            else:
                # loop
                os.environ["MLX_MOE_EP_LOCAL_FFN"] = "loop"
                med = bench_experts(model, dispatched, args.warmup, args.iters)

            results[label] = med
            row += f" | {med:>12.2f}"

        # Speedup columns
        loop_med = results.get("loop", float("nan"))
        for label in mode_labels[1:]:
            med = results.get(label, float("nan"))
            if loop_med > 0 and med > 0:
                sp = loop_med / med
                row += f" | {sp:>11.2f}x"
                if N <= 64:
                    decode_speedups[label].append(sp)
                if N >= 256:
                    prefill_speedups[label].append(sp)
            else:
                row += f" | {'N/A':>12s}"

        print(row)

    # ── summary ──
    print()
    print("Geomean speedup vs loop (higher = faster than loop):")
    for label in mode_labels[1:]:
        d = geomean(decode_speedups[label])
        p = geomean(prefill_speedups[label])
        print(f"  {label:>15s}:  Decode(N<=64) {d:.3f}x  Prefill(N>=256) {p:.3f}x")
    print()

    # Clean up env
    os.environ.pop("MLX_MOE_EP_LOCAL_FFN", None)
    os.environ.pop("MLX_MOE_EP_LOCAL_FFN_CHUNK_E", None)


if __name__ == "__main__":
    main()
