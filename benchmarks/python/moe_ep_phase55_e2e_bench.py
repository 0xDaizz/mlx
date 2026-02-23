#!/usr/bin/env python3
# Copyright © 2026 Apple Inc.

"""Phase 5.5 E2E MoE EP Forward benchmark — loop vs batched local FFN.

Measures the full MixtureOfExperts EP forward pass on 2 ranks, comparing
loop (sequential per-expert) vs batched (3D fused matmul) expert FFN modes.

Run:
    mlx.launch --backend jaccl --hostfile hosts.json -- \\
        python3 benchmarks/python/moe_ep_phase55_e2e_bench.py [--scale small|medium]

Options:
    --scale small|medium   Model scale (default: small)
    --warmup N             Warmup iterations before timing (default: 5)
    --iters N              Timing iterations (default: 10)
"""

import argparse
import math
import os
import sys
import time

import numpy as np

import mlx.core as mx
from mlx.nn.layers.moe import MixtureOfExperts


# ── scale configs ────────────────────────────────────────────────────────────

SCALES = {
    "small": dict(
        E_total=16,
        D=1024,
        expert_dim=4096,
        top_k=4,
        N_values=[1, 8, 64, 128, 256, 512],
    ),
    "medium": dict(
        E_total=32,
        D=2048,
        expert_dim=8192,
        top_k=8,
        N_values=[1, 8, 64, 128, 256, 512, 1024],
    ),
}


# ── helpers ──────────────────────────────────────────────────────────────────


def geomean(values):
    """Geometric mean of positive values; returns nan if none."""
    valid = [v for v in values if v is not None and not math.isnan(v) and v > 0]
    if not valid:
        return float("nan")
    return float(np.exp(np.mean(np.log(valid))))


def rdma_warmup(group, rank):
    """Warm up JACCL RDMA to prevent garbage data on first blocking calls."""
    if rank == 0:
        print("  Warming up JACCL RDMA ...", flush=True)
    buf = mx.ones(1)
    for _ in range(5):
        buf = mx.distributed.all_sum(buf, group=group)
        mx.eval(buf)
        mx.synchronize()
    if rank == 0:
        print("  JACCL warmup done.", flush=True)


def barrier(group):
    s = mx.distributed.all_sum(mx.array(0.0), group=group)
    mx.eval(s)
    mx.synchronize()


# ── benchmark core ───────────────────────────────────────────────────────────


def bench_mode(model, N, D, dtype, warmup, iters, group):
    """Run warmup then timed iters for model(x); return list of times in ms."""
    x = mx.random.normal((N, D)).astype(dtype)
    mx.eval(x)

    # warmup
    for _ in range(warmup):
        out, _ = model(x)
        mx.eval(out)
        mx.synchronize()

    # timed
    times = []
    for _ in range(iters):
        # Recreate input each iter to avoid caching artifacts
        xi = mx.random.normal((N, D)).astype(dtype)
        mx.eval(xi)
        barrier(group)

        t0 = time.perf_counter()
        out, _ = model(xi)
        mx.eval(out)
        mx.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)

    return times


# ── table printing ───────────────────────────────────────────────────────────


def print_header(cfg, ws, scale_name, warmup, iters):
    E = cfg["E_total"]
    D = cfg["D"]
    edim = cfg["expert_dim"]
    top_k = cfg["top_k"]
    print()
    print(f"Phase 5.5 E2E MoE EP Forward — loop vs batched ({ws}-rank)")
    print(
        f"E={E} D={D} expert_dim={edim} top_k={top_k} cf=1.25 "
        f"dtype=float16 ws={ws}  scale={scale_name}  "
        f"warmup={warmup}  iters={iters}"
    )
    print()
    hdr = f"{'N':>6s} | {'Loop(ms)':>9s} | {'Batched(ms)':>11s} | {'Speedup':>7s} | {'Saved(ms)':>9s}"
    sep = f"{'':->6s}-+-{'':->9s}-+-{'':->11s}-+-{'':->7s}-+-{'':->9s}"
    print(hdr)
    print(sep)


def print_row(N, loop_med, bat_med):
    if bat_med > 0 and not math.isnan(bat_med) and not math.isnan(loop_med):
        speedup = loop_med / bat_med
        saved = loop_med - bat_med
        speedup_str = f"{speedup:>6.2f}x"
        saved_str = f"{saved:>9.2f}"
    else:
        speedup_str = f"{'N/A':>7s}"
        saved_str = f"{'N/A':>9s}"

    loop_str = f"{loop_med:>9.2f}" if not math.isnan(loop_med) else f"{'N/A':>9s}"
    bat_str = f"{bat_med:>11.2f}" if not math.isnan(bat_med) else f"{'N/A':>11s}"

    print(f"{N:>6d} | {loop_str} | {bat_str} | {speedup_str} | {saved_str}")


# ── main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Phase 5.5 E2E MoE EP Forward benchmark — loop vs batched"
    )
    parser.add_argument(
        "--scale",
        choices=["small", "medium"],
        default="small",
        help="Model scale (default: small)",
    )
    parser.add_argument(
        "--warmup", type=int, default=5, help="Warmup iterations (default: 5)"
    )
    parser.add_argument(
        "--iters", type=int, default=10, help="Timing iterations (default: 10)"
    )
    args = parser.parse_args()

    # ── distributed init ────────────────────────────────────────────────────
    group = mx.distributed.init()
    rank = group.rank()
    ws = group.size()

    if ws < 2:
        if rank == 0:
            print(f"ERROR: This benchmark requires >= 2 ranks, got {ws}", file=sys.stderr)
        sys.exit(1)

    # Suppress output on non-root ranks
    def log(*a, **kw):
        if rank == 0:
            print(*a, **kw)

    cfg = SCALES[args.scale]
    E_total = cfg["E_total"]
    D = cfg["D"]
    expert_dim = cfg["expert_dim"]
    top_k = cfg["top_k"]
    N_values = cfg["N_values"]
    dtype = mx.float16

    if E_total % ws != 0:
        log(f"ERROR: E_total={E_total} not divisible by ws={ws}", file=sys.stderr)
        sys.exit(1)

    log(f"Initialized: rank={rank} ws={ws} scale={args.scale}")

    # ── RDMA warmup ─────────────────────────────────────────────────────────
    if rank == 0:
        rdma_warmup(group, rank)
    else:
        # Non-rank-0 participates silently
        buf = mx.ones(1)
        for _ in range(5):
            buf = mx.distributed.all_sum(buf, group=group)
            mx.eval(buf)
            mx.synchronize()

    barrier(group)

    # ── model creation ──────────────────────────────────────────────────────
    log("  Creating MixtureOfExperts model ...", flush=True)

    model = MixtureOfExperts(
        hidden_dim=D,
        expert_dim=expert_dim,
        num_experts=E_total,
        top_k=top_k,
        capacity_factor=1.25,
        ep_impl="cpp",
        ep_backend="auto",
    )

    # Cast all parameters to float16
    model.set_dtype(dtype)

    # Materialize weights before benchmarking
    params = model.parameters()
    mx.eval(params)
    mx.synchronize()
    log("  Model ready.", flush=True)

    # ── benchmark ───────────────────────────────────────────────────────────

    # results[N] = {"loop": median_ms, "batched": median_ms}
    results = {N: {} for N in N_values}

    for mode in ["loop", "batched"]:
        os.environ["MLX_MOE_EP_LOCAL_FFN"] = mode
        # Invalidate stacked weight cache so the new mode takes effect
        model._cached_stacked = None

        log(f"\n  Benchmarking mode='{mode}' ...", flush=True)

        for N in N_values:
            barrier(group)
            log(f"    N={N:>5d} ...", end="", flush=True)

            try:
                times = bench_mode(
                    model, N, D, dtype, args.warmup, args.iters, group
                )
                med = float(np.median(times))
                p90 = float(np.percentile(times, 90))
                results[N][mode] = med
                log(f" med={med:.2f}ms p90={p90:.2f}ms", flush=True)
            except Exception as ex:
                results[N][mode] = float("nan")
                log(f" ERROR: {ex}", flush=True)

    # ── print comparison table ──────────────────────────────────────────────
    if rank == 0:
        print_header(cfg, ws, args.scale, args.warmup, args.iters)

        decode_speedups = []
        prefill_speedups = []

        for N in N_values:
            loop_med = results[N].get("loop", float("nan"))
            bat_med = results[N].get("batched", float("nan"))
            print_row(N, loop_med, bat_med)

            if not math.isnan(loop_med) and not math.isnan(bat_med) and bat_med > 0:
                sp = loop_med / bat_med
                if N <= 64:
                    decode_speedups.append(sp)
                if N >= 256:
                    prefill_speedups.append(sp)

        print()
        print("Geomean speedup (loop / batched — higher = batched is faster):")
        if decode_speedups:
            print(f"  Decode  (N<=64):   {geomean(decode_speedups):.3f}x")
        else:
            print("  Decode  (N<=64):   N/A")
        if prefill_speedups:
            print(f"  Prefill (N>=256):  {geomean(prefill_speedups):.3f}x")
        else:
            print("  Prefill (N>=256):  N/A")
        print()

    barrier(group)


if __name__ == "__main__":
    main()
