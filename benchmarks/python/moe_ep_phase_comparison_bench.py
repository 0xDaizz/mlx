#!/usr/bin/env python3
# Copyright © 2026 Apple Inc.

"""Phase Comparison Benchmark — MoE Expert Parallelism.

Decomposes MoE EP performance into two independent measurements:

  Part 1 — Dispatch+Combine at Kimi K2.5 scale
    Measures C++ primitive overhead directly, without expert weights.
    E=384, D=7168, top_k=8, cf=1.25, dtype=float16
    Backends: cpu (Phase 3), metal (Phase 4), auto (Phase 5)

  Part 2 — Expert FFN at medium scale
    Measures loop vs batched expert FFN with a real MixtureOfExperts model.
    E=64, D=2048, expert_dim=8192, top_k=8, cf=1.25, dtype=float16
    Modes: loop (Phase 5), batched (Phase 5.5)

Run:
    mlx.launch --backend jaccl --hostfile hosts.json -- \\
        python3 benchmarks/python/moe_ep_phase_comparison_bench.py

Options:
    --warmup N         warmup iterations before timing (default 5)
    --iters N          timing iterations (default 10)
"""

import argparse
import math
import os
import subprocess
import sys
import time

import numpy as np

import mlx.core as mx
from mlx.nn.layers.moe import MixtureOfExperts


# ── helpers ──────────────────────────────────────────────────────────────────


def barrier(group):
    """Synchronize all ranks with a no-op all_sum."""
    s = mx.distributed.all_sum(mx.array(0.0), group=group)
    mx.eval(s)
    mx.synchronize()


def geomean(values):
    """Geometric mean of positive finite values; returns NaN if none valid."""
    valid = [v for v in values if v is not None and np.isfinite(v) and v > 0]
    if not valid:
        return float("nan")
    return float(np.exp(np.mean(np.log(valid))))


def compute_capacity(N, top_k, cf, E):
    """Per-expert capacity: ceil(N * top_k * cf / E), minimum 1."""
    return max(1, math.ceil(N * top_k * cf / E))


def sync_capacity(cap, group):
    """Synchronize capacity across ranks (take max so all ranks agree)."""
    cap_arr = mx.array(cap, dtype=mx.int32)
    cap_arr = mx.distributed.all_max(cap_arr, group=group)
    mx.eval(cap_arr)
    return cap_arr.item()


# ── JACCL RDMA warmup ────────────────────────────────────────────────────────


def jaccl_warmup(group, rank):
    """Warm up JACCL RDMA connections to avoid garbage on first calls."""
    if rank == 0:
        print("  Warming up JACCL RDMA ...", flush=True)
    buf = mx.ones((1,), dtype=mx.float32)
    mx.eval(buf)
    for _ in range(5):
        buf = mx.distributed.all_sum(buf, group=group)
        mx.eval(buf)
        mx.synchronize()
    a2a_buf = mx.ones((group.size(),), dtype=mx.float32)
    mx.eval(a2a_buf)
    for _ in range(3):
        a2a_buf = mx.distributed.all_to_all(a2a_buf, group=group)
        mx.eval(a2a_buf)
        mx.synchronize()
    barrier(group)
    if rank == 0:
        print("  JACCL warmup done.", flush=True)


# ── Part 1: Dispatch+Combine timing ──────────────────────────────────────────


def time_dispatch_combine(tokens, expert_indices, gate_weights, E_total, cap,
                          group, backend, warmup, iters):
    """Time dispatch+combine for a single backend.

    Returns median latency in ms. Raises on error.
    """
    def run_once():
        dispatched, route_idx = mx.distributed.moe_dispatch_exchange(
            tokens,
            expert_indices,
            num_experts=E_total,
            capacity=cap,
            group=group,
            backend=backend,
        )
        mx.eval(dispatched, route_idx)

        # Simulate expert output: pass dispatched tokens through unchanged.
        # This isolates dispatch+combine overhead from actual FFN computation.
        expert_out = dispatched

        output = mx.distributed.moe_combine_exchange(
            expert_out,
            route_idx,
            gate_weights,
            tokens,
            num_experts=E_total,
            capacity=cap,
            group=group,
            backend=backend,
        )
        mx.eval(output)

    # Warmup
    for _ in range(warmup):
        run_once()
        mx.synchronize()

    # Timed iterations
    times = []
    for _ in range(iters):
        barrier(group)
        t0 = time.perf_counter()
        run_once()
        mx.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)

    return float(np.median(times))


def run_part1(group, rank, warmup, iters):
    """Part 1: Dispatch+Combine at Kimi K2.5 scale.

    Returns:
        results: dict[(backend, N)] -> median_ms
        caps:    dict[N] -> capacity
        N_values: list of N
    """
    # Kimi K2.5 scale parameters
    E_total = 384
    D = 7168
    top_k = 8
    cf = 1.25
    dtype = mx.float16

    N_values = [1, 8, 64, 128, 256, 512, 1024, 2048, 4096]
    backends = ["cpu", "metal", "auto"]
    backend_labels = ["P3 CPU", "P4 Metal", "P5 Auto"]

    if rank == 0:
        print()
        print("=" * 72)
        print("Part 1: Dispatch+Combine (Kimi K2.5 Scale)")
        print(f"E={E_total} D={D} top_k={top_k} cf={cf} dtype=float16 "
              f"ws={group.size()}")
        print(f"warmup={warmup} iters={iters}")
        print("=" * 72)
        print()

    results = {}
    caps = {}

    for N in N_values:
        cap = compute_capacity(N, top_k, cf, E_total)
        cap = sync_capacity(cap, group)
        caps[N] = cap

        # Create synthetic routing data for this N
        tokens = mx.random.normal((N, D)).astype(dtype)
        expert_indices = mx.random.randint(0, E_total, shape=(N, top_k)).astype(mx.int32)
        gate_weights = mx.softmax(
            mx.random.normal((N, top_k)), axis=-1
        ).astype(mx.float32)
        mx.eval(tokens, expert_indices, gate_weights)

        for backend, label in zip(backends, backend_labels):
            barrier(group)

            if rank == 0:
                print(f"  N={N:>5d} cap={cap:>4d} [{label}] ...",
                      end="", flush=True)

            try:
                med_ms = time_dispatch_combine(
                    tokens, expert_indices, gate_weights,
                    E_total, cap, group, backend, warmup, iters,
                )
                results[(backend, N)] = med_ms
                if rank == 0:
                    print(f" {med_ms:.2f}ms", flush=True)
            except Exception as ex:
                results[(backend, N)] = float("nan")
                if rank == 0:
                    print(f" ERR({ex})", flush=True)

    return results, caps, N_values, backends, backend_labels


def print_part1_table(results, caps, N_values, backends, backend_labels, rank, ws):
    """Print Part 1 results table and geomean summary."""
    if rank != 0:
        return

    # Detect auto-pick by comparing auto vs cpu/metal latency
    def infer_auto_pick(N):
        cpu_t = results.get(("cpu", N), float("nan"))
        metal_t = results.get(("metal", N), float("nan"))
        auto_t = results.get(("auto", N), float("nan"))
        if not np.isfinite(auto_t):
            return "?"
        if not np.isfinite(cpu_t) or not np.isfinite(metal_t):
            return "?"
        if abs(auto_t - cpu_t) < abs(auto_t - metal_t):
            return "CPU"
        return "Metal"

    print()
    print("Phase Comparison — Dispatch+Combine (Kimi K2.5 Scale)")
    print(f"E=384 D=7168 top_k=8 cf=1.25 dtype=float16 ws={ws}")
    print()

    # Header
    print(f"{'':>7} | {'':>5} | {'P3 CPU(ms)':>12} | {'P4 Metal(ms)':>13} "
          f"| {'P5 Auto(ms)':>12} | {'Auto pick':>10}")
    print("-" * 7 + "+" + "-" * 7 + "+" + "-" * 14 + "+" + "-" * 15
          + "+" + "-" * 14 + "+" + "-" * 12)

    for N in N_values:
        cap = caps[N]
        cpu_t = results.get(("cpu", N), float("nan"))
        metal_t = results.get(("metal", N), float("nan"))
        auto_t = results.get(("auto", N), float("nan"))
        auto_pick = infer_auto_pick(N)

        def fmt(v):
            if not np.isfinite(v):
                return f"{'N/A':>12}"
            return f"{v:>12.2f}"

        print(f"{N:>7d} | {cap:>5d} | {fmt(cpu_t)} | {fmt(metal_t):>13} "
              f"| {fmt(auto_t)} | {auto_pick:>10}")

    # Geomean speedups vs Phase 3 (CPU)
    print()
    print("Geomean speedup vs Phase 3 (CPU baseline — higher = faster):")

    decode_ns = [N for N in N_values if N <= 64]
    prefill_ns = [N for N in N_values if N >= 256]

    def speedup(backend, ns):
        vals = []
        for N in ns:
            ref = results.get(("cpu", N), float("nan"))
            tgt = results.get((backend, N), float("nan"))
            if np.isfinite(ref) and np.isfinite(tgt) and tgt > 0:
                vals.append(ref / tgt)
        return geomean(vals)

    for segment, ns, label in [
        ("Decode  (N<=64) ", decode_ns, "Decode"),
        ("Prefill (N>=256)", prefill_ns, "Prefill"),
    ]:
        gm_p4 = speedup("metal", ns)
        gm_p5 = speedup("auto", ns)
        p4_str = f"{gm_p4:.2f}x" if np.isfinite(gm_p4) else "N/A"
        p5_str = f"{gm_p5:.2f}x" if np.isfinite(gm_p5) else "N/A"
        print(f"  {segment}:  P4/P3={p4_str}  P5/P3={p5_str}")


# ── Part 2: Expert FFN timing ─────────────────────────────────────────────────


def bench_expert_ffn(model, N, dtype, warmup, iters, group):
    """Time model forward pass for a single N.

    Returns list of elapsed times in ms.
    """
    D = model.hidden_dim

    # Warmup
    for _ in range(warmup):
        x = mx.random.normal((N, D)).astype(dtype)
        mx.eval(x)
        out, _ = model(x)
        mx.eval(out)
        mx.synchronize()

    # Timed iterations
    times = []
    for _ in range(iters):
        x = mx.random.normal((N, D)).astype(dtype)
        mx.eval(x)
        barrier(group)
        t0 = time.perf_counter()
        out, _ = model(x)
        mx.eval(out)
        mx.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)

    return times


def run_part2(group, rank, warmup, iters):
    """Part 2: Expert FFN at medium scale — loop vs batched.

    Returns:
        results: dict[(mode, N)] -> median_ms
        caps:    dict[N] -> capacity
        N_values: list of N
    """
    # Medium scale parameters
    E_total = 64
    D = 2048
    expert_dim = 8192
    top_k = 8
    cf = 1.25
    dtype = mx.float16

    N_values = [1, 8, 64, 128, 256, 512, 1024]
    modes = ["loop", "batched"]

    if rank == 0:
        print()
        print("=" * 72)
        print("Part 2: Expert FFN (Medium Scale)")
        print(f"E={E_total} D={D} expert_dim={expert_dim} top_k={top_k} "
              f"cf={cf} dtype=float16 ws={group.size()}")
        print(f"warmup={warmup} iters={iters}")
        print("=" * 72)

    if rank == 0:
        print()
        print("  Building MixtureOfExperts model ...", flush=True)

    model = MixtureOfExperts(
        hidden_dim=D,
        expert_dim=expert_dim,
        num_experts=E_total,
        top_k=top_k,
        capacity_factor=cf,
        ep_impl="cpp",
        ep_backend="auto",
    )
    model.set_dtype(dtype)
    mx.eval(model.parameters())
    mx.synchronize()

    E_local = E_total // group.size()
    if rank == 0:
        print(f"  Model ready: E_local={E_local} D={D} expert_dim={expert_dim}")
        print()

    results = {}
    caps = {}

    for mode in modes:
        os.environ["MLX_MOE_EP_LOCAL_FFN"] = mode
        # Invalidate stacked weight cache so the new mode takes effect
        model._cached_stacked = None

        if rank == 0:
            print(f"  Benchmarking mode='{mode}' ...")

        for N in N_values:
            cap = compute_capacity(N, top_k, cf, E_total)
            cap = sync_capacity(cap, group)
            caps[N] = cap

            barrier(group)

            if rank == 0:
                print(f"    N={N:>5d} cap={cap:>4d} ...", end="", flush=True)

            try:
                times = bench_expert_ffn(model, N, dtype, warmup, iters, group)
                med = float(np.median(times))
                results[(mode, N)] = med
                if rank == 0:
                    p90 = float(np.percentile(times, 90))
                    print(f" med={med:.2f}ms p90={p90:.2f}ms", flush=True)
            except Exception as ex:
                results[(mode, N)] = float("nan")
                if rank == 0:
                    print(f" ERR({ex})", flush=True)

    # Restore default
    os.environ.pop("MLX_MOE_EP_LOCAL_FFN", None)

    return results, caps, N_values


def print_part2_table(results, caps, N_values, rank, ws):
    """Print Part 2 results table and geomean summary."""
    if rank != 0:
        return

    print()
    print("Phase Comparison — Expert FFN (Medium Scale)")
    print(f"E=64 D=2048 expert_dim=8192 top_k=8 cf=1.25 dtype=float16 ws={ws}")
    print()

    # Header
    print(f"{'':>7} | {'':>5} | {'P5 Loop(ms)':>12} | {'P5.5 Batch(ms)':>15} "
          f"| {'Speedup':>8} | {'Saved(ms)':>10}")
    print("-" * 7 + "+" + "-" * 7 + "+" + "-" * 14 + "+" + "-" * 17
          + "+" + "-" * 10 + "+" + "-" * 12)

    decode_speedups = []
    prefill_speedups = []

    for N in N_values:
        cap = caps.get(N, 0)
        loop_t = results.get(("loop", N), float("nan"))
        bat_t = results.get(("batched", N), float("nan"))

        if np.isfinite(loop_t) and np.isfinite(bat_t) and bat_t > 0:
            speedup = loop_t / bat_t
            saved = loop_t - bat_t
            sp_str = f"{speedup:>7.2f}x"
            sv_str = f"{saved:>10.2f}"
            if N <= 64:
                decode_speedups.append(speedup)
            if N >= 256:
                prefill_speedups.append(speedup)
        else:
            sp_str = f"{'N/A':>8}"
            sv_str = f"{'N/A':>10}"

        loop_str = f"{loop_t:>12.2f}" if np.isfinite(loop_t) else f"{'N/A':>12}"
        bat_str = f"{bat_t:>15.2f}" if np.isfinite(bat_t) else f"{'N/A':>15}"

        print(f"{N:>7d} | {cap:>5d} | {loop_str} | {bat_str} "
              f"| {sp_str} | {sv_str}")

    # Geomean summary
    print()
    print("Geomean speedup (loop / batched — higher = batched is faster):")
    if decode_speedups:
        gm_dec = geomean(decode_speedups)
        print(f"  Decode  (N<=64):   {gm_dec:.3f}x")
    else:
        print("  Decode  (N<=64):   N/A")
    if prefill_speedups:
        gm_pre = geomean(prefill_speedups)
        print(f"  Prefill (N>=256):  {gm_pre:.3f}x")
    else:
        print("  Prefill (N>=256):  N/A")


# ── main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3/4/5/5.5 Comparison Benchmark — MoE EP"
    )
    parser.add_argument(
        "--warmup", type=int, default=5,
        help="Warmup iterations before timing (default 5)",
    )
    parser.add_argument(
        "--iters", type=int, default=10,
        help="Timing iterations (default 10)",
    )
    args = parser.parse_args()

    # ── Distributed init ─────────────────────────────────────────────────
    group = mx.distributed.init()
    rank = group.rank()
    ws = group.size()

    if ws < 2:
        if rank == 0:
            print(f"ERROR: This benchmark requires >= 2 ranks, got {ws}.",
                  file=sys.stderr)
        sys.exit(1)

    def log(msg=""):
        if rank == 0:
            print(msg, flush=True)

    # ── Header ───────────────────────────────────────────────────────────
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        commit = "unknown"

    log("=" * 72)
    log("MoE EP Phase Comparison Benchmark")
    log(f"commit={commit}  ws={ws}  warmup={args.warmup}  iters={args.iters}")
    log("=" * 72)

    # ── Check C++ path availability ───────────────────────────────────────
    has_cpp = hasattr(mx.distributed, "moe_dispatch_exchange")
    if not has_cpp:
        log("[SKIP] moe_dispatch_exchange not available in this build.")
        log("Rebuild with Metal/distributed support enabled.")
        sys.exit(1)

    # ── RDMA warmup ──────────────────────────────────────────────────────
    jaccl_warmup(group, rank)

    # ── Part 1: Dispatch+Combine (Kimi K2.5) ─────────────────────────────
    p1_results, p1_caps, p1_N, p1_backends, p1_labels = run_part1(
        group, rank, warmup=args.warmup, iters=args.iters,
    )
    print_part1_table(p1_results, p1_caps, p1_N, p1_backends, p1_labels, rank, ws)

    barrier(group)

    # ── Part 2: Expert FFN (medium scale) ────────────────────────────────
    p2_results, p2_caps, p2_N = run_part2(
        group, rank, warmup=args.warmup, iters=args.iters,
    )
    print_part2_table(p2_results, p2_caps, p2_N, rank, ws)

    barrier(group)

    # ── Legend ───────────────────────────────────────────────────────────
    log()
    log("=" * 72)
    log("Notes:")
    log("  Part 1 measures C++ primitive overhead only (no expert weights).")
    log("    P3 CPU   : backend=cpu   (v3 blocking protocol baseline)")
    log("    P4 Metal : backend=metal (Metal GPU kernels)")
    log("    P5 Auto  : backend=auto  (auto policy, 3-zone threshold)")
    log("  Part 2 measures full MoE forward pass at medium scale.")
    log("    P5 Loop  : MLX_MOE_EP_LOCAL_FFN=loop    (sequential per-expert)")
    log("    P5.5 Btch: MLX_MOE_EP_LOCAL_FFN=batched (3D fused matmul)")
    log("  Speedup = baseline_time / faster_time  (>1.0x means faster).")
    log(f"  Timing: median of {args.iters} iters, {args.warmup} warmup each.")
    log("=" * 72)


if __name__ == "__main__":
    main()
