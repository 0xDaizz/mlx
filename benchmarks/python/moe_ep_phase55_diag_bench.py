#!/usr/bin/env python3
# Copyright © 2026 Apple Inc.

"""Phase 5.5 Diagnostic Benchmark — Expert FFN Method Comparison.

Measures the gap between the current sequential expert loop and optimized
alternatives (batched 3D matmul, gather_mm, ideal dense SwiGLU) to validate
whether batched matmul optimization is worth pursuing.

Kimi K2.5 scale: E=384, top_k=8, D=7168, expert_dim=28672, dtype=float16

Run (single device, no distributed):
    python3 benchmarks/python/moe_ep_phase55_diag_bench.py

Options:
    --warmup N      warmup iterations (default 3)
    --iters N       bench iterations (default 10)
"""

import math
import time

import mlx.core as mx
from mlx.nn.layers.activations import silu

# ── Parameters (Kimi K2.5 scale) ────────────────────────────────────────────

E_total = 384
ws = 2
E_local = E_total // ws  # 192
D = 7168
expert_dim = D * 4  # 28672
top_k = 8
cf = 1.25
dtype = mx.float16
N_values = [1, 8, 64, 128, 256, 512, 1024, 2048, 4096]
warmup_iters = 3
bench_iters = 10


# ── Helpers ──────────────────────────────────────────────────────────────────


def compute_capacity(N):
    return max(1, math.ceil(N * top_k * cf / E_total))


def make_stacked_weights():
    """Create stacked weight tensors for batched matmul."""
    w_gate = mx.random.normal((E_local, D, expert_dim)).astype(dtype)
    w_up = mx.random.normal((E_local, D, expert_dim)).astype(dtype)
    w_down = mx.random.normal((E_local, expert_dim, D)).astype(dtype)
    mx.eval(w_gate, w_up, w_down)
    return w_gate, w_up, w_down


def make_individual_weights():
    """Create individual expert weight tuples (simulating Expert modules)."""
    experts = []
    for _ in range(E_local):
        wg = mx.random.normal((D, expert_dim)).astype(dtype)
        wu = mx.random.normal((D, expert_dim)).astype(dtype)
        wd = mx.random.normal((expert_dim, D)).astype(dtype)
        experts.append((wg, wu, wd))
    mx.eval(*[w for e in experts for w in e])
    return experts


def median(lst):
    s = sorted(lst)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


def p90(lst):
    s = sorted(lst)
    idx = int(len(s) * 0.9)
    return s[min(idx, len(s) - 1)]


# ── Benchmark functions ──────────────────────────────────────────────────────


def bench_loop(dispatched, experts, iters):
    """Benchmark: sequential expert loop (current implementation)."""
    times = []
    for _ in range(iters):
        mx.eval(dispatched)
        t0 = time.perf_counter()
        outputs = []
        for i, (wg, wu, wd) in enumerate(experts):
            x = dispatched[i]
            h = silu(x @ wg) * (x @ wu)
            outputs.append(h @ wd)
        result = mx.stack(outputs, axis=0)
        mx.eval(result)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return times


def bench_batched_3d(dispatched, w_gate, w_up, w_down, iters):
    """Benchmark: batched 3D matmul."""
    times = []
    for _ in range(iters):
        mx.eval(dispatched)
        t0 = time.perf_counter()
        gate = dispatched @ w_gate  # [E, cap, expert_dim]
        up = dispatched @ w_up
        h = silu(gate) * up
        out = h @ w_down  # [E, cap, D]
        mx.eval(out)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return times


def bench_gather_mm(dispatched, w_gate, w_up, w_down, iters):
    """Benchmark: gather_mm approach (3D batched, no indices needed)."""
    times = []
    for _ in range(iters):
        mx.eval(dispatched)
        t0 = time.perf_counter()
        gate = mx.gather_mm(dispatched, w_gate)
        up = mx.gather_mm(dispatched, w_up)
        h = silu(gate) * up
        out = mx.gather_mm(h, w_down)
        mx.eval(out)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return times


def bench_ideal_dense(N, iters):
    """Benchmark: ideal dense SwiGLU (theoretical best, single matmul)."""
    tokens_per_rank = N * top_k // ws
    x = mx.random.normal((tokens_per_rank, D)).astype(dtype)
    wg = mx.random.normal((D, expert_dim)).astype(dtype)
    wu = mx.random.normal((D, expert_dim)).astype(dtype)
    wd = mx.random.normal((expert_dim, D)).astype(dtype)
    mx.eval(x, wg, wu, wd)

    times = []
    for _ in range(iters):
        mx.eval(x)
        t0 = time.perf_counter()
        h = silu(x @ wg) * (x @ wu)
        out = h @ wd
        mx.eval(out)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return times


def compute_empty_expert_ratio(N):
    """Compute fraction of local experts that receive 0 tokens (random routing)."""
    cap = compute_capacity(N)
    expert_indices = mx.random.randint(0, E_total, shape=(N, top_k))
    mx.eval(expert_indices)
    idx_list = expert_indices.tolist()

    counts = [0] * E_local
    for n in range(N):
        for k in range(top_k):
            eid = idx_list[n][k]
            if eid < E_local:
                counts[eid % E_local] += 1

    empty = sum(1 for c in counts if c == 0)
    return empty / E_local


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    print("=" * 160)
    print("Phase 5.5 Diagnostic Benchmark — Expert FFN Method Comparison")
    print(
        f"E_total={E_total} E_local={E_local} D={D} expert_dim={expert_dim} "
        f"top_k={top_k} cf={cf} dtype={dtype}"
    )
    print(f"warmup={warmup_iters} bench={bench_iters}")
    print("=" * 160)

    print("\nCreating weights...")
    w_gate, w_up, w_down = make_stacked_weights()
    experts = make_individual_weights()
    print("Weights created.")

    print(
        f"\n{'N':>6} | {'cap':>5} | {'cap_tot':>7} | "
        f"{'Loop(ms)':>10} | {'Batch3D(ms)':>12} | {'GatherMM(ms)':>13} | {'Ideal(ms)':>10} | "
        f"{'Loop/Ideal':>10} | {'Batch/Ideal':>11} | {'GathMM/Ideal':>12} | "
        f"{'Loop-Batch(ms)':>14} | {'Empty%':>7}"
    )
    print("-" * 160)

    results = []

    for N in N_values:
        cap = compute_capacity(N)
        cap_total = ws * cap

        # Create dispatched input [E_local, cap_total, D]
        dispatched = mx.random.normal((E_local, cap_total, D)).astype(dtype)
        mx.eval(dispatched)

        # Warmup all methods
        for _ in range(warmup_iters):
            outputs = []
            for i, (wg, wu, wd) in enumerate(experts):
                x = dispatched[i]
                outputs.append(silu(x @ wg) * (x @ wu) @ wd)
            mx.eval(mx.stack(outputs))

            mx.eval(silu(dispatched @ w_gate) * (dispatched @ w_up) @ w_down)

        # Benchmark each method
        t_loop = bench_loop(dispatched, experts, bench_iters)
        t_batch = bench_batched_3d(dispatched, w_gate, w_up, w_down, bench_iters)

        try:
            t_gmm = bench_gather_mm(dispatched, w_gate, w_up, w_down, bench_iters)
            gmm_med = median(t_gmm)
        except Exception as e:
            print(f"  [gather_mm failed for N={N}: {e}]")
            t_gmm = None
            gmm_med = float("nan")

        t_ideal = bench_ideal_dense(N, bench_iters)

        empty_ratio = compute_empty_expert_ratio(N)

        loop_med = median(t_loop)
        batch_med = median(t_batch)
        ideal_med = median(t_ideal)

        loop_ratio = loop_med / ideal_med if ideal_med > 0 else float("nan")
        batch_ratio = batch_med / ideal_med if ideal_med > 0 else float("nan")
        gmm_ratio = gmm_med / ideal_med if ideal_med > 0 else float("nan")
        gap_ms = loop_med - batch_med

        results.append(
            {
                "N": N,
                "cap": cap,
                "loop_med": loop_med,
                "batch_med": batch_med,
                "gmm_med": gmm_med,
                "ideal_med": ideal_med,
                "loop_ratio": loop_ratio,
                "batch_ratio": batch_ratio,
                "gap_ms": gap_ms,
                "empty_ratio": empty_ratio,
            }
        )

        print(
            f"{N:>6} | {cap:>5} | {cap_total:>7} | "
            f"{loop_med:>10.2f} | {batch_med:>12.2f} | {gmm_med:>13.2f} | {ideal_med:>10.2f} | "
            f"{loop_ratio:>10.2f}x | {batch_ratio:>11.2f}x | {gmm_ratio:>12.2f}x | "
            f"{gap_ms:>14.2f} | {empty_ratio:>6.1%}"
        )

    # ── Go/No-Go Analysis ────────────────────────────────────────────────────

    print("\n" + "=" * 160)
    print("Go/No-Go Analysis")
    print("=" * 160)

    # Check relative criterion: loop/ideal >= 1.15 (15% gap)
    prefill_results = [r for r in results if r["N"] >= 256]
    if prefill_results:
        avg_loop_ratio = sum(r["loop_ratio"] for r in prefill_results) / len(
            prefill_results
        )
        avg_batch_ratio = sum(r["batch_ratio"] for r in prefill_results) / len(
            prefill_results
        )
        relative_pass = avg_loop_ratio >= 1.15
        print(
            f"\n  Relative criterion (prefill N>=256): loop/ideal avg = {avg_loop_ratio:.2f}x"
        )
        print(f"    Threshold: >= 1.15x  -->  {'PASS' if relative_pass else 'FAIL'}")
        print(f"    Batched/ideal avg = {avg_batch_ratio:.2f}x")

    # Check absolute criterion: loop-batch gap >= 5ms at N=1024
    r1024 = next((r for r in results if r["N"] == 1024), None)
    if r1024:
        abs_pass = r1024["gap_ms"] >= 5.0
        print(f"\n  Absolute criterion (N=1024): loop - batch = {r1024['gap_ms']:.2f} ms")
        print(f"    Threshold: >= 5ms  -->  {'PASS' if abs_pass else 'FAIL'}")

    # Overall verdict
    if prefill_results and r1024:
        if relative_pass and abs_pass:
            verdict = "GO — Batched matmul optimization is worthwhile"
        elif relative_pass or abs_pass:
            verdict = "MARGINAL — One criterion met; consider cost/benefit"
        else:
            verdict = "NO-GO — Both criteria unmet; need different strategy"
        print(f"\n  Verdict: {verdict}")

    print("\n" + "=" * 160)
    print("Interpretation guide:")
    print("  Loop/Ideal  = overhead of sequential Python expert loop vs theoretical best")
    print("  Batch/Ideal = overhead of batched 3D matmul vs theoretical best")
    print("  Loop-Batch  = absolute time saved by switching from loop to batched")
    print("  Empty%      = fraction of experts with 0 tokens (wasted compute in batched)")
    print("=" * 160)


if __name__ == "__main__":
    main()
