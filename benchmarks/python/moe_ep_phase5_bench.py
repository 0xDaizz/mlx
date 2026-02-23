#!/usr/bin/env python3
# Copyright © 2026 Apple Inc.

"""Phase 5 Inference Productionization benchmark — MoE Expert Parallelism.

Tests Phase 5 features: auto backend selection, warmup API, stats API, and soak testing.

Test suites:
  1. Auto vs CPU vs Metal — dispatch+combine across N=1..4096, comparing latencies.
  2. Warmup test — first-call latency with and without moe_ep_warmup().
  3. Stats dump — print moe_ep_stats() metrics after the benchmark.
  4. Soak mode — repeated iterations with backend="auto", check for NaN/inf.

Kimi K2.5 scale defaults: E=384, top_k=8, D=7168, capacity_factor=1.25, dtype=float16

Run:
    mlx.launch --backend jaccl --hostfile hosts.json -- \\
        python3 benchmarks/python/moe_ep_phase5_bench.py

Options:
    --warmup N      warmup iterations before timing (default 10)
    --iters N       timing iterations (default 30)
    --soak N        soak test iterations, 0 = skip (default 0)
    --E N           total number of experts (default 384)
    --D N           hidden dimension (default 7168)
    --top_k N       top-k experts per token (default 8)
    --cf F          capacity factor for dynamic cap (default 1.25)
    --capacity N    fixed per-expert capacity; 0 = dynamic (default 0)
"""

import argparse
import math
import subprocess
import sys
import time
import traceback

import numpy as np

import mlx.core as mx


# ── helpers ─────────────────────────────────────────────────────────────────


def time_fn(fn, warmup, iters):
    """Time a function, returning (median_ms, p90_ms, min_ms, all_times).

    CRITICAL: mx.synchronize() is called INSIDE the warmup loop to ensure
    each warmup iteration fully completes before the next one starts.
    Without this, blocking_sendrecv can return garbage data.
    """
    for _ in range(warmup):
        y = fn()
        if isinstance(y, (tuple, list)):
            mx.eval(*y)
        else:
            mx.eval(y)
        mx.synchronize()

    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        y = fn()
        if isinstance(y, (tuple, list)):
            mx.eval(*y)
        else:
            mx.eval(y)
        mx.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return (
        float(np.median(ts)),
        float(np.percentile(ts, 90)),
        float(np.min(ts)),
        ts,
    )


def time_fn_single(fn):
    """Time a single call (no warmup), returning elapsed_ms."""
    t0 = time.perf_counter()
    y = fn()
    if isinstance(y, (tuple, list)):
        mx.eval(*y)
    else:
        mx.eval(y)
    mx.synchronize()
    return (time.perf_counter() - t0) * 1e3


def barrier(group):
    """Synchronize ranks."""
    s = mx.distributed.all_sum(mx.array(0.0), group=group)
    mx.eval(s)
    mx.synchronize()


def geomean(values):
    """Compute geometric mean of a list of positive values, ignoring NaN."""
    valid = [v for v in values if v is not None and not np.isnan(v) and v > 0]
    if not valid:
        return float("nan")
    return float(np.exp(np.mean(np.log(valid))))


def compute_capacity(N, top_k, cf, E):
    """Compute per-expert capacity."""
    return max(1, math.ceil(N * top_k * cf / E))


def check_nan_inf(arr):
    """Check if array contains NaN or Inf values. Returns (has_nan, has_inf)."""
    flat = arr.reshape(-1).astype(mx.float32)
    mx.eval(flat)
    has_nan = bool(mx.any(mx.isnan(flat)).item())
    has_inf = bool(mx.any(mx.isinf(flat)).item())
    return has_nan, has_inf


# ── formatting ──────────────────────────────────────────────────────────────


def fmt_ms(v, width=9):
    """Format milliseconds to string, handling NaN/None."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return f"{'N/A':^{width}}"
    return f"{v:>{width}.2f}"


def fmt_speedup(v, width=7):
    """Format speedup ratio, handling NaN/None."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return f"{'N/A':^{width}}"
    return f"{v:>{width}.2f}x"


# ── table printing ──────────────────────────────────────────────────────────


def print_table(title, headers, rows, col_widths=None, col_formats=None):
    """Print a formatted table.

    Args:
        title: table title string
        headers: list of column header strings
        rows: list of lists (one per row)
        col_widths: list of int widths (default 10 per column)
        col_formats: list of format types: "int", "ms", "speedup", "str"
    """
    if not rows:
        return

    n_cols = len(headers)
    if col_widths is None:
        col_widths = [10] * n_cols
    if col_formats is None:
        col_formats = ["ms"] * n_cols

    print(f"\n{title}")

    # Build header / separator
    hdr_cells = []
    seps = []
    for i, hdr in enumerate(headers):
        w = col_widths[i]
        hdr_cells.append(f" {hdr:^{w}} ")
        seps.append("-" * (w + 2))

    print("+" + "+".join(seps) + "+")
    print("|" + "|".join(hdr_cells) + "|")
    print("+" + "+".join(seps) + "+")

    for row in rows:
        cells = []
        for i, val in enumerate(row):
            w = col_widths[i]
            fmt = col_formats[i]
            if fmt == "int":
                cells.append(f" {val:>{w}} ")
            elif fmt == "str":
                cells.append(f" {str(val):>{w}} ")
            elif fmt == "speedup":
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    cells.append(f" {'N/A':^{w}} ")
                else:
                    s = f"{val:.2f}x"
                    cells.append(f" {s:>{w}} ")
            else:  # ms
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    cells.append(f" {'N/A':^{w}} ")
                else:
                    cells.append(f" {val:>{w}.2f} ")
        print("|" + "|".join(cells) + "|")

    print("+" + "+".join(seps) + "+")


# ── JACCL RDMA warmup ──────────────────────────────────────────────────────


def jaccl_warmup(group, dtype, rank):
    """Warm up JACCL RDMA to prevent garbage data on first calls."""
    if rank == 0:
        print("  Warming up JACCL RDMA ...", flush=True)
    warmup_buf = mx.random.normal((64, 64)).astype(dtype)
    mx.eval(warmup_buf)
    for _ in range(5):
        warmup_buf = mx.distributed.all_sum(warmup_buf, group=group)
        mx.eval(warmup_buf)
        mx.synchronize()
    warmup_a2a = mx.random.normal((group.size(), 32)).astype(dtype)
    mx.eval(warmup_a2a)
    for _ in range(3):
        warmup_a2a = mx.distributed.all_to_all(warmup_a2a, group=group)
        mx.eval(warmup_a2a)
        mx.synchronize()
    barrier(group)
    if rank == 0:
        print("  JACCL warmup done.", flush=True)


# ── Suite 1: Auto vs CPU vs Metal comparison ─────────────────────────────


def make_dispatch_combine_fn(tokens, expert_indices, weights_f32,
                             E, capacity, group, backend):
    """Create a closure that runs dispatch+combine with the given backend."""
    def fn():
        dispatched, route_idx = mx.distributed.moe_dispatch_exchange(
            tokens, expert_indices,
            num_experts=E, capacity=capacity,
            group=group, backend=backend,
        )
        route_idx = mx.stop_gradient(route_idx)
        combined = mx.distributed.moe_combine_exchange(
            dispatched, route_idx, weights_f32, tokens,
            num_experts=E, capacity=capacity,
            group=group, backend=backend,
        )
        return combined
    return fn


def run_backend_comparison(group, E, D, top_k, cf, capacity, dtype,
                           warmup, iters, rank):
    """Suite 1: Compare auto, cpu, metal backends across N values.

    Args:
        capacity: 0 = dynamic (compute per-N using cf), >0 = fixed.
    """
    N_VALUES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

    has_cpp = hasattr(mx.distributed, "moe_dispatch_exchange")
    if not has_cpp:
        if rank == 0:
            print("\n[SKIP] Suite 1: moe_dispatch_exchange not available.")
        return

    backends = ["cpu", "metal", "auto"]

    if rank == 0:
        print("\n" + "=" * 90)
        print("Suite 1: Auto vs CPU vs Metal Backend Comparison")
        print("=" * 90)

    # Collect results: list of (N, cap, backend, med, p90, min)
    all_results = []

    for N in N_VALUES:
        barrier(group)

        # Resolve capacity for this N
        if capacity == 0:
            cap_n = compute_capacity(N, top_k, cf, E)
        else:
            cap_n = capacity
        # Synchronize capacity across ranks
        cap_arr = mx.array(cap_n, dtype=mx.int32)
        cap_arr = mx.distributed.all_max(cap_arr, group=group)
        mx.eval(cap_arr)
        cap_n = cap_arr.item()

        # Allocate inputs
        tokens = mx.random.normal((N, D)).astype(dtype)
        expert_indices = mx.random.randint(0, E, shape=(N, top_k)).astype(mx.int32)
        weights_raw = mx.random.normal((N, top_k))
        weights = mx.softmax(weights_raw, axis=-1).astype(mx.float32)
        mx.eval(tokens, expert_indices, weights)

        if rank == 0:
            print(f"  N={N:>5d} cap={cap_n:>4d} ...", end="", flush=True)

        for backend in backends:
            barrier(group)
            try:
                fn = make_dispatch_combine_fn(
                    tokens, expert_indices, weights,
                    E, cap_n, group, backend,
                )
                med_ms, p90_ms, min_ms, _ = time_fn(fn, warmup, iters)
                all_results.append((N, cap_n, backend, med_ms, p90_ms, min_ms))
            except Exception as ex:
                all_results.append((N, cap_n, backend, float("nan"), float("nan"), float("nan")))
                if rank == 0:
                    print(f" [{backend} ERR: {ex}]", end="", flush=True)

        if rank == 0:
            print(" done", flush=True)

    # Print results table on rank 0
    if rank == 0:
        headers = ["N", "cap", "backend", "med_ms", "p90_ms", "min_ms"]
        col_widths = [6, 6, 8, 10, 10, 10]
        col_formats = ["int", "int", "str", "ms", "ms", "ms"]
        rows = []
        for N, cap_n, backend, med_ms, p90_ms, min_ms in all_results:
            rows.append([N, cap_n, backend, med_ms, p90_ms, min_ms])
        print_table(
            "Backend Comparison — dispatch+combine latency (ms)",
            headers, rows, col_widths, col_formats,
        )

        # Condensed side-by-side table
        print_condensed_comparison(all_results, N_VALUES, backends)


def print_condensed_comparison(all_results, n_values, backends):
    """Print a condensed N x backend table with speedup columns."""
    # Organize results into dict: (N, backend) -> med_ms
    lookup = {}
    cap_lookup = {}
    for N, cap_n, backend, med_ms, _, _ in all_results:
        lookup[(N, backend)] = med_ms
        cap_lookup[N] = cap_n

    headers = ["N", "cap", "cpu_ms", "metal_ms", "auto_ms", "cpu/mtl", "auto pick"]
    col_widths = [6, 6, 9, 9, 9, 8, 10]
    col_formats = ["int", "int", "ms", "ms", "ms", "speedup", "str"]

    rows = []
    cpu_vs_mtl_all = []

    for N in n_values:
        cpu_t = lookup.get((N, "cpu"), float("nan"))
        metal_t = lookup.get((N, "metal"), float("nan"))
        auto_t = lookup.get((N, "auto"), float("nan"))

        # cpu/metal speedup
        if (not np.isnan(cpu_t) and not np.isnan(metal_t)
                and metal_t > 0 and cpu_t > 0):
            ratio = cpu_t / metal_t
            cpu_vs_mtl_all.append(ratio)
        else:
            ratio = float("nan")

        # Determine which backend auto likely picked
        if not np.isnan(auto_t) and not np.isnan(cpu_t) and not np.isnan(metal_t):
            # auto should be close to whichever is faster
            if abs(auto_t - cpu_t) < abs(auto_t - metal_t):
                auto_pick = "cpu"
            else:
                auto_pick = "metal"
        else:
            auto_pick = "?"

        rows.append([N, cap_lookup.get(N, 0), cpu_t, metal_t, auto_t, ratio, auto_pick])

    print_table(
        "\nCondensed Comparison (median_ms)",
        headers, rows, col_widths, col_formats,
    )

    # Geomean summary
    if cpu_vs_mtl_all:
        # Split by decode vs prefill
        decode_n = {1, 2, 4, 8, 16, 32, 64}
        prefill_n = {256, 512, 1024, 2048, 4096}

        decode_ratios = []
        prefill_ratios = []

        for N in n_values:
            cpu_t = lookup.get((N, "cpu"), float("nan"))
            metal_t = lookup.get((N, "metal"), float("nan"))
            if (not np.isnan(cpu_t) and not np.isnan(metal_t)
                    and metal_t > 0 and cpu_t > 0):
                r = cpu_t / metal_t
                if N in decode_n:
                    decode_ratios.append(r)
                elif N in prefill_n:
                    prefill_ratios.append(r)

        print(f"\n  cpu/metal geomean (all):     {geomean(cpu_vs_mtl_all):.2f}x")
        if decode_ratios:
            print(f"  cpu/metal geomean (decode):  {geomean(decode_ratios):.2f}x")
        if prefill_ratios:
            print(f"  cpu/metal geomean (prefill): {geomean(prefill_ratios):.2f}x")
        print(f"  (>1 = Metal faster, <1 = CPU faster)")


# ── Suite 2: Warmup test ─────────────────────────────────────────────────


def run_warmup_test(group, E, D, top_k, cf, capacity, dtype, rank):
    """Suite 2: Measure first-call latency with and without moe_ep_warmup.

    Args:
        capacity: 0 = dynamic (compute from cf), >0 = fixed.
    """
    has_cpp = hasattr(mx.distributed, "moe_dispatch_exchange")
    has_warmup_api = hasattr(mx.distributed, "moe_ep_warmup")

    if not has_cpp:
        if rank == 0:
            print("\n[SKIP] Suite 2: moe_dispatch_exchange not available.")
        return

    if rank == 0:
        print("\n" + "=" * 90)
        print("Suite 2: Warmup API Test — First-call Latency")
        print("=" * 90)

    if not has_warmup_api:
        if rank == 0:
            print("  [INFO] moe_ep_warmup() not available — "
                  "running cold-start measurement only.\n")

    N_TEST = 256  # representative mid-batch size

    # Resolve capacity for N_TEST
    if capacity == 0:
        capacity = compute_capacity(N_TEST, top_k, cf, E)
    cap_arr = mx.array(capacity, dtype=mx.int32)
    cap_arr = mx.distributed.all_max(cap_arr, group=group)
    mx.eval(cap_arr)
    capacity = cap_arr.item()

    # Allocate test inputs
    tokens = mx.random.normal((N_TEST, D)).astype(dtype)
    expert_indices = mx.random.randint(0, E, shape=(N_TEST, top_k)).astype(mx.int32)
    weights_raw = mx.random.normal((N_TEST, top_k))
    weights = mx.softmax(weights_raw, axis=-1).astype(mx.float32)
    mx.eval(tokens, expert_indices, weights)

    results = []

    # -- Test A: cold start (no warmup API), measure first 5 calls --
    for backend in ["cpu", "metal"]:
        barrier(group)
        if rank == 0:
            print(f"  Measuring cold-start ({backend}) ...", flush=True)

        cold_times = []
        for i in range(5):
            # Use fresh tensors each iteration to avoid caching effects
            t_in = mx.random.normal((N_TEST, D)).astype(dtype)
            ei_in = mx.random.randint(0, E, shape=(N_TEST, top_k)).astype(mx.int32)
            w_in = mx.softmax(mx.random.normal((N_TEST, top_k)), axis=-1).astype(mx.float32)
            mx.eval(t_in, ei_in, w_in)
            barrier(group)

            fn = make_dispatch_combine_fn(
                t_in, ei_in, w_in, E, capacity, group, backend,
            )
            elapsed = time_fn_single(fn)
            cold_times.append(elapsed)

        results.append((f"{backend}_cold_1st", cold_times[0]))
        results.append((f"{backend}_cold_2nd", cold_times[1]))
        results.append((f"{backend}_cold_avg5", float(np.mean(cold_times))))

    # -- Test B: with warmup API (if available) --
    if has_warmup_api:
        for backend in ["cpu", "metal"]:
            barrier(group)
            if rank == 0:
                print(f"  Calling moe_ep_warmup() then measuring ({backend}) ...",
                      flush=True)

            # Call the warmup API
            mx.distributed.moe_ep_warmup(
                group=group,
                num_experts=E,
                capacity=capacity,
                hidden_dim=D,
            )
            mx.synchronize()

            warm_times = []
            for i in range(5):
                t_in = mx.random.normal((N_TEST, D)).astype(dtype)
                ei_in = mx.random.randint(0, E, shape=(N_TEST, top_k)).astype(mx.int32)
                w_in = mx.softmax(mx.random.normal((N_TEST, top_k)), axis=-1).astype(mx.float32)
                mx.eval(t_in, ei_in, w_in)
                barrier(group)

                fn = make_dispatch_combine_fn(
                    t_in, ei_in, w_in, E, capacity, group, backend,
                )
                elapsed = time_fn_single(fn)
                warm_times.append(elapsed)

            results.append((f"{backend}_warm_1st", warm_times[0]))
            results.append((f"{backend}_warm_2nd", warm_times[1]))
            results.append((f"{backend}_warm_avg5", float(np.mean(warm_times))))

    # Print results
    if rank == 0:
        headers = ["test", "latency_ms"]
        col_widths = [20, 12]
        col_formats = ["str", "ms"]
        rows = [[label, val] for label, val in results]
        print_table(
            f"Warmup Test Results (N={N_TEST})",
            headers, rows, col_widths, col_formats,
        )

        # Print improvement summary if warmup API was available
        if has_warmup_api:
            for backend in ["cpu", "metal"]:
                cold_1st = dict(results).get(f"{backend}_cold_1st", float("nan"))
                warm_1st = dict(results).get(f"{backend}_warm_1st", float("nan"))
                if not np.isnan(cold_1st) and not np.isnan(warm_1st) and warm_1st > 0:
                    improvement = cold_1st / warm_1st
                    print(f"  {backend}: cold_1st/warm_1st = {improvement:.2f}x improvement")


# ── Suite 3: Stats dump ──────────────────────────────────────────────────


def run_stats_dump(group, rank):
    """Suite 3: Print moe_ep_stats() metrics."""
    has_stats = hasattr(mx.distributed, "moe_ep_stats")

    if rank == 0:
        print("\n" + "=" * 90)
        print("Suite 3: MoE EP Statistics")
        print("=" * 90)

    if not has_stats:
        if rank == 0:
            print("  [SKIP] moe_ep_stats() not available in this build.\n")
        return

    barrier(group)
    stats = mx.distributed.moe_ep_stats()

    if rank == 0:
        if isinstance(stats, dict):
            # Known metric keys (print in order if present)
            known_keys = [
                "dispatch_calls",
                "combine_calls",
                "cpu_backend_calls",
                "metal_backend_calls",
                "auto_backend_calls",
                "total_tokens_dispatched",
                "total_tokens_combined",
                "total_dispatch_ms",
                "total_combine_ms",
                "avg_dispatch_ms",
                "avg_combine_ms",
                "nan_count",
                "inf_count",
            ]
            max_key_len = max(
                len(k) for k in (list(stats.keys()) + known_keys) if k
            )

            for key in known_keys:
                if key in stats:
                    val = stats[key]
                    print(f"  {key:<{max_key_len}} : {val}")

            # Print any extra keys not in known_keys
            for key, val in sorted(stats.items()):
                if key not in known_keys:
                    print(f"  {key:<{max_key_len}} : {val}")
        else:
            # If it returns something else (e.g., a string), just print it
            print(f"  {stats}")
        print()


# ── Suite 4: Soak test ───────────────────────────────────────────────────


def run_soak_test(group, E, D, top_k, cf, capacity, dtype,
                  soak_iters, rank):
    """Suite 4: Soak test — run N iterations with backend='auto',
    check outputs for NaN/Inf.

    Args:
        capacity: 0 = dynamic (compute per-N from cf), >0 = fixed.
    """
    has_cpp = hasattr(mx.distributed, "moe_dispatch_exchange")
    if not has_cpp:
        if rank == 0:
            print("\n[SKIP] Suite 4: moe_dispatch_exchange not available.")
        return

    if soak_iters <= 0:
        if rank == 0:
            print("\n[SKIP] Suite 4: --soak 0 (soak test disabled).")
        return

    if rank == 0:
        print("\n" + "=" * 90)
        print(f"Suite 4: Soak Test — {soak_iters} iterations, backend='auto'")
        print("=" * 90)

    # Vary N across iterations to stress-test different paths
    N_CHOICES = [1, 4, 16, 64, 128, 256, 512, 1024]

    nan_failures = 0
    inf_failures = 0
    total_checks = 0
    error_details = []  # list of (iter, N, issue)
    iter_times = []

    # Determine which backend string to use
    # If auto is supported, use it; otherwise fall back to cpu
    test_backend = "auto"

    for it in range(soak_iters):
        N = N_CHOICES[it % len(N_CHOICES)]

        # Resolve capacity for this N
        if capacity == 0:
            cap_n = compute_capacity(N, top_k, cf, E)
        else:
            cap_n = capacity

        tokens = mx.random.normal((N, D)).astype(dtype)
        expert_indices = mx.random.randint(0, E, shape=(N, top_k)).astype(mx.int32)
        weights_raw = mx.random.normal((N, top_k))
        weights = mx.softmax(weights_raw, axis=-1).astype(mx.float32)
        mx.eval(tokens, expert_indices, weights)

        barrier(group)
        t0 = time.perf_counter()

        try:
            fn = make_dispatch_combine_fn(
                tokens, expert_indices, weights,
                E, cap_n, group, test_backend,
            )
            result = fn()
            mx.eval(result)
            mx.synchronize()
            elapsed = (time.perf_counter() - t0) * 1e3
            iter_times.append(elapsed)

            # Check for NaN/Inf
            has_nan, has_inf = check_nan_inf(result)
            total_checks += 1

            if has_nan:
                nan_failures += 1
                error_details.append((it, N, "NaN"))
            if has_inf:
                inf_failures += 1
                error_details.append((it, N, "Inf"))

        except Exception as ex:
            elapsed = (time.perf_counter() - t0) * 1e3
            iter_times.append(elapsed)
            error_details.append((it, N, f"Exception: {ex}"))
            total_checks += 1

        # Progress report every 10% or at least every 100 iterations
        report_interval = max(1, soak_iters // 10)
        if rank == 0 and ((it + 1) % report_interval == 0 or it == soak_iters - 1):
            print(
                f"  [{it + 1:>{len(str(soak_iters))}}/{soak_iters}] "
                f"NaN={nan_failures} Inf={inf_failures} "
                f"last_ms={elapsed:.2f}",
                flush=True,
            )

    # Print summary
    if rank == 0:
        print(f"\n  --- Soak Test Summary ---")
        print(f"  Total iterations:  {soak_iters}")
        print(f"  Total checks:      {total_checks}")
        print(f"  NaN failures:      {nan_failures}")
        print(f"  Inf failures:      {inf_failures}")
        total_failures = nan_failures + inf_failures + (
            len(error_details) - nan_failures - inf_failures
        )
        print(f"  Total failures:    {total_failures}")

        if iter_times:
            print(f"  Latency mean:      {np.mean(iter_times):.2f} ms")
            print(f"  Latency std:       {np.std(iter_times):.2f} ms")
            print(f"  Latency min:       {np.min(iter_times):.2f} ms")
            print(f"  Latency max:       {np.max(iter_times):.2f} ms")
            print(f"  Latency p99:       {np.percentile(iter_times, 99):.2f} ms")

        if error_details:
            print(f"\n  First 10 errors:")
            for it_num, n_val, issue in error_details[:10]:
                print(f"    iter={it_num} N={n_val}: {issue}")

        status = "PASS" if total_failures == 0 else "FAIL"
        print(f"\n  Soak result: {status}")
        print()


# ── main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Phase 5 Inference Productionization benchmark — MoE EP"
    )
    parser.add_argument("--warmup", type=int, default=10,
                        help="Warmup iterations before timing (default 10)")
    parser.add_argument("--iters", type=int, default=30,
                        help="Timing iterations (default 30)")
    parser.add_argument("--soak", type=int, default=0,
                        help="Soak test iterations, 0 = skip (default 0)")
    parser.add_argument("--E", type=int, default=384,
                        help="Total number of experts (default 384)")
    parser.add_argument("--D", type=int, default=7168,
                        help="Hidden dimension (default 7168)")
    parser.add_argument("--top_k", type=int, default=8,
                        help="Top-k experts per token (default 8)")
    parser.add_argument("--cf", type=float, default=1.25,
                        help="Capacity factor (default 1.25)")
    parser.add_argument("--capacity", type=int, default=0,
                        help="Fixed per-expert capacity; 0 = dynamic (default 0)")
    args = parser.parse_args()

    group = mx.distributed.init()
    rank = group.rank()
    ws = group.size()

    if ws < 2:
        raise RuntimeError(
            f"This benchmark requires >=2 ranks, got {ws}"
        )

    E = args.E
    D = args.D
    top_k = args.top_k
    cf = args.cf
    fixed_capacity = args.capacity  # 0 = dynamic
    warmup = args.warmup
    iters = args.iters
    dtype = mx.float16

    # Feature availability checks
    has_cpp = hasattr(mx.distributed, "moe_dispatch_exchange")
    has_warmup_api = hasattr(mx.distributed, "moe_ep_warmup")
    has_stats_api = hasattr(mx.distributed, "moe_ep_stats")

    # Print header on rank 0
    if rank == 0:
        try:
            commit = (
                subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
                .decode()
                .strip()
            )
        except Exception:
            commit = "unknown"

        print("=" * 90)
        cap_str = f"dynamic(cf={cf})" if fixed_capacity == 0 else f"fixed={fixed_capacity}"
        print(
            f"Phase 5 Benchmark — Inference Productionization  "
            f"(E={E}, top_k={top_k}, D={D}, cap={cap_str})"
        )
        print(
            f"  commit={commit}  world_size={ws}  "
            f"dtype=float16  warmup={warmup}  iters={iters}  soak={args.soak}"
        )
        print(f"  API availability:")
        print(f"    moe_dispatch_exchange: {'YES' if has_cpp else 'NO'}")
        print(f"    moe_ep_warmup:         {'YES' if has_warmup_api else 'NO'}")
        print(f"    moe_ep_stats:          {'YES' if has_stats_api else 'NO'}")
        print("=" * 90)

    # ── JACCL warmup (always needed) ────────────────────────────────────
    jaccl_warmup(group, dtype, rank)

    # ── Resolve capacity ─────────────────────────────────────────────────
    if fixed_capacity > 0:
        # Synchronize fixed capacity across ranks
        cap_arr = mx.array(fixed_capacity, dtype=mx.int32)
        cap_arr = mx.distributed.all_max(cap_arr, group=group)
        mx.eval(cap_arr)
        capacity = cap_arr.item()
        if rank == 0:
            E_local = E // ws
            print(f"  Fixed capacity={capacity}, E_local={E_local}")
    else:
        capacity = 0  # sentinel: Suite 1 computes per-N
        if rank == 0:
            E_local = E // ws
            print(f"  Dynamic capacity (cf={cf}), E_local={E_local}")

    # ── Suite 1: Backend comparison ─────────────────────────────────────
    run_backend_comparison(group, E, D, top_k, cf, capacity, dtype,
                           warmup, iters, rank)

    # ── Suite 2: Warmup test ────────────────────────────────────────────
    run_warmup_test(group, E, D, top_k, cf, capacity, dtype, rank)

    # ── Suite 3: Stats dump ─────────────────────────────────────────────
    run_stats_dump(group, rank)

    # ── Suite 4: Soak test ──────────────────────────────────────────────
    run_soak_test(group, E, D, top_k, cf, capacity, dtype,
                  args.soak, rank)

    # ── Final stats dump (after all tests) ──────────────────────────────
    if rank == 0:
        print("\n" + "=" * 90)
        print("Final Statistics (after all tests)")
        print("=" * 90)
    run_stats_dump(group, rank)

    # ── Legend ──────────────────────────────────────────────────────────
    if rank == 0:
        print("=" * 90)
        print("Legend:")
        print("  cpu_ms    = C++ fused dispatch+combine, CPU scatter/gather")
        print("  metal_ms  = C++ fused dispatch+combine, Metal GPU kernels")
        print("  auto_ms   = C++ fused dispatch+combine, auto backend selection")
        print("  cpu/mtl   = cpu_time / metal_time (>1 = Metal faster)")
        print("  auto pick = inferred backend choice for auto mode")
        print(f"  All times in milliseconds (median of {iters} iterations)")
        print("=" * 90)

    barrier(group)


if __name__ == "__main__":
    main()
