#!/usr/bin/env python3
# Copyright © 2026 Apple Inc.

"""Diagnostic benchmark for MoE Expert Parallelism — overhead, crossover, and routing analysis.

Answers three focused questions:
  1. Overhead isolation:  cpu vs auto backend at small N (decode-like)
  2. Fine-grained crossover sweep:  cpu vs metal, find exact crossover N
  3. Overflow and routing analysis:  capacity overflow, remote ratio, utilization

Kimi K2.5 scale defaults: E=384, top_k=8, D=7168, capacity=32, dtype=float16

Run:
    mlx.launch --backend jaccl --hostfile hosts.json -- \
        python3 benchmarks/python/moe_ep_diag_bench.py --warmup 10 --iters 30
"""

import argparse
import math
import subprocess
import time
import traceback

import mlx.core as mx
import numpy as np

# ── helpers ─────────────────────────────────────────────────────────────────


def time_fn(fn, warmup, iters):
    """Time a function, returning (median_ms, p90_ms, all_times).

    CRITICAL: mx.synchronize() is called INSIDE the warmup loop to ensure
    each warmup iteration fully completes before the next one starts.
    Without this, blocking_sendrecv can return garbage data.

    This is the EXACT Phase 4 timing methodology.
    """
    for _ in range(warmup):
        y = fn()
        if isinstance(y, (tuple, list)):
            mx.eval(*y)
        else:
            mx.eval(y)
        mx.synchronize()  # critical: must be inside the loop

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
    return float(np.median(ts)), float(np.percentile(ts, 90)), ts


def time_fn_split(fn_dispatch, fn_combine, warmup, iters):
    """Time dispatch and combine separately, returning per-phase medians.

    Returns: (dispatch_median_ms, combine_median_ms, combined_median_ms)
    """
    # Warmup both together
    for _ in range(warmup):
        d_out = fn_dispatch()
        if isinstance(d_out, (tuple, list)):
            mx.eval(*d_out)
        else:
            mx.eval(d_out)
        mx.synchronize()

        c_out = fn_combine(d_out)
        if isinstance(c_out, (tuple, list)):
            mx.eval(*c_out)
        else:
            mx.eval(c_out)
        mx.synchronize()

    # Measure dispatch only
    dispatch_ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        d_out = fn_dispatch()
        if isinstance(d_out, (tuple, list)):
            mx.eval(*d_out)
        else:
            mx.eval(d_out)
        mx.synchronize()
        dispatch_ts.append((time.perf_counter() - t0) * 1e3)

    # Measure combine only (using last dispatch output)
    d_out = fn_dispatch()
    if isinstance(d_out, (tuple, list)):
        mx.eval(*d_out)
    else:
        mx.eval(d_out)
    mx.synchronize()

    combine_ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        c_out = fn_combine(d_out)
        if isinstance(c_out, (tuple, list)):
            mx.eval(*c_out)
        else:
            mx.eval(c_out)
        mx.synchronize()
        combine_ts.append((time.perf_counter() - t0) * 1e3)

    # Measure combined (dispatch+combine together)
    combined_ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        d_out = fn_dispatch()
        if isinstance(d_out, (tuple, list)):
            mx.eval(*d_out)
        else:
            mx.eval(d_out)
        mx.synchronize()

        c_out = fn_combine(d_out)
        if isinstance(c_out, (tuple, list)):
            mx.eval(*c_out)
        else:
            mx.eval(c_out)
        mx.synchronize()
        combined_ts.append((time.perf_counter() - t0) * 1e3)

    return (
        float(np.median(dispatch_ts)),
        float(np.median(combine_ts)),
        float(np.median(combined_ts)),
    )


def barrier(group):
    """Synchronize ranks."""
    s = mx.distributed.all_sum(mx.array(0.0), group=group)
    mx.eval(s)
    mx.synchronize()


def geomean(values):
    """Compute geometric mean of positive values, ignoring NaN."""
    valid = [v for v in values if v is not None and not np.isnan(v) and v > 0]
    if not valid:
        return float("nan")
    return float(np.exp(np.mean(np.log(valid))))


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


# ── formatting ──────────────────────────────────────────────────────────────


def fmt_ms(v, width=9):
    """Format milliseconds, handling NaN/None."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return f"{'N/A':^{width}}"
    return f"{v:>{width}.2f}"


def fmt_speedup(v, width=7):
    """Format speedup ratio, handling NaN/None."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return f"{'N/A':^{width}}"
    return f"{v:>{width}.2f}x"


def fmt_pct(v, width=7):
    """Format percentage, handling NaN/None."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return f"{'N/A':^{width}}"
    return f"{v:>{width}.1f}%"


# ── table printing ──────────────────────────────────────────────────────────


def print_table(title, headers, rows, col_widths=None, col_formats=None):
    """Print a formatted table.

    col_formats: list of "int", "ms", "speedup", "str", "pct"
    """
    if not rows:
        return

    n_cols = len(headers)
    if col_widths is None:
        col_widths = [10] * n_cols
    if col_formats is None:
        col_formats = ["ms"] * n_cols

    print(f"\n{title}")

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
            elif fmt == "pct":
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    cells.append(f" {'N/A':^{w}} ")
                else:
                    s = f"{val:.1f}%"
                    cells.append(f" {s:>{w}} ")
            else:  # ms
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    cells.append(f" {'N/A':^{w}} ")
                else:
                    cells.append(f" {val:>{w}.2f} ")
        print("|" + "|".join(cells) + "|")

    print("+" + "+".join(seps) + "+")


# ── dispatch/combine factory ────────────────────────────────────────────────


def make_dispatch_fn(tokens, expert_indices, E, capacity, group, backend):
    """Create a closure that runs dispatch only."""

    def fn():
        dispatched, route_idx = mx.distributed.moe_dispatch_exchange(
            tokens,
            expert_indices,
            num_experts=E,
            capacity=capacity,
            group=group,
            backend=backend,
        )
        return dispatched, route_idx

    return fn


def make_combine_fn(weights_f32, tokens, E, capacity, group, backend):
    """Create a closure factory that runs combine given dispatch output."""

    def fn(dispatch_out):
        dispatched, route_idx = dispatch_out
        route_idx = mx.stop_gradient(route_idx)
        combined = mx.distributed.moe_combine_exchange(
            dispatched,
            route_idx,
            weights_f32,
            tokens,
            num_experts=E,
            capacity=capacity,
            group=group,
            backend=backend,
        )
        return combined

    return fn


def make_dispatch_combine_fn(
    tokens, expert_indices, weights_f32, E, capacity, group, backend
):
    """Create a closure that runs dispatch+combine together."""

    def fn():
        dispatched, route_idx = mx.distributed.moe_dispatch_exchange(
            tokens,
            expert_indices,
            num_experts=E,
            capacity=capacity,
            group=group,
            backend=backend,
        )
        route_idx = mx.stop_gradient(route_idx)
        combined = mx.distributed.moe_combine_exchange(
            dispatched,
            route_idx,
            weights_f32,
            tokens,
            num_experts=E,
            capacity=capacity,
            group=group,
            backend=backend,
        )
        return combined

    return fn


# ── Question 1: Overhead isolation ──────────────────────────────────────────


def run_overhead_isolation(group, E, D, top_k, capacity, dtype, warmup, iters, rank):
    """Q1: Compare cpu vs auto overhead at small N values.

    Uses exact Phase 4 timing methodology (time_fn with median + p90).
    Also reports dispatch-only and combine-only timings.
    """
    N_VALUES = [1, 4, 8, 16, 32, 64]

    if rank == 0:
        print("\n" + "=" * 100)
        print("Q1: Overhead Isolation — cpu vs auto at decode-like N")
        print(f"    E={E}, D={D}, top_k={top_k}, capacity={capacity}")
        print("=" * 100)

    rows_combined = []
    rows_split = []

    for N in N_VALUES:
        barrier(group)

        tokens = mx.random.normal((N, D)).astype(dtype)
        expert_indices = mx.random.randint(0, E, shape=(N, top_k)).astype(mx.int32)
        weights_raw = mx.random.normal((N, top_k))
        weights_f32 = mx.softmax(weights_raw, axis=-1).astype(mx.float32)
        mx.eval(tokens, expert_indices, weights_f32)

        if rank == 0:
            print(f"  N={N:>4d} ...", end="", flush=True)

        results = {}  # backend -> (median, p90)

        for backend in ["cpu", "auto"]:
            barrier(group)
            try:
                fn = make_dispatch_combine_fn(
                    tokens,
                    expert_indices,
                    weights_f32,
                    E,
                    capacity,
                    group,
                    backend,
                )
                med, p90, _ = time_fn(fn, warmup, iters)
                results[backend] = (med, p90)
            except Exception as ex:
                results[backend] = (float("nan"), float("nan"))
                if rank == 0:
                    print(f" [{backend} ERR: {ex}]", end="", flush=True)

        # Compute delta
        cpu_med = results["cpu"][0]
        auto_med = results["auto"][0]
        if not np.isnan(cpu_med) and not np.isnan(auto_med) and cpu_med > 0:
            delta_ms = auto_med - cpu_med
            delta_pct = (auto_med / cpu_med - 1.0) * 100.0
        else:
            delta_ms = float("nan")
            delta_pct = float("nan")

        rows_combined.append(
            [
                N,
                cpu_med,
                results["cpu"][1],
                auto_med,
                results["auto"][1],
                delta_ms,
                delta_pct,
            ]
        )

        # Split timings (dispatch-only, combine-only, combined) for cpu
        barrier(group)
        try:
            fn_d = make_dispatch_fn(tokens, expert_indices, E, capacity, group, "cpu")
            fn_c = make_combine_fn(weights_f32, tokens, E, capacity, group, "cpu")
            d_med, c_med, dc_med = time_fn_split(fn_d, fn_c, warmup, iters)
            rows_split.append([N, "cpu", d_med, c_med, dc_med])
        except Exception:
            rows_split.append([N, "cpu", float("nan"), float("nan"), float("nan")])

        # Split timings for auto
        barrier(group)
        try:
            fn_d = make_dispatch_fn(tokens, expert_indices, E, capacity, group, "auto")
            fn_c = make_combine_fn(weights_f32, tokens, E, capacity, group, "auto")
            d_med, c_med, dc_med = time_fn_split(fn_d, fn_c, warmup, iters)
            rows_split.append([N, "auto", d_med, c_med, dc_med])
        except Exception:
            rows_split.append([N, "auto", float("nan"), float("nan"), float("nan")])

        if rank == 0:
            print(" done", flush=True)

    if rank == 0:
        print_table(
            "Q1a: Combined dispatch+combine latency — cpu vs auto (Phase 4 methodology)",
            ["N", "cpu_med", "cpu_p90", "auto_med", "auto_p90", "delta_ms", "delta_%"],
            rows_combined,
            [6, 9, 9, 9, 9, 9, 8],
            ["int", "ms", "ms", "ms", "ms", "ms", "pct"],
        )

        print_table(
            "Q1b: Split timings — dispatch-only, combine-only, combined",
            ["N", "backend", "disp_ms", "comb_ms", "d+c_ms"],
            rows_split,
            [6, 7, 9, 9, 9],
            ["int", "str", "ms", "ms", "ms"],
        )

        # Summary
        deltas = [r[5] for r in rows_combined if not np.isnan(r[5])]
        delta_pcts = [r[6] for r in rows_combined if not np.isnan(r[6])]
        if deltas:
            print(f"\n  auto overhead summary (N=1..64):")
            print(f"    mean delta:  {np.mean(deltas):+.3f} ms")
            print(f"    max delta:   {np.max(deltas):+.3f} ms")
            print(f"    mean delta%: {np.mean(delta_pcts):+.1f}%")


# ── Question 2: Fine-grained crossover sweep ───────────────────────────────


def run_crossover_sweep(group, E, D, top_k, capacity, dtype, warmup, iters, rank):
    """Q2: Find the exact N where Metal becomes faster than CPU.

    Tests a fine-grained grid of N values and reports both backends.
    """
    N_VALUES = [
        64,
        80,
        96,
        112,
        128,
        144,
        160,
        176,
        192,
        208,
        224,
        240,
        256,
        320,
        384,
        512,
    ]

    if rank == 0:
        print("\n" + "=" * 100)
        print("Q2: Fine-Grained Crossover Sweep — cpu vs metal")
        print(f"    E={E}, D={D}, top_k={top_k}, capacity={capacity}")
        print("=" * 100)

    rows = []
    crossover_n = None  # first N where metal is faster

    for N in N_VALUES:
        barrier(group)

        tokens = mx.random.normal((N, D)).astype(dtype)
        expert_indices = mx.random.randint(0, E, shape=(N, top_k)).astype(mx.int32)
        weights_raw = mx.random.normal((N, top_k))
        weights_f32 = mx.softmax(weights_raw, axis=-1).astype(mx.float32)
        mx.eval(tokens, expert_indices, weights_f32)

        if rank == 0:
            print(f"  N={N:>4d} ...", end="", flush=True)

        results = {}

        for backend in ["cpu", "metal"]:
            barrier(group)
            try:
                # Combined dispatch+combine
                fn = make_dispatch_combine_fn(
                    tokens,
                    expert_indices,
                    weights_f32,
                    E,
                    capacity,
                    group,
                    backend,
                )
                med, p90, _ = time_fn(fn, warmup, iters)
                results[backend] = (med, p90)
            except Exception as ex:
                results[backend] = (float("nan"), float("nan"))
                if rank == 0:
                    print(f" [{backend} ERR: {ex}]", end="", flush=True)

        # Also get split timings for deeper analysis
        split_results = {}
        for backend in ["cpu", "metal"]:
            barrier(group)
            try:
                fn_d = make_dispatch_fn(
                    tokens, expert_indices, E, capacity, group, backend
                )
                fn_c = make_combine_fn(weights_f32, tokens, E, capacity, group, backend)
                d_med, c_med, dc_med = time_fn_split(fn_d, fn_c, warmup, iters)
                split_results[backend] = (d_med, c_med)
            except Exception:
                split_results[backend] = (float("nan"), float("nan"))

        cpu_med = results["cpu"][0]
        metal_med = results["metal"][0]

        if not np.isnan(cpu_med) and not np.isnan(metal_med) and metal_med > 0:
            ratio = cpu_med / metal_med
            winner = "metal" if ratio > 1.0 else "cpu"
            if crossover_n is None and ratio > 1.0:
                crossover_n = N
        else:
            ratio = float("nan")
            winner = "?"

        cpu_d = split_results.get("cpu", (float("nan"), float("nan")))[0]
        cpu_c = split_results.get("cpu", (float("nan"), float("nan")))[1]
        mtl_d = split_results.get("metal", (float("nan"), float("nan")))[0]
        mtl_c = split_results.get("metal", (float("nan"), float("nan")))[1]

        rows.append(
            [
                N,
                cpu_med,
                metal_med,
                ratio,
                winner,
                cpu_d,
                cpu_c,
                mtl_d,
                mtl_c,
            ]
        )

        if rank == 0:
            print(f" cpu={cpu_med:.2f} metal={metal_med:.2f} -> {winner}", flush=True)

    if rank == 0:
        print_table(
            "Q2a: Combined dispatch+combine — cpu vs metal",
            ["N", "cpu_ms", "metal_ms", "cpu/mtl", "winner"],
            [[r[0], r[1], r[2], r[3], r[4]] for r in rows],
            [6, 9, 9, 8, 7],
            ["int", "ms", "ms", "speedup", "str"],
        )

        print_table(
            "Q2b: Split dispatch/combine — cpu vs metal",
            ["N", "cpu_disp", "cpu_comb", "mtl_disp", "mtl_comb"],
            [[r[0], r[5], r[6], r[7], r[8]] for r in rows],
            [6, 9, 9, 9, 9],
            ["int", "ms", "ms", "ms", "ms"],
        )

        if crossover_n is not None:
            print(
                f"\n  >>> Crossover point: N={crossover_n} "
                f"(Metal becomes faster at this N)"
            )
        else:
            print(
                f"\n  >>> No crossover found in range "
                f"[{N_VALUES[0]}, {N_VALUES[-1]}] — CPU faster throughout"
            )

        # Compute data sizes at crossover for reference
        E_local = E // group.size()
        ws = group.size()
        if crossover_n is not None:
            bytes_per_elem = 2  # float16
            dispatch_buf_bytes = E_local * (ws * capacity) * D * bytes_per_elem
            token_bytes = crossover_n * D * bytes_per_elem
            print(
                f"      dispatch_buf = {dispatch_buf_bytes / 1024 / 1024:.1f} MB, "
                f"token_data = {token_bytes / 1024 / 1024:.1f} MB"
            )


# ── Question 3: Overflow and routing analysis ───────────────────────────────


def run_routing_analysis(group, E, D, top_k, capacity, dtype, warmup, iters, rank):
    """Q3: Analyze capacity overflow, remote routing ratio, and buffer utilization.

    For each N, runs dispatch and inspects route_indices to compute:
      - overflow_ratio:  fraction of (N * top_k) slots that hit -1 (capacity overflow)
      - remote_ratio:    fraction of valid tokens routed to non-local experts
      - utilization:     fraction of dispatch buffer slots actually filled
    """
    N_VALUES = [1, 64, 256, 512, 1024, 2048, 4096]

    ws = group.size()
    E_local = E // ws
    my_rank = group.rank()
    # Local expert range: [my_rank * E_local, (my_rank + 1) * E_local)
    local_start = my_rank * E_local
    local_end = local_start + E_local

    if rank == 0:
        print("\n" + "=" * 100)
        print("Q3: Overflow and Routing Analysis")
        print(
            f"    E={E}, E_local={E_local}, D={D}, top_k={top_k}, capacity={capacity}"
        )
        print(f"    world_size={ws}, rank={my_rank}")
        print(f"    cap_total = ws * capacity = {ws * capacity}")
        print(f"    dispatch buffer shape: [{E_local}, {ws * capacity}, {D}]")
        print("=" * 100)

    rows = []

    for N in N_VALUES:
        barrier(group)

        tokens = mx.random.normal((N, D)).astype(dtype)
        expert_indices = mx.random.randint(0, E, shape=(N, top_k)).astype(mx.int32)
        weights_raw = mx.random.normal((N, top_k))
        weights_f32 = mx.softmax(weights_raw, axis=-1).astype(mx.float32)
        mx.eval(tokens, expert_indices, weights_f32)

        if rank == 0:
            print(f"  N={N:>5d} ...", end="", flush=True)

        # Run dispatch to get route_indices
        barrier(group)
        try:
            dispatched, route_idx = mx.distributed.moe_dispatch_exchange(
                tokens,
                expert_indices,
                num_experts=E,
                capacity=capacity,
                group=group,
                backend="cpu",
            )
            mx.eval(dispatched, route_idx)
            mx.synchronize()

            # Convert route_idx to numpy for analysis
            ri = np.array(route_idx)  # shape [N, top_k], int32, -1 = overflow
            ei = np.array(expert_indices)  # shape [N, top_k]

            total_slots = N * top_k

            # 1. Overflow ratio: count -1 entries
            overflow_count = int(np.sum(ri == -1))
            overflow_ratio = overflow_count / total_slots * 100.0

            # 2. Remote ratio: among valid (non -1) entries, count those
            #    routed to non-local experts
            valid_mask = ri != -1
            valid_count = int(np.sum(valid_mask))

            if valid_count > 0:
                # Expert indices for valid slots
                valid_experts = ei[valid_mask]
                remote_mask = (valid_experts < local_start) | (
                    valid_experts >= local_end
                )
                remote_count = int(np.sum(remote_mask))
                remote_ratio = remote_count / valid_count * 100.0
            else:
                remote_count = 0
                remote_ratio = float("nan")

            # 3. Utilization: valid slots / total dispatch buffer capacity
            cap_total = ws * capacity
            total_buffer_slots = E_local * cap_total
            utilization = (
                valid_count / total_buffer_slots * 100.0
                if total_buffer_slots > 0
                else 0.0
            )

            # Also time dispatch+combine for latency context
            barrier(group)
            fn = make_dispatch_combine_fn(
                tokens,
                expert_indices,
                weights_f32,
                E,
                capacity,
                group,
                "cpu",
            )
            med_cpu, _, _ = time_fn(fn, warmup, iters)

            barrier(group)
            fn = make_dispatch_combine_fn(
                tokens,
                expert_indices,
                weights_f32,
                E,
                capacity,
                group,
                "metal",
            )
            med_metal, _, _ = time_fn(fn, warmup, iters)

            rows.append(
                [
                    N,
                    total_slots,
                    overflow_count,
                    overflow_ratio,
                    valid_count,
                    remote_count,
                    remote_ratio,
                    utilization,
                    med_cpu,
                    med_metal,
                ]
            )

        except Exception as ex:
            if rank == 0:
                print(f" ERR: {ex}", end="", flush=True)
                traceback.print_exc()
            rows.append(
                [
                    N,
                    N * top_k,
                    0,
                    float("nan"),
                    0,
                    0,
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    float("nan"),
                ]
            )

        if rank == 0:
            print(" done", flush=True)

    if rank == 0:
        print_table(
            "Q3a: Routing Statistics",
            ["N", "slots", "overflow", "oflow%", "valid", "remote", "remote%", "util%"],
            [[r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]] for r in rows],
            [6, 7, 8, 7, 7, 7, 8, 7],
            ["int", "int", "int", "pct", "int", "int", "pct", "pct"],
        )

        print_table(
            "Q3b: Latency alongside routing",
            ["N", "oflow%", "remote%", "util%", "cpu_ms", "metal_ms"],
            [[r[0], r[3], r[6], r[7], r[8], r[9]] for r in rows],
            [6, 7, 8, 7, 9, 9],
            ["int", "pct", "pct", "pct", "ms", "ms"],
        )

        # Print capacity analysis summary
        print("\n  Routing analysis summary:")
        print(f"    capacity per expert = {capacity}")
        print(f"    expected tokens/expert = N * top_k / E = N * {top_k} / {E}")
        for N in N_VALUES:
            expected = N * top_k / E
            print(
                f"      N={N:>5d}: expected {expected:.2f} tokens/expert "
                f"(cap={capacity}, {'OK' if expected <= capacity else 'OVERFLOW'})"
            )


# ── main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Diagnostic benchmark — MoE EP overhead, crossover, and routing"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warmup iterations before timing (default 5)",
    )
    parser.add_argument(
        "--iters", type=int, default=20, help="Timing iterations (default 20)"
    )
    args = parser.parse_args()

    group = mx.distributed.init()
    rank = group.rank()
    ws = group.size()

    if ws < 2:
        raise RuntimeError(f"This benchmark requires >=2 ranks, got {ws}")

    # Kimi K2.5 scale parameters
    E = 384
    D = 7168
    top_k = 8
    capacity = 32
    warmup = args.warmup
    iters = args.iters
    dtype = mx.float16

    has_cpp = hasattr(mx.distributed, "moe_dispatch_exchange")
    if not has_cpp:
        raise RuntimeError(
            "moe_dispatch_exchange not available — "
            "this benchmark requires the C++ fused MoE path."
        )

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

        E_local = E // ws
        cap_total = ws * capacity
        buf_mb = E_local * cap_total * D * 2 / 1024 / 1024  # float16

        print("=" * 100)
        print(f"Diagnostic Benchmark — MoE EP Overhead, Crossover & Routing Analysis")
        print(f"  E={E}, top_k={top_k}, D={D}, capacity={capacity}, dtype=float16")
        print(f"  commit={commit}  world_size={ws}  " f"warmup={warmup}  iters={iters}")
        print(
            f"  E_local={E_local}  cap_total={cap_total}  "
            f"dispatch_buf~{buf_mb:.0f}MB"
        )
        print("=" * 100)

    # ── JACCL warmup ──────────────────────────────────────────────────────
    jaccl_warmup(group, dtype, rank)

    # ── Q1: Overhead isolation ────────────────────────────────────────────
    run_overhead_isolation(group, E, D, top_k, capacity, dtype, warmup, iters, rank)

    # ── Q2: Crossover sweep ───────────────────────────────────────────────
    run_crossover_sweep(group, E, D, top_k, capacity, dtype, warmup, iters, rank)

    # ── Q3: Routing analysis ──────────────────────────────────────────────
    run_routing_analysis(group, E, D, top_k, capacity, dtype, warmup, iters, rank)

    # ── Stats dump (if available) ─────────────────────────────────────────
    if rank == 0:
        print("\n" + "=" * 100)
        print("MoE EP Stats (if available)")
        print("=" * 100)

    has_stats = hasattr(mx.distributed, "moe_ep_stats")
    if has_stats:
        barrier(group)
        stats = mx.distributed.moe_ep_stats()
        if rank == 0:
            if isinstance(stats, dict):
                max_key_len = max(len(k) for k in stats.keys()) if stats else 0
                for key, val in sorted(stats.items()):
                    print(f"  {key:<{max_key_len}} : {val}")
            else:
                print(f"  {stats}")
    else:
        if rank == 0:
            print("  moe_ep_stats() not available in this build.")

    # ── Legend ─────────────────────────────────────────────────────────────
    if rank == 0:
        print("\n" + "=" * 100)
        print("Legend:")
        print("  cpu_ms     = C++ fused dispatch+combine, CPU scatter/gather")
        print("  metal_ms   = C++ fused dispatch+combine, Metal GPU kernels")
        print("  auto_ms    = C++ fused dispatch+combine, auto backend selection")
        print("  cpu/mtl    = cpu_time / metal_time (>1 = Metal faster)")
        print(
            "  delta_ms   = auto_time - cpu_time (overhead of auto backend selection)"
        )
        print("  delta_%    = (auto / cpu - 1) * 100% (relative overhead)")
        print(
            "  oflow%     = overflow ratio: tokens hitting capacity limit / (N * top_k)"
        )
        print(
            "  remote%    = remote ratio: tokens sent to non-local experts / valid tokens"
        )
        print("  util%      = utilization: filled slots / (E_local * ws * capacity)")
        print(f"  All times are median of {iters} iterations, in milliseconds")
        print("=" * 100)

    barrier(group)


if __name__ == "__main__":
    main()
