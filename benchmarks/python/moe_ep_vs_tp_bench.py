#!/usr/bin/env python3
# Copyright © 2026 Apple Inc.

"""EP vs TP E2E comparison benchmark for MoE layers (SwiGLU FFN).

Measures 5 components to build end-to-end latency estimates for Expert
Parallelism (EP) vs Tensor Parallelism (TP) in a Mixture-of-Experts layer:

  1. TP communication:  all_sum(N × D)  — all-reduce cost in TP MoE
  2. TP compute:        SwiGLU FFN with sharded weights (D, intermediate/ws)
  3. EP communication:  moe_dispatch_exchange + moe_combine_exchange (C++ fused)
  4. EP compute:        SwiGLU FFN with full weights (D, intermediate) on N*top_k/ws tokens
  5. Dense MLP:         SwiGLU FFN baseline on N tokens (reference)

E2E estimates:
  TP_total = top_k × SwiGLU_sharded(N) + all_sum(N,D)
  EP_total = dispatch_combine(N)       + SwiGLU_full(N*top_k/ws)

Kimi K2.5 scale defaults: E=384, D=7168, top_k=8, intermediate=28672,
                           cf=1.25, dtype=float16

Run:
    mlx.launch --backend jaccl --hostfile hosts.json -- \\
        python3 benchmarks/python/moe_ep_vs_tp_bench.py

Options:
    --E 384  --D 7168  --top-k 8  --intermediate 0 (0 = 4*D)
    --cf 1.25  --warmup 10  --iters 30
"""

import argparse
import math
import subprocess
import time

import numpy as np

import mlx.core as mx


# ── helpers ─────────────────────────────────────────────────────────────────


def time_fn(fn, warmup, iters):
    """Time a function, returning median_ms.

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
    return float(np.median(ts))


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


# ── SwiGLU FFN ──────────────────────────────────────────────────────────────


def swiglu_ffn(x, w_gate, w_up, w_down):
    """SwiGLU FFN: 3 matmuls with SiLU gating.

    SiLU(x) = x * sigmoid(x)
    out = SiLU(x @ w_gate) * (x @ w_up) @ w_down
    """
    gate = x @ w_gate          # (N, D) @ (D, inter) -> (N, inter)
    up = x @ w_up              # (N, D) @ (D, inter) -> (N, inter)
    h = gate * mx.sigmoid(gate) * up   # SiLU(gate) * up
    return h @ w_down          # (N, inter) @ (inter, D) -> (N, D)


# ── table printing ──────────────────────────────────────────────────────────


def print_table(title, columns, rows):
    """Print a formatted table.

    Args:
        title: table title string
        columns: list of (header, width, fmt_type) tuples
                 fmt_type: "int" | "ms" | "speedup" | "str"
        rows: list of lists (one per row, matching columns)
    """
    if not rows:
        return

    print(f"\n{title}")

    # Build header / separator
    hdr_cells = []
    seps = []
    for hdr, w, _ in columns:
        hdr_cells.append(f" {hdr:^{w}} ")
        seps.append("-" * (w + 2))

    print("+" + "+".join(seps) + "+")
    print("|" + "|".join(hdr_cells) + "|")
    print("+" + "+".join(seps) + "+")

    for row in rows:
        cells = []
        for i, (_, w, fmt) in enumerate(columns):
            val = row[i]
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


# ── main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="EP vs TP E2E comparison benchmark for MoE layers (SwiGLU FFN)"
    )
    parser.add_argument("--E", type=int, default=384,
                        help="Total number of experts (default 384)")
    parser.add_argument("--D", type=int, default=7168,
                        help="Hidden dimension (default 7168)")
    parser.add_argument("--top-k", type=int, default=8,
                        help="Top-k experts per token (default 8)")
    parser.add_argument("--intermediate", type=int, default=0,
                        help="FFN intermediate dim; 0 = 4*D (default 0)")
    parser.add_argument("--cf", type=float, default=1.25,
                        help="Capacity factor for dynamic capacity (default 1.25)")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Warmup iterations (default 10)")
    parser.add_argument("--iters", type=int, default=30,
                        help="Timing iterations (default 30)")
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
    intermediate = args.intermediate if args.intermediate > 0 else 4 * D
    cf = args.cf
    warmup = args.warmup
    iters = args.iters
    dtype = mx.float16

    N_VALUES = [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

    has_cpp = hasattr(mx.distributed, "moe_dispatch_exchange")

    # ── Print header ───────────────────────────────────────────────────
    if rank == 0:
        try:
            commit = (
                subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
                .decode()
                .strip()
            )
        except Exception:
            commit = "unknown"

        sep = "=" * 79
        print(sep)
        print("EP vs TP E2E Comparison — MoE Layer (SwiGLU)")
        print(
            f"  commit={commit}  world_size={ws}  dtype=float16"
            f"  warmup={warmup}  iters={iters}"
        )
        print(
            f"  E={E}  D={D}  top_k={top_k}  intermediate={intermediate}"
            f"  cf={cf}"
        )
        print(
            f"  C++ fused EP path: {'available' if has_cpp else 'NOT available'}"
        )
        print(sep)

    # ── JACCL warmup ───────────────────────────────────────────────────
    jaccl_warmup(group, dtype, rank)

    # ── Allocate SwiGLU weights (once, before N sweep) ─────────────────
    if rank == 0:
        print("\n  Allocating SwiGLU weights ...", flush=True)

    # Sharded weights (for TP): intermediate dimension divided by ws
    inter_shard = intermediate // ws
    w_gate_s = mx.random.normal((D, inter_shard)).astype(dtype) * 0.02
    w_up_s = mx.random.normal((D, inter_shard)).astype(dtype) * 0.02
    w_down_s = mx.random.normal((inter_shard, D)).astype(dtype) * 0.02

    # Full weights (for EP and dense baseline)
    w_gate_f = mx.random.normal((D, intermediate)).astype(dtype) * 0.02
    w_up_f = mx.random.normal((D, intermediate)).astype(dtype) * 0.02
    w_down_f = mx.random.normal((intermediate, D)).astype(dtype) * 0.02

    mx.eval(w_gate_s, w_up_s, w_down_s, w_gate_f, w_up_f, w_down_f)

    if rank == 0:
        shard_mb = 3 * D * inter_shard * 2 / 1024 / 1024  # 3 matrices, float16=2B
        full_mb = 3 * D * intermediate * 2 / 1024 / 1024
        # w_down has transposed shape but same element count
        print(
            f"  Sharded weights: 3 x ({D},{inter_shard}) + ({inter_shard},{D})"
            f" ~ {shard_mb:.0f} MB"
        )
        print(
            f"  Full weights:    3 x ({D},{intermediate}) + ({intermediate},{D})"
            f" ~ {full_mb:.0f} MB"
        )
        print("  Weights allocated.\n", flush=True)

    # ── Sweep across N values ──────────────────────────────────────────
    suite1_rows = []  # Component measurements
    suite2_rows = []  # E2E estimates

    for N in N_VALUES:
        barrier(group)

        capacity = compute_capacity(N, top_k, cf, E)
        # Synchronize capacity across ranks
        cap_arr = mx.array(capacity, dtype=mx.int32)
        cap_arr = mx.distributed.all_max(cap_arr, group=group)
        mx.eval(cap_arr)
        capacity = cap_arr.item()

        N_ep = max(1, N * top_k // ws)

        if rank == 0:
            print(f"  N={N:>5d}  cap={capacity:>4d}  N_ep={N_ep:>5d} ...",
                  end="", flush=True)

        # ── 1. TP communication: all_sum(N, D) ────────────────────────
        x_allsum = mx.random.normal((N, D)).astype(dtype)
        mx.eval(x_allsum)
        barrier(group)

        med_allsum = time_fn(
            lambda: mx.distributed.all_sum(x_allsum, group=group),
            warmup, iters,
        )

        # ── 2. TP compute: SwiGLU with sharded weights on N tokens ────
        x_tp = mx.random.normal((N, D)).astype(dtype)
        mx.eval(x_tp)
        barrier(group)

        med_swiglu_shard = time_fn(
            lambda: swiglu_ffn(x_tp, w_gate_s, w_up_s, w_down_s),
            warmup, iters,
        )

        # ── 3. EP communication: dispatch + combine (auto backend) ────
        if has_cpp:
            tokens_ep = mx.random.normal((N, D)).astype(dtype)
            expert_indices = mx.random.randint(
                0, E, shape=(N, top_k)
            ).astype(mx.int32)
            weights_raw = mx.random.normal((N, top_k))
            weights_f32 = mx.softmax(weights_raw, axis=-1).astype(mx.float32)
            mx.eval(tokens_ep, expert_indices, weights_f32)
            barrier(group)

            fn_ep = make_dispatch_combine_fn(
                tokens_ep, expert_indices, weights_f32,
                E, capacity, group, "auto",
            )
            try:
                med_ep_dc = time_fn(fn_ep, warmup, iters)
            except Exception as ex:
                med_ep_dc = float("nan")
                if rank == 0:
                    print(f" [EP err: {ex}]", end="", flush=True)
        else:
            med_ep_dc = float("nan")

        # ── 4. EP compute: SwiGLU with full weights on N_ep tokens ────
        x_ep_compute = mx.random.normal((N_ep, D)).astype(dtype)
        mx.eval(x_ep_compute)
        barrier(group)

        med_swiglu_full = time_fn(
            lambda: swiglu_ffn(x_ep_compute, w_gate_f, w_up_f, w_down_f),
            warmup, iters,
        )

        # ── 5. Dense MLP baseline: SwiGLU with full weights on N tokens
        x_dense = mx.random.normal((N, D)).astype(dtype)
        mx.eval(x_dense)
        barrier(group)

        med_dense = time_fn(
            lambda: swiglu_ffn(x_dense, w_gate_f, w_up_f, w_down_f),
            warmup, iters,
        )

        # ── Collect Suite 1 row ────────────────────────────────────────
        suite1_rows.append([
            N, capacity, med_allsum, med_swiglu_shard,
            med_ep_dc, med_swiglu_full, med_dense,
        ])

        # ── Compute E2E estimates for Suite 2 ──────────────────────────
        tp_est = top_k * med_swiglu_shard + med_allsum
        if not np.isnan(med_ep_dc):
            ep_est = med_ep_dc + med_swiglu_full
            ep_tp_ratio = ep_est / tp_est if tp_est > 0 else float("nan")
            winner = "EP" if ep_tp_ratio <= 1.0 else "TP"
        else:
            ep_est = float("nan")
            ep_tp_ratio = float("nan")
            winner = "N/A"

        suite2_rows.append([
            N, capacity, tp_est, ep_est, ep_tp_ratio, winner,
        ])

        if rank == 0:
            if not np.isnan(ep_tp_ratio):
                print(
                    f" TP={tp_est:.2f}ms  EP={ep_est:.2f}ms"
                    f"  EP/TP={ep_tp_ratio:.2f}x  -> {winner}",
                    flush=True,
                )
            else:
                print(
                    f" TP={tp_est:.2f}ms  EP=N/A",
                    flush=True,
                )

    # ── Print results (rank 0 only) ────────────────────────────────────
    if rank == 0:
        # Suite 1: Component Measurements
        s1_cols = [
            ("N",            6, "int"),
            ("cap",          5, "int"),
            ("all_sum",      9, "ms"),
            ("swiglu_shrd", 11, "ms"),
            ("ep_d+c",       9, "ms"),
            ("swiglu_ful",  10, "ms"),
            ("dense_mlp",   10, "ms"),
        ]
        print_table("Suite 1: Component Measurements", s1_cols, suite1_rows)

        # Suite 2: E2E Estimates
        s2_cols = [
            ("N",       6, "int"),
            ("cap",     5, "int"),
            ("TP_est",  9, "ms"),
            ("EP_est",  9, "ms"),
            ("EP/TP",   8, "speedup"),
            ("winner",  8, "str"),
        ]
        print_table("Suite 2: E2E Estimates (median ms)", s2_cols, suite2_rows)

        # ── Geomean summaries ──────────────────────────────────────────
        all_ratios = [r[4] for r in suite2_rows
                      if not np.isnan(r[4]) and r[4] > 0]
        decode_ratios = [r[4] for r in suite2_rows
                         if r[0] <= 64
                         and not np.isnan(r[4]) and r[4] > 0]
        prefill_ratios = [r[4] for r in suite2_rows
                          if r[0] >= 256
                          and not np.isnan(r[4]) and r[4] > 0]

        gm_all = geomean(all_ratios)
        gm_decode = geomean(decode_ratios)
        gm_prefill = geomean(prefill_ratios)

        print()
        print(f"  EP/TP geomean (all):     {gm_all:.2f}x")
        print(f"  EP/TP geomean (decode):  {gm_decode:.2f}x  (N<=64)")
        print(f"  EP/TP geomean (prefill): {gm_prefill:.2f}x  (N>=256)")

        # Judgment
        if not np.isnan(gm_all):
            if gm_all <= 1.0:
                judgment = f"EP wins vs TP — geomean EP/TP = {gm_all:.2f} (<= 1.0 = EP wins)"
            else:
                judgment = f"EP loses vs TP — geomean EP/TP = {gm_all:.2f} (> 1.0 = TP wins)"
        else:
            judgment = "N/A (EP measurements unavailable)"

        print()
        print(f"  Judgment: {judgment}")

        # ── Legend ─────────────────────────────────────────────────────
        print()
        sep = "=" * 79
        print(sep)
        print("Legend:")
        print("  all_sum      = all_reduce communication cost for TP (N x D tensor)")
        print(f"  swiglu_shrd  = SwiGLU FFN with sharded weights"
              f" (D={D}, inter={intermediate // ws})")
        print(f"  ep_d+c       = moe_dispatch_exchange + moe_combine_exchange"
              f" (backend=auto, cf={cf})")
        print(f"  swiglu_ful   = SwiGLU FFN with full weights"
              f" (D={D}, inter={intermediate}) on N*top_k/ws tokens")
        print(f"  dense_mlp    = SwiGLU FFN baseline (full weights, N tokens)")
        print(f"  TP_est       = top_k({top_k}) x swiglu_shrd + all_sum")
        print(f"  EP_est       = ep_d+c + swiglu_ful")
        print(f"  EP/TP        = EP_est / TP_est  (<= 1.0 means EP is faster)")
        print(f"  All times in milliseconds (median of {iters} iterations)")
        print(sep)

    barrier(group)


if __name__ == "__main__":
    main()
