# Copyright © 2026 Apple Inc.

"""Phase 3 v3 protocol benchmark — Kimi K2.5 scale MoE Expert Parallelism.

Compares Python dispatch/combine, Phase 1 C++ ref (fixed all_to_all), and
Phase 2/3 C++ fused dispatch+combine across three suites:

  Suite A: Decode-like latency  (N = 1, 4, 8, 16, 32, 64)
  Suite B: Mid-batch            (N = 256, 512, 1024)
  Suite C: Prefill-like         (N = 2048, 4096)

Kimi K2.5 scale: E=384, top_k=8, D=7168, capacity_factor=1.25, dtype=float16

Run:
    mlx.launch --backend jaccl --hostfile hosts.json -- \\
        python3 benchmarks/python/moe_ep_phase3_bench.py

Options:
    --warmup 10 --iters 30  (defaults)
    --suites A B C          (choose subsets)
"""

import argparse
import math
import subprocess
import time
import traceback

import numpy as np

import mlx.core as mx
from mlx.nn.layers.moe import expert_dispatch, expert_combine


# ── helpers ─────────────────────────────────────────────────────────────────

def time_fn(fn, warmup, iters):
    """Time a function, returning (median_ms, p90_ms)."""
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
    return np.median(ts), np.percentile(ts, 90)


def barrier(group):
    """Synchronize ranks."""
    s = mx.distributed.all_sum(mx.array(0.0), group=group)
    mx.eval(s)
    mx.synchronize()


def fmt_ms(v):
    """Format milliseconds to string, handling NaN."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "  N/A  "
    return f"{v:7.2f}"


def compute_capacity(N, top_k, cf, E):
    """Compute per-expert capacity."""
    return max(1, math.ceil(N * top_k * cf / E))


# ── benchmark for one N value ───────────────────────────────────────────────

def bench_one_n(
    N, D, E, top_k, cf, dtype, group, warmup, iters, has_cpp
):
    """Run all measurements for a single N value.

    Returns dict with keys:
        N, capacity, tp_allsum, ep_a2a_x2, ep_d_c_py, ep_p1ref, ep_d_c_cpp,
        dense_mlp, comm_fixed_kb, comm_var_kb, comm_savings_pct
    """
    ws = group.size()
    E_local = E // ws
    capacity = compute_capacity(N, top_k, cf, E)

    # Synchronize capacity across ranks
    cap_arr = mx.array(capacity, dtype=mx.int32)
    cap_arr = mx.distributed.all_max(cap_arr, group=group)
    mx.eval(cap_arr)
    capacity = cap_arr.item()

    cap_total = ws * capacity  # per-expert capacity after exchange

    result = {"N": N, "capacity": capacity}

    # ── Allocate input tensors ──────────────────────────────────────────
    tokens = mx.random.normal((N, D)).astype(dtype)
    expert_indices = mx.random.randint(0, E, shape=(N, top_k)).astype(mx.int32)
    weights_raw = mx.random.normal((N, top_k))
    weights = mx.softmax(weights_raw, axis=-1)
    weights_f32 = weights.astype(mx.float32)

    x_allsum = mx.random.normal((N, D)).astype(dtype)
    flat_a2a = mx.random.normal((ws, N * D // ws)).astype(dtype)

    # Dense MLP proxy (4x expansion like SwiGLU)
    expert_dim = D * 4
    w1 = (mx.random.normal((D, expert_dim)) * 0.02).astype(dtype)
    w2 = (mx.random.normal((expert_dim, D)) * 0.02).astype(dtype)

    # Phase 1 ref buffer (fixed-size all_to_all)
    # Each direction sends E_local * capacity * D elements
    # Use float16 for the ref buffer to match the actual dtype
    send_buf_ref = mx.zeros(
        (ws, E_local * capacity * D), dtype=dtype
    )

    mx.eval(
        tokens, expert_indices, weights, weights_f32,
        x_allsum, flat_a2a, w1, w2, send_buf_ref,
    )

    # ── 1. TP all_sum ───────────────────────────────────────────────────
    barrier(group)
    tp_med, _ = time_fn(
        lambda: mx.distributed.all_sum(x_allsum, group=group),
        warmup, iters,
    )
    result["tp_allsum"] = tp_med

    # ── 2. EP all_to_all x2 ────────────────────────────────────────────
    barrier(group)

    def ep_a2a():
        y = mx.distributed.all_to_all(flat_a2a, group=group)
        z = mx.distributed.all_to_all(y, group=group)
        return z

    ep_a2a_med, _ = time_fn(ep_a2a, warmup, iters)
    result["ep_a2a_x2"] = ep_a2a_med

    # ── 3. EP dispatch+combine Python ───────────────────────────────────
    barrier(group)

    def ep_dc_py():
        dispatched, meta = expert_dispatch(
            tokens, expert_indices, weights,
            num_experts=E, capacity_factor=cf, group=group,
        )
        combined = expert_combine(dispatched, meta, tokens, group=group)
        return combined

    ep_dc_py_med, _ = time_fn(ep_dc_py, warmup, iters)
    result["ep_d_c_py"] = ep_dc_py_med

    # ── 4. Phase 1 ref (fixed all_to_all x2) ───────────────────────────
    barrier(group)

    def ep_p1ref():
        y = mx.distributed.all_to_all(send_buf_ref, group=group)
        z = mx.distributed.all_to_all(y, group=group)
        return z

    ep_p1ref_med, _ = time_fn(ep_p1ref, warmup, iters)
    result["ep_p1ref"] = ep_p1ref_med

    # ── 5. EP dispatch+combine C++ fused ────────────────────────────────
    if has_cpp:
        barrier(group)

        def ep_dc_cpp():
            dispatched, route_idx = mx.distributed.moe_dispatch_exchange(
                tokens, expert_indices,
                num_experts=E, capacity=capacity,
                group=group, backend="cpu",
            )
            route_idx = mx.stop_gradient(route_idx)
            combined = mx.distributed.moe_combine_exchange(
                dispatched, route_idx, weights_f32, tokens,
                num_experts=E, capacity=capacity,
                group=group, backend="cpu",
            )
            return combined

        ep_dc_cpp_med, _ = time_fn(ep_dc_cpp, warmup, iters)
        result["ep_d_c_cpp"] = ep_dc_cpp_med
    else:
        result["ep_d_c_cpp"] = float("nan")

    # ── 6. Dense MLP (local, single device) ─────────────────────────────
    barrier(group)

    def dense_mlp():
        h = mx.maximum(tokens @ w1, 0)
        return h @ w2

    mlp_med, _ = time_fn(dense_mlp, warmup, iters)
    result["dense_mlp"] = mlp_med

    # ── Communication savings estimate ──────────────────────────────────
    bytes_per_elem = 2 if dtype == mx.float16 else 4  # float16 = 2B
    fixed_bytes = E_local * capacity * D * bytes_per_elem  # Phase 1 fixed a2a per dir
    est_remote_tokens = N * top_k // ws  # uniform routing estimate
    var_bytes = est_remote_tokens * D * bytes_per_elem  # payload
    var_bytes += est_remote_tokens * 2 * 4  # meta (2 x int32 per token)

    result["comm_fixed_kb"] = fixed_bytes / 1024
    result["comm_var_kb"] = var_bytes / 1024
    if fixed_bytes > 0:
        result["comm_savings_pct"] = (1 - var_bytes / fixed_bytes) * 100
    else:
        result["comm_savings_pct"] = 0.0

    return result


# ── table printing ──────────────────────────────────────────────────────────

def print_table(title, rows, columns):
    """Print a formatted table.

    columns: list of (header, key, width) tuples.
    rows: list of dicts.
    """
    if not rows:
        return

    print(f"\n{title}")

    # Build header / separator / data lines
    headers = []
    seps = []
    for hdr, _, w in columns:
        headers.append(f" {hdr:^{w}} ")
        seps.append("-" * (w + 2))

    print("+" + "+".join(seps) + "+")
    print("|" + "|".join(headers) + "|")
    print("+" + "+".join(seps) + "+")

    for row in rows:
        cells = []
        for _, key, w in columns:
            val = row.get(key)
            if key in ("N", "capacity"):
                cell = f" {val:>{w}} "
            elif val is None or (isinstance(val, float) and np.isnan(val)):
                cell = f" {'N/A':^{w}} "
            elif key == "comm_savings_pct":
                cell = f" {val:>{w}.0f}% "
            elif key in ("comm_fixed_kb", "comm_var_kb"):
                cell = f" {val:>{w-2}.0f}KB "
            else:
                cell = f" {val:>{w}.2f} "
            cells.append(cell)
        print("|" + "|".join(cells) + "|")

    print("+" + "+".join(seps) + "+")


# ── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 3 v3 protocol benchmark — Kimi K2.5 scale"
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument(
        "--suites", nargs="+", default=["A", "B", "C"],
        help="Suites to run: A (decode), B (mid-batch), C (prefill)",
    )
    parser.add_argument("--e", type=int, default=384, help="Total experts")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--d", type=int, default=7168, help="Hidden dim")
    parser.add_argument("--cf", type=float, default=1.25, help="Capacity factor")
    args = parser.parse_args()

    group = mx.distributed.init()
    if group.size() < 2:
        raise RuntimeError(
            f"This benchmark requires >=2 ranks, got {group.size()}"
        )

    E, top_k, D, cf = args.e, args.top_k, args.d, args.cf
    warmup, iters = args.warmup, args.iters
    dtype = mx.float16

    has_cpp = hasattr(mx.distributed, "moe_dispatch_exchange")

    # Suite definitions: (suite_name, description, N_values)
    suite_defs = {
        "A": ("Suite A — Decode-like latency", [1, 4, 8, 16, 32, 64]),
        "B": ("Suite B — Mid-batch", [256, 512, 1024]),
        "C": ("Suite C — Prefill-like", [2048, 4096]),
    }

    # Table columns: (header, key, width)
    columns = [
        ("N", "N", 6),
        ("cap", "capacity", 6),
        ("tp_allsum", "tp_allsum", 9),
        ("ep_a2a_x2", "ep_a2a_x2", 9),
        ("ep_d+c_py", "ep_d_c_py", 9),
        ("ep_p1ref", "ep_p1ref", 9),
        ("ep_d+c_cpp", "ep_d_c_cpp", 10),
        ("dense_mlp", "dense_mlp", 9),
        ("save%", "comm_savings_pct", 5),
    ]

    # Print header on rank 0 only
    if group.rank() == 0:
        try:
            commit = (
                subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
                .decode()
                .strip()
            )
        except Exception:
            commit = "unknown"

        dtype_str = "float16" if dtype == mx.float16 else "float32"
        print("=" * 100)
        print(
            f"Phase 3 Benchmark — Kimi K2.5  "
            f"(E={E}, top_k={top_k}, D={D}, dtype={dtype_str})"
        )
        print(
            f"  commit={commit}  world_size={group.size()}  "
            f"cf={cf}  warmup={warmup}  iters={iters}"
        )
        print(
            f"  C++ fused path: {'available' if has_cpp else 'NOT available'}"
        )
        print("=" * 100)

    # Run suites
    for suite_key in args.suites:
        suite_key = suite_key.upper()
        if suite_key not in suite_defs:
            if group.rank() == 0:
                print(f"\nUnknown suite '{suite_key}', skipping.")
            continue

        title, n_values = suite_defs[suite_key]
        rows = []

        for N in n_values:
            barrier(group)
            try:
                if group.rank() == 0:
                    cap_est = compute_capacity(N, top_k, cf, E)
                    # Estimate memory for dispatch buffer:
                    # ws * E_local * capacity * D * sizeof(dtype)
                    E_local = E // group.size()
                    buf_bytes = (
                        group.size() * E_local * cap_est * D * 2  # float16 = 2B
                    )
                    print(
                        f"  Running N={N:>5d} "
                        f"(cap={cap_est}, buf~{buf_bytes/1024/1024:.0f}MB) ...",
                        flush=True,
                    )

                row = bench_one_n(
                    N, D, E, top_k, cf, dtype, group, warmup, iters, has_cpp
                )
                rows.append(row)

            except Exception as ex:
                if group.rank() == 0:
                    print(f"    SKIPPED N={N}: {ex}")
                    traceback.print_exc()
                # Still synchronize so all ranks agree
                barrier(group)
                continue

        if group.rank() == 0:
            print_table(title, rows, columns)

    # Summary: communication savings
    if group.rank() == 0:
        print("\n" + "=" * 100)
        print("Communication estimate (per direction, uniform routing):")
        print(
            f"  Fixed (Phase 1):   E_local * cap * D * 2B  "
            f"(E_local={E // group.size()})"
        )
        print(
            f"  Variable (Phase 3): N * top_k / ws * (D * 2B + 2 * 4B)  "
            f"(top_k={top_k})"
        )
        print(
            f"  Savings come from sending only routed tokens "
            f"instead of full capacity buffers."
        )
        print("=" * 100)


if __name__ == "__main__":
    main()
