# Copyright © 2026 Apple Inc.

"""EP vs TP micro-benchmark for MoE Expert Parallelism.

Requires exactly 2 devices. Run with:
    mlx.launch --backend jaccl --hostfile hosts.json \\
        benchmarks/python/moe_ep_tp_bench.py \\
        --n 2048 --d 128 --e 8 --top-k 2
"""

import argparse
import math
import subprocess
import time

import numpy as np

import mlx.core as mx
from mlx.nn.layers.moe import expert_dispatch, expert_combine


def time_fn(fn, warmup, iters):
    """Time a function. fn() should return tensors to eval."""
    for _ in range(warmup):
        y = fn()
        if isinstance(y, tuple):
            mx.eval(*y)
        else:
            mx.eval(y)
    mx.synchronize()

    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        y = fn()
        if isinstance(y, tuple):
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


def main():
    parser = argparse.ArgumentParser(description="EP vs TP benchmark")
    parser.add_argument("--n", type=int, default=2048)
    parser.add_argument("--d", type=int, default=128)
    parser.add_argument("--e", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--capacity-factor", type=float, default=1.25)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()

    group = mx.distributed.init()
    if group.size() != 2:
        raise RuntimeError(
            f"moe_ep_tp_bench requires exactly 2 ranks, got {group.size()}"
        )

    N, D, E, top_k = args.n, args.d, args.e, args.top_k
    cf = args.capacity_factor
    warmup, iters = args.warmup, args.iters

    # Pre-create input tensors (not timed)
    tokens = mx.random.normal((N, D))
    expert_indices = mx.random.randint(0, E, (N, top_k))
    weights_raw = mx.random.normal((N, top_k))
    weights = mx.softmax(weights_raw, axis=-1)

    # For TP all_sum proxy
    x_allsum = mx.random.normal((N, D))

    # For EP all_to_all proxy
    flat_a2a = mx.random.normal((group.size(), N * D // group.size()))

    # For dense MLP proxy
    expert_dim = D * 4  # typical 4x expansion
    w1 = mx.random.normal((D, expert_dim)) * 0.02
    w2 = mx.random.normal((expert_dim, D)) * 0.02

    mx.eval(tokens, expert_indices, weights, x_allsum, flat_a2a, w1, w2)

    # Initial sync
    barrier(group)

    # 1. TP all_sum
    barrier(group)
    tp_med, tp_p90 = time_fn(
        lambda: mx.distributed.all_sum(x_allsum, group=group),
        warmup,
        iters,
    )

    # 2. EP all_to_all x2
    barrier(group)

    def ep_a2a():
        y = mx.distributed.all_to_all(flat_a2a, group=group)
        z = mx.distributed.all_to_all(y, group=group)
        return z

    ep_a2a_med, ep_a2a_p90 = time_fn(ep_a2a, warmup, iters)

    # Capacity used by expert_dispatch and C++ fused path
    capacity = math.ceil(N * top_k / E * cf)
    E_local = E // group.size()

    # 3. EP dispatch + combine
    barrier(group)

    def ep_dc():
        dispatched, meta = expert_dispatch(
            tokens,
            expert_indices,
            weights,
            num_experts=E,
            capacity_factor=cf,
            group=group,
        )
        # Identity expert (no computation)
        combined = expert_combine(dispatched, meta, tokens, group=group)
        return combined

    ep_dc_med, ep_dc_p90 = time_fn(ep_dc, warmup, iters)

    # 4b. Phase 1 C++ ref: all_to_all(fixed dispatch buf) x2
    # Pre-allocate the full dispatch buffer (same shape Phase 1 C++ would send).
    # This approximates Phase 1 C++ communication cost without Python routing overhead.
    send_buf_ref = mx.zeros(
        (group.size(), E_local * capacity * D), dtype=mx.float32
    )
    mx.eval(send_buf_ref)

    barrier(group)

    def ep_a2a_ref():
        y = mx.distributed.all_to_all(send_buf_ref, group=group)
        z = mx.distributed.all_to_all(y, group=group)
        return z

    ep_p1ref_med, ep_p1ref_p90 = time_fn(ep_a2a_ref, warmup, iters)

    # 5. EP dispatch + combine (C++ fused, Phase 2)
    barrier(group)
    has_cpp = hasattr(mx.distributed, "moe_dispatch_exchange")

    if has_cpp:
        def ep_dc_cpp():
            dispatched, route_idx = mx.distributed.moe_dispatch_exchange(
                tokens, expert_indices,
                num_experts=E, capacity=capacity,
                group=group,
            )
            route_idx = mx.stop_gradient(route_idx)
            # Identity expert (no computation)
            combined = mx.distributed.moe_combine_exchange(
                dispatched, route_idx, weights, tokens,
                num_experts=E, capacity=capacity,
                group=group,
            )
            return combined

        ep_dc_cpp_med, ep_dc_cpp_p90 = time_fn(ep_dc_cpp, warmup, iters)
    else:
        ep_dc_cpp_med, ep_dc_cpp_p90 = float('nan'), float('nan')

    # 4. Dense MLP (local, single device)
    barrier(group)

    def dense_mlp():
        h = mx.maximum(tokens @ w1, 0)  # ReLU activation
        return h @ w2

    mlp_med, mlp_p90 = time_fn(dense_mlp, warmup, iters)

    # Print results (rank 0 only)
    if group.rank() == 0:
        try:
            commit = (
                subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
                .decode()
                .strip()
            )
        except Exception:
            commit = "unknown"

        ws = group.size()
        tp_compute_ms = mlp_med / ws
        ep_compute_ms = mlp_med * top_k / ws
        tp_est_ms = tp_med + tp_compute_ms
        ep_est_ms = ep_dc_med + ep_compute_ms

        print(f"[EP vs TP] commit={commit} N={N} D={D} E={E} top_k={top_k} cap={capacity}")
        print(f"  tp_all_sum:   median={tp_med:.2f}ms  p90={tp_p90:.2f}ms")
        print(f"  ep_a2a_x2:    median={ep_a2a_med:.2f}ms  p90={ep_a2a_p90:.2f}ms")
        print(f"  ep_d+c_py:        median={ep_dc_med:.2f}ms  p90={ep_dc_p90:.2f}ms")
        print(f"  ep_d+c_cpp_p1ref: median={ep_p1ref_med:.2f}ms  p90={ep_p1ref_p90:.2f}ms")
        print(f"  ep_d+c_cpp_p2:    median={ep_dc_cpp_med:.2f}ms  p90={ep_dc_cpp_p90:.2f}ms")
        if has_cpp:
            p2_vs_p1ref = ep_p1ref_med / max(ep_dc_cpp_med, 1e-6)
            p2_vs_py = ep_dc_med / max(ep_dc_cpp_med, 1e-6)
            # Estimate communication bytes
            fixed_bytes = E_local * capacity * D * 4  # float32, Phase 1 fixed a2a
            est_remote_tokens = N * top_k // group.size()  # uniform routing estimate
            var_bytes = est_remote_tokens * D * 4          # payload
            var_bytes += est_remote_tokens * 2 * 4         # meta (2 x int32)
            print(f"  p2_vs_p1ref:      {p2_vs_p1ref:.2f}x speedup over Phase 1 C++ ref")
            print(f"  p2_vs_py:         {p2_vs_py:.2f}x speedup over Python")
            print(f"  comm_fixed_a2a:   {fixed_bytes/1024:.1f}KB/dir (Phase 1 fixed)")
            print(f"  comm_var_est:     {var_bytes/1024:.1f}KB/dir (Phase 2 est, uniform routing)")
            print(f"  comm_savings_est: {(1 - var_bytes/fixed_bytes)*100:.0f}% (estimated)")
        print(f"  dense_mlp:        median={mlp_med:.2f}ms  p90={mlp_p90:.2f}ms")
        print(f"  ---")
        comm_ratio = ep_a2a_med / max(tp_med, 1e-6)
        speedup = tp_est_ms / max(ep_est_ms, 1e-6)
        print(f"  comm_ratio:        {comm_ratio:.2f}x (ep_a2a / tp_allsum)")
        print(f"  ep_vs_tp_speedup:  {speedup:.2f}x (tp_est / ep_est)")


if __name__ == "__main__":
    main()
