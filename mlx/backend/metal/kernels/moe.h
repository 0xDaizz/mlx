// Copyright © 2026 Apple Inc.

#pragma once

// MoE Expert Parallelism Metal Kernels
//
// These kernels accelerate the scatter/gather operations in
// MoeDispatchExchange and MoeCombineExchange primitives.

// Kernel 1: moe_dispatch_local
// Scatters LOCAL tokens into the dispatched buffer using a precomputed slot_map.
// slot_map[nk] = flat_idx into dispatched[E_local, cap_total, D], or -1 to skip.
//
// Grid: (D_ceil, valid_count, 1)  where D_ceil = ceil(D / ELEM_PER_THREAD)
// Group: (min(D_ceil, 256), 1, 1)
//
// Note: valid_count = number of (n,k) pairs with slot_map >= 0
// nk_indices[i] = original n*top_k+k index for the i-th valid entry
template <typename T>
[[kernel]] void moe_dispatch_local(
    const device T* tokens        [[buffer(0)]],  // [N, D]
    device T* dispatched           [[buffer(1)]],  // [E_local * cap_total * D] flat
    const device int* slot_map     [[buffer(2)]],  // [valid_count] flat_idx values
    const device int* nk_indices   [[buffer(3)]],  // [valid_count] n*top_k+k originals
    constant int& D                [[buffer(4)]],
    constant int& top_k            [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]) {
  int d = gid.x;
  int i = gid.y;
  if (d >= D) return;

  int flat_idx = slot_map[i];
  int nk = nk_indices[i];
  int n = nk / top_k;

  dispatched[static_cast<long>(flat_idx) * D + d] =
      tokens[static_cast<long>(n) * D + d];
}

// Kernel 2: moe_dispatch_scatter_remote
// Scatters received remote tokens (from RDMA exchange) into dispatched buffer.
// recv_meta[i] = (local_expert, pos) packed as meta32
// recv_payload is contiguous [cnt, D] after meta extraction on CPU.
//
// Grid: (D_ceil, cnt, 1)
// Group: (min(D_ceil, 256), 1, 1)
template <typename T>
[[kernel]] void moe_dispatch_scatter_remote(
    const device T* recv_payload   [[buffer(0)]],  // [cnt, D]
    device T* dispatched           [[buffer(1)]],  // [E_local * cap_total * D] flat
    const device int* recv_flat_idx [[buffer(2)]],  // [cnt] precomputed flat indices
    constant int& D                [[buffer(3)]],
    constant int& cnt              [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
  int d = gid.x;
  int i = gid.y;
  if (d >= D || i >= cnt) return;

  int flat_idx = recv_flat_idx[i];
  dispatched[static_cast<long>(flat_idx) * D + d] =
      recv_payload[static_cast<long>(i) * D + d];
}

// Kernel 3: moe_combine_gather_remote
// Gathers expert outputs for peer's requested tokens into a send buffer.
// For each request i, lookup expert_out at flat index and copy to send_results.
//
// Grid: (D_ceil, cnt, 1)
// Group: (min(D_ceil, 256), 1, 1)
template <typename T>
[[kernel]] void moe_combine_gather_remote(
    const device T* expert_out     [[buffer(0)]],  // [E_local * cap_total * D] flat
    device T* send_results         [[buffer(1)]],  // [cnt, D]
    const device int* eo_flat_idx  [[buffer(2)]],  // [cnt] precomputed flat indices
    constant int& D                [[buffer(3)]],
    constant int& cnt              [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
  int d = gid.x;
  int i = gid.y;
  if (d >= D || i >= cnt) return;

  int flat_idx = eo_flat_idx[i];
  send_results[static_cast<long>(i) * D + d] =
      expert_out[static_cast<long>(flat_idx) * D + d];
}

// Kernel 4: moe_combine_weighted_sum
// Performs weighted accumulation of expert outputs per token.
// For each token n, sums weights[n,k] * src[k_data_idx, d] for k=0..top_k-1.
// Uses float32 accumulation for precision regardless of input dtype.
//
// data_src contains interleaved local and remote results indexed by src_idx.
// src_idx[n * top_k + k] = index into data_src for that (n,k) pair, or -1 to skip.
//
// Grid: (D_ceil, N, 1)
// Group: (min(D_ceil, 256), 1, 1)
template <typename T>
[[kernel]] void moe_combine_weighted_sum(
    const device T* data_src       [[buffer(0)]],  // [total_entries, D]
    device T* output               [[buffer(1)]],  // [N, D]
    const device T* original       [[buffer(2)]],  // [N, D] fallback
    const device float* weights    [[buffer(3)]],  // [N, top_k]
    const device int* src_idx      [[buffer(4)]],  // [N * top_k]
    constant int& D                [[buffer(5)]],
    constant int& N                [[buffer(6)]],
    constant int& top_k            [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]) {
  int d = gid.x;
  int n = gid.y;
  if (d >= D || n >= N) return;

  float accum = 0.0f;
  bool has_valid = false;

  for (int k = 0; k < top_k; k++) {
    int idx = src_idx[n * top_k + k];
    if (idx >= 0) {
      has_valid = true;
      float w = weights[n * top_k + k];
      accum += w * static_cast<float>(data_src[static_cast<long>(idx) * D + d]);
    }
  }

  if (has_valid) {
    output[static_cast<long>(n) * D + d] = static_cast<T>(accum);
  } else {
    output[static_cast<long>(n) * D + d] = original[static_cast<long>(n) * D + d];
  }
}
