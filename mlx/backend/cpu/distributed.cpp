// Copyright © 2024 Apple Inc.

#include <cassert>
#include <cstring>

#include "mlx/allocator.h"
#include "mlx/backend/cpu/copy.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/distributed/primitives.h"
#include "mlx/types/half_types.h"

namespace mlx::core::distributed {

std::pair<array, bool> ensure_row_contiguous(const array& arr, Stream stream) {
  if (arr.flags().row_contiguous) {
    return {arr, false};
  } else {
    return {contiguous_copy_cpu(arr, stream), true};
  }
};

void AllReduce::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 1);
  assert(outputs.size() == 1);

  auto donate_or_copy = [s = stream()](const array& in, array& out) {
    if (in.flags().row_contiguous) {
      if (in.is_donatable()) {
        out.copy_shared_buffer(in);
      } else {
        out.set_data(allocator::malloc(out.nbytes()));
      }
      return in;
    } else {
      array arr_copy = contiguous_copy_cpu(in, s);
      out.copy_shared_buffer(arr_copy);
      return arr_copy;
    }
  };

  auto in = donate_or_copy(inputs[0], outputs[0]);
  switch (reduce_type_) {
    case Sum:
      distributed::detail::all_sum(group(), in, outputs[0], stream());
      break;
    case Max:
      distributed::detail::all_max(group(), in, outputs[0], stream());
      break;
    case Min:
      distributed::detail::all_min(group(), in, outputs[0], stream());
      break;
    default:
      throw std::runtime_error(
          "Only all reduce sum, min and max are supported for now");
  }
}

void AllGather::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 1);
  assert(outputs.size() == 1);

  auto [in, copied] = ensure_row_contiguous(inputs[0], stream());
  outputs[0].set_data(allocator::malloc(outputs[0].nbytes()));
  distributed::detail::all_gather(group(), in, outputs[0], stream());
  if (copied) {
    auto& enc = cpu::get_command_encoder(stream());
    enc.add_temporary(in);
  }
}

void Send::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 1);
  assert(outputs.size() == 1);

  auto [in, copied] = ensure_row_contiguous(inputs[0], stream());
  distributed::detail::send(group(), in, dst_, stream());
  outputs[0].copy_shared_buffer(inputs[0]);
  if (copied) {
    auto& enc = cpu::get_command_encoder(stream());
    enc.add_temporary(in);
  }
}

void Recv::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 0);
  assert(outputs.size() == 1);

  outputs[0].set_data(allocator::malloc(outputs[0].nbytes()));
  distributed::detail::recv(group(), outputs[0], src_, stream());
}

void ReduceScatter::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  throw std::runtime_error("[ReduceScatter] Not implemented yet.");
}

void AllToAll::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 1);
  assert(outputs.size() == 1);
  auto [in, copied] = ensure_row_contiguous(inputs[0], stream());
  outputs[0].set_data(allocator::malloc(outputs[0].nbytes()));
  distributed::detail::all_to_all(group(), in, outputs[0], stream());
  if (copied) {
    auto& enc = cpu::get_command_encoder(stream());
    enc.add_temporary(in);
  }
}

// Helper: make a row-contiguous array that shares the buffer of `parent`,
// starting at byte offset `byte_offset`, with shape `shape`.
// The caller must ensure `parent` stays alive while the returned array is used.
static array make_subview(
    const array& parent,
    const Shape& shape,
    size_t byte_offset,
    Dtype dtype) {
  // Compute strides for row-contiguous layout
  Strides strides(shape.size());
  if (!shape.empty()) {
    strides.back() = 1;
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
      strides[i] = strides[i + 1] * shape[i + 1];
    }
  }
  size_t num_elems = 1;
  for (auto s : shape) num_elems *= s;
  array::Flags flags;
  flags.contiguous = true;
  flags.row_contiguous = true;
  flags.col_contiguous = (shape.size() <= 1 || num_elems <= 1);

  // byte_offset in terms of elements
  int64_t elem_offset = static_cast<int64_t>(byte_offset / size_of(dtype));
  array view(shape, dtype, nullptr, {});
  view.copy_shared_buffer(parent, strides, flags, num_elems, elem_offset);
  return view;
}

void MoeDispatchExchange::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 2);
  assert(outputs.size() == 2);

  auto [tokens_in, tok_copied] = ensure_row_contiguous(inputs[0], stream());
  auto [indices_in, idx_copied] = ensure_row_contiguous(inputs[1], stream());

  // Allocate outputs before dispatch so callers can depend on their pointers
  outputs[0].set_data(allocator::malloc(outputs[0].nbytes()));
  outputs[1].set_data(allocator::malloc(outputs[1].nbytes()));

  int N = tokens_in.shape(0);
  int D = tokens_in.shape(1);
  int top_k = indices_in.shape(1);
  int world_size = group().size();
  int num_experts = num_experts_;
  int capacity = capacity_;
  Group grp = group();
  size_t elem_size = tokens_in.itemsize();
  Dtype dtype = tokens_in.dtype();

  // Capture raw pointers; arrays kept alive via add_temporary below
  const void* tok_raw = tokens_in.data<void>();
  const int32_t* idx_raw = indices_in.data<int32_t>();
  void* out0_raw = outputs[0].data<void>();
  int32_t* out1_raw = outputs[1].data<int32_t>();

  auto& enc = cpu::get_command_encoder(stream());

  // All data access is inside enc.dispatch so it runs after Metal GPU
  // command buffers for upstream ops have been committed and synchronized.
  enc.dispatch([tok_raw, idx_raw, out0_raw, out1_raw,
                N, D, top_k, world_size, num_experts, capacity,
                elem_size, dtype, grp]() mutable {
    int experts_per_device = num_experts / world_size;
    int cap_total = world_size * capacity; // total capacity slots per expert

    // Initialize route_indices to -1
    int32_t* route_ptr = out1_raw;
    std::fill(route_ptr, route_ptr + N * top_k, int32_t(-1));

    // Zero-initialize the output dispatched buffer
    // Output shape: [experts_per_device, cap_total, D]
    size_t out_nbytes = (size_t)experts_per_device * cap_total * D * elem_size;
    std::memset(out0_raw, 0, out_nbytes);

    const auto* tok_bytes = static_cast<const uint8_t*>(tok_raw);
    auto* out_bytes = static_cast<uint8_t*>(out0_raw);
    std::vector<int> expert_counts(num_experts, 0);

    // world_size == 1: local-only path (no send/recv)
    // New route_idx layout: flat_idx = local_expert * cap_total + dest_rank * capacity + pos
    // For world_size=1: dest_rank=0, cap_total=capacity
    //   => flat_idx = local_expert * capacity + pos  (same formula as old for ws=1)
    if (world_size == 1) {
      for (int k = 0; k < top_k; k++) {
        for (int n = 0; n < N; n++) {
          int eid = idx_raw[n * top_k + k];
          if (eid < 0 || eid >= num_experts) continue;
          int pos = expert_counts[eid]++;
          if (pos < capacity) {
            // dest_rank=0, local_expert=eid, cap_total=capacity
            int flat_idx = eid * cap_total + 0 * capacity + pos;
            route_ptr[n * top_k + k] = flat_idx;
            std::memcpy(
                out_bytes + flat_idx * D * elem_size,
                tok_bytes + n * D * elem_size,
                D * elem_size);
          }
          // else: route stays -1
        }
      }
      return;
    }

    // world_size == 2: variable exchange + local bypass
    if (world_size == 2) {
      int my_rank = grp.rank();
      int peer = 1 - my_rank;

      // Allocate send payload and meta buffers (worst case: all N*top_k tokens remote)
      int max_send = N * top_k;
      size_t send_payload_nbytes = (size_t)std::max(max_send, 1) * D * elem_size;
      size_t send_meta_nbytes = (size_t)std::max(max_send, 1) * 2 * sizeof(int32_t);

      array send_payload_arr(Shape{std::max(max_send, 1), D}, dtype, nullptr, {});
      send_payload_arr.set_data(allocator::malloc(send_payload_nbytes));

      array send_meta_arr(Shape{std::max(max_send, 1), 2}, int32, nullptr, {});
      send_meta_arr.set_data(allocator::malloc(send_meta_nbytes));

      auto* send_payload_bytes =
          static_cast<uint8_t*>(send_payload_arr.data<void>());
      auto* send_meta_ptr = send_meta_arr.data<int32_t>();

      int send_count = 0;

      // Dispatch: k-outer, n-inner deterministic loop
      for (int k = 0; k < top_k; k++) {
        for (int n = 0; n < N; n++) {
          int eid = idx_raw[n * top_k + k];
          if (eid < 0 || eid >= num_experts) continue;
          int dest_rank = eid / experts_per_device;
          int local_expert = eid % experts_per_device;
          int pos = expert_counts[eid]++;
          if (pos >= capacity) {
            // overflow: route stays -1
            continue;
          }
          // New layout: flat_idx = local_expert * cap_total + dest_rank * capacity + pos
          int flat_idx = local_expert * cap_total + dest_rank * capacity + pos;
          route_ptr[n * top_k + k] = flat_idx;

          if (dest_rank == my_rank) {
            // LOCAL: directly scatter into dispatched output
            std::memcpy(
                out_bytes + flat_idx * D * elem_size,
                tok_bytes + n * D * elem_size,
                D * elem_size);
          } else {
            // REMOTE: pack into send buffer
            std::memcpy(
                send_payload_bytes + send_count * D * elem_size,
                tok_bytes + n * D * elem_size,
                D * elem_size);
            send_meta_ptr[send_count * 2 + 0] = local_expert;
            send_meta_ptr[send_count * 2 + 1] = pos;
            send_count++;
          }
        }
      }

      // Step 1: Exchange remote token counts
      array send_count_arr(Shape{1}, int32, nullptr, {});
      send_count_arr.set_data(allocator::malloc(sizeof(int32_t)));
      send_count_arr.data<int32_t>()[0] = static_cast<int32_t>(send_count);

      array recv_count_arr(Shape{1}, int32, nullptr, {});
      recv_count_arr.set_data(allocator::malloc(sizeof(int32_t)));
      recv_count_arr.data<int32_t>()[0] = 0;

      // Both ranks exchange counts using blocking API with rank-parity
      auto* raw = grp.raw_group().get();
      bool recv_first = (grp.rank() % 2 != 0);

      if (recv_first) {
        raw->blocking_recv(recv_count_arr, peer);
        raw->blocking_send(send_count_arr, peer);
      } else {
        raw->blocking_send(send_count_arr, peer);
        raw->blocking_recv(recv_count_arr, peer);
      }
      int peer_send_count = recv_count_arr.data<int32_t>()[0];

      // Step 2: Exchange payload tokens using blocking API with rank-parity
      array recv_payload_arr(Shape{std::max(peer_send_count, 1), D}, dtype, nullptr, {});
      recv_payload_arr.set_data(
          allocator::malloc((size_t)std::max(peer_send_count, 1) * D * elem_size));

      if (recv_first) {
        if (peer_send_count > 0) {
          array payload_recv_view =
              make_subview(recv_payload_arr, {peer_send_count, D}, 0, dtype);
          raw->blocking_recv(payload_recv_view, peer);
        }
        if (send_count > 0) {
          array payload_send_view =
              make_subview(send_payload_arr, {send_count, D}, 0, dtype);
          raw->blocking_send(payload_send_view, peer);
        }
      } else {
        if (send_count > 0) {
          array payload_send_view =
              make_subview(send_payload_arr, {send_count, D}, 0, dtype);
          raw->blocking_send(payload_send_view, peer);
        }
        if (peer_send_count > 0) {
          array payload_recv_view =
              make_subview(recv_payload_arr, {peer_send_count, D}, 0, dtype);
          raw->blocking_recv(payload_recv_view, peer);
        }
      }

      // Step 3: Exchange metadata using blocking API with rank-parity
      array recv_meta_arr(Shape{std::max(peer_send_count, 1), 2}, int32, nullptr, {});
      recv_meta_arr.set_data(
          allocator::malloc(
              (size_t)std::max(peer_send_count, 1) * 2 * sizeof(int32_t)));

      if (recv_first) {
        if (peer_send_count > 0) {
          array meta_recv_view =
              make_subview(recv_meta_arr, {peer_send_count, 2}, 0, int32);
          raw->blocking_recv(meta_recv_view, peer);
        }
        if (send_count > 0) {
          array meta_send_view =
              make_subview(send_meta_arr, {send_count, 2}, 0, int32);
          raw->blocking_send(meta_send_view, peer);
        }
      } else {
        if (send_count > 0) {
          array meta_send_view =
              make_subview(send_meta_arr, {send_count, 2}, 0, int32);
          raw->blocking_send(meta_send_view, peer);
        }
        if (peer_send_count > 0) {
          array meta_recv_view =
              make_subview(recv_meta_arr, {peer_send_count, 2}, 0, int32);
          raw->blocking_recv(meta_recv_view, peer);
        }
      }

      // Step 4: Scatter received remote tokens into output
      const auto* recv_payload_bytes =
          static_cast<const uint8_t*>(recv_payload_arr.data<void>());
      const auto* recv_meta_ptr = recv_meta_arr.data<int32_t>();

      for (int i = 0; i < peer_send_count; i++) {
        int local_expert = recv_meta_ptr[i * 2 + 0];
        int slot_pos = recv_meta_ptr[i * 2 + 1];
        if (local_expert < 0 || local_expert >= experts_per_device ||
            slot_pos < 0 || slot_pos >= capacity) {
          throw std::runtime_error(
              "[MoeDispatchExchange] received out-of-bounds metadata: "
              "local_expert=" + std::to_string(local_expert) +
              " slot_pos=" + std::to_string(slot_pos));
        }
        // flat_idx in output: local_expert * cap_total + peer * capacity + slot_pos
        int recv_flat_idx = local_expert * cap_total + peer * capacity + slot_pos;
        std::memcpy(
            out_bytes + recv_flat_idx * D * elem_size,
            recv_payload_bytes + i * D * elem_size,
            D * elem_size);
      }
      return;
    }

    // world_size > 2: fallback to existing fixed all_to_all
    {
      int slots_per_device = experts_per_device * capacity;
      int total_slots = world_size * slots_per_device;
      size_t send_nbytes = (size_t)total_slots * D * elem_size;

      // Allocate send buffer: [total_slots, D] (layout: [W, E, C, D])
      array send_arr(Shape{total_slots, D}, dtype, nullptr, {});
      send_arr.set_data(allocator::malloc(send_nbytes));
      std::memset(send_arr.data<void>(), 0, send_nbytes);

      auto* send_bytes = static_cast<uint8_t*>(send_arr.data<void>());

      // Dispatch: k-outer, n-inner for deterministic slot assignment
      // Use NEW route_idx layout: flat_idx = local_expert * cap_total + dest_rank * capacity + pos
      // The send buffer uses old layout [W, E, C, D] for all_to_all compatibility
      for (int k = 0; k < top_k; k++) {
        for (int n = 0; n < N; n++) {
          int eid = idx_raw[n * top_k + k];
          if (eid < 0 || eid >= num_experts) continue;
          int pos = expert_counts[eid]++;
          if (pos < capacity) {
            int dest_rank = eid / experts_per_device;
            int local_expert = eid % experts_per_device;
            // Old send buffer layout for all_to_all: [W, E, C, D]
            int send_flat =
                dest_rank * slots_per_device + local_expert * capacity + pos;
            // New route_idx layout
            int new_flat_idx =
                local_expert * cap_total + dest_rank * capacity + pos;
            route_ptr[n * top_k + k] = new_flat_idx;
            std::memcpy(
                send_bytes + send_flat * D * elem_size,
                tok_bytes + n * D * elem_size,
                D * elem_size);
          }
          // else: route stays -1
        }
      }

      // Allocate recv buffer
      array recv_arr(Shape{total_slots, D}, dtype, nullptr, {});
      recv_arr.set_data(allocator::malloc(send_nbytes));

      // All-to-all exchange using blocking API
      grp.raw_group()->blocking_all_to_all(send_arr, recv_arr);

      // recv_arr layout: [world_size, experts_per_device, capacity, D]
      // output layout:   [experts_per_device, world_size * capacity, D]
      // out[e, w*capacity+c, d] = recv[w, e, c, d]
      const auto* recv_bytes =
          static_cast<const uint8_t*>(recv_arr.data<void>());

      for (int w = 0; w < world_size; w++) {
        for (int e = 0; e < experts_per_device; e++) {
          for (int c = 0; c < capacity; c++) {
            int recv_row = w * slots_per_device + e * capacity + c;
            int out_row = e * cap_total + w * capacity + c;
            std::memcpy(
                out_bytes + out_row * D * elem_size,
                recv_bytes + recv_row * D * elem_size,
                D * elem_size);
          }
        }
      }
      // send_arr and recv_arr go out of scope here; their allocator memory
      // is freed via the array destructor.
    }
  });

  // Keep input arrays alive until the dispatched lambda has executed
  enc.add_temporary(tokens_in);
  enc.add_temporary(indices_in);
}

void MoeCombineExchange::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 4);
  assert(outputs.size() == 1);

  // inputs: expert_outputs [E_local, cap_total, D],
  //         route_indices [N, top_k] int32,
  //         weights [N, top_k] float32,
  //         original_tokens [N, D]
  auto [expert_out, eo_copied] = ensure_row_contiguous(inputs[0], stream());
  auto [route_idx, ri_copied] = ensure_row_contiguous(inputs[1], stream());
  auto [weights_in, w_copied] = ensure_row_contiguous(inputs[2], stream());
  auto [orig_tok, ot_copied] = ensure_row_contiguous(inputs[3], stream());

  int experts_per_device = expert_out.shape(0);
  int cap_total = expert_out.shape(1);
  int D = expert_out.shape(2);
  int N = orig_tok.shape(0);
  int top_k = route_idx.shape(1);
  int world_size = group().size();
  int capacity = capacity_;
  Group grp = group();
  size_t elem_size = expert_out.itemsize();
  Dtype dtype = expert_out.dtype();

  // Allocate output before dispatch
  outputs[0].set_data(allocator::malloc(outputs[0].nbytes()));

  // Capture raw pointers; arrays kept alive via add_temporary below
  const void* eo_raw = expert_out.data<void>();
  const int32_t* ri_raw = route_idx.data<int32_t>();
  const float* w_raw = weights_in.data<float>();
  const void* orig_raw = orig_tok.data<void>();
  void* out0_raw = outputs[0].data<void>();

  auto& enc = cpu::get_command_encoder(stream());

  // All data access is inside enc.dispatch so it runs after Metal GPU
  // command buffers for upstream ops have been committed and synchronized.
  enc.dispatch([eo_raw, ri_raw, w_raw, orig_raw, out0_raw,
                experts_per_device, cap_total, D, N, top_k, world_size,
                capacity, elem_size, dtype, grp]() mutable {

    // world_size == 1: local-only path (no send/recv)
    // route_idx flat_idx = local_expert * cap_total + 0 * capacity + pos
    //                    = local_expert * capacity + pos  (cap_total == capacity for ws=1)
    // expert_outputs is indexed directly by flat_idx
    if (world_size == 1) {
      switch (dtype) {
        case float32: {
          const auto* eo_f = static_cast<const float*>(eo_raw);
          auto* out_f = static_cast<float*>(out0_raw);
          const auto* orig_f = static_cast<const float*>(orig_raw);
          for (int n = 0; n < N; n++) {
            float* dst = out_f + n * D;
            std::fill(dst, dst + D, 0.0f);
            bool has_valid = false;
            for (int k = 0; k < top_k; k++) {
              int flat_idx = ri_raw[n * top_k + k];
              if (flat_idx >= 0) {
                has_valid = true;
                float w = w_raw[n * top_k + k];
                const float* src = eo_f + flat_idx * D;
                for (int d = 0; d < D; d++) dst[d] += w * src[d];
              }
            }
            if (!has_valid) {
              std::memcpy(dst, orig_f + n * D, D * sizeof(float));
            }
          }
          break;
        }
        case float16: {
          const auto* eo_h = static_cast<const float16_t*>(eo_raw);
          auto* out_h = static_cast<float16_t*>(out0_raw);
          const auto* orig_h = static_cast<const float16_t*>(orig_raw);
          std::vector<float> accum(D);
          for (int n = 0; n < N; n++) {
            std::fill(accum.begin(), accum.end(), 0.0f);
            bool has_valid = false;
            for (int k = 0; k < top_k; k++) {
              int flat_idx = ri_raw[n * top_k + k];
              if (flat_idx >= 0) {
                has_valid = true;
                float w = w_raw[n * top_k + k];
                const float16_t* src = eo_h + flat_idx * D;
                for (int d = 0; d < D; d++) {
                  accum[d] += w * static_cast<float>(src[d]);
                }
              }
            }
            float16_t* dst = out_h + n * D;
            if (has_valid) {
              for (int d = 0; d < D; d++) dst[d] = float16_t(accum[d]);
            } else {
              std::memcpy(dst, orig_h + n * D, D * sizeof(float16_t));
            }
          }
          break;
        }
        case bfloat16: {
          const auto* eo_h = static_cast<const bfloat16_t*>(eo_raw);
          auto* out_h = static_cast<bfloat16_t*>(out0_raw);
          const auto* orig_h = static_cast<const bfloat16_t*>(orig_raw);
          std::vector<float> accum(D);
          for (int n = 0; n < N; n++) {
            std::fill(accum.begin(), accum.end(), 0.0f);
            bool has_valid = false;
            for (int k = 0; k < top_k; k++) {
              int flat_idx = ri_raw[n * top_k + k];
              if (flat_idx >= 0) {
                has_valid = true;
                float w = w_raw[n * top_k + k];
                const bfloat16_t* src = eo_h + flat_idx * D;
                for (int d = 0; d < D; d++) {
                  accum[d] += w * static_cast<float>(src[d]);
                }
              }
            }
            bfloat16_t* dst = out_h + n * D;
            if (has_valid) {
              for (int d = 0; d < D; d++) dst[d] = bfloat16_t(accum[d]);
            } else {
              std::memcpy(dst, orig_h + n * D, D * sizeof(bfloat16_t));
            }
          }
          break;
        }
        default:
          throw std::runtime_error(
              "[MoeCombineExchange] Unsupported dtype. Use float32, float16, or bfloat16.");
      }
      return;
    }

    // world_size == 2: variable exchange + local bypass
    if (world_size == 2) {
      int my_rank = grp.rank();
      int peer = 1 - my_rank;

      const auto* eo_bytes = static_cast<const uint8_t*>(eo_raw);

      // Collect remote (n, k) pairs in k-outer, n-inner order
      // These are tokens we routed to peer, in dispatch they were sent out,
      // now we expect results back from peer for these.
      std::vector<std::pair<int, int>> remote_nk;
      remote_nk.reserve(N * top_k);

      for (int k = 0; k < top_k; k++) {
        for (int n = 0; n < N; n++) {
          int flat_idx = ri_raw[n * top_k + k];
          if (flat_idx < 0) continue;
          // Decode dest_rank from flat_idx (new layout)
          // flat_idx = local_expert * cap_total + dest_rank * capacity + pos
          int remainder = flat_idx % cap_total;
          int dest_rank = remainder / capacity;
          if (dest_rank != my_rank) {
            remote_nk.push_back({n, k});
          }
        }
      }

      int my_remote_count = static_cast<int>(remote_nk.size());

      // Exchange counts: send how many results we expect from peer,
      // recv how many results peer expects from us (= how many tokens peer sent us)
      array send_count_arr(Shape{1}, int32, nullptr, {});
      send_count_arr.set_data(allocator::malloc(sizeof(int32_t)));
      send_count_arr.data<int32_t>()[0] = my_remote_count;

      array recv_count_arr(Shape{1}, int32, nullptr, {});
      recv_count_arr.set_data(allocator::malloc(sizeof(int32_t)));
      recv_count_arr.data<int32_t>()[0] = 0;

      auto* raw = grp.raw_group().get();
      bool recv_first = (grp.rank() % 2 != 0);

      if (recv_first) {
        raw->blocking_recv(recv_count_arr, peer);
        raw->blocking_send(send_count_arr, peer);
      } else {
        raw->blocking_send(send_count_arr, peer);
        raw->blocking_recv(recv_count_arr, peer);
      }
      int peer_remote_count = recv_count_arr.data<int32_t>()[0];
      // peer_remote_count = number of tokens peer sent us in dispatch
      //                   = number of results we must send back

      // Exchange meta: send our remote (local_expert, pos) pairs to peer
      // so peer can scatter our results into their combined output.
      // Recv peer's (local_expert, pos) pairs to look up results in our expert_outputs.
      //
      // For each entry in remote_nk: decode local_expert and pos from route_idx
      // (these are local_expert and pos AS SEEN BY peer, i.e., peer's expert index and slot)
      int send_meta_count = my_remote_count;
      array send_result_meta(Shape{std::max(send_meta_count, 1), 2}, int32, nullptr, {});
      send_result_meta.set_data(
          allocator::malloc(
              (size_t)std::max(send_meta_count, 1) * 2 * sizeof(int32_t)));
      auto* send_result_meta_ptr = send_result_meta.data<int32_t>();

      for (int i = 0; i < send_meta_count; i++) {
        auto [n, k] = remote_nk[i];
        int flat_idx = ri_raw[n * top_k + k];
        // flat_idx = local_expert * cap_total + dest_rank * capacity + pos
        int local_expert = flat_idx / cap_total;
        int remainder = flat_idx % cap_total;
        int pos = remainder % capacity;
        send_result_meta_ptr[i * 2 + 0] = local_expert;
        send_result_meta_ptr[i * 2 + 1] = pos;
      }

      int recv_meta_count = peer_remote_count;
      array recv_result_meta(Shape{std::max(recv_meta_count, 1), 2}, int32, nullptr, {});
      recv_result_meta.set_data(
          allocator::malloc(
              (size_t)std::max(recv_meta_count, 1) * 2 * sizeof(int32_t)));

      if (recv_first) {
        if (recv_meta_count > 0) {
          array meta_recv_view =
              make_subview(recv_result_meta, {recv_meta_count, 2}, 0, int32);
          raw->blocking_recv(meta_recv_view, peer);
        }
        if (send_meta_count > 0) {
          array meta_send_view =
              make_subview(send_result_meta, {send_meta_count, 2}, 0, int32);
          raw->blocking_send(meta_send_view, peer);
        }
      } else {
        if (send_meta_count > 0) {
          array meta_send_view =
              make_subview(send_result_meta, {send_meta_count, 2}, 0, int32);
          raw->blocking_send(meta_send_view, peer);
        }
        if (recv_meta_count > 0) {
          array meta_recv_view =
              make_subview(recv_result_meta, {recv_meta_count, 2}, 0, int32);
          raw->blocking_recv(meta_recv_view, peer);
        }
      }

      // Pack results for peer: for each of peer's tokens (indexed by recv_result_meta),
      // look up expert_outputs at: local_expert * cap_total + my_rank * capacity + pos
      const auto* recv_meta_ptr = recv_result_meta.data<int32_t>();

      array send_results(Shape{std::max(recv_meta_count, 1), D}, dtype, nullptr, {});
      send_results.set_data(
          allocator::malloc(
              (size_t)std::max(recv_meta_count, 1) * D * elem_size));
      auto* send_results_bytes = static_cast<uint8_t*>(send_results.data<void>());

      for (int i = 0; i < recv_meta_count; i++) {
        int local_expert = recv_meta_ptr[i * 2 + 0];
        int slot_pos = recv_meta_ptr[i * 2 + 1];
        if (local_expert < 0 || local_expert >= experts_per_device ||
            slot_pos < 0 || slot_pos >= capacity) {
          throw std::runtime_error(
              "[MoeCombineExchange] received out-of-bounds result metadata: "
              "local_expert=" + std::to_string(local_expert) +
              " slot_pos=" + std::to_string(slot_pos));
        }
        // Expert output for peer's token: slot from peer in our layout
        // = local_expert * cap_total + peer * capacity + slot_pos
        int eo_flat = local_expert * cap_total + peer * capacity + slot_pos;
        std::memcpy(
            send_results_bytes + i * D * elem_size,
            eo_bytes + eo_flat * D * elem_size,
            D * elem_size);
      }

      // Exchange results
      array recv_results(Shape{std::max(send_meta_count, 1), D}, dtype, nullptr, {});
      recv_results.set_data(
          allocator::malloc(
              (size_t)std::max(send_meta_count, 1) * D * elem_size));

      if (recv_first) {
        if (send_meta_count > 0) {
          array res_recv_view =
              make_subview(recv_results, {send_meta_count, D}, 0, dtype);
          raw->blocking_recv(res_recv_view, peer);
        }
        if (recv_meta_count > 0) {
          array res_send_view =
              make_subview(send_results, {recv_meta_count, D}, 0, dtype);
          raw->blocking_send(res_send_view, peer);
        }
      } else {
        if (recv_meta_count > 0) {
          array res_send_view =
              make_subview(send_results, {recv_meta_count, D}, 0, dtype);
          raw->blocking_send(res_send_view, peer);
        }
        if (send_meta_count > 0) {
          array res_recv_view =
              make_subview(recv_results, {send_meta_count, D}, 0, dtype);
          raw->blocking_recv(res_recv_view, peer);
        }
      }

      // Build a lookup: for each (n*top_k+k) that is remote, what is its
      // index in recv_results? remote_nk and recv_results are in same order.
      std::vector<int> remote_recv_idx(N * top_k, -1);
      for (int i = 0; i < static_cast<int>(remote_nk.size()); i++) {
        auto [n, k] = remote_nk[i];
        remote_recv_idx[n * top_k + k] = i;
      }

      // Weighted combine: for each token n, accumulate local + remote results
      switch (dtype) {
        case float32: {
          const auto* eo_f = static_cast<const float*>(eo_raw);
          const auto* recv_f =
              static_cast<const float*>(recv_results.data<void>());
          auto* out_f = static_cast<float*>(out0_raw);
          const auto* orig_f = static_cast<const float*>(orig_raw);

          for (int n = 0; n < N; n++) {
            float* dst = out_f + n * D;
            std::fill(dst, dst + D, 0.0f);
            bool has_valid = false;
            for (int k = 0; k < top_k; k++) {
              int flat_idx = ri_raw[n * top_k + k];
              if (flat_idx < 0) continue;
              has_valid = true;
              float w = w_raw[n * top_k + k];
              int rri = remote_recv_idx[n * top_k + k];
              if (rri >= 0) {
                // Remote result
                const float* src = recv_f + rri * D;
                for (int d = 0; d < D; d++) dst[d] += w * src[d];
              } else {
                // Local result: read directly from expert_outputs
                const float* src = eo_f + flat_idx * D;
                for (int d = 0; d < D; d++) dst[d] += w * src[d];
              }
            }
            if (!has_valid) {
              std::memcpy(dst, orig_f + n * D, D * sizeof(float));
            }
          }
          break;
        }
        case float16: {
          const auto* eo_h = static_cast<const float16_t*>(eo_raw);
          const auto* recv_h =
              static_cast<const float16_t*>(recv_results.data<void>());
          auto* out_h = static_cast<float16_t*>(out0_raw);
          const auto* orig_h = static_cast<const float16_t*>(orig_raw);
          std::vector<float> accum(D);

          for (int n = 0; n < N; n++) {
            std::fill(accum.begin(), accum.end(), 0.0f);
            bool has_valid = false;
            for (int k = 0; k < top_k; k++) {
              int flat_idx = ri_raw[n * top_k + k];
              if (flat_idx < 0) continue;
              has_valid = true;
              float w = w_raw[n * top_k + k];
              int rri = remote_recv_idx[n * top_k + k];
              if (rri >= 0) {
                const float16_t* src = recv_h + rri * D;
                for (int d = 0; d < D; d++) {
                  accum[d] += w * static_cast<float>(src[d]);
                }
              } else {
                const float16_t* src = eo_h + flat_idx * D;
                for (int d = 0; d < D; d++) {
                  accum[d] += w * static_cast<float>(src[d]);
                }
              }
            }
            float16_t* dst = out_h + n * D;
            if (has_valid) {
              for (int d = 0; d < D; d++) dst[d] = float16_t(accum[d]);
            } else {
              std::memcpy(dst, orig_h + n * D, D * sizeof(float16_t));
            }
          }
          break;
        }
        case bfloat16: {
          const auto* eo_h = static_cast<const bfloat16_t*>(eo_raw);
          const auto* recv_h =
              static_cast<const bfloat16_t*>(recv_results.data<void>());
          auto* out_h = static_cast<bfloat16_t*>(out0_raw);
          const auto* orig_h = static_cast<const bfloat16_t*>(orig_raw);
          std::vector<float> accum(D);

          for (int n = 0; n < N; n++) {
            std::fill(accum.begin(), accum.end(), 0.0f);
            bool has_valid = false;
            for (int k = 0; k < top_k; k++) {
              int flat_idx = ri_raw[n * top_k + k];
              if (flat_idx < 0) continue;
              has_valid = true;
              float w = w_raw[n * top_k + k];
              int rri = remote_recv_idx[n * top_k + k];
              if (rri >= 0) {
                const bfloat16_t* src = recv_h + rri * D;
                for (int d = 0; d < D; d++) {
                  accum[d] += w * static_cast<float>(src[d]);
                }
              } else {
                const bfloat16_t* src = eo_h + flat_idx * D;
                for (int d = 0; d < D; d++) {
                  accum[d] += w * static_cast<float>(src[d]);
                }
              }
            }
            bfloat16_t* dst = out_h + n * D;
            if (has_valid) {
              for (int d = 0; d < D; d++) dst[d] = bfloat16_t(accum[d]);
            } else {
              std::memcpy(dst, orig_h + n * D, D * sizeof(bfloat16_t));
            }
          }
          break;
        }
        default:
          throw std::runtime_error(
              "[MoeCombineExchange] Unsupported dtype. Use float32, float16, or bfloat16.");
      }
      return;
    }

    // world_size > 2: fallback to existing all_to_all
    {
      int total_slots = experts_per_device * cap_total; // E * W * C
      size_t total_bytes = (size_t)total_slots * D * elem_size;
      int slots_per_device = experts_per_device * capacity;

      // Reverse transpose: [E, W*C, D] -> [W, E, C, D]
      // send_arr[w, e, c, d] = expert_out[e, w*capacity+c, d]
      array send_arr(Shape{total_slots, D}, dtype, nullptr, {});
      send_arr.set_data(allocator::malloc(total_bytes));

      auto* send_bytes = static_cast<uint8_t*>(send_arr.data<void>());
      const auto* eo_bytes = static_cast<const uint8_t*>(eo_raw);

      for (int e = 0; e < experts_per_device; e++) {
        for (int w = 0; w < world_size; w++) {
          for (int c = 0; c < capacity; c++) {
            int eo_row = e * cap_total + w * capacity + c;
            int send_row = w * slots_per_device + e * capacity + c;
            std::memcpy(
                send_bytes + send_row * D * elem_size,
                eo_bytes + eo_row * D * elem_size,
                D * elem_size);
          }
        }
      }

      array recv_arr(Shape{total_slots, D}, dtype, nullptr, {});
      recv_arr.set_data(allocator::malloc(total_bytes));

      grp.raw_group()->blocking_all_to_all(send_arr, recv_arr);

      // recv_arr: [total_slots, D] flat, layout: [W, E, C, D]
      // route_idx uses NEW layout: flat_idx = local_expert * cap_total + dest_rank * cap + pos
      // Map new route_idx -> old recv_row:
      //   local_expert = flat_idx / cap_total
      //   dest_rank    = (flat_idx % cap_total) / capacity
      //   pos          = (flat_idx % cap_total) % capacity
      //   recv_row     = dest_rank * slots_per_device + local_expert * capacity + pos

      switch (dtype) {
        case float32: {
          const auto* recv_f = static_cast<const float*>(recv_arr.data<void>());
          auto* out_f = static_cast<float*>(out0_raw);
          const auto* orig_f = static_cast<const float*>(orig_raw);

          for (int n = 0; n < N; n++) {
            float* dst = out_f + n * D;
            std::fill(dst, dst + D, 0.0f);
            bool has_valid = false;
            for (int k = 0; k < top_k; k++) {
              int flat_idx = ri_raw[n * top_k + k];
              if (flat_idx >= 0) {
                has_valid = true;
                float w = w_raw[n * top_k + k];
                int local_expert = flat_idx / cap_total;
                int remainder = flat_idx % cap_total;
                int dest_rank = remainder / capacity;
                int pos = remainder % capacity;
                int recv_row = dest_rank * slots_per_device +
                               local_expert * capacity + pos;
                const float* src = recv_f + recv_row * D;
                for (int d = 0; d < D; d++) dst[d] += w * src[d];
              }
            }
            if (!has_valid) {
              std::memcpy(dst, orig_f + n * D, D * sizeof(float));
            }
          }
          break;
        }
        case float16: {
          const auto* recv_h =
              static_cast<const float16_t*>(recv_arr.data<void>());
          auto* out_h = static_cast<float16_t*>(out0_raw);
          const auto* orig_h = static_cast<const float16_t*>(orig_raw);

          std::vector<float> accum(D);
          for (int n = 0; n < N; n++) {
            std::fill(accum.begin(), accum.end(), 0.0f);
            bool has_valid = false;
            for (int k = 0; k < top_k; k++) {
              int flat_idx = ri_raw[n * top_k + k];
              if (flat_idx >= 0) {
                has_valid = true;
                float w = w_raw[n * top_k + k];
                int local_expert = flat_idx / cap_total;
                int remainder = flat_idx % cap_total;
                int dest_rank = remainder / capacity;
                int pos = remainder % capacity;
                int recv_row = dest_rank * slots_per_device +
                               local_expert * capacity + pos;
                const float16_t* src = recv_h + recv_row * D;
                for (int d = 0; d < D; d++) {
                  accum[d] += w * static_cast<float>(src[d]);
                }
              }
            }
            float16_t* dst = out_h + n * D;
            if (has_valid) {
              for (int d = 0; d < D; d++) {
                dst[d] = float16_t(accum[d]);
              }
            } else {
              std::memcpy(dst, orig_h + n * D, D * sizeof(float16_t));
            }
          }
          break;
        }
        case bfloat16: {
          const auto* recv_h =
              static_cast<const bfloat16_t*>(recv_arr.data<void>());
          auto* out_h = static_cast<bfloat16_t*>(out0_raw);
          const auto* orig_h = static_cast<const bfloat16_t*>(orig_raw);

          std::vector<float> accum(D);
          for (int n = 0; n < N; n++) {
            std::fill(accum.begin(), accum.end(), 0.0f);
            bool has_valid = false;
            for (int k = 0; k < top_k; k++) {
              int flat_idx = ri_raw[n * top_k + k];
              if (flat_idx >= 0) {
                has_valid = true;
                float w = w_raw[n * top_k + k];
                int local_expert = flat_idx / cap_total;
                int remainder = flat_idx % cap_total;
                int dest_rank = remainder / capacity;
                int pos = remainder % capacity;
                int recv_row = dest_rank * slots_per_device +
                               local_expert * capacity + pos;
                const bfloat16_t* src = recv_h + recv_row * D;
                for (int d = 0; d < D; d++) {
                  accum[d] += w * static_cast<float>(src[d]);
                }
              }
            }
            bfloat16_t* dst = out_h + n * D;
            if (has_valid) {
              for (int d = 0; d < D; d++) {
                dst[d] = bfloat16_t(accum[d]);
              }
            } else {
              std::memcpy(dst, orig_h + n * D, D * sizeof(bfloat16_t));
            }
          }
          break;
        }
        default:
          throw std::runtime_error(
              "[MoeCombineExchange] Unsupported dtype. Use float32, float16, or bfloat16.");
      }
      // send_arr and recv_arr go out of scope here; their allocator memory
      // is freed via the array destructor.
    }
  });

  // Keep input arrays alive until the dispatched lambda has executed
  enc.add_temporary(expert_out);
  enc.add_temporary(route_idx);
  enc.add_temporary(weights_in);
  enc.add_temporary(orig_tok);
}
} // namespace mlx::core::distributed
