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
  Stream strm = stream();
  size_t elem_size = tokens_in.itemsize();
  Dtype dtype = tokens_in.dtype();

  // Capture raw pointers; arrays kept alive via add_temporary below
  const void* tok_raw = tokens_in.data<void>();
  const int32_t* idx_raw = indices_in.data<int32_t>();
  void* out0_raw = outputs[0].data<void>();
  int32_t* out1_raw = outputs[1].data<int32_t>();

  auto& enc = cpu::get_command_encoder(stream());
  enc.set_input_array(inputs[0]);
  enc.set_input_array(inputs[1]);
  enc.set_output_array(outputs[0]);
  enc.set_output_array(outputs[1]);

  // All data access is inside enc.dispatch so it runs after Metal GPU
  // command buffers for upstream ops have been committed and synchronized.
  enc.dispatch([tok_raw, idx_raw, out0_raw, out1_raw,
                N, D, top_k, world_size, num_experts, capacity,
                elem_size, dtype, grp, strm]() mutable {
    int experts_per_device = num_experts / world_size;
    int slots_per_device = experts_per_device * capacity;
    int total_slots = world_size * slots_per_device;
    size_t send_nbytes = (size_t)total_slots * D * elem_size;

    // Allocate send buffer: [total_slots, D] (layout: [W, E, C, D])
    array send_arr(Shape{total_slots, D}, dtype, nullptr, {});
    send_arr.set_data(allocator::malloc(send_nbytes));
    std::memset(send_arr.data<void>(), 0, send_nbytes);

    // Initialize route_indices to -1
    int32_t* route_ptr = out1_raw;
    std::fill(route_ptr, route_ptr + N * top_k, int32_t(-1));

    auto* send_bytes = static_cast<uint8_t*>(send_arr.data<void>());
    const auto* tok_bytes = static_cast<const uint8_t*>(tok_raw);
    std::vector<int> expert_counts(num_experts, 0);

    // Dispatch: k-outer, n-inner for deterministic slot assignment
    for (int k = 0; k < top_k; k++) {
      for (int n = 0; n < N; n++) {
        int eid = idx_raw[n * top_k + k];
        if (eid < 0 || eid >= num_experts) continue;
        int pos = expert_counts[eid]++;
        if (pos < capacity) {
          int dest_rank = eid / experts_per_device;
          int local_expert = eid % experts_per_device;
          int flat_idx =
              dest_rank * slots_per_device + local_expert * capacity + pos;
          route_ptr[n * top_k + k] = flat_idx;
          std::memcpy(
              send_bytes + flat_idx * D * elem_size,
              tok_bytes + n * D * elem_size,
              D * elem_size);
        }
        // else: route stays -1
      }
    }

    // Allocate recv buffer
    array recv_arr(Shape{total_slots, D}, dtype, nullptr, {});
    recv_arr.set_data(allocator::malloc(send_nbytes));

    // All-to-all exchange
    if (world_size > 1) {
      distributed::detail::all_to_all(grp, send_arr, recv_arr, strm);
    } else {
      std::memcpy(recv_arr.data<void>(), send_arr.data<void>(), send_nbytes);
    }

    // recv_arr layout: [world_size, experts_per_device, capacity, D]
    // output layout:   [experts_per_device, world_size * capacity, D]
    // out[e, w*capacity+c, d] = recv[w, e, c, d]
    int cap_total = world_size * capacity;
    auto* out_bytes = static_cast<uint8_t*>(out0_raw);
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
  Stream strm = stream();
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
  enc.set_input_array(inputs[0]);
  enc.set_input_array(inputs[1]);
  enc.set_input_array(inputs[2]);
  enc.set_input_array(inputs[3]);
  enc.set_output_array(outputs[0]);

  // All data access is inside enc.dispatch so it runs after Metal GPU
  // command buffers for upstream ops have been committed and synchronized.
  enc.dispatch([eo_raw, ri_raw, w_raw, orig_raw, out0_raw,
                experts_per_device, cap_total, D, N, top_k, world_size,
                capacity, elem_size, dtype, grp, strm]() mutable {
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

    if (world_size > 1) {
      distributed::detail::all_to_all(grp, send_arr, recv_arr, strm);
    } else {
      std::memcpy(recv_arr.data<void>(), send_arr.data<void>(), total_bytes);
    }

    // recv_arr: [total_slots, D] flat, layout: [W, E, C, D]
    // Gather: combined[n] = sum_k(weights[n,k] * recv[route_indices[n,k]])
    // If all route_indices[n,:] == -1: combined[n] = original_tokens[n]
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
              const float* src = recv_f + flat_idx * D;
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
              const float16_t* src = recv_h + flat_idx * D;
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
              const bfloat16_t* src = recv_h + flat_idx * D;
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
  });

  // Keep input arrays alive until the dispatched lambda has executed
  enc.add_temporary(expert_out);
  enc.add_temporary(route_idx);
  enc.add_temporary(weights_in);
  enc.add_temporary(orig_tok);
}
} // namespace mlx::core::distributed
