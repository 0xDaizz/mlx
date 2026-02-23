// Copyright © 2024 Apple Inc.

#include <cmath>
#include <cstdlib>
#include <mutex>
#include <set>
#include <tuple>

#include "mlx/backend/metal/metal.h"
#include "mlx/distributed/moe_metrics.h"
#include "mlx/distributed/moe_warmup.h"
#include "mlx/distributed/ops.h"
#include "mlx/ops.h"
#include "mlx/transforms.h"

namespace mlx::core::distributed {

namespace {

struct WarmupKey {
  uintptr_t group_ptr;
  int rank;
  std::string backend_str; // "rdma" or "metal"
  Dtype dtype;
  int d_bucket; // floor(log2(hidden_dim))

  bool operator<(const WarmupKey& o) const {
    return std::tie(group_ptr, rank, backend_str, dtype, d_bucket) <
        std::tie(o.group_ptr, o.rank, o.backend_str, o.dtype, o.d_bucket);
  }
};

std::mutex warmup_mutex;
std::set<WarmupKey> warmup_done;

bool already_done(const WarmupKey& key) {
  std::lock_guard<std::mutex> lock(warmup_mutex);
  return !warmup_done.insert(key).second;
}

} // namespace

void moe_ep_warmup(
    std::optional<Group> group_,
    int num_experts,
    int capacity,
    int hidden_dim,
    Dtype dtype) {
  // Environment variable kill-switch
  const char* env = std::getenv("MLX_MOE_EP_WARMUP");
  if (env && std::string(env) == "0") {
    return;
  }

  // Configurable round count
  int rounds = 5;
  const char* rounds_env = std::getenv("MLX_MOE_EP_WARMUP_ROUNDS");
  if (rounds_env) {
    try {
      int r = std::stoi(rounds_env);
      if (r > 0) {
        rounds = r;
      }
    } catch (...) {
    }
  }

  Group group = group_.has_value() ? group_.value() : distributed::init();
  int ws = group.size();
  int rank = group.rank();
  auto* raw_ptr = group.raw_group().get();
  uintptr_t gptr = reinterpret_cast<uintptr_t>(raw_ptr);

  // === Stage 1: RDMA warmup (ws > 1) ===
  if (ws > 1) {
    WarmupKey rdma_key{gptr, rank, "rdma", dtype, 0};
    if (!already_done(rdma_key)) {
      // all_sum rounds
      auto dummy = mlx::core::ones({4});
      for (int i = 0; i < rounds; i++) {
        auto r = distributed::all_sum(dummy, group);
        mlx::core::eval({r});
      }
      // all_to_all rounds — shape[0] must be divisible by world_size
      try {
        auto a2a_dummy = mlx::core::zeros({ws * 4});
        for (int i = 0; i < rounds; i++) {
          auto r = distributed::all_to_all(a2a_dummy, group);
          mlx::core::eval({r});
        }
      } catch (const std::runtime_error&) {
        // Backend doesn't support all_to_all, skip warmup stage
      }
    }
  }

  // === Stage 2: Metal kernel JIT compile (if available and params provided)
  // ===
  if (num_experts > 0 && capacity > 0 && hidden_dim > 0 &&
      metal::is_available()) {
    int d_bucket = static_cast<int>(std::floor(std::log2(hidden_dim)));
    WarmupKey metal_key{gptr, rank, "metal", dtype, d_bucket};
    if (!already_done(metal_key)) {
      // Tiny tensors to trigger JIT compile without real data movement
      auto tok = mlx::core::zeros({1, hidden_dim}, dtype);
      auto idx = mlx::core::zeros({1, 1}, int32);

      // Dispatch with metal backend
      auto [disp, ri] = distributed::moe_dispatch_exchange(
          tok,
          idx,
          num_experts,
          capacity,
          group,
          /*deterministic=*/true,
          /*backend=*/"metal");
      mlx::core::eval({disp, ri});

      // Combine with metal backend
      auto weights = mlx::core::ones({1, 1}, float32);
      auto combined = distributed::moe_combine_exchange(
          disp,
          ri,
          weights,
          tok,
          num_experts,
          capacity,
          group,
          /*deterministic=*/true,
          /*backend=*/"metal");
      mlx::core::eval({combined});
    }
  }

  // Record metric
  MoeMetrics::global().warmup_completed.fetch_add(1, std::memory_order_relaxed);
}

} // namespace mlx::core::distributed
