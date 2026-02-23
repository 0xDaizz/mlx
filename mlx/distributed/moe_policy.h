// Copyright © 2024 Apple Inc.

#pragma once

#include <atomic>
#include <cstdint>

#include "mlx/distributed/primitives.h"

namespace mlx::core::distributed {

struct MoePolicy {
  // Cooldown state
  std::atomic<int64_t> cooldown_until_ms{0};
  std::atomic<int64_t> cooldown_calls_remaining{0};

  /// Resolve the backend for a given workload.
  /// 3-zone policy: small N -> CPU, large N -> Metal, middle -> byte threshold.
  /// Cooldown overrides to CPU if EITHER time or calls remaining is active
  /// (OR semantics).
  MoeBackend resolve(int N, int top_k, int D, int elem_size);

  /// Record a Metal failure and activate cooldown.
  void record_metal_failure();

  /// Tick cooldown calls counter (called on every resolve).
  void tick_cooldown();

  /// Global singleton.
  static MoePolicy& global();
};

} // namespace mlx::core::distributed
