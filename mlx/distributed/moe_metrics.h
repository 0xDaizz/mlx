// Copyright © 2024 Apple Inc.

#pragma once

#include <atomic>
#include <cstdint>
#include <string>
#include <unordered_map>

namespace mlx::core::distributed {

struct MoeMetrics {
  // --- Counters ---
  std::atomic<uint64_t> dispatch_calls{0};
  std::atomic<uint64_t> combine_calls{0};
  std::atomic<uint64_t> cpu_backend_calls{0};
  std::atomic<uint64_t> metal_backend_calls{0};
  std::atomic<uint64_t> fallback_to_cpu_count{0};
  std::atomic<uint64_t> total_tokens_dispatched{0};
  std::atomic<uint64_t> warmup_completed{0};

  // --- Traffic/Routing ---
  std::atomic<uint64_t> overflow_count{0};
  std::atomic<uint64_t> remote_tokens_total{0};
  std::atomic<uint64_t> local_tokens_total{0};

  // --- Latency (microseconds, last-value store) ---
  std::atomic<uint64_t> dispatch_route_cpu_us{0};
  std::atomic<uint64_t> dispatch_comm_us{0};
  std::atomic<uint64_t> combine_comm_us{0};
  std::atomic<uint64_t> dispatch_total_us{0};
  std::atomic<uint64_t> combine_total_us{0};

  /// Reset all counters to 0.
  void reset();

  /// Return a snapshot of all metrics as string->uint64 map.
  /// This is pure C++ (no Python types). Python binding converts to dict.
  std::unordered_map<std::string, uint64_t> snapshot() const;

  /// Global singleton.
  static MoeMetrics& global();
};

} // namespace mlx::core::distributed
