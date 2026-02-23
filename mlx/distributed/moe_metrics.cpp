// Copyright © 2024 Apple Inc.

#include "mlx/distributed/moe_metrics.h"

namespace mlx::core::distributed {

void MoeMetrics::reset() {
  dispatch_calls.store(0, std::memory_order_relaxed);
  combine_calls.store(0, std::memory_order_relaxed);
  cpu_backend_calls.store(0, std::memory_order_relaxed);
  metal_backend_calls.store(0, std::memory_order_relaxed);
  fallback_to_cpu_count.store(0, std::memory_order_relaxed);
  total_tokens_dispatched.store(0, std::memory_order_relaxed);
  warmup_completed.store(0, std::memory_order_relaxed);

  overflow_count.store(0, std::memory_order_relaxed);
  remote_tokens_total.store(0, std::memory_order_relaxed);
  local_tokens_total.store(0, std::memory_order_relaxed);

  dispatch_route_cpu_us.store(0, std::memory_order_relaxed);
  dispatch_comm_us.store(0, std::memory_order_relaxed);
  combine_comm_us.store(0, std::memory_order_relaxed);
  dispatch_total_us.store(0, std::memory_order_relaxed);
  combine_total_us.store(0, std::memory_order_relaxed);
}

std::unordered_map<std::string, uint64_t> MoeMetrics::snapshot() const {
  return {
      {"dispatch_calls", dispatch_calls.load(std::memory_order_relaxed)},
      {"combine_calls", combine_calls.load(std::memory_order_relaxed)},
      {"cpu_backend_calls", cpu_backend_calls.load(std::memory_order_relaxed)},
      {"metal_backend_calls",
       metal_backend_calls.load(std::memory_order_relaxed)},
      {"fallback_to_cpu_count",
       fallback_to_cpu_count.load(std::memory_order_relaxed)},
      {"total_tokens_dispatched",
       total_tokens_dispatched.load(std::memory_order_relaxed)},
      {"warmup_completed", warmup_completed.load(std::memory_order_relaxed)},
      {"overflow_count", overflow_count.load(std::memory_order_relaxed)},
      {"remote_tokens_total",
       remote_tokens_total.load(std::memory_order_relaxed)},
      {"local_tokens_total",
       local_tokens_total.load(std::memory_order_relaxed)},
      {"dispatch_route_cpu_us",
       dispatch_route_cpu_us.load(std::memory_order_relaxed)},
      {"dispatch_comm_us", dispatch_comm_us.load(std::memory_order_relaxed)},
      {"combine_comm_us", combine_comm_us.load(std::memory_order_relaxed)},
      {"dispatch_total_us", dispatch_total_us.load(std::memory_order_relaxed)},
      {"combine_total_us", combine_total_us.load(std::memory_order_relaxed)},
  };
}

MoeMetrics& MoeMetrics::global() {
  static MoeMetrics instance;
  return instance;
}

} // namespace mlx::core::distributed
