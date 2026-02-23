// Copyright © 2024 Apple Inc.

#pragma once

#include <optional>

#include "mlx/distributed/distributed.h"
#include "mlx/dtype.h"

namespace mlx::core::distributed {

/// Warm up MoE EP infrastructure: RDMA paths, Metal kernel JIT cache,
/// and memory allocator. Idempotent — repeated calls with the same
/// parameters are no-ops.
///
/// Args:
///   group: distributed group (nullopt -> global group)
///   num_experts: total experts across all devices (0 -> skip Metal warmup)
///   capacity: per-expert capacity (0 -> skip Metal warmup)
///   hidden_dim: D dimension (0 -> skip Metal warmup)
///   dtype: data type for Metal warmup tensors (default float16)
void moe_ep_warmup(
    std::optional<Group> group,
    int num_experts,
    int capacity,
    int hidden_dim,
    Dtype dtype);

} // namespace mlx::core::distributed
