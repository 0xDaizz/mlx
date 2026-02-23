// Copyright © 2024 Apple Inc.

#include <chrono>
#include <cstdlib>

#include "mlx/backend/metal/metal.h"
#include "mlx/distributed/moe_policy.h"

namespace mlx::core::distributed {

namespace {

int64_t now_ms() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

int64_t env_int(const char* name, int64_t def) {
  const char* v = std::getenv(name);
  if (!v)
    return def;
  try {
    return std::stoll(v);
  } catch (...) {
    return def;
  }
}

} // namespace

MoeBackend MoePolicy::resolve(int N, int top_k, int D, int elem_size) {
  // 1. Environment variable override (highest priority)
  const char* env = std::getenv("MLX_MOE_EP_BACKEND");
  if (env) {
    std::string val(env);
    if (val == "cpu")
      return MoeBackend::Cpu;
    if (val == "metal")
      return MoeBackend::Metal;
    // "auto" or unknown falls through to heuristic
  }

  // 2. Metal availability check
  if (!metal::is_available()) {
    return MoeBackend::Cpu;
  }

  // 3. Tick cooldown calls counter
  tick_cooldown();

  // 4. Cooldown check (OR semantics: EITHER condition forces CPU)
  {
    int64_t until = cooldown_until_ms.load(std::memory_order_relaxed);
    int64_t calls = cooldown_calls_remaining.load(std::memory_order_relaxed);
    if (now_ms() < until || calls > 0) {
      return MoeBackend::Cpu;
    }
  }

  // 5. Zone 1: small N -> CPU
  int64_t cpu_n_max = env_int("MLX_MOE_EP_CPU_N_MAX", 64);
  if (N <= cpu_n_max) {
    return MoeBackend::Cpu;
  }

  // 6. Zone 2: large N -> Metal
  int64_t gpu_n_min = env_int("MLX_MOE_EP_GPU_N_MIN", 256);
  if (N >= gpu_n_min) {
    return MoeBackend::Metal;
  }

  // 7. Zone 3: middle zone -> byte threshold
  int64_t gpu_switch_bytes = env_int("MLX_MOE_EP_GPU_SWITCH_BYTES", 2 * 1024 * 1024);
  int64_t work_bytes =
      static_cast<int64_t>(N) * top_k * D * elem_size;
  return (work_bytes >= gpu_switch_bytes) ? MoeBackend::Metal : MoeBackend::Cpu;
}

void MoePolicy::record_metal_failure() {
  int64_t cooldown_ms = env_int("MLX_MOE_EP_COOLDOWN_MS", 5000);
  int64_t cooldown_calls = env_int("MLX_MOE_EP_COOLDOWN_CALLS", 1000);

  cooldown_until_ms.store(
      now_ms() + cooldown_ms, std::memory_order_relaxed);
  cooldown_calls_remaining.store(cooldown_calls, std::memory_order_relaxed);
}

void MoePolicy::tick_cooldown() {
  int64_t calls = cooldown_calls_remaining.load(std::memory_order_relaxed);
  while (calls > 0) {
    if (cooldown_calls_remaining.compare_exchange_weak(
            calls, calls - 1, std::memory_order_relaxed)) {
      break;
    }
    // calls is updated by compare_exchange_weak on failure
  }
}

MoePolicy& MoePolicy::global() {
  static MoePolicy instance;
  return instance;
}

} // namespace mlx::core::distributed
