# Copyright © 2026 Apple Inc.

import math
from typing import NamedTuple, Optional

import mlx.core as mx
from mlx.nn.layers.activations import silu
from mlx.nn.layers.base import Module
from mlx.nn.layers.linear import Linear


class DispatchMeta(NamedTuple):
    """Metadata for expert dispatch/combine round-trip."""
    expert_indices: mx.array      # [N, top_k] expert assignments
    weights: mx.array             # [N, top_k] routing weights
    positions: mx.array           # [N, top_k] slot positions in dispatch buffer
    overflow_mask: mx.array       # [N, 1] True if token overflowed capacity
    num_experts: int
    capacity: int
    world_size: int


class TopKRouter(Module):
    """Top-K expert router with load balancing auxiliary loss.

    Routes each token to the top-k experts based on a learned gate.

    Args:
        hidden_dim: Input hidden dimension.
        num_experts: Total number of experts.
        top_k: Number of experts per token. Default: ``2``.
        capacity_factor: Capacity scaling factor. Default: ``1.25``.
        aux_loss_coeff: Coefficient for load balancing loss. Default: ``0.01``.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        top_k: int = 2,
        capacity_factor: float = 1.25,
        aux_loss_coeff: float = 0.01,
    ):
        super().__init__()
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        if top_k > num_experts:
            raise ValueError(
                f"top_k ({top_k}) must not exceed num_experts ({num_experts})")
        self.gate = Linear(hidden_dim, num_experts, bias=False)
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.aux_loss_coeff = aux_loss_coeff

    def __call__(self, x: mx.array):
        """Route tokens to experts.

        Args:
            x: Input tensor of shape ``[N, hidden_dim]``.

        Returns:
            Tuple of (weights, indices, aux_loss):
                - weights: ``[N, top_k]`` normalized routing weights
                - indices: ``[N, top_k]`` expert indices (integers in [0, num_experts))
                - aux_loss: scalar load balancing loss
        """
        # x: [N, D] -> logits: [N, num_experts]
        logits = self.gate(x)
        probs = mx.softmax(logits, axis=-1)

        # Get top-k experts per token
        # mx.argpartition gives indices of top-k elements (unordered within top-k)
        # We negate probs to get the largest values
        neg_probs = -probs
        top_k_indices = mx.argpartition(neg_probs, kth=self.top_k - 1, axis=-1)[
            :, : self.top_k
        ]

        # Stop gradient on discrete routing decisions
        expert_indices = mx.stop_gradient(top_k_indices)

        # Gather the weights for selected experts
        # Use take_along_axis to gather probs at the top-k indices
        weights = mx.take_along_axis(probs, expert_indices, axis=-1)

        # Normalize weights so they sum to 1 per token
        weights = weights / mx.maximum(weights.sum(axis=-1, keepdims=True), 1e-6)

        # Compute auxiliary load balancing loss
        aux_loss = self._load_balance_loss(probs, expert_indices)

        return weights, expert_indices, aux_loss

    def _load_balance_loss(self, probs: mx.array, expert_indices: mx.array) -> mx.array:
        """GShard load balancing loss: num_experts * sum(f_e * P_e).

        Args:
            probs: [N, num_experts] routing probabilities.
            expert_indices: [N, top_k] selected expert indices.

        Returns:
            Scalar auxiliary loss.
        """
        num_tokens = probs.shape[0]

        # f_e: fraction of tokens routed to each expert
        # Create one-hot and sum across top_k selections
        one_hot = mx.zeros_like(probs)
        for k in range(self.top_k):
            indices_k = expert_indices[:, k]  # [N]
            rows = mx.arange(num_tokens)
            one_hot = one_hot.at[rows, indices_k].add(1.0)
        f_e = mx.mean(one_hot, axis=0) / self.top_k  # [num_experts]

        # P_e: mean routing probability per expert
        P_e = mx.mean(probs, axis=0)  # [num_experts]

        # GShard loss
        loss = self.aux_loss_coeff * self.num_experts * mx.sum(f_e * P_e)
        return loss


def _compute_capacity(
    num_tokens: int,
    top_k: int,
    capacity_factor: float,
    num_experts: int,
) -> int:
    """Compute expert buffer capacity."""
    if num_experts <= 0:
        raise ValueError(f"num_experts must be positive, got {num_experts}")
    return max(1, math.ceil(num_tokens * top_k * capacity_factor / num_experts))


class Expert(Module):
    """Single expert network with SwiGLU activation.

    Args:
        hidden_dim: Input/output hidden dimension.
        expert_dim: Expert intermediate dimension.
    """

    def __init__(self, hidden_dim: int, expert_dim: int):
        super().__init__()
        self.w_gate = Linear(hidden_dim, expert_dim, bias=False)
        self.w_up = Linear(hidden_dim, expert_dim, bias=False)
        self.w_down = Linear(expert_dim, hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass with SwiGLU activation.

        Args:
            x: Input tensor of shape ``[..., hidden_dim]``.

        Returns:
            Output tensor of shape ``[..., hidden_dim]``.
        """
        return self.w_down(silu(self.w_gate(x)) * self.w_up(x))


def expert_dispatch(
    tokens: mx.array,
    expert_indices: mx.array,
    weights: mx.array,
    num_experts: int,
    capacity_factor: float,
    group: Optional["mx.distributed.Group"] = None,
) -> tuple:
    """Dispatch tokens to experts across devices.

    Args:
        tokens: [N, D] input tokens.
        expert_indices: [N, top_k] expert assignments.
        weights: [N, top_k] routing weights.
        num_experts: Total number of experts across all devices.
        capacity_factor: Capacity scaling factor.
        group: Distributed group. If None, uses local-only dispatch.

    Returns:
        Tuple of (dispatched, meta):
            - dispatched: [experts_per_device, capacity, D] expert inputs for this device
            - meta: DispatchMeta for use with expert_combine
    """
    num_tokens, hidden_dim = tokens.shape
    top_k = expert_indices.shape[1]

    world_size = group.size() if group is not None else 1
    if world_size > 1 and num_experts % world_size != 0:
        raise ValueError(
            f"num_experts ({num_experts}) must be divisible by "
            f"world_size ({world_size})"
        )
    experts_per_device = num_experts // world_size
    capacity = _compute_capacity(num_tokens, top_k, capacity_factor, num_experts)

    # In distributed mode, synchronize capacity across ranks so all ranks
    # use the same buffer size for all_to_all. Different ranks may have
    # different token counts (e.g. uneven last batch).
    if world_size > 1 and group is not None:
        cap_arr = mx.array(capacity, dtype=mx.int32)
        cap_arr = mx.distributed.all_max(cap_arr, group=group)
        mx.eval(cap_arr)
        capacity = cap_arr.item()

    # Build dispatch buffer: [world_size, experts_per_device, capacity, D]
    dispatch_buffer = mx.zeros(
        (world_size, experts_per_device, capacity, hidden_dim),
        dtype=tokens.dtype,
    )

    # Compute positions using cumulative sum per expert
    # For each (token, k) pair, determine which device and local expert slot
    overflow_mask = mx.zeros((num_tokens, 1), dtype=mx.bool_)
    positions = mx.full((num_tokens, top_k), -1, dtype=mx.int32)

    # Track running count per expert across all top-k columns to avoid
    # position collisions when a token is routed to the same expert by
    # multiple top-k selections.
    expert_counts = mx.zeros((num_experts,), dtype=mx.int32)

    for k in range(top_k):
        indices_k = expert_indices[:, k]  # [N]

        # Compute position within each expert's capacity buffer
        for e in range(num_experts):
            mask_e = (indices_k == e)  # [N] bool
            # Offset cumsum by the running count for this expert
            cum_pos = (
                mx.cumsum(mask_e.astype(mx.int32), axis=0) - 1 + expert_counts[e]
            )
            # Only place tokens that fit within capacity
            valid = mask_e & (cum_pos < capacity)
            overflow_k = mask_e & (cum_pos >= capacity)
            overflow_mask = overflow_mask | overflow_k.reshape(-1, 1)

            d = e // experts_per_device
            le = e % experts_per_device

            # Scatter tokens into dispatch buffer
            # For valid tokens going to expert e:
            valid_pos = mx.where(valid, cum_pos, mx.array(-1, dtype=mx.int32))
            positions = mx.where(
                (indices_k == e).reshape(-1, 1) & (mx.arange(top_k) == k).reshape(1, -1),
                mx.broadcast_to(valid_pos.reshape(-1, 1), (num_tokens, top_k)),
                positions,
            )

            # Update running count for this expert
            expert_counts = expert_counts.at[e].add(
                mask_e.astype(mx.int32).sum()
            )

            # Evaluate once to get concrete values for the scatter loop
            # This avoids per-iteration mx.eval() from bool(mx.array)
            mx.eval(valid, cum_pos)

            # Scatter unweighted tokens (weights applied in expert_combine)
            for n_idx in range(num_tokens):
                if valid[n_idx].item():
                    p = cum_pos[n_idx].item()
                    dispatch_buffer = dispatch_buffer.at[d, le, p].add(
                        tokens[n_idx]
                    )

    meta = DispatchMeta(
        expert_indices=expert_indices,
        weights=weights,
        positions=positions,
        overflow_mask=overflow_mask,
        num_experts=num_experts,
        capacity=capacity,
        world_size=world_size,
    )

    # All-to-all exchange if distributed
    if world_size > 1 and group is not None:
        flat = dispatch_buffer.reshape(world_size, -1)
        exchanged = mx.distributed.all_to_all(flat, group=group)
        dispatched = exchanged.reshape(world_size, experts_per_device, capacity, hidden_dim)
        # Each device processes experts_per_device experts,
        # data from all devices combined
        # Reshape: [world_size, experts_per_device, capacity, D] -> [experts_per_device, world_size * capacity, D]
        dispatched = mx.transpose(dispatched, axes=(1, 0, 2, 3)).reshape(
            experts_per_device, world_size * capacity, hidden_dim
        )
    else:
        # Local only: [1, experts_per_device, capacity, D] -> [experts_per_device, capacity, D]
        dispatched = dispatch_buffer.squeeze(0)

    return dispatched, meta


def expert_combine(
    expert_outputs: mx.array,
    meta: DispatchMeta,
    original_tokens: mx.array,
    group: Optional["mx.distributed.Group"] = None,
) -> mx.array:
    """Combine expert outputs back to token order.

    Args:
        expert_outputs: [experts_per_device, capacity_total, D] expert output tokens.
        meta: DispatchMeta from expert_dispatch.
        original_tokens: [N, D] original input tokens for residual.
        group: Distributed group.

    Returns:
        [N, D] combined output tokens.
    """
    world_size = meta.world_size
    experts_per_device = meta.num_experts // world_size
    capacity = meta.capacity
    hidden_dim = original_tokens.shape[-1]
    num_tokens = original_tokens.shape[0]

    if world_size > 1 and group is not None:
        # Reshape back for all_to_all: [experts_per_device, world_size * capacity, D]
        # -> [world_size, experts_per_device, capacity, D]
        reshaped = expert_outputs.reshape(
            experts_per_device, world_size, capacity, hidden_dim
        )
        reshaped = mx.transpose(reshaped, axes=(1, 0, 2, 3))
        flat = reshaped.reshape(world_size, -1)
        exchanged = mx.distributed.all_to_all(flat, group=group)
        result_buffer = exchanged.reshape(world_size, experts_per_device, capacity, hidden_dim)
    else:
        result_buffer = expert_outputs.reshape(1, experts_per_device, capacity, hidden_dim)

    # Gather results back to original token positions
    combined = mx.zeros_like(original_tokens)
    top_k = meta.expert_indices.shape[1]

    for k in range(top_k):
        indices_k = meta.expert_indices[:, k]
        positions_k = meta.positions[:, k]
        weights_k = meta.weights[:, k]  # [N] routing weights for k-th selection
        device_idx = indices_k // experts_per_device
        local_expert = indices_k % experts_per_device

        # Evaluate once to get concrete values for the gather loop
        mx.eval(positions_k, device_idx, local_expert)

        for n_idx in range(num_tokens):
            pos = positions_k[n_idx].item()
            if pos >= 0:
                d = device_idx[n_idx].item()
                le = local_expert[n_idx].item()
                combined = combined.at[n_idx].add(
                    weights_k[n_idx] * result_buffer[d, le, pos]
                )

    # Apply overflow residual
    combined = mx.where(meta.overflow_mask, original_tokens, combined)

    return combined


class MixtureOfExperts(Module):
    """Mixture of Experts layer with Expert Parallelism support.

    Args:
        hidden_dim: Input/output hidden dimension.
        expert_dim: Expert intermediate dimension.
        num_experts: Total number of experts.
        top_k: Number of experts per token. Default: ``2``.
        capacity_factor: Capacity scaling factor. Default: ``1.25``.
        aux_loss_coeff: Load balance loss coefficient. Default: ``0.01``.
    """

    def __init__(
        self,
        hidden_dim: int,
        expert_dim: int,
        num_experts: int,
        top_k: int = 2,
        capacity_factor: float = 1.25,
        aux_loss_coeff: float = 0.01,
    ):
        super().__init__()

        if num_experts <= 0:
            raise ValueError(f"num_experts must be positive, got {num_experts}")
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        if top_k > num_experts:
            raise ValueError(
                f"top_k ({top_k}) must not exceed num_experts ({num_experts})")

        # Determine distributed context
        self._world_size = 1
        self._group = None
        try:
            group = mx.distributed.init(strict=False)
            if group.size() > 1:
                # Probe all_to_all support; some backends (ring, NCCL)
                # do not implement it and would crash on every forward pass.
                try:
                    test = mx.distributed.all_to_all(
                        mx.zeros((group.size(),)), group=group
                    )
                    mx.eval(test)
                    self._world_size = group.size()
                    self._group = group
                except RuntimeError:
                    # Backend doesn't support all_to_all, fall back to local-only
                    pass
        except Exception:
            pass

        if num_experts % self._world_size != 0:
            raise ValueError(
                f"num_experts ({num_experts}) must be divisible by "
                f"world_size ({self._world_size})"
            )

        self.hidden_dim = hidden_dim
        self.expert_dim = expert_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor

        # Router
        self.router = TopKRouter(
            hidden_dim, num_experts, top_k, capacity_factor, aux_loss_coeff
        )

        # Local experts for this device
        experts_per_device = num_experts // self._world_size
        self.experts = [
            Expert(hidden_dim, expert_dim) for _ in range(experts_per_device)
        ]

    def __call__(self, x: mx.array):
        """Forward pass.

        Args:
            x: Input tensor of shape ``[N, hidden_dim]``.

        Returns:
            Tuple of (output, aux_loss):
                - output: ``[N, hidden_dim]`` combined expert outputs
                - aux_loss: scalar load balancing loss
        """
        # Route
        weights, expert_indices, aux_loss = self.router(x)

        # Dispatch
        dispatched, meta = expert_dispatch(
            x, expert_indices, weights,
            self.num_experts, self.capacity_factor,
            group=self._group,
        )

        # Run local experts
        expert_outputs = self._run_local_experts(dispatched)

        # Combine
        output = expert_combine(
            expert_outputs, meta, x,
            group=self._group,
        )

        return output, aux_loss

    def _run_local_experts(self, dispatched: mx.array) -> mx.array:
        """Run local experts on dispatched tokens.

        Args:
            dispatched: [experts_per_device, capacity_total, D] dispatched inputs.

        Returns:
            [experts_per_device, capacity_total, D] expert outputs.
        """
        outputs = []
        for i, expert in enumerate(self.experts):
            expert_input = dispatched[i]  # [capacity_total, D]
            expert_output = expert(expert_input)  # [capacity_total, D]
            outputs.append(expert_output)
        return mx.stack(outputs, axis=0)
