# Copyright © 2026 Apple Inc.

import math
import unittest

import mlx.core as mx
import mlx.nn as nn
import mlx_tests

# Imports will be validated once the MoE module is implemented.
# If the module path changes, update accordingly.
from mlx.nn.layers.moe import (
    TopKRouter,
    Expert,
    MixtureOfExperts,
    _compute_capacity,
    expert_dispatch,
    expert_combine,
    DispatchMeta,
)


class TestTopKRouter(mlx_tests.MLXTestCase):
    def test_output_shapes(self):
        """Router should produce correct output shapes."""
        hidden_dim, num_experts, top_k = 64, 8, 2
        router = TopKRouter(hidden_dim, num_experts, top_k)
        x = mx.random.normal((16, hidden_dim))
        weights, indices, aux_loss = router(x)
        mx.eval(weights, indices, aux_loss)

        self.assertEqual(weights.shape, (16, top_k))
        self.assertEqual(indices.shape, (16, top_k))
        self.assertEqual(aux_loss.shape, ())

    def test_index_range(self):
        """Expert indices should be in [0, num_experts)."""
        hidden_dim, num_experts, top_k = 64, 8, 2
        router = TopKRouter(hidden_dim, num_experts, top_k)
        x = mx.random.normal((32, hidden_dim))
        _, indices, _ = router(x)
        mx.eval(indices)

        self.assertTrue(mx.all(indices >= 0).item())
        self.assertTrue(mx.all(indices < num_experts).item())

    def test_weights_sum(self):
        """Routing weights should approximately sum to 1 per token."""
        hidden_dim, num_experts, top_k = 64, 8, 2
        router = TopKRouter(hidden_dim, num_experts, top_k)
        x = mx.random.normal((16, hidden_dim))
        weights, _, _ = router(x)
        mx.eval(weights)

        weight_sums = weights.sum(axis=-1)
        self.assertTrue(
            mx.allclose(weight_sums, mx.ones_like(weight_sums), atol=1e-5).item()
        )

    def test_gradient_flow(self):
        """Gradients should flow through the router gate."""
        hidden_dim, num_experts, top_k = 32, 4, 2
        router = TopKRouter(hidden_dim, num_experts, top_k)
        x = mx.random.normal((8, hidden_dim))

        def loss_fn(model, x):
            weights, _, aux_loss = model(x)
            return weights.sum() + aux_loss

        loss, grads = nn.value_and_grad(router, loss_fn)(router, x)
        mx.eval(loss, grads)

        # Gate weight should have non-zero gradient
        self.assertTrue(mx.any(grads["gate"]["weight"] != 0).item())

    def test_aux_loss_positive(self):
        """Auxiliary loss should be non-negative."""
        hidden_dim, num_experts, top_k = 64, 8, 2
        router = TopKRouter(hidden_dim, num_experts, top_k)
        x = mx.random.normal((16, hidden_dim))
        _, _, aux_loss = router(x)
        mx.eval(aux_loss)

        self.assertTrue(aux_loss.item() >= 0)

    def test_different_top_k_values(self):
        """Router should work with different top_k values."""
        hidden_dim, num_experts = 64, 8
        x = mx.random.normal((16, hidden_dim))

        for top_k in [1, 2, 4]:
            router = TopKRouter(hidden_dim, num_experts, top_k)
            weights, indices, aux_loss = router(x)
            mx.eval(weights, indices, aux_loss)

            self.assertEqual(weights.shape, (16, top_k))
            self.assertEqual(indices.shape, (16, top_k))

    def test_single_token(self):
        """Router should handle a single-token input."""
        hidden_dim, num_experts, top_k = 64, 8, 2
        router = TopKRouter(hidden_dim, num_experts, top_k)
        x = mx.random.normal((1, hidden_dim))
        weights, indices, aux_loss = router(x)
        mx.eval(weights, indices, aux_loss)

        self.assertEqual(weights.shape, (1, top_k))
        self.assertEqual(indices.shape, (1, top_k))

    def test_top_k_validation(self):
        """Should raise error for invalid top_k."""
        with self.assertRaises(ValueError):
            TopKRouter(64, 8, top_k=0)
        with self.assertRaises(ValueError):
            TopKRouter(64, 8, top_k=9)

    def test_empty_batch(self):
        """Router should handle zero-token input without NaN."""
        router = TopKRouter(64, 8, top_k=2)
        x = mx.zeros((0, 64))
        weights, indices, aux_loss = router(x)
        mx.eval(weights, indices, aux_loss)
        self.assertEqual(weights.shape, (0, 2))
        self.assertEqual(indices.shape, (0, 2))
        self.assertTrue(mx.isfinite(aux_loss).item())
        self.assertEqual(aux_loss.item(), 0.0)


class TestExpert(mlx_tests.MLXTestCase):
    def test_output_shape(self):
        """Expert should preserve input/output dimensions."""
        hidden_dim, expert_dim = 64, 128
        expert = Expert(hidden_dim, expert_dim)
        x = mx.random.normal((8, hidden_dim))
        out = expert(x)
        mx.eval(out)

        self.assertEqual(out.shape, (8, hidden_dim))

    def test_gradient_flow(self):
        """Gradients should flow through expert."""
        hidden_dim, expert_dim = 32, 64
        expert = Expert(hidden_dim, expert_dim)
        x = mx.random.normal((4, hidden_dim))

        def loss_fn(model, x):
            return model(x).sum()

        loss, grads = nn.value_and_grad(expert, loss_fn)(expert, x)
        mx.eval(loss, grads)

        self.assertTrue(mx.any(grads["w_gate"]["weight"] != 0).item())
        self.assertTrue(mx.any(grads["w_up"]["weight"] != 0).item())
        self.assertTrue(mx.any(grads["w_down"]["weight"] != 0).item())

    def test_single_token(self):
        """Expert should handle single-token input."""
        hidden_dim, expert_dim = 64, 128
        expert = Expert(hidden_dim, expert_dim)
        x = mx.random.normal((1, hidden_dim))
        out = expert(x)
        mx.eval(out)

        self.assertEqual(out.shape, (1, hidden_dim))

    def test_empty_input(self):
        """Expert should handle zero-token input."""
        hidden_dim, expert_dim = 64, 128
        expert = Expert(hidden_dim, expert_dim)
        x = mx.zeros((0, hidden_dim))
        out = expert(x)
        mx.eval(out)

        self.assertEqual(out.shape, (0, hidden_dim))


class TestComputeCapacity(mlx_tests.MLXTestCase):
    def test_basic(self):
        """Test capacity computation."""
        # 16 tokens, top_k=2, factor=1.25, 8 experts
        cap = _compute_capacity(16, 2, 1.25, 8)
        expected = max(1, math.ceil(16 * 2 * 1.25 / 8))  # ceil(5.0) = 5
        self.assertEqual(cap, expected)

    def test_minimum_one(self):
        """Capacity should be at least 1."""
        cap = _compute_capacity(0, 2, 1.0, 8)
        self.assertEqual(cap, 1)

    def test_exact_division(self):
        """Test when division is exact."""
        # 8 tokens, top_k=1, factor=1.0, 4 experts -> ceil(2.0) = 2
        cap = _compute_capacity(8, 1, 1.0, 4)
        self.assertEqual(cap, 2)

    def test_large_capacity_factor(self):
        """Larger capacity factor yields larger capacity."""
        cap_low = _compute_capacity(16, 2, 1.0, 8)
        cap_high = _compute_capacity(16, 2, 2.0, 8)
        self.assertGreaterEqual(cap_high, cap_low)


class TestMixtureOfExperts(mlx_tests.MLXTestCase):
    def test_forward_shape(self):
        """MoE forward should produce correct output shape."""
        hidden_dim, expert_dim, num_experts = 64, 128, 4
        moe = MixtureOfExperts(hidden_dim, expert_dim, num_experts, top_k=2)
        x = mx.random.normal((8, hidden_dim))
        output, aux_loss = moe(x)
        mx.eval(output, aux_loss)

        self.assertEqual(output.shape, (8, hidden_dim))
        self.assertEqual(aux_loss.shape, ())

    def test_backward(self):
        """MoE should support backward pass."""
        hidden_dim, expert_dim, num_experts = 32, 64, 4
        moe = MixtureOfExperts(hidden_dim, expert_dim, num_experts, top_k=2)
        x = mx.random.normal((4, hidden_dim))

        def loss_fn(model, x):
            output, aux_loss = model(x)
            return output.sum() + aux_loss

        loss, grads = nn.value_and_grad(moe, loss_fn)(moe, x)
        mx.eval(loss, grads)

        # At least the router should have gradients
        self.assertIsNotNone(grads["router"]["gate"]["weight"])

    def test_parameter_count(self):
        """Verify parameter structure."""
        hidden_dim, expert_dim, num_experts = 64, 128, 4
        moe = MixtureOfExperts(hidden_dim, expert_dim, num_experts, top_k=2)

        params = moe.parameters()
        # Should have router and experts
        self.assertIn("router", params)
        self.assertIn("experts", params)
        # Single process: all experts are local
        self.assertEqual(len(params["experts"]), num_experts)

    def test_validation_error(self):
        """Should raise error for invalid num_experts or top_k."""
        with self.assertRaises(Exception):
            MixtureOfExperts(64, 128, 0)
        with self.assertRaises(ValueError):
            MixtureOfExperts(64, 128, 4, top_k=0)
        with self.assertRaises(ValueError):
            MixtureOfExperts(64, 128, 4, top_k=5)

    def test_different_top_k(self):
        """MoE should work with different top_k values."""
        hidden_dim, expert_dim, num_experts = 64, 128, 4
        x = mx.random.normal((8, hidden_dim))

        for top_k in [1, 2]:
            moe = MixtureOfExperts(
                hidden_dim, expert_dim, num_experts, top_k=top_k
            )
            output, aux_loss = moe(x)
            mx.eval(output, aux_loss)

            self.assertEqual(output.shape, (8, hidden_dim))

    def test_deterministic_with_seed(self):
        """Same seed should produce same results."""
        hidden_dim, expert_dim, num_experts = 32, 64, 4
        x = mx.random.normal((4, hidden_dim))

        moe1 = MixtureOfExperts(hidden_dim, expert_dim, num_experts, top_k=2)
        moe2 = MixtureOfExperts(hidden_dim, expert_dim, num_experts, top_k=2)
        moe2.update(moe1.parameters())

        out1, loss1 = moe1(x)
        out2, loss2 = moe2(x)
        mx.eval(out1, out2, loss1, loss2)

        self.assertTrue(mx.allclose(out1, out2).item())
        self.assertTrue(mx.allclose(loss1, loss2).item())

    def test_large_batch(self):
        """MoE should handle larger batch sizes."""
        hidden_dim, expert_dim, num_experts = 64, 128, 8
        moe = MixtureOfExperts(hidden_dim, expert_dim, num_experts, top_k=2)
        x = mx.random.normal((128, hidden_dim))
        output, aux_loss = moe(x)
        mx.eval(output, aux_loss)

        self.assertEqual(output.shape, (128, hidden_dim))

    def test_partial_overflow_preserves_valid_routes(self):
        """Tokens with at least one valid route should not be replaced by residual."""
        hidden_dim = 4
        num_experts = 2
        capacity = 1
        top_k = 2

        # token0: expert 0 valid (pos=0), expert 1 overflow (pos=-1)
        # token1: both routes overflow (pos=-1, -1)
        positions = mx.array([[0, -1], [-1, -1]], dtype=mx.int32)
        expert_indices = mx.array([[0, 1], [0, 1]], dtype=mx.int32)
        weights = mx.array([[0.6, 0.4], [0.5, 0.5]])
        overflow_mask = mx.array([[True], [True]])

        meta = DispatchMeta(
            expert_indices=expert_indices,
            weights=weights,
            positions=positions,
            overflow_mask=overflow_mask,
            num_experts=num_experts,
            capacity=capacity,
            world_size=1,
        )

        # expert_outputs: [num_experts, capacity, hidden_dim]
        expert_outputs = mx.ones((num_experts, capacity, hidden_dim)) * 10.0
        original_tokens = mx.zeros((2, hidden_dim))

        combined = expert_combine(expert_outputs, meta, original_tokens)
        mx.eval(combined)

        # Verify bug reproduction condition: token0 has overflow_mask=True
        # but should still use expert output because it has a valid route.
        self.assertTrue(meta.overflow_mask[0].item())
        has_valid = (meta.positions[0] >= 0).any().item()
        self.assertTrue(has_valid)

        # token0: weight=0.6 * expert_output=10.0 → expected 6.0 per dim
        expected_token0 = mx.full((hidden_dim,), 0.6 * 10.0)
        self.assertTrue(mx.allclose(combined[0], expected_token0).item())
        # token1: all overflow → should be original (zeros)
        self.assertTrue(mx.array_equal(combined[1], original_tokens[1]).item())


class TestVectorizedDispatchCombine(mlx_tests.MLXTestCase):
    def test_dispatch_combine_duplicate_expert_across_k(self):
        """Same expert selected by both top_k slots should not collide positions."""
        N, D = 4, 8
        num_experts = 4
        capacity_factor = 2.0  # generous capacity

        tokens = mx.random.normal((N, D))
        # Force token 0 and token 1 to route to the same expert (expert 0) for both k=0 and k=1
        expert_indices = mx.array([
            [0, 0],  # token 0: expert 0 twice
            [0, 0],  # token 1: expert 0 twice
            [1, 2],  # token 2: different experts
            [3, 1],  # token 3: different experts
        ], dtype=mx.int32)
        weights = mx.array([
            [0.6, 0.4],
            [0.5, 0.5],
            [0.7, 0.3],
            [0.8, 0.2],
        ])

        dispatched, meta = expert_dispatch(
            tokens, expert_indices, weights,
            num_experts=num_experts, capacity_factor=capacity_factor,
        )
        mx.eval(dispatched, *meta)

        # Positions for token 0 and token 1 should be different across k
        # (expert_counts accumulation ensures no collision)
        pos_token0 = meta.positions[0]  # [top_k]
        pos_token1 = meta.positions[1]  # [top_k]

        # All positions should be >= 0 (no overflow with generous capacity)
        self.assertTrue(mx.all(meta.positions >= 0).item(),
                        f"Expected all valid positions, got {meta.positions}")

        # For tokens routed to same expert: k=0 and k=1 positions must differ
        self.assertNotEqual(pos_token0[0].item(), pos_token0[1].item(),
                            "Same expert positions should differ across k")

        # Round-trip test: dispatch then combine with identity expert
        expert_outputs = dispatched  # identity
        combined = expert_combine(expert_outputs, meta, tokens)
        mx.eval(combined)
        # Combined should not contain NaN
        self.assertTrue(mx.all(mx.isfinite(combined)).item())

    def test_dispatch_combine_overflow_boundary(self):
        """Capacity boundary: first 2 tokens fit, last 2 overflow."""
        N, D = 4, 8
        num_experts = 2

        tokens = mx.ones((N, D))  # all-ones for easy verification
        # All tokens go to expert 0 for k=0, expert 1 for k=1
        expert_indices = mx.array([
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
        ], dtype=mx.int32)
        weights = mx.array([
            [0.6, 0.4],
            [0.6, 0.4],
            [0.6, 0.4],
            [0.6, 0.4],
        ])

        # capacity = max(1, ceil(4 * 2 * capacity_factor / 2))
        # With capacity_factor = 0.5: ceil(4 * 2 * 0.5 / 2) = ceil(2.0) = 2
        dispatched, meta = expert_dispatch(
            tokens, expert_indices, weights,
            num_experts=num_experts, capacity_factor=0.5,
        )
        mx.eval(dispatched, *meta)

        capacity = meta.capacity
        self.assertEqual(capacity, 2)

        # For k=0 (expert 0): tokens 0,1 should have positions 0,1; tokens 2,3 overflow
        positions_k0 = meta.positions[:, 0]
        mx.eval(positions_k0)
        self.assertEqual(positions_k0[0].item(), 0)
        self.assertEqual(positions_k0[1].item(), 1)
        self.assertEqual(positions_k0[2].item(), -1)  # overflow
        self.assertEqual(positions_k0[3].item(), -1)  # overflow

        # Overflow mask should be True for tokens 2 and 3
        self.assertTrue(meta.overflow_mask[2].item())
        self.assertTrue(meta.overflow_mask[3].item())

    def test_dispatch_combine_empty_batch(self):
        """N=0 input should produce correct shapes without errors."""
        D = 8
        num_experts = 4

        tokens = mx.zeros((0, D))
        expert_indices = mx.zeros((0, 2), dtype=mx.int32)
        weights = mx.zeros((0, 2))

        dispatched, meta = expert_dispatch(
            tokens, expert_indices, weights,
            num_experts=num_experts, capacity_factor=1.25,
        )
        mx.eval(dispatched, *meta)

        # Shape checks
        self.assertEqual(meta.positions.shape, (0, 2))
        self.assertEqual(meta.overflow_mask.shape, (0, 1))
        self.assertEqual(dispatched.shape[0], num_experts)  # experts_per_device
        self.assertEqual(dispatched.shape[-1], D)

        # Round-trip with combine
        expert_outputs = dispatched
        combined = expert_combine(expert_outputs, meta, tokens)
        mx.eval(combined)
        self.assertEqual(combined.shape, (0, D))

    def test_combine_all_invalid_residual(self):
        """All routes invalid → combined should equal original_tokens."""
        N, D = 4, 8
        num_experts = 2
        capacity = 2

        original_tokens = mx.random.normal((N, D))
        expert_outputs = mx.random.normal((num_experts, capacity, D))

        # Manually construct meta with all-invalid positions
        positions = mx.full((N, 2), -1, dtype=mx.int32)
        expert_indices = mx.array([[0, 1]] * N, dtype=mx.int32)
        weights = mx.array([[0.5, 0.5]] * N)
        overflow_mask = mx.ones((N, 1), dtype=mx.bool_)

        meta = DispatchMeta(
            expert_indices=expert_indices,
            weights=weights,
            positions=positions,
            overflow_mask=overflow_mask,
            num_experts=num_experts,
            capacity=capacity,
            world_size=1,
        )

        combined = expert_combine(expert_outputs, meta, original_tokens)
        mx.eval(combined)

        self.assertTrue(mx.allclose(combined, original_tokens).item())


if __name__ == "__main__":
    unittest.main()
