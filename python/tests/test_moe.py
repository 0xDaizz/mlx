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


if __name__ == "__main__":
    unittest.main()
