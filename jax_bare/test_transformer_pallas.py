"""Unit tests comparing standard JAX transformer with Pallas kernel version.

This test suite verifies that the Pallas-based transformer implementation produces
results that are numerically equivalent to the standard JAX implementation.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from transformer import (
    TransformerConfig,
    init_transformer_params,
    transformer_block,
    layer_norm,
    attention,
    multi_head_attention,
    feed_forward,
)

from transformer_pallas import (
    transformer_block_pallas,
    attention_pallas,
    multi_head_attention_pallas,
    feed_forward_pallas,
)

# Set random seed for reproducibility
SEED = 42
RTOL = 1e-4
ATOL = 1e-4


class TestElementwiseKernels:
    """Test individual elementwise Pallas kernels."""

    def test_add(self):
        """Test addition kernel."""
        from kernels import elementwise

        x = jnp.ones((4, 8))
        y = jnp.ones((4, 8)) * 2

        result = elementwise.add(x, y)
        expected = x + y

        assert jnp.allclose(result, expected, rtol=RTOL, atol=ATOL)

    def test_mul(self):
        """Test multiplication kernel."""
        from kernels import elementwise

        x = jnp.ones((4, 8)) * 3
        y = jnp.ones((4, 8)) * 2

        result = elementwise.mul(x, y)
        expected = x * y

        assert jnp.allclose(result, expected, rtol=RTOL, atol=ATOL)

    def test_exp(self):
        """Test exponential kernel."""
        from kernels import elementwise

        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        result = elementwise.exp(x)
        expected = jnp.exp(x)

        assert jnp.allclose(result, expected, rtol=RTOL, atol=ATOL)

    def test_tanh(self):
        """Test tanh kernel."""
        from kernels import elementwise

        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        result = elementwise.tanh(x)
        expected = jnp.tanh(x)

        assert jnp.allclose(result, expected, rtol=RTOL, atol=ATOL)


class TestMatmulKernels:
    """Test matrix multiplication Pallas kernels."""

    def test_simple_matmul(self):
        """Test simple 2D matmul."""
        from kernels import matmul

        x = jax.random.normal(jax.random.PRNGKey(SEED), (4, 8))
        y = jax.random.normal(jax.random.PRNGKey(SEED + 1), (8, 16))

        result = matmul.matmul_simple(x, y)
        expected = jnp.matmul(x, y)

        assert jnp.allclose(result, expected, rtol=RTOL, atol=ATOL)

    def test_batch_matmul(self):
        """Test batched matmul."""
        from kernels import matmul

        x = jax.random.normal(jax.random.PRNGKey(SEED), (2, 16, 64))
        y = jax.random.normal(jax.random.PRNGKey(SEED + 1), (2, 64, 64))

        result = matmul.batch_matmul(x, y)
        expected = jnp.matmul(x, y)

        assert jnp.allclose(result, expected, rtol=RTOL, atol=ATOL)

    def test_dot_general_linear_projection(self):
        """Test dot_general for linear projection pattern."""
        from kernels import matmul

        x = jax.random.normal(jax.random.PRNGKey(SEED), (2, 16, 64))
        w = jax.random.normal(jax.random.PRNGKey(SEED + 1), (64, 64))

        result = matmul.dot_general_pallas(
            x, w, dimension_numbers=(([2], [0]), ([], []))
        )
        expected = jax.lax.dot_general(
            x, w, dimension_numbers=(([2], [0]), ([], []))
        )

        assert jnp.allclose(result, expected, rtol=RTOL, atol=ATOL)

    def test_dot_general_attention(self):
        """Test dot_general for attention pattern."""
        from kernels import matmul

        q = jax.random.normal(jax.random.PRNGKey(SEED), (2, 4, 16, 16))
        k = jax.random.normal(jax.random.PRNGKey(SEED + 1), (2, 4, 16, 16))

        result = matmul.dot_general_pallas(
            q, k, dimension_numbers=(([3], [2]), ([0, 1], [0, 1]))
        )
        expected = jax.lax.dot_general(
            q, k, dimension_numbers=(([3], [2]), ([0, 1], [0, 1]))
        )

        assert jnp.allclose(result, expected, rtol=RTOL, atol=ATOL)


class TestReduceKernels:
    """Test reduction Pallas kernels."""

    def test_reduce_sum(self):
        """Test reduce_sum kernel."""
        from kernels import reduce

        x = jax.random.normal(jax.random.PRNGKey(SEED), (2, 4, 8))

        result = reduce.reduce_sum(x, axes=(2,))
        expected = jnp.sum(x, axis=2)

        assert jnp.allclose(result, expected, rtol=RTOL, atol=ATOL)

    def test_reduce_max(self):
        """Test reduce_max kernel."""
        from kernels import reduce

        x = jax.random.normal(jax.random.PRNGKey(SEED), (2, 4, 8, 8))

        result = reduce.reduce_max(x, axes=(3,))
        expected = jnp.max(x, axis=3)

        assert jnp.allclose(result, expected, rtol=RTOL, atol=ATOL)


class TestTransposeKernels:
    """Test transpose Pallas kernels."""

    def test_transpose_2d(self):
        """Test 2D transpose."""
        from kernels import transpose

        x = jax.random.normal(jax.random.PRNGKey(SEED), (4, 8))

        result = transpose.transpose_pallas(x, (1, 0))
        expected = jnp.transpose(x, (1, 0))

        assert jnp.allclose(result, expected, rtol=RTOL, atol=ATOL)

    def test_transpose_4d(self):
        """Test 4D transpose (attention pattern)."""
        from kernels import transpose

        x = jax.random.normal(jax.random.PRNGKey(SEED), (2, 16, 4, 16))

        result = transpose.transpose_pallas(x, (0, 2, 1, 3))
        expected = jnp.transpose(x, (0, 2, 1, 3))

        assert jnp.allclose(result, expected, rtol=RTOL, atol=ATOL)


class TestReshapeKernels:
    """Test reshape Pallas kernels."""

    def test_reshape_split_heads(self):
        """Test reshape for splitting attention heads."""
        from kernels import reshape

        x = jax.random.normal(jax.random.PRNGKey(SEED), (2, 16, 64))

        result = reshape.reshape(x, (2, 16, 4, 16))
        expected = x.reshape(2, 16, 4, 16)

        assert jnp.allclose(result, expected, rtol=RTOL, atol=ATOL)

    def test_reshape_merge_heads(self):
        """Test reshape for merging attention heads."""
        from kernels import reshape

        x = jax.random.normal(jax.random.PRNGKey(SEED), (2, 16, 4, 16))

        result = reshape.reshape(x, (2, 16, 64))
        expected = x.reshape(2, 16, 64)

        assert jnp.allclose(result, expected, rtol=RTOL, atol=ATOL)


class TestBroadcastKernels:
    """Test broadcast Pallas kernels."""

    def test_broadcast_add_dimension(self):
        """Test broadcasting by adding a dimension."""
        from kernels import broadcast

        x = jax.random.normal(jax.random.PRNGKey(SEED), (2, 16))

        result = broadcast.broadcast_in_dim_pallas(
            x, shape=(2, 16, 1), broadcast_dimensions=(0, 1)
        )
        expected = jax.lax.broadcast_in_dim(
            x, shape=(2, 16, 1), broadcast_dimensions=(0, 1)
        )

        assert jnp.allclose(result, expected, rtol=RTOL, atol=ATOL)

    def test_broadcast_bias(self):
        """Test broadcasting bias pattern."""
        from kernels import broadcast

        bias = jax.random.normal(jax.random.PRNGKey(SEED), (256,))

        result = broadcast.broadcast_in_dim_pallas(
            bias, shape=(1, 1, 256), broadcast_dimensions=(2,)
        )
        expected = jax.lax.broadcast_in_dim(
            bias, shape=(1, 1, 256), broadcast_dimensions=(2,)
        )

        assert jnp.allclose(result, expected, rtol=RTOL, atol=ATOL)


class TestTransformerComponents:
    """Test transformer components with Pallas kernels."""

    @pytest.fixture
    def config(self):
        """Create a small transformer config for testing."""
        return TransformerConfig(
            d_model=64,
            n_heads=4,
            d_ff=256,
            seq_len=16
        )

    @pytest.fixture
    def params(self, config):
        """Initialize transformer parameters."""
        key = jax.random.PRNGKey(SEED)
        return init_transformer_params(config, key)

    @pytest.fixture
    def input_data(self, config):
        """Create sample input data."""
        key = jax.random.PRNGKey(SEED)
        return jax.random.normal(key, (2, config.seq_len, config.d_model))

    def test_feed_forward(self, input_data, params):
        """Test feed-forward network."""
        # Standard JAX version
        output_jax = feed_forward(
            input_data,
            params['ff_w1'],
            params['ff_b1'],
            params['ff_w2'],
            params['ff_b2']
        )

        # Pallas version
        output_pallas = feed_forward_pallas(
            input_data,
            params['ff_w1'],
            params['ff_b1'],
            params['ff_w2'],
            params['ff_b2']
        )

        # Compare
        assert output_jax.shape == output_pallas.shape
        # GELU implementation might have small numerical differences
        assert jnp.allclose(output_jax, output_pallas, rtol=1e-3, atol=1e-3)

    def test_multi_head_attention(self, input_data, params, config):
        """Test multi-head attention."""
        # Standard JAX version
        output_jax = multi_head_attention(
            input_data,
            params['w_q'],
            params['w_k'],
            params['w_v'],
            params['w_o'],
            config
        )

        # Pallas version
        output_pallas = multi_head_attention_pallas(
            input_data,
            params['w_q'],
            params['w_k'],
            params['w_v'],
            params['w_o'],
            config
        )

        # Compare
        assert output_jax.shape == output_pallas.shape
        assert jnp.allclose(output_jax, output_pallas, rtol=1e-3, atol=1e-3)

    def test_transformer_block(self, input_data, params, config):
        """Test full transformer block."""
        # Standard JAX version
        output_jax = transformer_block(input_data, params, config)

        # Pallas version
        output_pallas = transformer_block_pallas(input_data, params, config)

        # Compare shapes
        assert output_jax.shape == output_pallas.shape

        # Compare values (allow slightly larger tolerance due to accumulation)
        assert jnp.allclose(output_jax, output_pallas, rtol=1e-2, atol=1e-2), \
            f"Max diff: {jnp.max(jnp.abs(output_jax - output_pallas))}"


def test_full_transformer_comparison():
    """End-to-end test comparing JAX and Pallas implementations."""
    # Setup
    config = TransformerConfig(
        d_model=64,
        n_heads=4,
        d_ff=256,
        seq_len=16
    )

    key = jax.random.PRNGKey(SEED)
    params = init_transformer_params(config, key)
    input_data = jax.random.normal(key, (2, config.seq_len, config.d_model))

    # Run both versions
    output_jax = transformer_block(input_data, params, config)
    output_pallas = transformer_block_pallas(input_data, params, config)

    # Verify
    assert output_jax.shape == output_pallas.shape
    print(f"\nJAX output mean: {jnp.mean(output_jax):.6f}, std: {jnp.std(output_jax):.6f}")
    print(f"Pallas output mean: {jnp.mean(output_pallas):.6f}, std: {jnp.std(output_pallas):.6f}")
    print(f"Max absolute difference: {jnp.max(jnp.abs(output_jax - output_pallas)):.6e}")
    print(f"Mean absolute difference: {jnp.mean(jnp.abs(output_jax - output_pallas)):.6e}")

    # Allow larger tolerance for full transformer
    assert jnp.allclose(output_jax, output_pallas, rtol=0.1, atol=0.1)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
