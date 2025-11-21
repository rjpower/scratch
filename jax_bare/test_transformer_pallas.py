"""Unit tests comparing standard JAX transformer with Pallas kernel version.

This test suite verifies that the Pallas-based transformer implementation produces
results that are numerically equivalent to the standard JAX implementation.

The tests use the transformation mechanism to convert JAX functions to use
Pallas kernels, ensuring that the same code works with both backends.
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

from transform import to_pallas, apply_pallas, create_pallas_version

# Set random seed for reproducibility
SEED = 42
RTOL = 1e-3
ATOL = 1e-3


class TestKernelOperations:
    """Test individual Pallas kernels."""

    def test_elementwise_add(self):
        """Test addition kernel."""
        from kernels import elementwise

        x = jnp.ones((4, 8))
        y = jnp.ones((4, 8)) * 2

        result = elementwise.add(x, y)
        expected = x + y

        assert jnp.allclose(result, expected, rtol=RTOL, atol=ATOL)

    def test_elementwise_mul(self):
        """Test multiplication kernel."""
        from kernels import elementwise

        x = jnp.ones((4, 8)) * 3
        y = jnp.ones((4, 8)) * 2

        result = elementwise.mul(x, y)
        expected = x * y

        assert jnp.allclose(result, expected, rtol=RTOL, atol=ATOL)

    def test_matmul_simple(self):
        """Test simple 2D matmul."""
        from kernels import matmul

        x = jax.random.normal(jax.random.PRNGKey(SEED), (4, 8))
        y = jax.random.normal(jax.random.PRNGKey(SEED + 1), (8, 16))

        result = matmul.matmul_simple(x, y)
        expected = jnp.matmul(x, y)

        assert jnp.allclose(result, expected, rtol=RTOL, atol=ATOL)

    def test_softmax(self):
        """Test softmax kernel."""
        from kernels import softmax

        x = jax.random.normal(jax.random.PRNGKey(SEED), (2, 4, 8))

        result = softmax.softmax(x, axis=-1)
        expected = jax.nn.softmax(x, axis=-1)

        assert jnp.allclose(result, expected, rtol=RTOL, atol=ATOL)

    def test_gelu(self):
        """Test GELU kernel."""
        from kernels import gelu

        x = jax.random.normal(jax.random.PRNGKey(SEED), (4, 8))

        result = gelu.gelu(x)
        expected = jax.nn.gelu(x)

        assert jnp.allclose(result, expected, rtol=RTOL, atol=ATOL)

    def test_mean(self):
        """Test mean kernel."""
        from kernels import mean

        x = jax.random.normal(jax.random.PRNGKey(SEED), (2, 4, 8))

        result = mean.mean(x, axis=-1, keepdims=True)
        expected = jnp.mean(x, axis=-1, keepdims=True)

        assert jnp.allclose(result, expected, rtol=RTOL, atol=ATOL)

    def test_var(self):
        """Test variance kernel."""
        from kernels import var

        x = jax.random.normal(jax.random.PRNGKey(SEED), (2, 4, 8))

        result = var.var(x, axis=-1, keepdims=True)
        expected = jnp.var(x, axis=-1, keepdims=True)

        assert jnp.allclose(result, expected, rtol=RTOL, atol=ATOL)


class TestTransformerComponents:
    """Test transformer components with Pallas transformation."""

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

    def test_layer_norm_transform(self, input_data):
        """Test layer norm with Pallas transformation."""
        # Standard JAX version
        output_jax = layer_norm(input_data)

        # Pallas version using transformation
        output_pallas = apply_pallas(layer_norm, input_data)

        # Compare
        assert output_jax.shape == output_pallas.shape
        assert jnp.allclose(output_jax, output_pallas, rtol=RTOL, atol=ATOL), \
            f"Max diff: {jnp.max(jnp.abs(output_jax - output_pallas))}"

    def test_attention_transform(self, config):
        """Test attention with Pallas transformation."""
        key = jax.random.PRNGKey(SEED)
        batch_size = 2
        n_heads = config.n_heads
        seq_len = config.seq_len
        d_k = config.d_model // n_heads

        # Create query, key, value
        query = jax.random.normal(key, (batch_size, n_heads, seq_len, d_k))
        key_input = jax.random.normal(jax.random.PRNGKey(SEED + 1), (batch_size, n_heads, seq_len, d_k))
        value = jax.random.normal(jax.random.PRNGKey(SEED + 2), (batch_size, n_heads, seq_len, d_k))

        # Standard JAX version
        output_jax = attention(query, key_input, value)

        # Pallas version
        output_pallas = apply_pallas(attention, query, key_input, value)

        # Compare
        assert output_jax.shape == output_pallas.shape
        assert jnp.allclose(output_jax, output_pallas, rtol=RTOL, atol=ATOL), \
            f"Max diff: {jnp.max(jnp.abs(output_jax - output_pallas))}"

    def test_feed_forward_transform(self, input_data, params):
        """Test feed-forward network with Pallas transformation."""
        # Standard JAX version
        output_jax = feed_forward(
            input_data,
            params['ff_w1'],
            params['ff_b1'],
            params['ff_w2'],
            params['ff_b2']
        )

        # Pallas version
        output_pallas = apply_pallas(
            feed_forward,
            input_data,
            params['ff_w1'],
            params['ff_b1'],
            params['ff_w2'],
            params['ff_b2']
        )

        # Compare
        assert output_jax.shape == output_pallas.shape
        assert jnp.allclose(output_jax, output_pallas, rtol=RTOL, atol=ATOL), \
            f"Max diff: {jnp.max(jnp.abs(output_jax - output_pallas))}"

    def test_multi_head_attention_transform(self, input_data, params, config):
        """Test multi-head attention with Pallas transformation."""
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
        output_pallas = apply_pallas(
            multi_head_attention,
            input_data,
            params['w_q'],
            params['w_k'],
            params['w_v'],
            params['w_o'],
            config
        )

        # Compare
        assert output_jax.shape == output_pallas.shape
        assert jnp.allclose(output_jax, output_pallas, rtol=RTOL, atol=ATOL), \
            f"Max diff: {jnp.max(jnp.abs(output_jax - output_pallas))}"

    def test_transformer_block_transform(self, input_data, params, config):
        """Test full transformer block with Pallas transformation."""
        # Standard JAX version
        output_jax = transformer_block(input_data, params, config)

        # Pallas version using transformation
        output_pallas = apply_pallas(transformer_block, input_data, params, config)

        # Compare shapes
        assert output_jax.shape == output_pallas.shape

        # Compare values
        max_diff = jnp.max(jnp.abs(output_jax - output_pallas))
        mean_diff = jnp.mean(jnp.abs(output_jax - output_pallas))

        print(f"\nTransformer block comparison:")
        print(f"  Output shape: {output_jax.shape}")
        print(f"  Max absolute difference: {max_diff:.6e}")
        print(f"  Mean absolute difference: {mean_diff:.6e}")
        print(f"  JAX output mean: {jnp.mean(output_jax):.6f}, std: {jnp.std(output_jax):.6f}")
        print(f"  Pallas output mean: {jnp.mean(output_pallas):.6f}, std: {jnp.std(output_pallas):.6f}")

        assert jnp.allclose(output_jax, output_pallas, rtol=RTOL, atol=ATOL), \
            f"Max diff: {max_diff}"


def test_transformation_decorator():
    """Test the @to_pallas decorator."""
    # Define a simple function
    def simple_func(x, y):
        result = jnp.matmul(x, y)
        return result + jnp.exp(result)

    # Create Pallas version using decorator
    pallas_func = to_pallas(simple_func)

    # Test data
    key = jax.random.PRNGKey(SEED)
    x = jax.random.normal(key, (4, 8))
    y = jax.random.normal(jax.random.PRNGKey(SEED + 1), (8, 16))

    # Standard JAX version
    result_jax = simple_func(x, y)

    # Pallas version
    result_pallas = pallas_func(x, y)

    # Compare
    assert result_jax.shape == result_pallas.shape
    assert jnp.allclose(result_jax, result_pallas, rtol=RTOL, atol=ATOL)


def test_create_pallas_version():
    """Test creating a permanent Pallas version of a function."""
    # Create Pallas version of transformer_block
    transformer_block_pallas = create_pallas_version(transformer_block)

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
    assert jnp.allclose(output_jax, output_pallas, rtol=RTOL, atol=ATOL)


def test_full_transformer_equivalence():
    """End-to-end test ensuring JAX and Pallas implementations are equivalent."""
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

    # Run JAX version
    output_jax = transformer_block(input_data, params, config)

    # Run Pallas version using transformation
    output_pallas = apply_pallas(transformer_block, input_data, params, config)

    # Detailed comparison
    max_diff = jnp.max(jnp.abs(output_jax - output_pallas))
    mean_diff = jnp.mean(jnp.abs(output_jax - output_pallas))
    rel_diff = max_diff / (jnp.max(jnp.abs(output_jax)) + 1e-10)

    print("\n" + "=" * 80)
    print("FULL TRANSFORMER EQUIVALENCE TEST")
    print("=" * 80)
    print(f"Input shape: {input_data.shape}")
    print(f"Output shape: {output_jax.shape}")
    print(f"\nJAX output:")
    print(f"  Mean: {jnp.mean(output_jax):.6f}")
    print(f"  Std:  {jnp.std(output_jax):.6f}")
    print(f"  Min:  {jnp.min(output_jax):.6f}")
    print(f"  Max:  {jnp.max(output_jax):.6f}")
    print(f"\nPallas output:")
    print(f"  Mean: {jnp.mean(output_pallas):.6f}")
    print(f"  Std:  {jnp.std(output_pallas):.6f}")
    print(f"  Min:  {jnp.min(output_pallas):.6f}")
    print(f"  Max:  {jnp.max(output_pallas):.6f}")
    print(f"\nDifference metrics:")
    print(f"  Max absolute difference: {max_diff:.6e}")
    print(f"  Mean absolute difference: {mean_diff:.6e}")
    print(f"  Relative difference: {rel_diff:.6e}")
    print("=" * 80)

    # Verify equivalence
    assert output_jax.shape == output_pallas.shape, "Output shapes don't match"
    assert jnp.allclose(output_jax, output_pallas, rtol=RTOL, atol=ATOL), \
        f"Outputs don't match within tolerance (max diff: {max_diff:.6e})"

    print("✓ JAX and Pallas implementations produce equivalent results!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
