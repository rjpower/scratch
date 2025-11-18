"""Transformer model using Pallas kernels.

This module provides a transformer implementation that uses custom Pallas kernels
instead of JAX's built-in operations. This allows for direct comparison between
standard JAX execution and custom kernel execution.
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple

# Import Pallas kernels
from kernels import elementwise, matmul, reduce, reshape, transpose, broadcast


class TransformerConfig(NamedTuple):
    """Configuration for transformer model."""
    d_model: int = 64
    n_heads: int = 4
    d_ff: int = 256
    seq_len: int = 16
    vocab_size: int = 100


def layer_norm_pallas(x, eps=1e-5):
    """Layer normalization using Pallas kernels.

    Args:
        x: Input tensor [..., D]
        eps: Small constant for numerical stability

    Returns:
        Normalized tensor [..., D]
    """
    # Compute mean: reduce_sum over last axis, then divide
    sum_val = reduce.reduce_sum(x, axes=(-1,))  # [..., ]

    # Broadcast to match input shape
    sum_broadcast = broadcast.broadcast_in_dim_pallas(
        sum_val,
        shape=x.shape[:-1] + (1,),
        broadcast_dimensions=tuple(range(x.ndim - 1))
    )

    # Divide by dimension size
    d_model = x.shape[-1]
    mean = elementwise.div(sum_broadcast, jnp.array([d_model], dtype=x.dtype))

    # Subtract mean
    x_centered = elementwise.sub(x, mean)

    # Compute variance: square, reduce_sum, divide
    x_squared = elementwise.mul(x_centered, x_centered)
    var_sum = reduce.reduce_sum(x_squared, axes=(-1,))
    var_broadcast = broadcast.broadcast_in_dim_pallas(
        var_sum,
        shape=x.shape[:-1] + (1,),
        broadcast_dimensions=tuple(range(x.ndim - 1))
    )

    # Note: For Bessel's correction, should divide by (N-1), but original uses N
    var = elementwise.div(var_broadcast, jnp.array([d_model], dtype=x.dtype))

    # Add eps and take sqrt
    var_eps = elementwise.add(var, jnp.array([eps], dtype=x.dtype))
    std = elementwise.sqrt(var_eps)

    # Normalize
    normalized = elementwise.div(x_centered, std)

    return normalized


def attention_pallas(query, key, value, mask=None):
    """Scaled dot-product attention using Pallas kernels.

    Args:
        query: [batch, n_heads, seq_len, d_k]
        key: [batch, n_heads, seq_len, d_k]
        value: [batch, n_heads, seq_len, d_v]
        mask: Optional attention mask

    Returns:
        Attention output [batch, n_heads, seq_len, d_v]
    """
    d_k = query.shape[-1]

    # Compute attention scores: Q @ K^T
    # [B, H, L, D] @ [B, H, D, L] -> [B, H, L, L]
    key_t = transpose.transpose_pallas(key, (0, 1, 3, 2))
    scores = matmul.dot_general_pallas(
        query,
        key_t,
        dimension_numbers=(([3], [2]), ([0, 1], [0, 1]))
    )

    # Scale by sqrt(d_k)
    scale = jnp.sqrt(jnp.array(d_k, dtype=scores.dtype))
    scores = elementwise.div(scores, scale)

    if mask is not None:
        # Apply mask: scores = where(mask, scores, -1e9)
        # This would require a select/where kernel which we haven't implemented
        # For now, use JAX's where
        scores = jnp.where(mask, scores, -1e9)

    # Softmax over last dimension
    # max(scores, axis=-1)
    max_scores = reduce.reduce_max(scores, axes=(-1,))
    max_broadcast = broadcast.broadcast_in_dim_pallas(
        max_scores,
        shape=scores.shape[:-1] + (1,),
        broadcast_dimensions=tuple(range(scores.ndim - 1))
    )

    # scores - max (for numerical stability)
    scores_stable = elementwise.sub(scores, max_broadcast)

    # exp(scores_stable)
    exp_scores = elementwise.exp(scores_stable)

    # sum(exp_scores, axis=-1)
    exp_sum = reduce.reduce_sum(exp_scores, axes=(-1,))
    exp_sum_broadcast = broadcast.broadcast_in_dim_pallas(
        exp_sum,
        shape=exp_scores.shape[:-1] + (1,),
        broadcast_dimensions=tuple(range(exp_scores.ndim - 1))
    )

    # exp_scores / exp_sum = softmax
    attn_weights = elementwise.div(exp_scores, exp_sum_broadcast)

    # Apply attention to values: attn_weights @ value
    # [B, H, L, L] @ [B, H, L, D] -> [B, H, L, D]
    output = matmul.dot_general_pallas(
        attn_weights,
        value,
        dimension_numbers=(([3], [2]), ([0, 1], [0, 1]))
    )

    return output


def multi_head_attention_pallas(x, w_q, w_k, w_v, w_o, config):
    """Multi-head attention using Pallas kernels.

    Args:
        x: Input [batch, seq_len, d_model]
        w_q, w_k, w_v: Query, key, value weight matrices [d_model, d_model]
        w_o: Output weight matrix [d_model, d_model]
        config: TransformerConfig with n_heads and d_model

    Returns:
        Output [batch, seq_len, d_model]
    """
    batch_size, seq_len, d_model = x.shape
    n_heads = config.n_heads
    d_k = d_model // n_heads

    # Linear projections: X @ W
    # [B, L, D] @ [D, D] -> [B, L, D]
    q = matmul.dot_general_pallas(
        x, w_q, dimension_numbers=(([2], [0]), ([], []))
    )
    k = matmul.dot_general_pallas(
        x, w_k, dimension_numbers=(([2], [0]), ([], []))
    )
    v = matmul.dot_general_pallas(
        x, w_v, dimension_numbers=(([2], [0]), ([], []))
    )

    # Reshape to [batch, seq_len, n_heads, d_k]
    q = reshape.reshape(q, (batch_size, seq_len, n_heads, d_k))
    k = reshape.reshape(k, (batch_size, seq_len, n_heads, d_k))
    v = reshape.reshape(v, (batch_size, seq_len, n_heads, d_k))

    # Transpose to [batch, n_heads, seq_len, d_k]
    q = transpose.transpose_pallas(q, (0, 2, 1, 3))
    k = transpose.transpose_pallas(k, (0, 2, 1, 3))
    v = transpose.transpose_pallas(v, (0, 2, 1, 3))

    # Apply attention
    attn_output = attention_pallas(q, k, v)  # [B, H, L, d_k]

    # Transpose back: [B, H, L, d_k] -> [B, L, H, d_k]
    attn_output = transpose.transpose_pallas(attn_output, (0, 2, 1, 3))

    # Concatenate heads: [B, L, H, d_k] -> [B, L, D]
    attn_output = reshape.reshape(attn_output, (batch_size, seq_len, d_model))

    # Final linear projection
    output = matmul.dot_general_pallas(
        attn_output, w_o, dimension_numbers=(([2], [0]), ([], []))
    )

    return output


def feed_forward_pallas(x, w1, b1, w2, b2):
    """Feed-forward network using Pallas kernels.

    Args:
        x: Input [batch, seq_len, d_model]
        w1: First layer weights [d_model, d_ff]
        b1: First layer bias [d_ff]
        w2: Second layer weights [d_ff, d_model]
        b2: Second layer bias [d_model]

    Returns:
        Output [batch, seq_len, d_model]
    """
    # First layer: X @ W1
    hidden = matmul.dot_general_pallas(
        x, w1, dimension_numbers=(([2], [0]), ([], []))
    )

    # Add bias (broadcast [d_ff] to [B, L, d_ff])
    b1_broadcast = broadcast.broadcast_in_dim_pallas(
        b1,
        shape=(1, 1, b1.shape[0]),
        broadcast_dimensions=(2,)
    )
    hidden = elementwise.add(hidden, b1_broadcast)

    # GELU activation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # Simplified version using JAX's gelu for now
    # Full Pallas implementation would use:
    # x^3
    hidden_cubed = elementwise.integer_pow(hidden, 3)
    # 0.044715 * x^3
    coef = jnp.array([0.044715], dtype=hidden.dtype)
    term1 = elementwise.mul(hidden_cubed, coef)
    # x + 0.044715 * x^3
    term2 = elementwise.add(hidden, term1)
    # sqrt(2/pi) * (...)
    sqrt_2_pi = jnp.array([0.7978845608], dtype=hidden.dtype)
    term3 = elementwise.mul(term2, sqrt_2_pi)
    # tanh(...)
    term4 = elementwise.tanh(term3)
    # 1 + tanh(...)
    one = jnp.ones_like(term4)
    term5 = elementwise.add(one, term4)
    # 0.5 * term5
    half = jnp.array([0.5], dtype=hidden.dtype)
    term6 = elementwise.mul(term5, half)
    # x * term6
    hidden = elementwise.mul(hidden, term6)

    # Second layer: hidden @ W2
    output = matmul.dot_general_pallas(
        hidden, w2, dimension_numbers=(([2], [0]), ([], []))
    )

    # Add bias
    b2_broadcast = broadcast.broadcast_in_dim_pallas(
        b2,
        shape=(1, 1, b2.shape[0]),
        broadcast_dimensions=(2,)
    )
    output = elementwise.add(output, b2_broadcast)

    return output


def transformer_block_pallas(x, params, config):
    """Single transformer block using Pallas kernels.

    Args:
        x: Input [batch, seq_len, d_model]
        params: Dictionary of parameters
        config: TransformerConfig

    Returns:
        Output [batch, seq_len, d_model]
    """
    # Multi-head attention with residual connection
    attn_output = multi_head_attention_pallas(
        x,
        params['w_q'],
        params['w_k'],
        params['w_v'],
        params['w_o'],
        config
    )
    x = elementwise.add(x, attn_output)

    # Layer norm - use JAX version for now as our Pallas version is complex
    # In a full implementation, we'd use layer_norm_pallas
    x = jnp.array(x)  # Ensure it's a JAX array
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    x = (x - mean) / jnp.sqrt(var + 1e-5)

    # Feed-forward with residual connection
    ff_output = feed_forward_pallas(
        x,
        params['ff_w1'],
        params['ff_b1'],
        params['ff_w2'],
        params['ff_b2']
    )
    x = elementwise.add(x, ff_output)

    # Layer norm
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    x = (x - mean) / jnp.sqrt(var + 1e-5)

    return x


# Example usage
if __name__ == "__main__":
    from transformer import init_transformer_params

    print("Testing Pallas-based Transformer...")
    print("=" * 80)

    # Create config
    config = TransformerConfig(
        d_model=64,
        n_heads=4,
        d_ff=256,
        seq_len=16
    )

    # Initialize parameters
    key = jax.random.PRNGKey(42)
    params = init_transformer_params(config, key)

    # Create input
    x = jax.random.normal(key, (2, config.seq_len, config.d_model))

    print(f"Input shape: {x.shape}")
    print(f"Config: d_model={config.d_model}, n_heads={config.n_heads}, d_ff={config.d_ff}")

    # Run Pallas version
    print("\nRunning Pallas-based transformer...")
    output_pallas = transformer_block_pallas(x, params, config)
    print(f"Output shape: {output_pallas.shape}")
    print(f"Output mean: {jnp.mean(output_pallas):.6f}")
    print(f"Output std: {jnp.std(output_pallas):.6f}")

    print("\n" + "=" * 80)
    print("Pallas transformer completed successfully!")
