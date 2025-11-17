"""Simple Transformer model in JAX for Pallas kernel experimentation."""

import jax
import jax.numpy as jnp
from typing import NamedTuple


class TransformerConfig(NamedTuple):
    """Configuration for transformer model."""
    d_model: int = 64
    n_heads: int = 4
    d_ff: int = 256
    seq_len: int = 16
    vocab_size: int = 100


def layer_norm(x, eps=1e-5):
    """Layer normalization."""
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps)


def attention(query, key, value, mask=None):
    """Scaled dot-product attention.

    Args:
        query: [batch, n_heads, seq_len, d_k]
        key: [batch, n_heads, seq_len, d_k]
        value: [batch, n_heads, seq_len, d_v]
        mask: Optional attention mask

    Returns:
        Attention output [batch, n_heads, seq_len, d_v]
    """
    d_k = query.shape[-1]

    # Compute attention scores
    scores = jnp.matmul(query, jnp.transpose(key, (0, 1, 3, 2)))  # [B, H, L, L]
    scores = scores / jnp.sqrt(d_k)

    if mask is not None:
        scores = jnp.where(mask, scores, -1e9)

    # Softmax over last dimension
    attn_weights = jax.nn.softmax(scores, axis=-1)

    # Apply attention to values
    output = jnp.matmul(attn_weights, value)  # [B, H, L, d_v]
    return output


def multi_head_attention(x, w_q, w_k, w_v, w_o, config):
    """Multi-head attention.

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

    # Linear projections
    q = jnp.matmul(x, w_q)  # [B, L, d_model]
    k = jnp.matmul(x, w_k)
    v = jnp.matmul(x, w_v)

    # Reshape to [batch, n_heads, seq_len, d_k]
    q = q.reshape(batch_size, seq_len, n_heads, d_k).transpose(0, 2, 1, 3)
    k = k.reshape(batch_size, seq_len, n_heads, d_k).transpose(0, 2, 1, 3)
    v = v.reshape(batch_size, seq_len, n_heads, d_k).transpose(0, 2, 1, 3)

    # Apply attention
    attn_output = attention(q, k, v)  # [B, H, L, d_k]

    # Concatenate heads
    attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)

    # Final linear projection
    output = jnp.matmul(attn_output, w_o)
    return output


def feed_forward(x, w1, b1, w2, b2):
    """Feed-forward network.

    Args:
        x: Input [batch, seq_len, d_model]
        w1: First layer weights [d_model, d_ff]
        b1: First layer bias [d_ff]
        w2: Second layer weights [d_ff, d_model]
        b2: Second layer bias [d_model]

    Returns:
        Output [batch, seq_len, d_model]
    """
    # First layer
    hidden = jnp.matmul(x, w1) + b1
    hidden = jax.nn.gelu(hidden)

    # Second layer
    output = jnp.matmul(hidden, w2) + b2
    return output


def transformer_block(x, params, config):
    """Single transformer block.

    Args:
        x: Input [batch, seq_len, d_model]
        params: Dictionary of parameters
        config: TransformerConfig

    Returns:
        Output [batch, seq_len, d_model]
    """
    # Multi-head attention with residual connection
    attn_output = multi_head_attention(
        x,
        params['w_q'],
        params['w_k'],
        params['w_v'],
        params['w_o'],
        config
    )
    x = x + attn_output
    x = layer_norm(x)

    # Feed-forward with residual connection
    ff_output = feed_forward(
        x,
        params['ff_w1'],
        params['ff_b1'],
        params['ff_w2'],
        params['ff_b2']
    )
    x = x + ff_output
    x = layer_norm(x)

    return x


def init_transformer_params(config: TransformerConfig, key):
    """Initialize transformer parameters.

    Args:
        config: TransformerConfig
        key: JAX random key

    Returns:
        Dictionary of parameters
    """
    keys = jax.random.split(key, 8)

    d_model = config.d_model
    d_ff = config.d_ff

    # Xavier/Glorot initialization
    def init_weight(key, shape):
        scale = jnp.sqrt(2.0 / (shape[0] + shape[1]))
        return jax.random.normal(key, shape) * scale

    params = {
        'w_q': init_weight(keys[0], (d_model, d_model)),
        'w_k': init_weight(keys[1], (d_model, d_model)),
        'w_v': init_weight(keys[2], (d_model, d_model)),
        'w_o': init_weight(keys[3], (d_model, d_model)),
        'ff_w1': init_weight(keys[4], (d_model, d_ff)),
        'ff_b1': jnp.zeros(d_ff),
        'ff_w2': init_weight(keys[5], (d_ff, d_model)),
        'ff_b2': jnp.zeros(d_model),
    }

    return params


def dump_jaxpr(config: TransformerConfig):
    """Dump the jaxpr for the transformer block to see what ops are needed."""
    # Initialize params and input
    key = jax.random.PRNGKey(0)
    params = init_transformer_params(config, key)

    # Create sample input [batch=2, seq_len, d_model]
    x = jax.random.normal(key, (2, config.seq_len, config.d_model))

    # Get the jaxpr - need to use functools.partial to pass config
    from functools import partial
    jaxpr = jax.make_jaxpr(partial(transformer_block, config=config))(x, params)

    print("=" * 80)
    print("JAXPR for Transformer Block")
    print("=" * 80)
    print(jaxpr)
    print("=" * 80)

    # Extract unique primitives
    primitives = set()
    for eqn in jaxpr.jaxpr.eqns:
        primitives.add(eqn.primitive.name)

    print("\nUnique primitives (operations) needed:")
    for i, prim in enumerate(sorted(primitives), 1):
        print(f"{i}. {prim}")
    print("=" * 80)

    return jaxpr, sorted(primitives)


if __name__ == "__main__":
    # Create config
    config = TransformerConfig(
        d_model=64,
        n_heads=4,
        d_ff=256,
        seq_len=16
    )

    # Dump jaxpr
    jaxpr, primitives = dump_jaxpr(config)

    # Also run the model to make sure it works
    print("\nRunning model to verify correctness...")
    key = jax.random.PRNGKey(42)
    params = init_transformer_params(config, key)
    x = jax.random.normal(key, (2, config.seq_len, config.d_model))

    output = transformer_block(x, params, config)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output mean: {jnp.mean(output):.6f}")
    print(f"Output std: {jnp.std(output):.6f}")
    print("\nModel runs successfully!")
