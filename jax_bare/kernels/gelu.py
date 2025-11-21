"""Pallas kernel for GELU activation function."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

# Store references to original JAX operations to avoid recursion
_jax_power = jnp.power
_jax_tanh = jnp.tanh


def gelu_kernel(x_ref, out_ref):
    """Kernel for GELU activation function.

    GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

    Args:
        x_ref: Input array
        out_ref: Output array
    """
    x = x_ref[...]

    # Compute GELU using the approximation formula
    # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    x_cubed = _jax_power(x, 3)
    inner = 0.7978845608 * (x + 0.044715 * x_cubed)  # sqrt(2/pi) ≈ 0.7978845608
    tanh_inner = _jax_tanh(inner)
    out_ref[...] = 0.5 * x * (1.0 + tanh_inner)


def gelu(x: jax.Array) -> jax.Array:
    """GELU activation function using Pallas kernel.

    Args:
        x: Input array

    Returns:
        GELU activation applied element-wise
    """
    return pl.pallas_call(
        gelu_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        interpret=True
    )(x)
