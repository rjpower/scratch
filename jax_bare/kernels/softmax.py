"""Pallas kernel for softmax operation."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from functools import partial

# Store references to original JAX operations to avoid recursion
_jax_max = jnp.max
_jax_exp = jnp.exp
_jax_sum = jnp.sum


def softmax_kernel(x_ref, out_ref, *, axis: int):
    """Kernel for softmax along a specified axis.

    softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))

    Args:
        x_ref: Input array
        out_ref: Output array
        axis: Axis along which to compute softmax
    """
    x = x_ref[...]

    # For numerical stability, subtract max
    x_max = _jax_max(x, axis=axis, keepdims=True)
    x_shifted = x - x_max

    # Exponential
    x_exp = _jax_exp(x_shifted)

    # Sum over axis
    x_sum = _jax_sum(x_exp, axis=axis, keepdims=True)

    # Divide
    out_ref[...] = x_exp / x_sum


def softmax(x: jax.Array, axis: int = -1) -> jax.Array:
    """Softmax operation using Pallas kernel.

    Args:
        x: Input array
        axis: Axis along which to compute softmax (default: -1)

    Returns:
        Softmax probabilities along the specified axis
    """
    # Normalize axis
    if axis < 0:
        axis = x.ndim + axis

    return pl.pallas_call(
        partial(softmax_kernel, axis=axis),
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        interpret=True
    )(x)
