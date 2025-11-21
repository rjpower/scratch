"""Pallas kernel for softmax operation."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from functools import partial


def softmax_kernel(x_ref, out_ref, *, axis: int):
    """Kernel for softmax along a specified axis implemented with explicit loops.

    softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))

    This implementation uses three explicit passes:
    1. Find max for numerical stability
    2. Compute exp(x - max) and accumulate sum
    3. Normalize by dividing by sum

    Args:
        x_ref: Input array reference
        out_ref: Output array reference
        axis: Axis along which to compute softmax
    """
    x = x_ref[...]

    # Normalize negative axis
    if axis < 0:
        axis = x.ndim + axis

    # Move axis to last position for easier indexing
    x = jnp.moveaxis(x, axis, -1)
    size = x.shape[-1]

    # Pass 1: Find max value for numerical stability
    max_val = jnp.full(x.shape[:-1], -jnp.inf, dtype=x.dtype)
    for i in range(size):
        max_val = jnp.maximum(max_val, x[..., i])

    # Pass 2: Compute exp(x - max) and accumulate sum
    exp_vals = jnp.zeros_like(x)
    sum_exp = jnp.zeros(x.shape[:-1], dtype=x.dtype)
    for i in range(size):
        exp_i = jnp.exp(x[..., i] - max_val)
        exp_vals = exp_vals.at[..., i].set(exp_i)
        sum_exp = sum_exp + exp_i

    # Pass 3: Normalize
    result = jnp.zeros_like(x)
    for i in range(size):
        result = result.at[..., i].set(exp_vals[..., i] / sum_exp)

    # Move axis back to original position
    result = jnp.moveaxis(result, -1, axis)
    out_ref[...] = result


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
