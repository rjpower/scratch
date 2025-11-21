"""Pallas kernel for variance operation."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from functools import partial

# Store references to original JAX operations to avoid recursion
_jax_mean = jnp.mean
_jax_squeeze = jnp.squeeze


def var_kernel(x_ref, out_ref, *, axis: int, size: int):
    """Kernel for computing variance along a specified axis.

    Var(x) = mean((x - mean(x))^2)

    Uses explicit two-pass algorithm:
    1. First pass: compute mean by looping and accumulating sum
    2. Second pass: compute variance by looping and accumulating squared deviations

    Args:
        x_ref: Input array
        out_ref: Output array
        axis: Axis along which to compute variance
        size: Size of the dimension being reduced
    """
    x = x_ref[...]

    # Initialize accumulator for mean computation
    sum_val = jnp.zeros_like(out_ref[...])

    # First pass: compute mean
    for i in range(size):
        # Extract slice at position i along the reduction axis (keepdims=True style)
        slice_i = jax.lax.slice_in_dim(x, i, i+1, axis=axis)
        sum_val = sum_val + slice_i

    mean_val = sum_val / size

    # Initialize accumulator for variance computation
    sum_sq = jnp.zeros_like(out_ref[...])

    # Second pass: compute variance
    for i in range(size):
        # Extract slice at position i along the reduction axis
        slice_i = jax.lax.slice_in_dim(x, i, i+1, axis=axis)
        diff = slice_i - mean_val
        sum_sq = sum_sq + diff * diff

    out_ref[...] = sum_sq / size


def var(x: jax.Array, axis: int = -1, keepdims: bool = False) -> jax.Array:
    """Variance operation using Pallas kernel.

    Args:
        x: Input array
        axis: Axis along which to compute variance (default: -1)
        keepdims: Whether to keep the reduced dimension (default: False)

    Returns:
        Variance values along the specified axis
    """
    # Normalize axis
    if axis < 0:
        axis = x.ndim + axis

    size = x.shape[axis]

    result = pl.pallas_call(
        partial(var_kernel, axis=axis, size=size),
        out_shape=jax.ShapeDtypeStruct(
            x.shape[:axis] + (1,) + x.shape[axis+1:],
            x.dtype
        ),
        interpret=True
    )(x)

    if not keepdims:
        result = _jax_squeeze(result, axis=axis)

    return result
