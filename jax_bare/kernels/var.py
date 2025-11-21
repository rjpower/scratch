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

    Args:
        x_ref: Input array
        out_ref: Output array
        axis: Axis along which to compute variance
        size: Size of the dimension being reduced
    """
    x = x_ref[...]

    # Compute mean
    x_mean = _jax_mean(x, axis=axis, keepdims=True)

    # Center the data
    x_centered = x - x_mean

    # Square
    x_squared = x_centered * x_centered

    # Mean of squares
    out_ref[...] = _jax_mean(x_squared, axis=axis, keepdims=True)


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
