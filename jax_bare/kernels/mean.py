"""Pallas kernel for mean operation."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from functools import partial

# Store references to original JAX operations to avoid recursion
_jax_sum = jnp.sum
_jax_squeeze = jnp.squeeze


def mean_kernel(x_ref, out_ref, *, axis: int, size: int):
    """Kernel for computing mean along a specified axis.

    Args:
        x_ref: Input array
        out_ref: Output array
        axis: Axis along which to compute mean
        size: Size of the dimension being reduced
    """
    x = x_ref[...]

    # Sum along axis
    x_sum = _jax_sum(x, axis=axis, keepdims=True)

    # Divide by size
    out_ref[...] = x_sum / size


def mean(x: jax.Array, axis: int = -1, keepdims: bool = False) -> jax.Array:
    """Mean operation using Pallas kernel.

    Args:
        x: Input array
        axis: Axis along which to compute mean (default: -1)
        keepdims: Whether to keep the reduced dimension (default: False)

    Returns:
        Mean values along the specified axis
    """
    # Normalize axis
    if axis < 0:
        axis = x.ndim + axis

    size = x.shape[axis]

    # Compute output shape
    if keepdims:
        out_shape = list(x.shape)
        out_shape[axis] = 1
        out_shape = tuple(out_shape)
    else:
        out_shape = tuple(s for i, s in enumerate(x.shape) if i != axis)

    result = pl.pallas_call(
        partial(mean_kernel, axis=axis, size=size),
        out_shape=jax.ShapeDtypeStruct(
            x.shape[:axis] + (1,) + x.shape[axis+1:],
            x.dtype
        ),
        interpret=True
    )(x)

    if not keepdims:
        result = _jax_squeeze(result, axis=axis)

    return result
