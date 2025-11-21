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
    # Accumulator for the sum
    acc = jnp.float32(0.0)

    # Get the number of dimensions in the input
    ndim = len(x_ref.shape)

    # Build the index for this output element using program_id
    # For each non-reduced dimension, use program_id
    indices = []
    pid_counter = 0
    for dim in range(ndim):
        if dim == axis:
            # This will be our loop variable
            indices.append(0)
        else:
            indices.append(pl.program_id(pid_counter))
            pid_counter += 1

    # Sum over the reduction axis
    for i in range(size):
        indices[axis] = i
        acc += x_ref[tuple(indices)]

    # Compute mean by dividing by size
    mean_val = acc / size

    # Store the result
    # Output has same indices but the reduction axis is always 0 (since output shape has size 1 there)
    out_indices = list(indices)
    out_indices[axis] = 0
    out_ref[tuple(out_indices)] = mean_val


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

    # Compute output shape with reduced dimension having size 1
    kernel_out_shape = x.shape[:axis] + (1,) + x.shape[axis+1:]

    # Compute final output shape based on keepdims
    if keepdims:
        out_shape = kernel_out_shape
    else:
        out_shape = tuple(s for i, s in enumerate(x.shape) if i != axis)

    # Grid covers all output elements
    # One kernel invocation per output element
    grid = kernel_out_shape

    result = pl.pallas_call(
        partial(mean_kernel, axis=axis, size=size),
        out_shape=jax.ShapeDtypeStruct(kernel_out_shape, x.dtype),
        grid=grid,
        interpret=True
    )(x)

    if not keepdims:
        result = _jax_squeeze(result, axis=axis)

    return result
