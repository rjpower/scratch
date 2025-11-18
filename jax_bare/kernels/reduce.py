"""Pallas kernels for reduction operations (sum, max) on CPU.

This module provides Pallas-based implementations of reduction operations
that can replace JAX's built-in reduce operations in the transformer model.

The kernels support reducing along any single axis and are optimized for CPU execution.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
import functools
from typing import Tuple


def _reduce_sum_kernel(x_ref, out_ref, *, reduce_axis: int, reduce_size: int):
    """Kernel function for sum reduction along a single axis.

    This kernel computes the sum of elements along the specified reduction axis.
    Each kernel invocation handles one output element by summing all elements
    along the reduction dimension.

    Args:
        x_ref: Input array reference
        out_ref: Output array reference
        reduce_axis: The axis to reduce over (relative to input shape)
        reduce_size: Size of the dimension being reduced
    """
    # Accumulator for the sum
    acc = jnp.float32(0.0)

    # Get the number of dimensions in the input
    ndim = len(x_ref.shape)

    # Build the index for this output element using program_id
    # For each non-reduced dimension, use program_id
    indices = []
    pid_counter = 0
    for axis in range(ndim):
        if axis == reduce_axis:
            # This will be our loop variable
            indices.append(0)
        else:
            indices.append(pl.program_id(pid_counter))
            pid_counter += 1

    # Sum over the reduction axis
    for i in range(reduce_size):
        indices[reduce_axis] = i
        acc += x_ref[tuple(indices)]

    # Store the result
    # Output indices are the same as input indices without the reduction axis
    out_indices = tuple(idx for axis, idx in enumerate(indices) if axis != reduce_axis)
    out_ref[out_indices] = acc


def _reduce_max_kernel(x_ref, out_ref, *, reduce_axis: int, reduce_size: int):
    """Kernel function for max reduction along a single axis.

    This kernel computes the maximum value along the specified reduction axis.
    Each kernel invocation handles one output element by finding the max of all
    elements along the reduction dimension.

    Args:
        x_ref: Input array reference
        out_ref: Output array reference
        reduce_axis: The axis to reduce over (relative to input shape)
        reduce_size: Size of the dimension being reduced
    """
    # Initialize accumulator to negative infinity
    acc = jnp.float32(-jnp.inf)

    # Get the number of dimensions in the input
    ndim = len(x_ref.shape)

    # Build the index for this output element using program_id
    indices = []
    pid_counter = 0
    for axis in range(ndim):
        if axis == reduce_axis:
            # This will be our loop variable
            indices.append(0)
        else:
            indices.append(pl.program_id(pid_counter))
            pid_counter += 1

    # Find max over the reduction axis
    for i in range(reduce_size):
        indices[reduce_axis] = i
        acc = jnp.maximum(acc, x_ref[tuple(indices)])

    # Store the result
    out_indices = tuple(idx for axis, idx in enumerate(indices) if axis != reduce_axis)
    out_ref[out_indices] = acc


def reduce_sum(x: jax.Array, axes: Tuple[int, ...]) -> jax.Array:
    """Compute sum reduction along specified axes using Pallas kernel.

    This function reduces the input array by summing along the specified axes.
    Currently supports single-axis reductions only.

    Args:
        x: Input array of shape [..., reduce_dim, ...]
        axes: Tuple of axes to reduce over (currently only single axis supported)

    Returns:
        Array with specified axes reduced (summed)

    Examples:
        >>> import jax.numpy as jnp
        >>> # Reduce last axis of [B, L, D] -> [B, L]
        >>> x = jnp.ones((2, 4, 8))
        >>> result = reduce_sum(x, axes=(2,))
        >>> result.shape
        (2, 4)
        >>> result[0, 0]  # Sum of 8 ones
        8.0

        >>> # Reduce last axis of [B, H, L, L] -> [B, H, L]
        >>> x = jnp.ones((2, 4, 8, 8))
        >>> result = reduce_sum(x, axes=(3,))
        >>> result.shape
        (2, 4, 8)
        >>> result[0, 0, 0]  # Sum of 8 ones
        8.0

    Raises:
        ValueError: If multiple axes are specified (not yet supported)
    """
    if len(axes) != 1:
        raise ValueError(f"Only single-axis reduction supported, got axes={axes}")

    # Normalize negative axis indices
    reduce_axis = axes[0]
    if reduce_axis < 0:
        reduce_axis = len(x.shape) + reduce_axis

    # Input shape and reduce dimension size
    input_shape = x.shape
    reduce_size = input_shape[reduce_axis]

    # Output shape is input shape without the reduced axis
    output_shape = tuple(s for i, s in enumerate(input_shape) if i != reduce_axis)

    # Grid covers all output elements
    # One kernel invocation per output element
    grid = output_shape

    # Create the kernel with reduction parameters
    kernel = functools.partial(
        _reduce_sum_kernel,
        reduce_axis=reduce_axis,
        reduce_size=reduce_size
    )

    # Execute the Pallas kernel
    result = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(output_shape, x.dtype),
        grid=grid,
        interpret=True,
    )(x)

    return result


def reduce_max(x: jax.Array, axes: Tuple[int, ...]) -> jax.Array:
    """Compute max reduction along specified axes using Pallas kernel.

    This function reduces the input array by taking the maximum along the
    specified axes. Currently supports single-axis reductions only.

    Args:
        x: Input array of shape [..., reduce_dim, ...]
        axes: Tuple of axes to reduce over (currently only single axis supported)

    Returns:
        Array with specified axes reduced (max taken)

    Examples:
        >>> import jax.numpy as jnp
        >>> # Reduce last axis of [B, H, L, L] -> [B, H, L]
        >>> x = jnp.arange(64).reshape(2, 2, 4, 4)
        >>> result = reduce_max(x, axes=(3,))
        >>> result.shape
        (2, 2, 4)
        >>> # Each result is the max of 4 consecutive elements
        >>> result[0, 0, 0]  # max of [0, 1, 2, 3]
        3.0

        >>> # With different values
        >>> x = jnp.array([[[1, 5, 2, 8]]])  # Shape: [1, 1, 4]
        >>> result = reduce_max(x, axes=(2,))
        >>> result.shape
        (1, 1)
        >>> result[0, 0]  # max of [1, 5, 2, 8]
        8.0

    Raises:
        ValueError: If multiple axes are specified (not yet supported)
    """
    if len(axes) != 1:
        raise ValueError(f"Only single-axis reduction supported, got axes={axes}")

    # Normalize negative axis indices
    reduce_axis = axes[0]
    if reduce_axis < 0:
        reduce_axis = len(x.shape) + reduce_axis

    # Input shape and reduce dimension size
    input_shape = x.shape
    reduce_size = input_shape[reduce_axis]

    # Output shape is input shape without the reduced axis
    output_shape = tuple(s for i, s in enumerate(input_shape) if i != reduce_axis)

    # Grid covers all output elements
    grid = output_shape

    # Create the kernel with reduction parameters
    kernel = functools.partial(
        _reduce_max_kernel,
        reduce_axis=reduce_axis,
        reduce_size=reduce_size
    )

    # Execute the Pallas kernel (interpret=True required for CPU)
    result = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(output_shape, x.dtype),
        grid=grid,
        interpret=True,
    )(x)

    return result


# Example usage and verification
if __name__ == "__main__":
    print("=" * 80)
    print("Pallas Reduction Kernels - Examples and Verification")
    print("=" * 80)

    # Example 1: reduce_sum on [B, L, D] -> [B, L]
    print("\n1. reduce_sum on [B=2, L=4, D=8] along axis=2")
    x1 = jnp.ones((2, 4, 8))
    result1 = reduce_sum(x1, axes=(2,))
    expected1 = jnp.sum(x1, axis=2)
    print(f"   Input shape: {x1.shape}")
    print(f"   Output shape: {result1.shape}")
    print(f"   Result[0, 0]: {result1[0, 0]} (expected: 8.0)")
    print(f"   Matches jnp.sum: {jnp.allclose(result1, expected1)}")

    # Example 2: reduce_sum on [B, H, L, L] -> [B, H, L]
    print("\n2. reduce_sum on [B=2, H=4, L=8, L=8] along axis=3")
    x2 = jnp.ones((2, 4, 8, 8))
    result2 = reduce_sum(x2, axes=(3,))
    expected2 = jnp.sum(x2, axis=3)
    print(f"   Input shape: {x2.shape}")
    print(f"   Output shape: {result2.shape}")
    print(f"   Result[0, 0, 0]: {result2[0, 0, 0]} (expected: 8.0)")
    print(f"   Matches jnp.sum: {jnp.allclose(result2, expected2)}")

    # Example 3: reduce_max on [B, H, L, L] -> [B, H, L]
    print("\n3. reduce_max on [B=2, H=2, L=4, L=4] along axis=3")
    x3 = jnp.arange(128).reshape(2, 2, 4, 4).astype(jnp.float32)
    result3 = reduce_max(x3, axes=(3,))
    expected3 = jnp.max(x3, axis=3)
    print(f"   Input shape: {x3.shape}")
    print(f"   Output shape: {result3.shape}")
    print(f"   Result[0, 0, 0]: {result3[0, 0, 0]} (expected: {expected3[0, 0, 0]})")
    print(f"   Matches jnp.max: {jnp.allclose(result3, expected3)}")

    # Example 4: Reduce along different axes
    print("\n4. reduce_sum on [B=3, L=5, D=7] along axis=1")
    x4 = jnp.ones((3, 5, 7))
    result4 = reduce_sum(x4, axes=(1,))
    expected4 = jnp.sum(x4, axis=1)
    print(f"   Input shape: {x4.shape}")
    print(f"   Output shape: {result4.shape}")
    print(f"   Result[0, 0]: {result4[0, 0]} (expected: 5.0)")
    print(f"   Matches jnp.sum: {jnp.allclose(result4, expected4)}")

    # Example 5: Test with actual values
    print("\n5. reduce_max with varied values")
    x5 = jnp.array([
        [[1.0, 5.0, 2.0, 8.0],
         [3.0, 1.0, 9.0, 2.0]],
        [[4.0, 6.0, 1.0, 3.0],
         [7.0, 2.0, 5.0, 1.0]]
    ])  # Shape: [2, 2, 4]
    result5 = reduce_max(x5, axes=(2,))
    expected5 = jnp.max(x5, axis=2)
    print(f"   Input shape: {x5.shape}")
    print(f"   Output shape: {result5.shape}")
    print(f"   Result: {result5}")
    print(f"   Expected: {expected5}")
    print(f"   Matches jnp.max: {jnp.allclose(result5, expected5)}")

    print("\n" + "=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)
