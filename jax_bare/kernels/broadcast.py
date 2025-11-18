"""Pallas kernel implementation for broadcast_in_dim operation.

This module provides a CPU-based Pallas kernel for broadcasting arrays to new shapes,
replicating JAX's broadcast_in_dim primitive functionality.

Broadcasting allows arrays with smaller shapes to be replicated along specified
dimensions to match a larger shape. This is fundamental for element-wise operations
between arrays of different shapes.

Examples:
    Basic broadcasting from [2, 16] to [2, 16, 1]:
        >>> import jax.numpy as jnp
        >>> x = jnp.arange(32).reshape(2, 16)
        >>> result = broadcast_in_dim_pallas(
        ...     x,
        ...     shape=(2, 16, 1),
        ...     broadcast_dimensions=(0, 1)
        ... )
        >>> result.shape
        (2, 16, 1)

    Broadcasting from [256] to [1, 1, 256]:
        >>> x = jnp.arange(256)
        >>> result = broadcast_in_dim_pallas(
        ...     x,
        ...     shape=(1, 1, 256),
        ...     broadcast_dimensions=(2,)
        ... )
        >>> result.shape
        (1, 1, 256)

References:
    JAX broadcast_in_dim: https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.broadcast_in_dim.html
    Pallas documentation: https://docs.jax.dev/en/latest/pallas/index.html
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from typing import Tuple, Sequence
import functools


def _broadcast_kernel(
    input_ref,
    output_ref,
    *,
    input_shape: Tuple[int, ...],
    output_shape: Tuple[int, ...],
    broadcast_dimensions: Tuple[int, ...],
):
    """Pallas kernel function for broadcasting.

    This kernel maps each output position to its corresponding input position
    based on the broadcast_dimensions specification.

    Args:
        input_ref: Reference to input array
        output_ref: Reference to output array
        input_shape: Shape of the input array
        output_shape: Shape of the output array
        broadcast_dimensions: Tuple specifying which output dimensions correspond
            to which input dimensions. For example, (0, 2) means input dim 0
            maps to output dim 0, and input dim 1 maps to output dim 2.

    Algorithm:
        For each element in the output at position (i, j, k, ...):
        1. Map output indices back to input indices using broadcast_dimensions
        2. Copy the value from input to output

        For dimensions not in broadcast_dimensions, the input is implicitly size 1
        and gets broadcast (replicated) along that dimension.
    """
    # Get the current program position in the grid
    # The grid is sized according to the output shape
    output_indices = []
    for axis in range(len(output_shape)):
        output_indices.append(pl.program_id(axis))

    # Map output indices to input indices
    # Only dimensions specified in broadcast_dimensions are mapped
    input_indices = []
    for input_dim in range(len(input_shape)):
        # Find which output dimension this input dimension corresponds to
        output_dim = broadcast_dimensions[input_dim]
        input_indices.append(output_indices[output_dim])

    # Read from input and write to output
    # Convert lists to tuples for indexing
    input_idx_tuple = tuple(input_indices)
    output_idx_tuple = tuple(output_indices)

    output_ref[output_idx_tuple] = input_ref[input_idx_tuple]


def broadcast_in_dim_pallas(
    operand: jax.Array,
    *,
    shape: Tuple[int, ...],
    broadcast_dimensions: Tuple[int, ...],
) -> jax.Array:
    """Broadcast an array to a new shape using Pallas kernel.

    Broadcasts (replicates) an array to a new shape by specifying which
    dimensions of the output correspond to which dimensions of the input.

    This implementation uses JAX's Pallas framework to execute the broadcasting
    operation as a custom kernel on CPU.

    Args:
        operand: Input array to broadcast
        shape: Target shape for the output array
        broadcast_dimensions: Tuple of integers specifying the correspondence
            between input and output dimensions. The i-th element indicates
            which output dimension corresponds to the i-th input dimension.
            Must have length equal to operand.ndim.

    Returns:
        Broadcasted array with the specified shape

    Raises:
        ValueError: If broadcast_dimensions length doesn't match operand.ndim
        ValueError: If broadcast_dimensions contains invalid indices
        ValueError: If input dimension size doesn't match output dimension size
            for mapped dimensions

    Examples:
        Broadcast from [2, 16] to [2, 16, 1]:
            Input dims 0, 1 map to output dims 0, 1
            Output dim 2 is new (size 1)

            >>> x = jnp.arange(32).reshape(2, 16)
            >>> broadcast_in_dim_pallas(
            ...     x,
            ...     shape=(2, 16, 1),
            ...     broadcast_dimensions=(0, 1)
            ... )

        Broadcast from [256] to [1, 1, 256]:
            Input dim 0 maps to output dim 2
            Output dims 0, 1 are new (size 1 each)

            >>> x = jnp.arange(256)
            >>> broadcast_in_dim_pallas(
            ...     x,
            ...     shape=(1, 1, 256),
            ...     broadcast_dimensions=(2,)
            ... )

        Broadcast from [4] to [2, 3, 4]:
            Input dim 0 maps to output dim 2
            Output dims 0, 1 are new (replicated)

            >>> x = jnp.array([10, 20, 30, 40])
            >>> result = broadcast_in_dim_pallas(
            ...     x,
            ...     shape=(2, 3, 4),
            ...     broadcast_dimensions=(2,)
            ... )
            >>> result.shape
            (2, 3, 4)
            >>> # Each (i, j) position contains [10, 20, 30, 40]

    Notes:
        - The length of broadcast_dimensions must equal operand.ndim
        - Each value in broadcast_dimensions must be in [0, len(shape))
        - For each i, operand.shape[i] must equal shape[broadcast_dimensions[i]]
        - Dimensions in shape not mentioned in broadcast_dimensions are broadcast
          dimensions (must have size 1 in conceptual input or be new dimensions)
        - This is a CPU implementation using pallas.cpu
    """
    input_shape = operand.shape
    output_shape = shape

    # Validation
    if len(broadcast_dimensions) != len(input_shape):
        raise ValueError(
            f"broadcast_dimensions length ({len(broadcast_dimensions)}) must "
            f"equal operand ndim ({len(input_shape)})"
        )

    for i, (input_size, output_dim) in enumerate(zip(input_shape, broadcast_dimensions)):
        if output_dim < 0 or output_dim >= len(output_shape):
            raise ValueError(
                f"broadcast_dimensions[{i}] = {output_dim} is out of range "
                f"for output shape with {len(output_shape)} dimensions"
            )

        output_size = output_shape[output_dim]
        if input_size != output_size:
            raise ValueError(
                f"Incompatible shapes for broadcasting: "
                f"input dimension {i} has size {input_size} but "
                f"output dimension {output_dim} has size {output_size}"
            )

    # Create the kernel with static arguments
    kernel_with_args = functools.partial(
        _broadcast_kernel,
        input_shape=input_shape,
        output_shape=output_shape,
        broadcast_dimensions=broadcast_dimensions,
    )

    # Create output specification
    out_spec = jax.ShapeDtypeStruct(shape=output_shape, dtype=operand.dtype)

    # Create grid - one program per output element
    grid = output_shape

    # Call the Pallas kernel
    # Note: Using interpret mode for CPU execution
    result = pl.pallas_call(
        kernel_with_args,
        out_shape=out_spec,
        grid=grid,
        interpret=True,  # CPU execution mode
    )(operand)

    return result


def broadcast_in_dim_batched(
    operands: Sequence[jax.Array],
    *,
    shapes: Sequence[Tuple[int, ...]],
    broadcast_dimensions_list: Sequence[Tuple[int, ...]],
) -> Sequence[jax.Array]:
    """Broadcast multiple arrays in a single call.

    Convenience function for broadcasting multiple arrays with different
    target shapes and broadcast dimensions.

    Args:
        operands: Sequence of input arrays to broadcast
        shapes: Sequence of target shapes, one per operand
        broadcast_dimensions_list: Sequence of broadcast_dimensions tuples,
            one per operand

    Returns:
        Sequence of broadcasted arrays

    Example:
        >>> x1 = jnp.arange(32).reshape(2, 16)
        >>> x2 = jnp.arange(256)
        >>> results = broadcast_in_dim_batched(
        ...     operands=[x1, x2],
        ...     shapes=[(2, 16, 1), (1, 1, 256)],
        ...     broadcast_dimensions_list=[(0, 1), (2,)]
        ... )
    """
    if not (len(operands) == len(shapes) == len(broadcast_dimensions_list)):
        raise ValueError(
            "operands, shapes, and broadcast_dimensions_list must have same length"
        )

    results = []
    for operand, shape, broadcast_dims in zip(
        operands, shapes, broadcast_dimensions_list
    ):
        result = broadcast_in_dim_pallas(
            operand,
            shape=shape,
            broadcast_dimensions=broadcast_dims,
        )
        results.append(result)

    return results


# Convenience function that matches JAX's API more closely
broadcast_in_dim = broadcast_in_dim_pallas


if __name__ == "__main__":
    """Demonstration and testing of broadcast operations."""

    print("=" * 80)
    print("Pallas Broadcast Kernel Demonstration")
    print("=" * 80)

    # Example 1: Broadcast [2, 16] -> [2, 16, 1]
    print("\nExample 1: Broadcast [2, 16] -> [2, 16, 1]")
    print("-" * 40)
    x1 = jnp.arange(32).reshape(2, 16)
    print(f"Input shape: {x1.shape}")
    print(f"Input:\n{x1}")

    result1 = broadcast_in_dim_pallas(
        x1,
        shape=(2, 16, 1),
        broadcast_dimensions=(0, 1)
    )
    print(f"\nOutput shape: {result1.shape}")
    print(f"Output (first 2x4x1 slice):\n{result1[:, :4, :]}")
    print(f"Verification: All values preserved: {jnp.allclose(result1[:, :, 0], x1)}")

    # Example 2: Broadcast [256] -> [1, 1, 256]
    print("\n" + "=" * 80)
    print("Example 2: Broadcast [256] -> [1, 1, 256]")
    print("-" * 40)
    x2 = jnp.arange(256)
    print(f"Input shape: {x2.shape}")
    print(f"Input (first 10 elements): {x2[:10]}")

    result2 = broadcast_in_dim_pallas(
        x2,
        shape=(1, 1, 256),
        broadcast_dimensions=(2,)
    )
    print(f"\nOutput shape: {result2.shape}")
    print(f"Output (first 10 elements): {result2[0, 0, :10]}")
    print(f"Verification: All values preserved: {jnp.allclose(result2[0, 0, :], x2)}")

    # Example 3: Broadcast [4] -> [2, 3, 4]
    print("\n" + "=" * 80)
    print("Example 3: Broadcast [4] -> [2, 3, 4] (with replication)")
    print("-" * 40)
    x3 = jnp.array([10., 20., 30., 40.])
    print(f"Input shape: {x3.shape}")
    print(f"Input: {x3}")

    result3 = broadcast_in_dim_pallas(
        x3,
        shape=(2, 3, 4),
        broadcast_dimensions=(2,)
    )
    print(f"\nOutput shape: {result3.shape}")
    print(f"Output:\n{result3}")
    print(f"Verification: Each position replicates input: {jnp.allclose(result3[0, 0, :], x3)}")
    print(f"Verification: All slices identical: {jnp.allclose(result3[0, 0, :], result3[1, 2, :])}")

    # Example 4: Compare with JAX's native broadcast_in_dim
    print("\n" + "=" * 80)
    print("Example 4: Verification against JAX's native broadcast_in_dim")
    print("-" * 40)
    x4 = jnp.arange(12).reshape(3, 4)
    target_shape = (2, 3, 4)
    broadcast_dims = (1, 2)

    result_pallas = broadcast_in_dim_pallas(
        x4,
        shape=target_shape,
        broadcast_dimensions=broadcast_dims
    )
    result_jax = jax.lax.broadcast_in_dim(
        x4,
        shape=target_shape,
        broadcast_dimensions=broadcast_dims
    )

    print(f"Input shape: {x4.shape}")
    print(f"Target shape: {target_shape}")
    print(f"Broadcast dimensions: {broadcast_dims}")
    print(f"\nPallas result shape: {result_pallas.shape}")
    print(f"JAX result shape: {result_jax.shape}")
    print(f"Results match: {jnp.allclose(result_pallas, result_jax)}")
    print(f"Max absolute difference: {jnp.max(jnp.abs(result_pallas - result_jax))}")

    print("\n" + "=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)
