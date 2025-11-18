"""Pallas kernel for tensor reshape operations.

This module implements a Pallas kernel for reshaping tensors. While reshape is
typically a view operation (no data movement) in JAX, this Pallas implementation
provides an explicit kernel that can be used in custom kernel pipelines.

Key use cases in transformers:
- Split heads: [B, L, D] -> [B, L, n_heads, d_k]
- Merge heads: [B, L, n_heads, d_k] -> [B, L, D]

Note: For CPU execution, this uses interpret=True mode.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from typing import Tuple


def reshape_kernel(x_ref, o_ref):
    """Pallas kernel for reshape operation.

    This kernel reads from the input reference and writes to the output reference.
    The actual reshaping is handled by the indexing logic in the grid.

    Args:
        x_ref: Input array reference (Pallas Ref)
        o_ref: Output array reference (Pallas Ref)
    """
    # Get the linear index for this program
    idx = pl.program_id(0)

    # Read from input and write to output
    # Both references are already sliced to the appropriate element
    o_ref[...] = x_ref[...]


def reshape(x: jax.Array, new_shape: Tuple[int, ...]) -> jax.Array:
    """Reshape an array using a Pallas kernel.

    This function reshapes an array from its current shape to a new shape.
    The total number of elements must remain the same.

    Args:
        x: Input array to reshape
        new_shape: Tuple specifying the desired output shape

    Returns:
        Reshaped array with shape new_shape

    Raises:
        ValueError: If the total number of elements doesn't match

    Examples:
        >>> import jax.numpy as jnp
        >>> x = jnp.arange(24).reshape(2, 3, 4)
        >>> y = reshape(x, (2, 12))
        >>> y.shape
        (2, 12)

        >>> # Split heads: [B, L, D] -> [B, L, n_heads, d_k]
        >>> x = jnp.ones((2, 8, 64))  # batch=2, seq_len=8, d_model=64
        >>> y = reshape(x, (2, 8, 4, 16))  # n_heads=4, d_k=16
        >>> y.shape
        (2, 8, 4, 16)

        >>> # Merge heads: [B, L, n_heads, d_k] -> [B, L, D]
        >>> x = jnp.ones((2, 8, 4, 16))
        >>> y = reshape(x, (2, 8, 64))
        >>> y.shape
        (2, 8, 64)
    """
    # Validate that reshape is valid
    old_size = x.size
    new_size = 1
    for dim in new_shape:
        new_size *= dim

    if old_size != new_size:
        raise ValueError(
            f"Cannot reshape array of size {old_size} into shape {new_shape} "
            f"(size {new_size})"
        )

    # Simple kernel that just reshapes
    def _reshape_kernel(x_ref, out_ref):
        out_ref[...] = x_ref[...].reshape(new_shape)

    # Call the kernel
    result = pl.pallas_call(
        _reshape_kernel,
        out_shape=jax.ShapeDtypeStruct(new_shape, x.dtype),
        interpret=True,  # Required for CPU execution
    )(x)

    return result


def split_heads(x: jax.Array, n_heads: int) -> jax.Array:
    """Split the last dimension into multiple attention heads.

    This is a specialized reshape for transformer attention, converting
    [batch, seq_len, d_model] -> [batch, seq_len, n_heads, d_k]
    where d_k = d_model // n_heads.

    Args:
        x: Input tensor of shape [batch, seq_len, d_model]
        n_heads: Number of attention heads

    Returns:
        Tensor of shape [batch, seq_len, n_heads, d_k]

    Example:
        >>> x = jnp.ones((2, 8, 64))  # batch=2, seq_len=8, d_model=64
        >>> y = split_heads(x, n_heads=4)
        >>> y.shape
        (2, 8, 4, 16)
    """
    batch_size, seq_len, d_model = x.shape

    if d_model % n_heads != 0:
        raise ValueError(
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )

    d_k = d_model // n_heads
    new_shape = (batch_size, seq_len, n_heads, d_k)

    return reshape(x, new_shape)


def merge_heads(x: jax.Array) -> jax.Array:
    """Merge multiple attention heads back into a single dimension.

    This is a specialized reshape for transformer attention, converting
    [batch, seq_len, n_heads, d_k] -> [batch, seq_len, d_model]
    where d_model = n_heads * d_k.

    Args:
        x: Input tensor of shape [batch, seq_len, n_heads, d_k]

    Returns:
        Tensor of shape [batch, seq_len, d_model]

    Example:
        >>> x = jnp.ones((2, 8, 4, 16))  # batch=2, seq_len=8, n_heads=4, d_k=16
        >>> y = merge_heads(x)
        >>> y.shape
        (2, 8, 64)
    """
    batch_size, seq_len, n_heads, d_k = x.shape
    d_model = n_heads * d_k
    new_shape = (batch_size, seq_len, d_model)

    return reshape(x, new_shape)


# Example usage and testing
if __name__ == "__main__":
    print("Testing Pallas reshape kernel...")
    print("=" * 80)

    # Test 1: Basic reshape
    print("\nTest 1: Basic reshape")
    x = jnp.arange(24).reshape(2, 3, 4)
    print(f"Input shape: {x.shape}")
    y = reshape(x, (2, 12))
    print(f"Output shape: {y.shape}")
    print(f"Correct: {jnp.allclose(y, x.reshape(2, 12))}")

    # Test 2: Split heads for transformer
    print("\nTest 2: Split heads [B, L, D] -> [B, L, n_heads, d_k]")
    x = jnp.ones((2, 8, 64))  # batch=2, seq_len=8, d_model=64
    print(f"Input shape: {x.shape}")
    y = split_heads(x, n_heads=4)
    print(f"Output shape: {y.shape}")
    expected = x.reshape(2, 8, 4, 16)
    print(f"Correct: {jnp.allclose(y, expected)}")

    # Test 3: Merge heads for transformer
    print("\nTest 3: Merge heads [B, L, n_heads, d_k] -> [B, L, D]")
    x = jnp.ones((2, 8, 4, 16))
    print(f"Input shape: {x.shape}")
    y = merge_heads(x)
    print(f"Output shape: {y.shape}")
    expected = x.reshape(2, 8, 64)
    print(f"Correct: {jnp.allclose(y, expected)}")

    # Test 4: Round-trip split and merge
    print("\nTest 4: Round-trip split and merge")
    x_orig = jnp.arange(2 * 8 * 64).reshape(2, 8, 64).astype(jnp.float32)
    print(f"Original shape: {x_orig.shape}")
    x_split = split_heads(x_orig, n_heads=4)
    print(f"After split: {x_split.shape}")
    x_merged = merge_heads(x_split)
    print(f"After merge: {x_merged.shape}")
    print(f"Round-trip correct: {jnp.allclose(x_merged, x_orig)}")

    # Test 5: Different dtypes
    print("\nTest 5: Different dtypes")
    for dtype in [jnp.float32, jnp.float16, jnp.int32]:
        x = jnp.ones((4, 6), dtype=dtype)
        y = reshape(x, (2, 12))
        print(f"  {dtype}: {y.dtype} - Correct: {jnp.allclose(y, x.reshape(2, 12))}")

    # Test 6: Large tensor
    print("\nTest 6: Large tensor")
    x = jnp.ones((4, 16, 128))
    y = split_heads(x, n_heads=8)
    print(f"Input: {x.shape} -> Output: {y.shape}")
    expected = x.reshape(4, 16, 8, 16)
    print(f"Correct: {jnp.allclose(y, expected)}")

    print("\n" + "=" * 80)
    print("All tests completed!")
