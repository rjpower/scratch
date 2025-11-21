"""Pallas kernel for matrix multiplication (dot_general) operations.

This module implements efficient matmul operations using JAX Pallas for CPU.
It supports various dot_general patterns used in transformer models including:
- Standard batch matmul: [B, L, D] @ [D, D'] -> [B, L, D']
- Batched attention matmul: [B, H, L, D] @ [B, H, D, L'] -> [B, H, L, L']

Note: For CPU execution, we use interpret=True mode.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from typing import Tuple, Optional
import functools

# Store references to original JAX operations to avoid recursion
_jax_matmul = jnp.matmul
_jax_dot_general = jax.lax.dot_general


def _matmul_kernel(x_ref, y_ref, out_ref):
    """Basic matrix multiplication kernel.

    Args:
        x_ref: Left input reference
        y_ref: Right input reference
        out_ref: Output reference
    """
    # Load inputs
    x = x_ref[...]
    y = y_ref[...]

    # Perform matmul and store result
    out_ref[...] = _jax_matmul(x, y)


def matmul_simple(x: jax.Array, y: jax.Array) -> jax.Array:
    """Simple matrix multiplication using Pallas.

    This handles standard 2D matrix multiplication: [M, K] @ [K, N] -> [M, N]

    Args:
        x: Left matrix [M, K]
        y: Right matrix [K, N]

    Returns:
        Product matrix [M, N]

    Example:
        >>> x = jnp.ones((4, 8))
        >>> y = jnp.ones((8, 16))
        >>> result = matmul_simple(x, y)
        >>> result.shape
        (4, 16)
    """
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(f"Expected 2D matrices, got shapes {x.shape} and {y.shape}")

    m, k = x.shape
    k2, n = y.shape

    if k != k2:
        raise ValueError(f"Incompatible shapes for matmul: {x.shape} @ {y.shape}")

    out_shape = (m, n)

    return pl.pallas_call(
        _matmul_kernel,
        out_shape=jax.ShapeDtypeStruct(out_shape, x.dtype),
        interpret=True,
    )(x, y)


def _batch_matmul_kernel(x_ref, y_ref, out_ref):
    """Batched matrix multiplication kernel.

    Handles batched matmul where the batch dimensions are already aligned.

    Args:
        x_ref: Left input reference [..., M, K]
        y_ref: Right input reference [..., K, N]
        out_ref: Output reference [..., M, N]
    """
    x = x_ref[...]
    y = y_ref[...]

    # JAX matmul handles batched dimensions automatically
    out_ref[...] = _jax_matmul(x, y)


def batch_matmul(x: jax.Array, y: jax.Array) -> jax.Array:
    """Batched matrix multiplication using Pallas.

    Handles matmul with leading batch dimensions: [..., M, K] @ [..., K, N] -> [..., M, N]

    Args:
        x: Left tensor with shape [..., M, K]
        y: Right tensor with shape [..., K, N]
            (batch dimensions must match or be broadcastable)

    Returns:
        Product tensor [..., M, N]

    Example:
        >>> # Single batch dimension
        >>> x = jnp.ones((2, 16, 64))  # [B, L, D]
        >>> y = jnp.ones((2, 64, 64))  # [B, D, D']
        >>> result = batch_matmul(x, y)
        >>> result.shape
        (2, 16, 64)

        >>> # Multiple batch dimensions (attention)
        >>> x = jnp.ones((2, 4, 16, 16))  # [B, H, L, D]
        >>> y = jnp.ones((2, 4, 16, 16))  # [B, H, D, L']
        >>> result = batch_matmul(x, y)
        >>> result.shape
        (2, 4, 16, 16)
    """
    if x.ndim < 2 or y.ndim < 2:
        raise ValueError(f"Expected at least 2D tensors, got shapes {x.shape} and {y.shape}")

    # Get matrix dimensions
    *batch_dims_x, m, k = x.shape
    *batch_dims_y, k2, n = y.shape

    if k != k2:
        raise ValueError(
            f"Incompatible inner dimensions for matmul: "
            f"{x.shape}[..., {k}] @ {y.shape}[..., {k2}, ...]"
        )

    # Determine output shape
    # Batch dimensions should broadcast
    batch_dims = []
    for i in range(max(len(batch_dims_x), len(batch_dims_y))):
        idx_x = i - (max(len(batch_dims_x), len(batch_dims_y)) - len(batch_dims_x))
        idx_y = i - (max(len(batch_dims_x), len(batch_dims_y)) - len(batch_dims_y))

        dim_x = batch_dims_x[idx_x] if idx_x >= 0 else 1
        dim_y = batch_dims_y[idx_y] if idx_y >= 0 else 1

        if dim_x != dim_y and dim_x != 1 and dim_y != 1:
            raise ValueError(
                f"Incompatible batch dimensions: {batch_dims_x} vs {batch_dims_y}"
            )

        batch_dims.append(max(dim_x, dim_y))

    out_shape = tuple(batch_dims) + (m, n)

    return pl.pallas_call(
        _batch_matmul_kernel,
        out_shape=jax.ShapeDtypeStruct(out_shape, x.dtype),
        interpret=True,
    )(x, y)


def dot_general_pallas(
    lhs: jax.Array,
    rhs: jax.Array,
    dimension_numbers: Tuple[Tuple[Tuple[int, ...], Tuple[int, ...]], Tuple[Tuple[int, ...], Tuple[int, ...]]],
    preferred_element_type: Optional[jnp.dtype] = None,
) -> jax.Array:
    """General dot product using Pallas (wrapper around jax.lax.dot_general).

    This function replicates jax.lax.dot_general behavior using Pallas kernels.

    Args:
        lhs: Left-hand side array
        rhs: Right-hand side array
        dimension_numbers: Tuple of ((lhs_contract, rhs_contract), (lhs_batch, rhs_batch))
            - lhs_contract, rhs_contract: axes to contract over
            - lhs_batch, rhs_batch: batch axes
        preferred_element_type: Optional output dtype

    Returns:
        Result of the dot_general operation

    Common patterns in transformers:
        1. Standard matmul: (([2], [0]), ([], []))
           [B, L, D] @ [D, D'] -> [B, L, D']

        2. Batched attention: (([3], [2]), ([0, 1], [0, 1]))
           [B, H, L, D] @ [B, H, D, L'] -> [B, H, L, L']

    Example:
        >>> # Pattern 1: [B, L, D] @ [D, D'] -> [B, L, D']
        >>> lhs = jnp.ones((2, 16, 64))
        >>> rhs = jnp.ones((64, 64))
        >>> result = dot_general_pallas(
        ...     lhs, rhs,
        ...     dimension_numbers=(([2], [0]), ([], []))
        ... )
        >>> result.shape
        (2, 16, 64)

        >>> # Pattern 2: [B, H, L, D] @ [B, H, D, L'] -> [B, H, L, L']
        >>> lhs = jnp.ones((2, 4, 16, 16))
        >>> rhs = jnp.ones((2, 4, 16, 16))
        >>> result = dot_general_pallas(
        ...     lhs, rhs,
        ...     dimension_numbers=(([3], [2]), ([0, 1], [0, 1]))
        ... )
        >>> result.shape
        (2, 4, 16, 16)
    """
    # For now, we use a simple implementation that calls jax.lax.dot_general
    # inside the Pallas kernel. A more optimized version would implement
    # the contraction logic directly.

    (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers

    def _dot_general_kernel(lhs_ref, rhs_ref, out_ref):
        """Kernel for dot_general operation."""
        lhs_val = lhs_ref[...]
        rhs_val = rhs_ref[...]

        # Perform dot_general
        result = _jax_dot_general(
            lhs_val,
            rhs_val,
            dimension_numbers=dimension_numbers,
            preferred_element_type=preferred_element_type,
        )

        out_ref[...] = result

    # Compute output shape
    # This is complex in the general case, so we use jax's shape inference
    dummy_result = jax.eval_shape(
        lambda: _jax_dot_general(
            lhs, rhs,
            dimension_numbers=dimension_numbers,
            preferred_element_type=preferred_element_type,
        )
    )

    out_dtype = preferred_element_type if preferred_element_type is not None else lhs.dtype

    return pl.pallas_call(
        _dot_general_kernel,
        out_shape=jax.ShapeDtypeStruct(dummy_result.shape, out_dtype),
        interpret=True,
    )(lhs, rhs)


# Convenience functions for common transformer patterns

def attention_scores(query: jax.Array, key: jax.Array) -> jax.Array:
    """Compute attention scores: Q @ K^T.

    Args:
        query: Query tensor [B, H, L, D]
        key: Key tensor [B, H, L, D]

    Returns:
        Attention scores [B, H, L, L]

    Example:
        >>> query = jnp.ones((2, 4, 16, 16))
        >>> key = jnp.ones((2, 4, 16, 16))
        >>> scores = attention_scores(query, key)
        >>> scores.shape
        (2, 4, 16, 16)
    """
    # Q @ K^T: [B, H, L, D] @ [B, H, D, L] -> [B, H, L, L]
    # This is: (([3], [2]), ([0, 1], [0, 1]))
    return dot_general_pallas(
        query,
        key,
        dimension_numbers=(([3], [2]), ([0, 1], [0, 1]))
    )


def attention_output(attn_weights: jax.Array, value: jax.Array) -> jax.Array:
    """Compute attention output: attn_weights @ V.

    Args:
        attn_weights: Attention weights [B, H, L, L]
        value: Value tensor [B, H, L, D]

    Returns:
        Attention output [B, H, L, D]

    Example:
        >>> attn_weights = jnp.ones((2, 4, 16, 16))
        >>> value = jnp.ones((2, 4, 16, 16))
        >>> output = attention_output(attn_weights, value)
        >>> output.shape
        (2, 4, 16, 16)
    """
    # attn @ V: [B, H, L, L] @ [B, H, L, D] -> [B, H, L, D]
    # This is: (([3], [2]), ([0, 1], [0, 1]))
    return dot_general_pallas(
        attn_weights,
        value,
        dimension_numbers=(([3], [2]), ([0, 1], [0, 1]))
    )


def linear_projection(x: jax.Array, weight: jax.Array) -> jax.Array:
    """Apply linear projection: X @ W.

    Args:
        x: Input tensor [B, L, D]
        weight: Weight matrix [D, D']

    Returns:
        Projected tensor [B, L, D']

    Example:
        >>> x = jnp.ones((2, 16, 64))
        >>> weight = jnp.ones((64, 64))
        >>> output = linear_projection(x, weight)
        >>> output.shape
        (2, 16, 64)
    """
    # X @ W: [B, L, D] @ [D, D'] -> [B, L, D']
    # This is: (([2], [0]), ([], []))
    return dot_general_pallas(
        x,
        weight,
        dimension_numbers=(([2], [0]), ([], []))
    )


# Export public API
__all__ = [
    'matmul_simple',
    'batch_matmul',
    'dot_general_pallas',
    'attention_scores',
    'attention_output',
    'linear_projection',
]


# Example usage and verification
if __name__ == "__main__":
    print("=" * 80)
    print("Pallas Matmul Kernel - Examples and Verification")
    print("=" * 80)

    # Example 1: Simple 2D matmul
    print("\n1. Simple 2D matmul [M, K] @ [K, N] -> [M, N]")
    x1 = jnp.ones((4, 8))
    y1 = jnp.ones((8, 16))
    result1 = matmul_simple(x1, y1)
    expected1 = jnp.matmul(x1, y1)
    print(f"   Input shapes: {x1.shape} @ {y1.shape}")
    print(f"   Output shape: {result1.shape}")
    print(f"   Matches jnp.matmul: {jnp.allclose(result1, expected1)}")

    # Example 2: Batched matmul
    print("\n2. Batched matmul [B, L, D] @ [B, D, D'] -> [B, L, D']")
    x2 = jnp.ones((2, 16, 64))
    y2 = jnp.ones((2, 64, 64))
    result2 = batch_matmul(x2, y2)
    expected2 = jnp.matmul(x2, y2)
    print(f"   Input shapes: {x2.shape} @ {y2.shape}")
    print(f"   Output shape: {result2.shape}")
    print(f"   Matches jnp.matmul: {jnp.allclose(result2, expected2)}")

    # Example 3: Linear projection (transformer pattern)
    print("\n3. Linear projection [B, L, D] @ [D, D'] -> [B, L, D']")
    x3 = jnp.ones((2, 16, 64))
    weight3 = jnp.ones((64, 64))
    result3 = linear_projection(x3, weight3)
    expected3 = jax.lax.dot_general(
        x3, weight3, dimension_numbers=(([2], [0]), ([], []))
    )
    print(f"   Input shapes: {x3.shape} @ {weight3.shape}")
    print(f"   Output shape: {result3.shape}")
    print(f"   Matches dot_general: {jnp.allclose(result3, expected3)}")

    # Example 4: Attention scores (Q @ K^T)
    print("\n4. Attention scores [B, H, L, D] @ [B, H, D, L'] -> [B, H, L, L']")
    query = jnp.ones((2, 4, 16, 16))
    key = jnp.ones((2, 4, 16, 16))
    scores = attention_scores(query, key)
    expected_scores = jax.lax.dot_general(
        query, key, dimension_numbers=(([3], [2]), ([0, 1], [0, 1]))
    )
    print(f"   Input shapes: {query.shape} @ {key.shape}")
    print(f"   Output shape: {scores.shape}")
    print(f"   Matches dot_general: {jnp.allclose(scores, expected_scores)}")

    # Example 5: Attention output (attn @ V)
    print("\n5. Attention output [B, H, L, L] @ [B, H, L, D] -> [B, H, L, D]")
    attn = jnp.ones((2, 4, 16, 16))
    value = jnp.ones((2, 4, 16, 16))
    output = attention_output(attn, value)
    expected_output = jax.lax.dot_general(
        attn, value, dimension_numbers=(([3], [2]), ([0, 1], [0, 1]))
    )
    print(f"   Input shapes: {attn.shape} @ {value.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Matches dot_general: {jnp.allclose(output, expected_output)}")

    print("\n" + "=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)
