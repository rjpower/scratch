"""Pallas kernels for element-wise operations."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from functools import partial

# Store references to original JAX operations to avoid recursion
_jax_add = jnp.add
_jax_subtract = jnp.subtract
_jax_multiply = jnp.multiply
_jax_divide = jnp.divide
_jax_maximum = jnp.maximum
_jax_exp = jnp.exp
_jax_tanh = jnp.tanh
_jax_sqrt = jnp.sqrt
_jax_power = jnp.power


def elementwise_binary_kernel(x_ref, y_ref, out_ref, *, op):
    """Generic binary elementwise kernel.

    Args:
        x_ref: First input reference
        y_ref: Second input reference
        out_ref: Output reference
        op: Binary operation function (e.g., jnp.add, jnp.multiply)
    """
    x = x_ref[...]
    y = y_ref[...]
    out_ref[...] = op(x, y)


def elementwise_unary_kernel(x_ref, out_ref, *, op):
    """Generic unary elementwise kernel.

    Args:
        x_ref: Input reference
        out_ref: Output reference
        op: Unary operation function (e.g., jnp.exp, jnp.tanh)
    """
    x = x_ref[...]
    out_ref[...] = op(x)


def integer_pow_kernel(x_ref, out_ref, *, y):
    """Integer power kernel.

    Args:
        x_ref: Input reference
        out_ref: Output reference
        y: Integer exponent
    """
    x = x_ref[...]
    out_ref[...] = _jax_power(x, y)


# Binary operations
def add(x: jax.Array, y: jax.Array) -> jax.Array:
    """Element-wise addition using Pallas."""
    # Handle scalars by falling back to JAX
    if not hasattr(x, 'shape') or not hasattr(y, 'shape') or x.ndim == 0 or y.ndim == 0:
        return _jax_add(x, y)
    return pl.pallas_call(
        partial(elementwise_binary_kernel, op=_jax_add),
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        interpret=True
    )(x, y)


def sub(x: jax.Array, y: jax.Array) -> jax.Array:
    """Element-wise subtraction using Pallas."""
    # Handle scalars by falling back to JAX
    if not hasattr(x, 'shape') or not hasattr(y, 'shape') or x.ndim == 0 or y.ndim == 0:
        return _jax_subtract(x, y)
    return pl.pallas_call(
        partial(elementwise_binary_kernel, op=_jax_subtract),
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        interpret=True
    )(x, y)


def mul(x: jax.Array, y: jax.Array) -> jax.Array:
    """Element-wise multiplication using Pallas."""
    # Handle scalars by falling back to JAX
    if not hasattr(x, 'shape') or not hasattr(y, 'shape') or x.ndim == 0 or y.ndim == 0:
        return _jax_multiply(x, y)
    return pl.pallas_call(
        partial(elementwise_binary_kernel, op=_jax_multiply),
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        interpret=True
    )(x, y)


def div(x: jax.Array, y: jax.Array) -> jax.Array:
    """Element-wise division using Pallas."""
    # Handle scalars by falling back to JAX
    if not hasattr(x, 'shape') or not hasattr(y, 'shape') or x.ndim == 0 or y.ndim == 0:
        return _jax_divide(x, y)
    return pl.pallas_call(
        partial(elementwise_binary_kernel, op=_jax_divide),
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        interpret=True
    )(x, y)


def maximum(x: jax.Array, y: jax.Array) -> jax.Array:
    """Element-wise maximum using Pallas."""
    return pl.pallas_call(
        partial(elementwise_binary_kernel, op=_jax_maximum),
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        interpret=True
    )(x, y)


# Unary operations
def exp(x: jax.Array) -> jax.Array:
    """Element-wise exponential using Pallas."""
    # Handle scalars by falling back to JAX
    if not hasattr(x, 'shape') or x.ndim == 0:
        return _jax_exp(x)
    return pl.pallas_call(
        partial(elementwise_unary_kernel, op=_jax_exp),
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        interpret=True
    )(x)


def tanh(x: jax.Array) -> jax.Array:
    """Element-wise tanh using Pallas."""
    # Handle scalars by falling back to JAX
    if not hasattr(x, 'shape') or x.ndim == 0:
        return _jax_tanh(x)
    return pl.pallas_call(
        partial(elementwise_unary_kernel, op=_jax_tanh),
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        interpret=True
    )(x)


def sqrt(x: jax.Array) -> jax.Array:
    """Element-wise square root using Pallas."""
    # Handle scalars by falling back to JAX
    if not hasattr(x, 'shape') or x.ndim == 0:
        return _jax_sqrt(x)
    return pl.pallas_call(
        partial(elementwise_unary_kernel, op=_jax_sqrt),
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        interpret=True
    )(x)


def integer_pow(x: jax.Array, y: int) -> jax.Array:
    """Element-wise integer power using Pallas.

    Args:
        x: Input array
        y: Integer exponent

    Returns:
        x^y element-wise
    """
    return pl.pallas_call(
        partial(integer_pow_kernel, y=y),
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        interpret=True
    )(x)
