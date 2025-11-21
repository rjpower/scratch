"""Pallas kernels for element-wise operations."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from functools import partial


def elementwise_binary_kernel(x_ref, y_ref, out_ref, *, op):
    """Generic binary elementwise kernel.

    Args:
        x_ref: First input reference
        y_ref: Second input reference
        out_ref: Output reference
        op: Operation name ('add', 'sub', 'mul', 'div', 'max')
    """
    x = x_ref[...]
    y = y_ref[...]

    if op == 'add':
        out_ref[...] = x + y
    elif op == 'sub':
        out_ref[...] = x - y
    elif op == 'mul':
        out_ref[...] = x * y
    elif op == 'div':
        out_ref[...] = x / y
    elif op == 'max':
        out_ref[...] = jnp.where(x > y, x, y)


def elementwise_unary_kernel(x_ref, out_ref, *, op):
    """Generic unary elementwise kernel.

    Args:
        x_ref: Input reference
        out_ref: Output reference
        op: Operation name ('exp', 'tanh', 'sqrt')
    """
    x = x_ref[...]

    if op == 'exp':
        out_ref[...] = jnp.exp(x)
    elif op == 'tanh':
        out_ref[...] = jnp.tanh(x)
    elif op == 'sqrt':
        out_ref[...] = jnp.sqrt(x)


def integer_pow_kernel(x_ref, out_ref, *, y):
    """Integer power kernel.

    Args:
        x_ref: Input reference
        out_ref: Output reference
        y: Integer exponent
    """
    x = x_ref[...]
    out_ref[...] = jnp.power(x, y)


# Binary operations
def add(x: jax.Array, y: jax.Array) -> jax.Array:
    """Element-wise addition using Pallas."""
    # Handle scalars by falling back to JAX
    if not hasattr(x, 'shape') or not hasattr(y, 'shape') or x.ndim == 0 or y.ndim == 0:
        return x + y
    return pl.pallas_call(
        partial(elementwise_binary_kernel, op='add'),
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        interpret=True
    )(x, y)


def sub(x: jax.Array, y: jax.Array) -> jax.Array:
    """Element-wise subtraction using Pallas."""
    # Handle scalars by falling back to JAX
    if not hasattr(x, 'shape') or not hasattr(y, 'shape') or x.ndim == 0 or y.ndim == 0:
        return x - y
    return pl.pallas_call(
        partial(elementwise_binary_kernel, op='sub'),
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        interpret=True
    )(x, y)


def mul(x: jax.Array, y: jax.Array) -> jax.Array:
    """Element-wise multiplication using Pallas."""
    # Handle scalars by falling back to JAX
    if not hasattr(x, 'shape') or not hasattr(y, 'shape') or x.ndim == 0 or y.ndim == 0:
        return x * y
    return pl.pallas_call(
        partial(elementwise_binary_kernel, op='mul'),
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        interpret=True
    )(x, y)


def div(x: jax.Array, y: jax.Array) -> jax.Array:
    """Element-wise division using Pallas."""
    # Handle scalars by falling back to JAX
    if not hasattr(x, 'shape') or not hasattr(y, 'shape') or x.ndim == 0 or y.ndim == 0:
        return x / y
    return pl.pallas_call(
        partial(elementwise_binary_kernel, op='div'),
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        interpret=True
    )(x, y)


def maximum(x: jax.Array, y: jax.Array) -> jax.Array:
    """Element-wise maximum using Pallas."""
    return pl.pallas_call(
        partial(elementwise_binary_kernel, op='max'),
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        interpret=True
    )(x, y)


# Unary operations
def exp(x: jax.Array) -> jax.Array:
    """Element-wise exponential using Pallas."""
    # Handle scalars by falling back to JAX
    if not hasattr(x, 'shape') or x.ndim == 0:
        return jnp.exp(x)
    return pl.pallas_call(
        partial(elementwise_unary_kernel, op='exp'),
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        interpret=True
    )(x)


def tanh(x: jax.Array) -> jax.Array:
    """Element-wise tanh using Pallas."""
    # Handle scalars by falling back to JAX
    if not hasattr(x, 'shape') or x.ndim == 0:
        return jnp.tanh(x)
    return pl.pallas_call(
        partial(elementwise_unary_kernel, op='tanh'),
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        interpret=True
    )(x)


def sqrt(x: jax.Array) -> jax.Array:
    """Element-wise square root using Pallas."""
    # Handle scalars by falling back to JAX
    if not hasattr(x, 'shape') or x.ndim == 0:
        return jnp.sqrt(x)
    return pl.pallas_call(
        partial(elementwise_unary_kernel, op='sqrt'),
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
