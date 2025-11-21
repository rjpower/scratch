"""Backend module that provides JAX-compatible API using Pallas kernels.

This module provides drop-in replacements for JAX numpy and neural network
operations that use Pallas kernels under the hood.
"""

import jax.numpy as jnp
from kernels import (
    elementwise,
    matmul,
    reduce,
    reshape,
    transpose,
    broadcast,
    where as where_kernel,
    softmax as softmax_kernel,
    gelu as gelu_kernel,
    mean as mean_kernel,
    var as var_kernel,
)


# Elementwise operations
add = elementwise.add
subtract = elementwise.sub
multiply = elementwise.mul
divide = elementwise.div
exp = elementwise.exp
sqrt = elementwise.sqrt
tanh = elementwise.tanh


# Matrix operations
def matmul_op(x, y):
    """Matrix multiplication using Pallas kernels."""
    # Handle different cases
    if x.ndim == 2 and y.ndim == 2:
        return matmul.matmul_simple(x, y)
    elif x.ndim == 3 and y.ndim == 3:
        return matmul.batch_matmul(x, y)
    else:
        # For general matmul, use dot_general with appropriate dimension numbers
        # Contract last dim of x with second-to-last of y
        lhs_contract = [x.ndim - 1]
        rhs_contract = [max(0, y.ndim - 2)]

        # Batch dimensions
        num_batch = min(x.ndim - 2, y.ndim - 2)
        batch_dims = (list(range(num_batch)), list(range(num_batch)))

        dimension_numbers = ((lhs_contract, rhs_contract), batch_dims)
        return matmul.dot_general_pallas(x, y, dimension_numbers=dimension_numbers)


# Shape operations
reshape_op = reshape.reshape
transpose_op = transpose.transpose_pallas


# Conditional operations
where_op = where_kernel.where


# Reduction operations
def mean_op(x, axis=-1, keepdims=False):
    """Mean using Pallas kernel."""
    return mean_kernel.mean(x, axis=axis, keepdims=keepdims)


def var_op(x, axis=-1, keepdims=False):
    """Variance using Pallas kernel."""
    return var_kernel.var(x, axis=axis, keepdims=keepdims)


# Neural network operations
class nn:
    """Neural network operations using Pallas kernels."""

    @staticmethod
    def softmax(x, axis=-1):
        """Softmax using Pallas kernel."""
        return softmax_kernel.softmax(x, axis=axis)

    @staticmethod
    def gelu(x):
        """GELU using Pallas kernel."""
        return gelu_kernel.gelu(x)


# Create a namespace that mimics jax.numpy
class Backend:
    """Backend namespace providing JAX-compatible API with Pallas kernels."""

    # Elementwise
    add = add
    subtract = subtract
    multiply = multiply
    divide = divide
    exp = exp
    sqrt = sqrt
    tanh = tanh

    # Matrix
    matmul = matmul_op

    # Shape
    reshape = reshape_op
    transpose = transpose_op

    # Conditional
    where = where_op

    # Reduction
    mean = mean_op
    var = var_op

    # Neural network
    nn = nn


# Create default backend instance
pallas_backend = Backend()
