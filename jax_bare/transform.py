"""Transformation utilities to convert JAX functions to use Pallas kernels.

This module provides utilities to transform JAX computations to transparently
use Pallas kernels instead of standard JAX operations.
"""

import functools
import jax
import jax.numpy as jnp
from typing import Callable, Any, Dict
import pallas_backend


class OperationRegistry:
    """Registry mapping JAX operations to Pallas kernel equivalents."""

    def __init__(self):
        self._mappings = {
            # Elementwise operations
            jnp.add: pallas_backend.add,
            jnp.subtract: pallas_backend.subtract,
            jnp.multiply: pallas_backend.multiply,
            jnp.divide: pallas_backend.divide,
            jnp.exp: pallas_backend.exp,
            jnp.sqrt: pallas_backend.sqrt,
            jnp.tanh: pallas_backend.tanh,

            # Matrix operations
            jnp.matmul: pallas_backend.matmul_op,

            # Shape operations
            jnp.reshape: pallas_backend.reshape_op,
            jnp.transpose: pallas_backend.transpose_op,

            # Conditional operations
            jnp.where: pallas_backend.where_op,

            # Reduction operations
            jnp.mean: pallas_backend.mean_op,
            jnp.var: pallas_backend.var_op,

            # Neural network operations
            jax.nn.softmax: pallas_backend.nn.softmax,
            jax.nn.gelu: pallas_backend.nn.gelu,
        }

    def get_pallas_op(self, jax_op):
        """Get Pallas equivalent of a JAX operation."""
        return self._mappings.get(jax_op, jax_op)

    def register(self, jax_op, pallas_op):
        """Register a custom mapping."""
        self._mappings[jax_op] = pallas_op


# Global registry
_registry = OperationRegistry()


def tree_map_operations(func: Callable, op_mapping: Dict = None) -> Callable:
    """Transform a function to use alternative operations.

    This uses a context-based approach where we temporarily replace JAX
    operations with Pallas kernels during function execution.

    Args:
        func: Function to transform
        op_mapping: Optional custom operation mapping

    Returns:
        Transformed function that uses Pallas kernels
    """
    if op_mapping is None:
        op_mapping = _registry._mappings

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Store original operations
        saved_ops = {}

        # Replace JAX operations with Pallas equivalents
        for jax_op, pallas_op in op_mapping.items():
            if hasattr(jax_op, '__module__') and hasattr(jax_op, '__name__'):
                # Get the module and attribute name
                if 'jax.numpy' in str(jax_op.__module__):
                    attr_name = jax_op.__name__
                    if hasattr(jnp, attr_name):
                        saved_ops[(jnp, attr_name)] = getattr(jnp, attr_name)
                        setattr(jnp, attr_name, pallas_op)
                elif 'jax._src.nn' in str(jax_op.__module__) or 'jax.nn' in str(jax_op.__module__):
                    attr_name = jax_op.__name__
                    if hasattr(jax.nn, attr_name):
                        saved_ops[(jax.nn, attr_name)] = getattr(jax.nn, attr_name)
                        setattr(jax.nn, attr_name, pallas_op)

        try:
            # Execute function with Pallas operations
            result = func(*args, **kwargs)
        finally:
            # Restore original operations
            for (module, attr_name), original_op in saved_ops.items():
                setattr(module, attr_name, original_op)

        return result

    return wrapper


def to_pallas(func: Callable) -> Callable:
    """Decorator to transform a JAX function to use Pallas kernels.

    Usage:
        @to_pallas
        def my_transformer(x, params, config):
            # Uses JAX operations that get mapped to Pallas
            return transformer_block(x, params, config)

    Args:
        func: JAX function to transform

    Returns:
        Function that uses Pallas kernels
    """
    return tree_map_operations(func)


def apply_pallas(func: Callable, *args, **kwargs) -> Any:
    """Apply a JAX function with Pallas kernel transformations.

    Usage:
        result = apply_pallas(transformer_block, x, params, config)

    Args:
        func: JAX function to call
        *args: Positional arguments for func
        **kwargs: Keyword arguments for func

    Returns:
        Result of func(*args, **kwargs) with Pallas kernels
    """
    transformed = to_pallas(func)
    return transformed(*args, **kwargs)


def create_pallas_version(func: Callable) -> Callable:
    """Create a Pallas version of a JAX function.

    This returns a new function that always uses Pallas kernels.

    Usage:
        transformer_block_pallas = create_pallas_version(transformer_block)
        result = transformer_block_pallas(x, params, config)

    Args:
        func: JAX function to transform

    Returns:
        New function that uses Pallas kernels
    """
    return to_pallas(func)
