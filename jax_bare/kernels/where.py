"""Pallas kernel for conditional selection (where operation)."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

# Store reference to original JAX where to avoid recursion
_jax_where = jnp.where


def where_kernel(cond_ref, x_ref, y_ref, out_ref):
    """Kernel for conditional selection.

    Selects elements from x where cond is True, otherwise from y.

    Args:
        cond_ref: Boolean condition array
        x_ref: First input array (selected where cond is True)
        y_ref: Second input array (selected where cond is False)
        out_ref: Output array
    """
    cond = cond_ref[...]
    x = x_ref[...]
    y = y_ref[...]
    out_ref[...] = _jax_where(cond, x, y)


def where(cond: jax.Array, x: jax.Array, y: jax.Array) -> jax.Array:
    """Conditional selection using Pallas kernel.

    Args:
        cond: Boolean condition array
        x: Values to select where cond is True
        y: Values to select where cond is False

    Returns:
        Array with elements from x where cond is True, else from y
    """
    return pl.pallas_call(
        where_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        interpret=True
    )(cond, x, y)
