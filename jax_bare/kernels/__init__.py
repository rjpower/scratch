"""Pallas kernels for JAX transformer model.

This package provides Pallas implementations of common JAX primitives used in
transformer models, allowing for custom kernel execution on CPU.

Available kernels:
- elementwise: Binary and unary operations (add, mul, exp, tanh, etc.)
- matmul: Matrix multiplication and dot_general operations
- reduce: Reduction operations (sum, max)
- reshape: Tensor reshape operations
- transpose: Tensor transpose operations
- broadcast: Broadcasting operations
"""

# Import all kernel functions
from . import elementwise
from . import matmul
from . import reduce
from . import reshape
from . import transpose
from . import broadcast

__all__ = [
    'elementwise',
    'matmul',
    'reduce',
    'reshape',
    'transpose',
    'broadcast',
]
