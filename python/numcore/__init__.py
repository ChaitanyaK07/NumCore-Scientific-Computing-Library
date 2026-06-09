"""
numcore — C++ scientific computing library with Python bindings.

    from numcore import Vector, Matrix
    from numcore import linalg, ode, optim
"""
# Module is compiled as _NumCore (matches PYBIND11_MODULE(_NumCore, m))
from ._NumCore import (
    Matrix,
    Vector,
    matvec,
    linalg,
    ode,
    optim,
)

__version__ = "0.1.1"
__all__ = ["Matrix", "Vector", "matvec", "linalg", "ode", "optim"]
