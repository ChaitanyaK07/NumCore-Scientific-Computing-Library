# NumCore

NumCore is a lightweight scientific computing library written in modern C++ with Python bindings powered by `pybind11`.

It implements core numerical methods from scratch, including custom vector/matrix classes, linear algebra routines, ODE solvers, and optimization algorithms. The package is designed as both a usable numerical library and a learning-focused implementation of fundamental scientific computing algorithms.

> Install package: `numcore-scicomp`  
> Python import name: `numcore`

---

## Features

### Matrix and Vector Core

- Custom `Vector` and `Matrix` classes
- Operator overloading for natural arithmetic syntax
- Shape and bounds validation
- NumPy interoperability through `.numpy()`
- Python bindings for direct use from Python

### Linear Algebra

- Matrix-vector multiplication
- Linear system solving
- Determinant computation
- Matrix inverse
- Least-squares solver
- LU decomposition with partial pivoting

### ODE Solvers

- Forward Euler method
- Classical fourth-order Runge-Kutta method, RK4
- Adaptive Dormand-Prince RK45 solver

### Optimization

- Gradient Descent
- Adam optimizer
- Newton's method with numerical Hessian
- BFGS with Armijo line search

---

## Installation

### Install from PyPI

```bash
pip install numcore-scicomp
```

Then use it in Python as:

```python
import numcore as nc
```

### Install from source

```bash
git clone https://github.com/ChaitanyaK07/NumCore---Scientific-Computing-Library.git
cd NumCore---Scientific-Computing-Library

python -m pip install .
```

For development:

```bash
python -m pip install -e .
```

---

## Quick Start

```python
import numcore as nc

v = nc.Vector([3.0, 4.0])
print(v.norm())   # 5.0
```

---

## Python Examples

### Vector operations

```python
import numcore as nc

x = nc.Vector([1.0, 2.0, 3.0])

print(x.norm())      # 3.741657...
print(x.dot(x))      # 14.0
print((2.0 * x)[1])  # 4.0
```

### Matrix operations

```python
import numcore as nc

A = nc.Matrix(2, 2, [1.0, 2.0,
                     3.0, 4.0])

B = A @ A

print(B[(0, 0)])      # 7.0
print(B[(1, 1)])      # 22.0
print(A.T().numpy())  # convert to NumPy array
```

### Solve a linear system

```python
import numcore as nc

A = nc.Matrix(3, 3, [
    2.0,  1.0, -1.0,
   -3.0, -1.0,  2.0,
   -2.0,  1.0,  2.0,
])

b = nc.Vector([8.0, -11.0, -3.0])

x = nc.linalg.solve(A, b)

print(x.numpy())  # approximately [2.0, 3.0, -1.0]
```

### ODE solving with RK4

```python
import numcore as nc

def harmonic_oscillator(t, y):
    return nc.Vector([y[1], -y[0]])

sol = nc.ode.rk4(
    harmonic_oscillator,
    0.0,
    6.28,
    nc.Vector([1.0, 0.0]),
    0.01,
)

t = sol.t_array()
y = sol.y_array()

print(t.shape)
print(y.shape)
```

### Optimization with BFGS

```python
import numcore as nc

def rosenbrock(v):
    x = v[0]
    y = v[1]
    return (1.0 - x) ** 2 + 100.0 * (y - x * x) ** 2

def rosenbrock_grad(v):
    x = v[0]
    y = v[1]
    return nc.Vector([
        -2.0 * (1.0 - x) - 400.0 * x * (y - x * x),
        200.0 * (y - x * x),
    ])

res = nc.optim.bfgs(
    rosenbrock,
    rosenbrock_grad,
    nc.Vector([0.0, 0.0]),
)

print(res.x[0], res.x[1])      # approximately 1.0, 1.0
print(res.converged)
print(res.iterations)
```

---

## C++ Usage

NumCore can also be used directly as a C++ header-based library.

Example:

```cpp
#include <iostream>
#include "NumCore/vector.hpp"
#include "NumCore/matrix.hpp"
#include "NumCore/linalg.hpp"

int main() {
    Vector v({3.0, 4.0});
    std::cout << v.norm() << std::endl;

    Matrix A(2, 2, {2.0, 1.0,
                    5.0, 7.0});

    Vector b({11.0, 13.0});
    Vector x = linalg::solve(A, b);

    std::cout << x[0] << " " << x[1] << std::endl;
}
```

Compile with:

```bash
g++ main.cpp -I include -std=c++17 -o main
```

---

## Building from Source

### Python package build

```bash
python -m pip install .
```

### Manual CMake build

If CMake cannot find `pybind11`, pass the pybind11 CMake directory explicitly.

#### Windows PowerShell

```powershell
$pybind11_DIR = python -m pybind11 --cmakedir

cmake -S . -B build -Dpybind11_DIR="$pybind11_DIR"
cmake --build build --config Release
```

Run C++ tests:

```powershell
.\build\Release\tests_cpp.exe
```

#### Linux/macOS

```bash
pybind11_DIR=$(python -m pybind11 --cmakedir)

cmake -S . -B build -Dpybind11_DIR="$pybind11_DIR" -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Run C++ tests:

```bash
./build/tests_cpp
```

---

## Running Tests

### Python tests

```bash
python -m pytest tests/tests_python.py -q
```

### C++ tests

Windows:

```powershell
.\build\Release\tests_cpp.exe
```

Linux/macOS:

```bash
./build/tests_cpp
```

---

## Project Structure

```text
NumCore/
├── include/
│   └── NumCore/
│       ├── NumCore.hpp
│       ├── vector.hpp
│       ├── matrix.hpp
│       ├── linalg.hpp
│       ├── ode.hpp
│       └── optim.hpp
├── python/
│   ├── bindings.cpp
│   └── numcore/
│       └── __init__.py
├── tests/
│   ├── tests_cpp.cpp
│   └── tests_python.py
├── CMakeLists.txt
├── pyproject.toml
└── README.md
```

---

## Algorithms Implemented

| Category | Algorithm | Notes |
|---|---|---|
| Matrix/Vector | Core arithmetic | Operator overloading and validation |
| Linear Algebra | Matrix-vector multiplication | C++ implementation exposed to Python |
| Linear Algebra | Linear solve | LU-based solve with pivoting |
| Linear Algebra | Determinant | Uses elimination/LU-style logic |
| Linear Algebra | Matrix inverse | Solves against identity basis |
| Linear Algebra | Least squares | Normal-equation based implementation |
| ODE | Euler | Fixed-step first-order method |
| ODE | RK4 | Fixed-step fourth-order Runge-Kutta |
| ODE | RK45 | Adaptive Dormand-Prince method |
| Optimization | Gradient Descent | User-supplied objective and gradient |
| Optimization | Adam | Adaptive first/second moment optimizer |
| Optimization | Newton | Numerical Hessian with regularization |
| Optimization | BFGS | Quasi-Newton inverse-Hessian update |

---

## Notes

NumCore is implemented from scratch for clarity and educational value. It is not intended to replace mature high-performance numerical libraries such as NumPy, SciPy, Eigen, BLAS, or LAPACK for large-scale production workloads.

For small examples, learning, experimentation, and understanding how numerical algorithms are built internally, NumCore provides a compact C++/Python implementation.

---

## Roadmap

- [ ] More decomposition methods, including QR and Cholesky
- [ ] Sparse matrix support
- [ ] Conjugate Gradient and iterative solvers
- [ ] More robust benchmarking suite
- [ ] Prebuilt wheels for more Python versions and operating systems
- [ ] Expanded documentation and examples

---

## License

Add a license file before public production use. Recommended options for open-source numerical libraries include MIT, BSD-3-Clause, or Apache-2.0.
