# NumCore

NumCore is a C++ scientific computing library with Python bindings using pybind11.  
It implements core numerical methods including vector/matrix operations, linear algebra solvers, ODE solvers, and optimization algorithms — built from scratch with no BLAS/LAPACK dependency.

---

## Features

- **Linear Algebra** — LU decomposition (partial pivoting), `solve`, `det`, `inv`, `lstsq`
- **ODE Solvers** — Euler, RK4, Dormand-Prince RK45 (adaptive step)
- **Optimization** — Gradient Descent, Adam, Newton (numerical Hessian), BFGS (Armijo line search)
- **Matrix / Vector** — Custom C++ classes with operator overloading, shape validation, exception handling, and numpy interop

---

## Installation

```bash
# Prerequisites
pip install pybind11 scikit-build-core numpy

# Build and install
pip install .

# Or build manually (development)
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -Dpybind11_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())")
make -j$(nproc)
```

---

## How to Run Tests

```bash
# C++ unit tests (no pytest needed)
cd build && ./test_cpp

# Python tests
cd build && python3 -m pytest ../tests/test_python.py -v
```

---

## Python Usage

```python
from numcore import Vector, Matrix
from numcore import linalg, ode, optim

# Vector operations
x = Vector([1.0, 2.0, 3.0])
print(x.norm())          # 3.7416...
print(x.dot(x))          # 14.0

# Matrix operations
A = Matrix(2, 2, [3.0, 1.0, 1.0, 2.0])
print(A @ A)             # matrix multiply
print(A.T().numpy())     # as numpy array

# Solve  A x = b
A = Matrix(3, 3, [2,1,-1, -3,-1,2, -2,1,2])
b = Vector([8.0, -11.0, -3.0])
x = linalg.solve(A, b)   # x = (2, 3, -1)

# ODE — harmonic oscillator
def f(t, y):
    return Vector([y[1], -y[0]])

sol = ode.rk4(f, 0.0, 6.28, Vector([1.0, 0.0]), dt=0.01)
t_arr = sol.t_array()    # numpy array of times
y_arr = sol.y_array()    # (steps, 2) numpy array

# Adaptive step
sol = ode.rk45(f, 0.0, 6.28, Vector([1.0, 0.0]))

# Optimization
def loss(v):
    return (1 - v[0])**2 + 100*(v[1] - v[0]**2)**2

def grad(v):
    return Vector([-2*(1-v[0]) - 400*v[0]*(v[1]-v[0]**2),
                    200*(v[1]-v[0]**2)])

res = optim.bfgs(loss, grad, Vector([0.0, 0.0]))
print(res.x[0], res.x[1])    # ≈ 1.0, 1.0
print(res.converged, res.iterations)
```

---

## Algorithms Implemented

| Category | Algorithm | Notes |
|---|---|---|
| Linear Algebra | LU decomposition | Partial pivoting, correct det sign |
| Linear Algebra | Gaussian elimination | Gauss-Jordan variant |
| Linear Algebra | Least-squares (`lstsq`) | Via normal equations |
| ODE | Euler | First-order, fixed step |
| ODE | RK4 | Classic 4th-order Runge-Kutta |
| ODE | RK45 | Dormand-Prince, adaptive PI step control |
| Optimization | Gradient Descent | Fixed learning rate |
| Optimization | Adam | Adaptive moments, bias correction |
| Optimization | Newton | Numerical Hessian + Tikhonov regularization |
| Optimization | BFGS | Rank-2 inverse Hessian update, Armijo line search |

---

## Benchmarks

```bash
python benchmarks/bench_linalg.py
python benchmarks/bench_ode.py
python benchmarks/bench_optim.py
```

| Routine | numcore | NumPy/SciPy | Notes |
|---|---|---|---|
| `solve` 32×32 | ~0.02 ms | ~0.02 ms | Parity at small N |
| `bfgs` Rosenbrock | ~0.4 ms | ~2.8 ms | **7× faster** |
| `rk4` 10k steps | ~90 ms | — | Python callback cost |

NumPy uses OpenBLAS — faster at large dense matmul is expected and honest.  
For small problems and custom kernels, C++ wins.

---

## Demo

```bash
cd build && PYTHONPATH=. python ../examples/demo_numcore.py
```

---

## Future Work

- [ ] OpenMP parallelism for matmul
- [ ] Sparse matrix (CSR format) + Conjugate Gradient
- [ ] Eigen backend for dense BLAS-level performance
- [ ] CUDA support for GPU acceleration
