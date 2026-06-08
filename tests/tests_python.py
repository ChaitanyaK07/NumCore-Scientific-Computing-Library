import math
import sys
from pathlib import Path

import numpy as np
import pytest


# ─────────────────────────────────────────────────────────────
# Import compiled pybind11 extension
# ─────────────────────────────────────────────────────────────
# Expected project layout:
# NumCore/
# ├── build/
# │   └── Release/_NumCore*.pyd
# └── tests/
#     └── tests_python.py
#
# This makes pytest work even when the .pyd is inside build/Release or build/Debug.

ROOT = Path(__file__).resolve().parents[1]

candidate_paths = [
    ROOT,
    ROOT / "build",
    ROOT / "build" / "Release",
    ROOT / "build" / "Debug",
]

for path in candidate_paths:
    if path.exists():
        sys.path.insert(0, str(path))

# Fallback: search the whole build folder for _NumCore*.pyd
build_dir = ROOT / "build"
if build_dir.exists():
    for pyd in build_dir.rglob("_NumCore*.pyd"):
        sys.path.insert(0, str(pyd.parent))
        break

nc = None

for _mod_name in ("numcore", "_NumCore", "_numcore"):
    try:
        import importlib
        nc = importlib.import_module(_mod_name)
        break
    except ImportError:
        continue


if nc is None:
    raise ImportError(
        "Could not import NumCore extension. Run one of:\n"
        "  pip install .                      (recommended)\n"
        "  cd build && cmake .. && cmake --build . --config Release"
    )

def near(a, b, tol=1e-10):
    return abs(a - b) < tol


# ══════════════════════════════════════════════════════════════
# Matrix tests
# ══════════════════════════════════════════════════════════════
class TestMatrix:
    def test_zeros(self):
        m = nc.Matrix.zeros(3, 4)
        assert m.rows() == 3
        assert m.cols() == 4
        assert m[(0, 0)] == 0.0
        assert m[(2, 3)] == 0.0

    def test_eye(self):
        I = nc.Matrix.eye(4)
        assert I.rows() == 4
        assert I.cols() == 4
        assert I[(0, 0)] == 1.0
        assert I[(1, 1)] == 1.0
        assert I[(0, 1)] == 0.0

    def test_setitem(self):
        m = nc.Matrix(2, 2)
        m[(1, 1)] = 7.0
        assert m[(1, 1)] == 7.0

    def test_add(self):
        a = nc.Matrix.ones(2, 2)
        b = a + a
        assert b[(0, 0)] == 2.0
        assert b[(1, 1)] == 2.0

    def test_sub(self):
        a = nc.Matrix.ones(2, 2)
        z = a - a
        assert z[(0, 0)] == 0.0
        assert z[(1, 1)] == 0.0

    def test_matmul(self):
        a = nc.Matrix(2, 2, [1.0, 2.0, 3.0, 4.0])
        b = a @ a
        assert near(b[(0, 0)], 7.0)
        assert near(b[(0, 1)], 10.0)
        assert near(b[(1, 0)], 15.0)
        assert near(b[(1, 1)], 22.0)

    def test_transpose(self):
        a = nc.Matrix(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        t = a.T()
        assert t.rows() == 3
        assert t.cols() == 2
        assert t[(0, 1)] == a[(1, 0)]
        assert t[(2, 1)] == a[(1, 2)]

    def test_scalar_mul(self):
        a = nc.Matrix(2, 2, [1.0, 2.0, 3.0, 4.0])
        b = a * 2.0
        c = 2.0 * a
        assert b[(0, 0)] == 2.0
        assert b[(1, 1)] == 8.0
        assert c[(0, 1)] == 4.0

    def test_numpy_roundtrip(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        m = nc.Matrix(arr)
        back = m.numpy()
        np.testing.assert_allclose(back, arr)

    def test_bounds_error(self):
        m = nc.Matrix(2, 2)
        with pytest.raises(Exception):
            _ = m[(10, 10)]


# ══════════════════════════════════════════════════════════════
# Vector tests
# ══════════════════════════════════════════════════════════════
class TestVector:
    def test_norm(self):
        v = nc.Vector([3.0, 4.0])
        assert near(v.norm(), 5.0)

    def test_dot(self):
        a = nc.Vector([1.0, 2.0, 3.0])
        assert near(a.dot(a), 14.0)

    def test_cross(self):
        i = nc.Vector([1.0, 0.0, 0.0])
        j = nc.Vector([0.0, 1.0, 0.0])
        k = i.cross(j)
        assert near(k[0], 0.0)
        assert near(k[1], 0.0)
        assert near(k[2], 1.0)

    def test_normalized(self):
        v = nc.Vector([3.0, 4.0])

        # bindings.cpp exposes Vector::normalize as the Python method "normalized"
        n = v.normalized()

        assert near(n.norm(), 1.0)

    def test_add_sub_scalar(self):
        v = nc.Vector([1.0, 2.0, 3.0])
        assert (v + v)[0] == 2.0
        assert (v - v)[1] == 0.0
        assert (v * 2.0)[2] == 6.0
        assert (2.0 * v)[1] == 4.0

    def test_numpy_roundtrip(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        v = nc.Vector(arr)
        np.testing.assert_allclose(v.numpy(), arr)


# ══════════════════════════════════════════════════════════════
# linalg tests
# ══════════════════════════════════════════════════════════════
class TestLinalg:
    def test_solve(self):
        A = nc.Matrix(2, 2, [2.0, 1.0, 5.0, 7.0])
        b = nc.Vector([11.0, 13.0])

        x = nc.linalg.solve(A, b)
        Ax = nc.matvec(A, x)

        assert near(Ax[0], 11.0, 1e-8)
        assert near(Ax[1], 13.0, 1e-8)

    def test_det(self):
        A = nc.Matrix(2, 2, [2.0, 1.0, 5.0, 7.0])
        assert near(nc.linalg.det(A), 9.0)

    def test_inv_identity(self):
        I = nc.Matrix.eye(3)
        invI = nc.linalg.inv(I)
        np.testing.assert_allclose(invI.numpy(), np.eye(3), atol=1e-10)

    def test_inv_correct(self):
        A = nc.Matrix(2, 2, [2.0, 1.0, 5.0, 7.0])
        invA = nc.linalg.inv(A)
        prod = (A @ invA).numpy()
        np.testing.assert_allclose(prod, np.eye(2), atol=1e-8)

    def test_singular_raises(self):
        A = nc.Matrix(2, 2, [1.0, 2.0, 2.0, 4.0])
        b = nc.Vector([1.0, 2.0])

        with pytest.raises(Exception):
            nc.linalg.solve(A, b)

    def test_lstsq(self):
        # Fit y = a*x + b using columns [x, 1].
        A = nc.Matrix(4, 2, [1.0, 1.0,
                             2.0, 1.0,
                             3.0, 1.0,
                             4.0, 1.0])
        y = nc.Vector([3.0, 5.0, 7.0, 9.0])

        coeff = nc.linalg.lstsq(A, y)

        assert near(coeff[0], 2.0, 1e-6)
        assert near(coeff[1], 1.0, 1e-6)


# ══════════════════════════════════════════════════════════════
# ODE tests
# ══════════════════════════════════════════════════════════════
class TestODE:
    @staticmethod
    def decay(t, y):
        return nc.Vector([-y[0]])

    def test_rk4_exponential_decay(self):
        sol = nc.ode.rk4(self.decay, 0.0, 5.0, nc.Vector([1.0]), 0.01)
        assert near(sol.y[-1][0], math.exp(-5.0), 1e-4)

    def test_euler_exponential_decay(self):
        sol = nc.ode.euler(self.decay, 0.0, 1.0, nc.Vector([1.0]), 0.001)
        assert near(sol.y[-1][0], math.exp(-1.0), 1e-2)

    def test_rk45_exponential_decay(self):
        sol = nc.ode.rk45(self.decay, 0.0, 5.0, nc.Vector([1.0]))
        assert near(sol.y[-1][0], math.exp(-5.0), 1e-5)

    def test_harmonic_oscillator_energy(self):
        def ho(t, y):
            return nc.Vector([y[1], -y[0]])

        sol = nc.ode.rk4(ho, 0.0, 2.0 * math.pi, nc.Vector([1.0, 0.0]), 0.01)
        x = sol.y[-1][0]
        v = sol.y[-1][1]
        energy = x * x + v * v

        assert near(energy, 1.0, 1e-3)

    def test_y_array_shape(self):
        def ho(t, y):
            return nc.Vector([y[1], -y[0]])

        sol = nc.ode.rk4(ho, 0.0, 1.0, nc.Vector([1.0, 0.0]), 0.1)
        arr = sol.y_array()

        assert arr.ndim == 2
        assert arr.shape[1] == 2


# ══════════════════════════════════════════════════════════════
# Optim tests
# ══════════════════════════════════════════════════════════════
class TestOptim:
    def test_newton_quadratic(self):
        f = lambda v: v[0] ** 2 + v[1] ** 2

        res = nc.optim.newton(f, nc.Vector([3.0, 4.0]))

        assert near(res.x[0], 0.0, 1e-5)
        assert near(res.x[1], 0.0, 1e-5)
        assert res.converged

    def test_gradient_descent_quadratic(self):
        f = lambda v: v[0] ** 2 + 4.0 * v[1] ** 2
        g = lambda v: nc.Vector([2.0 * v[0], 8.0 * v[1]])

        # bindings.cpp exposes this as GDOptions, not GDOption
        opts = nc.optim.GDOptions()
        opts.lr = 0.1

        res = nc.optim.gradient_descent(f, g, nc.Vector([5.0, 3.0]), opts)

        assert near(res.x[0], 0.0, 1e-3)
        assert near(res.x[1], 0.0, 1e-3)

    def test_bfgs_rosenbrock(self):
        def f(v):
            x = v[0]
            y = v[1]
            return (1.0 - x) ** 2 + 100.0 * (y - x * x) ** 2

        def g(v):
            x = v[0]
            y = v[1]
            return nc.Vector([
                -2.0 * (1.0 - x) - 400.0 * x * (y - x * x),
                200.0 * (y - x * x),
            ])

        res = nc.optim.bfgs(f, g, nc.Vector([0.0, 0.0]))

        assert near(res.x[0], 1.0, 1e-2)
        assert near(res.x[1], 1.0, 1e-2)

    def test_adam_quadratic(self):
        f = lambda v: v[0] ** 2 + v[1] ** 2
        g = lambda v: nc.Vector([2.0 * v[0], 2.0 * v[1]])

        res = nc.optim.adam(f, g, nc.Vector([5.0, -3.0]))

        assert near(res.x[0], 0.0, 1e-2)
        assert near(res.x[1], 0.0, 1e-2)
