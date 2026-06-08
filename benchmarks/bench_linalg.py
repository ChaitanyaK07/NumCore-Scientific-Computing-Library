"""
Benchmark: numcore linear algebra vs NumPy
Run from project root: python benchmarks/bench_linalg.py
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

import numpy as np
try:
    import _numcore as nc
except ImportError:
    sys.exit("Build first: cd build && make")

def bench(label, fn, repeats=20):
    fn()  # warm-up
    t = min(time.perf_counter() - (lambda: (time.perf_counter(), fn()))[0]
            for _ in range(repeats))
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    ms = min(times) * 1000
    print(f"  {label:<50} {ms:8.4f} ms")
    return ms

print("\n" + "="*65)
print("  Matrix-vector multiply  A @ x")
print("="*65)
sizes = [10, 50, 100, 300]
for n in sizes:
    A_np = np.random.rand(n, n)
    x_np = np.random.rand(n)
    A_nc = nc.Matrix(n, n, A_np.ravel().tolist())
    x_nc = nc.Vector(x_np.tolist())

    t_nc = bench(f"numcore matvec  n={n}", lambda: nc.matvec(A_nc, x_nc))
    t_np = bench(f"numpy   matvec  n={n}", lambda: A_np @ x_np)
    print(f"  {'ratio':>50} {t_nc/t_np:8.2f}x  (numpy faster = >1)\n")

print("="*65)
print("  Linear solve  A x = b")
print("="*65)
for n in [10, 32, 64, 128]:
    rng = np.random.default_rng(0)
    A_np = rng.random((n,n)) + n * np.eye(n)
    b_np = rng.random(n)
    A_nc = nc.Matrix(n, n, A_np.ravel().tolist())
    b_nc = nc.Vector(b_np.tolist())

    t_nc = bench(f"numcore solve   n={n}", lambda: nc.linalg.solve(A_nc, b_nc))
    t_np = bench(f"numpy   solve   n={n}", lambda: np.linalg.solve(A_np, b_np))
    print(f"  {'ratio':>50} {t_nc/t_np:8.2f}x\n")

print("="*65)
print("  Matrix multiply  A @ A")
print("="*65)
for n in [32, 64, 128]:
    A_np = np.random.rand(n, n)
    A_nc = nc.Matrix(n, n, A_np.ravel().tolist())

    t_nc = bench(f"numcore matmul  n={n}", lambda: A_nc @ A_nc)
    t_np = bench(f"numpy   matmul  n={n}", lambda: A_np @ A_np)
    print(f"  {'ratio':>50} {t_nc/t_np:8.2f}x\n")

print("Note: NumPy uses OpenBLAS/MKL — faster at large N is expected.")
print("Goal here is to show understanding of benchmarking and scaling.\n")
