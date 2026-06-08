"""
Benchmark: numcore optimizers vs SciPy minimize
Run from project root: python benchmarks/bench_optim.py
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

import numpy as np
import scipy.optimize as spopt
try:
    import _numcore as nc
except ImportError:
    sys.exit("Build first: cd build && make")

def bench(label, fn, repeats=50):
    fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    ms = min(times) * 1000
    print(f"  {label:<50} {ms:8.4f} ms")
    return ms

# ── Rosenbrock ────────────────────────────────────────────────────────────────
def rosen_nc(v):   return (1-v[0])**2 + 100*(v[1]-v[0]**2)**2
def rosen_g_nc(v): return nc.Vector([-2*(1-v[0])-400*v[0]*(v[1]-v[0]**2), 200*(v[1]-v[0]**2)])
def rosen_np(x):   return (1-x[0])**2 + 100*(x[1]-x[0]**2)**2
def rosen_g_np(x): return np.array([-2*(1-x[0])-400*x[0]*(x[1]-x[0]**2), 200*(x[1]-x[0]**2)])

x0_nc = nc.Vector([0.0, 0.0])
x0_np = np.array([0.0, 0.0])

print("\n" + "="*65)
print("  Rosenbrock minimisation  (minimum at x=(1,1), f=0)")
print("="*65)

t_bfgs   = bench("numcore bfgs             ", lambda: nc.optim.bfgs(rosen_nc, rosen_g_nc, x0_nc))
t_newton = bench("numcore newton           ", lambda: nc.optim.newton(rosen_nc, x0_nc))
t_adam   = bench("numcore adam             ", lambda: nc.optim.adam(rosen_nc, rosen_g_nc, x0_nc))
t_sp     = bench("scipy   BFGS             ", lambda: spopt.minimize(rosen_np, x0_np, jac=rosen_g_np, method='BFGS'))
t_sp_nm  = bench("scipy   Nelder-Mead      ", lambda: spopt.minimize(rosen_np, x0_np, method='Nelder-Mead'))

res = nc.optim.bfgs(rosen_nc, rosen_g_nc, x0_nc)
print(f"\n  numcore BFGS result:  x=({res.x[0]:.6f}, {res.x[1]:.6f})  f={res.f_val:.2e}  iters={res.iterations}")

# ── Simple quadratic ──────────────────────────────────────────────────────────
def quad_nc(v):   return v[0]**2 + 4*v[1]**2
def quad_g_nc(v): return nc.Vector([2*v[0], 8*v[1]])
def quad_np(x):   return x[0]**2 + 4*x[1]**2
def quad_g_np(x): return np.array([2*x[0], 8*x[1]])

x0_nc2 = nc.Vector([5.0, 3.0])
x0_np2 = np.array([5.0, 3.0])

print("\n" + "="*65)
print("  Quadratic  f = x² + 4y²  (minimum at origin)")
print("="*65)

bench("numcore gradient_descent ", lambda: nc.optim.gradient_descent(quad_nc, quad_g_nc, x0_nc2))
bench("numcore adam             ", lambda: nc.optim.adam(quad_nc, quad_g_nc, x0_nc2))
bench("numcore newton           ", lambda: nc.optim.newton(quad_nc, x0_nc2))
bench("numcore bfgs             ", lambda: nc.optim.bfgs(quad_nc, quad_g_nc, x0_nc2))
bench("scipy   BFGS             ", lambda: spopt.minimize(quad_np, x0_np2, jac=quad_g_np, method='BFGS'))

print("\nNote: numcore BFGS avoids scipy overhead — competitive at small N.\n")
