"""
Benchmark: numcore ODE solvers vs SciPy solve_ivp
Run from project root: python benchmarks/bench_ode.py
"""
import sys, os, time, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

import numpy as np
import scipy.integrate as spint
try:
    import _numcore as nc
except ImportError:
    sys.exit("Build first: cd build && make")

def bench(label, fn, repeats=10):
    fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    ms = min(times) * 1000
    print(f"  {label:<50} {ms:8.3f} ms")
    return ms

print("\n" + "="*65)
print("  Exponential decay  y'=-y,  y(0)=1,  t in [0,10]")
print("="*65)

for dt, label in [(0.01, "dt=0.01"), (0.001, "dt=0.001")]:
    def run_nc():
        nc.ode.rk4(lambda t,y: nc.Vector([-y[0]]), 0.0, 10.0, nc.Vector([1.0]), dt)
    def run_sp():
        spint.solve_ivp(lambda t,y: [-y[0]], [0,10], [1.0], method='RK45', rtol=1e-6, atol=1e-8)

    t_nc = bench(f"numcore rk4     {label}", run_nc)
    t_sp = bench(f"scipy   RK45    (adaptive)", run_sp)
    exact = math.exp(-10.0)
    sol = nc.ode.rk4(lambda t,y: nc.Vector([-y[0]]), 0.0, 10.0, nc.Vector([1.0]), dt)
    err = abs(sol.y[-1][0] - exact)
    print(f"  numcore error vs exact: {err:.2e}   steps: {len(sol.t)-1}\n")

print("="*65)
print("  Harmonic oscillator  [x,v]' = [v, -x],  t in [0, 4π]")
print("="*65)

def ho_nc(t, y): return nc.Vector([y[1], -y[0]])
def ho_sp(t, y): return [y[1], -y[0]]

t_nc = bench("numcore rk4     dt=0.01 ", lambda: nc.ode.rk4(ho_nc, 0.0, 4*math.pi, nc.Vector([1.0,0.0]), 0.01))
t_nc45 = bench("numcore rk45    adaptive", lambda: nc.ode.rk45(ho_nc, 0.0, 4*math.pi, nc.Vector([1.0,0.0])))
t_sp = bench("scipy   RK45    adaptive", lambda: spint.solve_ivp(ho_sp, [0, 4*math.pi], [1.0,0.0], method='RK45'))

sol = nc.ode.rk45(ho_nc, 0.0, 4*math.pi, nc.Vector([1.0,0.0]))
energy = sol.y[-1][0]**2 + sol.y[-1][1]**2
print(f"\n  RK45 energy conservation (should be 1.0): {energy:.8f}")
print(f"  RK45 adaptive steps taken: {len(sol.t)-1}\n")

print("Note: numcore pays Python callback overhead per step.")
print("Adaptive solvers (rk45) minimise total steps for same accuracy.\n")
