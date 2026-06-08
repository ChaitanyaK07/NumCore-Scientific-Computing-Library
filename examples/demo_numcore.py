"""
NumCore — Polished Demo
=======================
Shows the four core capabilities:
  1. Vector and matrix operations
  2. Solve linear system  A x = b
  3. Solve an ODE with RK4
  4. Minimize a function with gradient descent / Newton / BFGS

Run from the build directory:
    PYTHONPATH=. python ../examples/demo_numcore.py
"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

try:
    import _numcore as nc
except ImportError:
    sys.exit("Build first: cd build && make")

SEP = "=" * 55

# ── 1. Vector and matrix operations ──────────────────────────
print(SEP)
print("1. Vector and Matrix Operations")
print(SEP)

x = nc.Vector([1.0, 2.0, 3.0])
y = nc.Vector([4.0, 5.0, 6.0])
print(f"  x          = {x}")
print(f"  y          = {y}")
print(f"  x + y      = {x + y}")
print(f"  x . y      = {x.dot(y)}")
print(f"  ||x||      = {x.norm():.6f}")
print(f"  x × y      = {x.cross(y)}")

A = nc.Matrix(2, 2, [3.0, 1.0, 1.0, 2.0])
B = nc.Matrix(2, 2, [1.0, 0.0, 0.0, 1.0])   # identity
print(f"\n  A =\n{A}")
print(f"  A @ I =\n{A @ B}")
print(f"  A^T =\n{A.T()}")

# ── 2. Linear system  A x = b ────────────────────────────────
print(SEP)
print("2. Solve  A x = b")
print(SEP)

A = nc.Matrix(3, 3, [
    2.0,  1.0, -1.0,
   -3.0, -1.0,  2.0,
   -2.0,  1.0,  2.0,
])
b = nc.Vector([8.0, -11.0, -3.0])
x = nc.linalg.solve(A, b)

print(f"  A x = b   →   x = ({x[0]:.4f}, {x[1]:.4f}, {x[2]:.4f})")
print(f"  Expected:      x = (2.0000, 3.0000, -1.0000)")
print(f"  det(A)     = {nc.linalg.det(A):.4f}")

# Verify residual
Ax = nc.matvec(A, x)
residual = math.sqrt(sum((Ax[i] - b[i])**2 for i in range(3)))
print(f"  ||Ax - b|| = {residual:.2e}")

# ── 3. ODE — RK4 ─────────────────────────────────────────────
print(SEP)
print("3. ODE Solver — Harmonic Oscillator (RK4)")
print(SEP)

# d²x/dt² + x = 0   →   state = [x, v]
# exact: x(t) = cos(t),  v(t) = -sin(t)
def f(t, y):
    return nc.Vector([y[1], -y[0]])

sol = nc.ode.rk4(f, 0.0, 2*math.pi, nc.Vector([1.0, 0.0]), dt=0.01)
x_final = sol.y[-1][0]
v_final = sol.y[-1][1]
energy  = x_final**2 + v_final**2  # should be 1

print(f"  x(0)=1, v(0)=0  →  integrate one full period (t=2π)")
print(f"  x(2π) = {x_final:+.6f}   (exact: +1.000000)")
print(f"  v(2π) = {v_final:+.6f}   (exact:  0.000000)")
print(f"  Energy conservation  x²+v² = {energy:.8f}  (exact: 1.0)")
print(f"  Steps: {len(sol.t)-1}")

# Also show adaptive RK45
sol45 = nc.ode.rk45(f, 0.0, 2*math.pi, nc.Vector([1.0, 0.0]))
e45 = sol45.y[-1][0]**2 + sol45.y[-1][1]**2
print(f"\n  RK45 adaptive: {len(sol45.t)-1} steps,  energy = {e45:.10f}")

# ── 4. Optimization ───────────────────────────────────────────
print(SEP)
print("4. Optimization — Rosenbrock  f(x,y)=(1-x)²+100(y-x²)²")
print(SEP)
print("   Global minimum at (1, 1),  f = 0")
print()

def loss(v):
    return (v[0] - 3)**2 + (v[1] + 2)**2

def loss_g(v):
    return nc.Vector([2*(v[0]-3), 2*(v[1]+2)])

def rosen(v):
    return (1-v[0])**2 + 100*(v[1]-v[0]**2)**2

def rosen_g(v):
    return nc.Vector([-2*(1-v[0])-400*v[0]*(v[1]-v[0]**2),
                       200*(v[1]-v[0]**2)])

x0 = nc.Vector([0.0, 0.0])

for name, res in [
    ("Gradient Descent", nc.optim.gradient_descent(rosen, rosen_g, x0)),
    ("Adam",             nc.optim.adam(rosen, rosen_g, x0)),
    ("Newton",           nc.optim.newton(rosen, x0)),
    ("BFGS",             nc.optim.bfgs(rosen, rosen_g, x0)),
]:
    status = "✓" if res.converged else "~"
    print(f"  [{status}] {name:<18} "
          f"x=({res.x[0]:+.5f}, {res.x[1]:+.5f})  "
          f"f={res.f_val:.2e}  iter={res.iterations}")

print(SEP)
print("All demos complete.")
print(SEP)
