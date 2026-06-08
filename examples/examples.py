"""
numcpp scientific examples
--------------------------
1. Harmonic oscillator  (ODE)
2. Logistic growth      (ODE)
3. Least-squares regression  (linalg)
4. Gradient descent / Adam   (optim)

Run from the build directory:  python3 ../examples/examples.py
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    import _numcpp as nc
except ImportError:
    sys.exit("Build _numcpp first: cd build && make")


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ══════════════════════════════════════════════════════════════
# 1. Harmonic oscillator: d²x/dt² + ω²x = 0
#    State: [x, v]   ω=1
# ══════════════════════════════════════════════════════════════
section("1. Harmonic Oscillator  (RK4 & RK45)")

omega = 1.0
def harmonic(t, y):
    return nc.Vector([y[1], -omega**2 * y[0]])

sol_rk4 = nc.ode.rk4(harmonic, 0.0, 4*math.pi, nc.Vector([1.0, 0.0]), dt=0.05)
sol_rk45 = nc.ode.rk45(harmonic, 0.0, 4*math.pi, nc.Vector([1.0, 0.0]))

# Energy = 0.5*(v^2 + omega^2*x^2) should be conserved = 0.5
def energy(state):
    x, v = state[0], state[1]
    return 0.5*(v**2 + omega**2 * x**2)

e_initial = energy(sol_rk4.y[0])
e_final_rk4  = energy(sol_rk4.y[-1])
e_final_rk45 = energy(sol_rk45.y[-1])

print(f"  Period: 2π ≈ {2*math.pi:.4f}")
print(f"  RK4  steps: {len(sol_rk4.t)-1},  energy drift: {abs(e_final_rk4 - e_initial):.2e}")
print(f"  RK45 steps: {len(sol_rk45.t)-1}, energy drift: {abs(e_final_rk45 - e_initial):.2e}")
print(f"  x(4π) RK4  = {sol_rk4.y[-1][0]:.6f}  (exact ≈ 1.0)")
print(f"  x(4π) RK45 = {sol_rk45.y[-1][0]:.6f}  (exact ≈ 1.0)")

# ══════════════════════════════════════════════════════════════
# 2. Logistic growth: dy/dt = r*y*(1 - y/K)
#    y(0) = y0,  r=1,  K=10,  exact: K/(1+(K/y0-1)*e^{-rt})
# ══════════════════════════════════════════════════════════════
section("2. Logistic Growth  (RK45 adaptive)")

r, K, y0_val = 1.0, 10.0, 0.1
def logistic(t, y):
    return nc.Vector([r * y[0] * (1.0 - y[0]/K)])

sol = nc.ode.rk45(logistic, 0.0, 10.0, nc.Vector([y0_val]))

def exact_logistic(t):
    return K / (1 + (K/y0_val - 1) * math.exp(-r*t))

print(f"  r={r}, K={K}, y(0)={y0_val}")
print(f"  Steps taken: {len(sol.t)-1}")
for t_check in [1.0, 5.0, 10.0]:
    # find closest t in solution
    idx = min(range(len(sol.t)), key=lambda i: abs(sol.t[i]-t_check))
    y_nc = sol.y[idx][0]
    y_ex = exact_logistic(sol.t[idx])
    print(f"  t={sol.t[idx]:.3f}:  numcpp={y_nc:.6f},  exact={y_ex:.6f},  err={abs(y_nc-y_ex):.2e}")

# ══════════════════════════════════════════════════════════════
# 3. Least-squares polynomial regression y = a0 + a1*x + a2*x²
# ══════════════════════════════════════════════════════════════
section("3. Polynomial Regression  (lstsq)")

# Ground truth: y = 1 + 2x - 0.5x²  with noise
import random
random.seed(42)
xs = [i*0.5 for i in range(20)]
true_coeffs = [1.0, 2.0, -0.5]
ys = [true_coeffs[0] + true_coeffs[1]*x + true_coeffs[2]*x*x + random.gauss(0, 0.3)
      for x in xs]

# Build Vandermonde matrix [1, x, x²]
A_data = []
for x in xs:
    A_data += [1.0, x, x*x]
A = nc.Matrix(len(xs), 3, A_data)
b = nc.Vector(ys)

coeffs = nc.linalg.lstsq(A, b)
print(f"  True coefficients:     a0={true_coeffs[0]:.4f}, a1={true_coeffs[1]:.4f}, a2={true_coeffs[2]:.4f}")
print(f"  Fitted coefficients:   a0={coeffs[0]:.4f}, a1={coeffs[1]:.4f}, a2={coeffs[2]:.4f}")

# Compute R²
y_mean = sum(ys)/len(ys)
ss_tot = sum((y - y_mean)**2 for y in ys)
y_pred = [coeffs[0] + coeffs[1]*x + coeffs[2]*x*x for x in xs]
ss_res = sum((y - yp)**2 for y, yp in zip(ys, y_pred))
print(f"  R² = {1 - ss_res/ss_tot:.6f}")

# ══════════════════════════════════════════════════════════════
# 4. Optimization: Rosenbrock with multiple methods
# ══════════════════════════════════════════════════════════════
section("4. Rosenbrock Minimisation")

def rosen(v):
    x, y = v[0], v[1]
    return (1-x)**2 + 100*(y - x**2)**2

def rosen_grad(v):
    x, y = v[0], v[1]
    return nc.Vector([-2*(1-x) - 400*x*(y - x**2),
                       200*(y - x**2)])

x0 = nc.Vector([-1.0, 1.0])

res_gd = nc.optim.gradient_descent(rosen, rosen_grad, x0)
res_adam = nc.optim.adam(rosen, rosen_grad, x0)
res_bfgs = nc.optim.bfgs(rosen, rosen_grad, x0)
res_newton = nc.optim.newton(rosen, x0)

print(f"  Starting point: ({x0[0]}, {x0[1]}),  f={rosen(x0):.2f}")
print(f"  Global minimum: (1, 1),              f=0")
print()
for name, res in [("Gradient Descent", res_gd), ("Adam", res_adam),
                  ("BFGS", res_bfgs), ("Newton", res_newton)]:
    status = "✓" if res.converged else "~"
    print(f"  [{status}] {name:<18} "
          f"x=({res.x[0]:+.5f}, {res.x[1]:+.5f})  "
          f"f={res.f_val:.2e}  iter={res.iterations}")

# ══════════════════════════════════════════════════════════════
# 5. Linear system solve verification
# ══════════════════════════════════════════════════════════════
section("5. Linear System Solve  (LU decomposition)")

A = nc.Matrix(3, 3, [
    2.0, 1.0, -1.0,
   -3.0,-1.0,  2.0,
   -2.0, 1.0,  2.0
])
b = nc.Vector([8.0, -11.0, -3.0])

x = nc.linalg.solve(A, b)
print(f"  A x = b,  exact solution: x=(2, 3, -1)")
print(f"  numcpp:  x=({x[0]:.6f}, {x[1]:.6f}, {x[2]:.6f})")

# Verify: Ax
Ax0 = A[(0,0)]*x[0] + A[(0,1)]*x[1] + A[(0,2)]*x[2]
Ax1 = A[(1,0)]*x[0] + A[(1,1)]*x[1] + A[(1,2)]*x[2]
Ax2 = A[(2,0)]*x[0] + A[(2,1)]*x[1] + A[(2,2)]*x[2]
print(f"  Residual ||Ax-b||: ({abs(Ax0-b[0]):.2e}, {abs(Ax1-b[1]):.2e}, {abs(Ax2-b[2]):.2e})")

det = nc.linalg.det(A)
print(f"  det(A) = {det:.6f}")

print("\nAll examples complete.")
