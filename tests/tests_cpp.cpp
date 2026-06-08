#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include "NumCore/NumCore.hpp"

using namespace NumCore;

static int passed = 0, failed = 0;

#define TEST(name, expr) do { \
    try { \
        bool ok = (expr); \
        if (ok) { ++passed; std::cout << "  PASS  " << (name) << "\n"; } \
        else    { ++failed; std::cout << "  FAIL  " << (name) << "\n"; } \
    } catch (std::exception& e) { \
        ++failed; std::cout << "  FAIL  " << (name) << " [exception: " << e.what() << "]\n"; } \
} while(0)


#define NEAR(a,b,tol) (std::abs((a)-(b)) < (tol))

void test_matrix(){

    std::cout << "\n[matrix]\n";

    Matrix A(2,2,0.0);

    A(0,0) = 1; A(0,1) = 2; A(1,0) = 3; A(1,1) = 4;
    TEST("element access", A(1,1) == 4.0);
    TEST("sum", A.sum() == 10);
    TEST("eye diagonal", Matrix::eye(3)(1,1) == 1 && Matrix::eye(3)(0,1) == 0);

    Matrix B = A + A;
    TEST("add", B(0,0) == 2 && B(1,1) == 8);

    Matrix C = A.matmul(A);
    TEST("matmul", NEAR(C(0,0), 7, 1e-10) && NEAR(C(1,1),22,1e-10));

    Matrix T = A.T();
    TEST("transpose", T(0,1) == A(1,0) && T(1,0) == A(0,1));
    TEST("scalar mul", (A*2)(0,0) == 2);

    try { A(5,5); ++failed; std::cout << "  FAIL  bounds check (no throw)\n"; }
    catch (...) { ++passed; std::cout << "  PASS  bounds check throws\n"; }

}


void test_vector() {
    std::cout << "\n[Vector]\n";

    Vector v{1.0, 2.0, 3.0};
    TEST("norm", NEAR(v.norm(), std::sqrt(14.0), 1e-10));
    TEST("dot",  NEAR(v.dot(v), 14.0, 1e-10));

    Vector u{3.0, 1.0, 0.0};
    Vector cross = v.cross(u);  // (2*0-3*1, 3*3-1*0, 1*1-2*3) = (-3,9,-5)
    TEST("cross", NEAR(cross[0],-3,1e-10) && NEAR(cross[1],9,1e-10) && NEAR(cross[2],-5,1e-10));

    Vector n = v.normalize();
    TEST("normalized norm", NEAR(n.norm(), 1.0, 1e-10));

    TEST("add", (v + v)[0] == 2.0);
    TEST("sub", (v - v)[1] == 0.0);
}

// ── linalg tests ─────────────────────────────────────────────────────────────
void test_linalg() {
    std::cout << "\n[linalg]\n";

    // solve [[2,1],[5,7]] x = [11,13] → x=[3.0,5.0]  — NO: [3.0,5.0] check
    Matrix A{{2,1},{5,7}};
    Vector b{11,13};
    Vector x = linalg::solve(A, b);
    // 2*3+1*5=11, 5*3+7*5=50≠13 — correct: x=[(11*7-1*13)/(2*7-1*5), ...]
    // 2*x0+x1=11, 5*x0+7*x1=13
    // det = 14-5=9, x0=(11*7-1*13)/9=(77-13)/9=64/9, x1=(2*13-11*5)/9=(26-55)/9=-29/9
    TEST("solve",
         NEAR(2*x[0]+x[1], 11.0, 1e-8) &&
         NEAR(5*x[0]+7*x[1], 13.0, 1e-8));

    double d = linalg::det(A);  // 2*7-1*5=9
    TEST("det", NEAR(d, 9.0, 1e-10));

    Matrix I3 = Matrix::eye(3);
    Matrix invI = linalg::inv(I3);
    TEST("inv(I)=I", NEAR(invI(0,0),1,1e-10) && NEAR(invI(0,1),0,1e-10));

    // inv of A: 1/9 * [[7,-1],[-5,2]]
    Matrix invA = linalg::inv(A);
    TEST("inv correct", NEAR(invA(0,0), 7.0/9, 1e-10) && NEAR(invA(1,1), 2.0/9, 1e-10));

    // verify A @ inv(A) ≈ I
    Matrix should_be_I = A.matmul(invA);
    TEST("A @ inv(A) = I", NEAR(should_be_I(0,0),1,1e-8) && NEAR(should_be_I(0,1),0,1e-8));

    // lstsq: fit y = a*x + b with 3 points (overdetermined)
    Matrix D{{1,1},{2,1},{3,1}};
    Vector yd{2.0, 4.0, 5.0};  // roughly 1.5x + 0.33
    Vector coeff = linalg::lstsq(D, yd);
    TEST("lstsq",
     NEAR(coeff[0], 1.5, 1e-8) &&
     NEAR(coeff[1], 2.0/3.0, 1e-8));
}

// ── ODE tests ─────────────────────────────────────────────────────────────────
void test_ode() {
    std::cout << "\n[ODE]\n";

    // dy/dt = -y, y(0)=1  → y(t) = e^{-t}
    auto f = [](double t, const Vector& y) -> Vector {
        (void)t;
        return Vector{-y[0]};
    };

    auto sol_rk4 = ode::rk4(f, 0.0, 5.0, Vector{1.0}, 0.01);
    double y_rk4 = sol_rk4.y.back()[0];
    double exact  = std::exp(-5.0);
    TEST("rk4 decay", NEAR(y_rk4, exact, 1e-4));

    auto sol_eu = ode::euler(f, 0.0, 1.0, Vector{1.0}, 0.001);
    TEST("euler decay", NEAR(sol_eu.y.back()[0], std::exp(-1.0), 1e-2));

    auto sol_45 = ode::rk45(f, 0.0, 5.0, Vector{1.0});
    TEST("rk45 decay", NEAR(sol_45.y.back()[0], exact, 1e-5));

    // harmonic oscillator: d²x/dt² + x = 0 → [x,v], conserve energy x²+v²=1
    auto ho = [](double t, const Vector& y) -> Vector {
        (void)t;
        return Vector{y[1], -y[0]};
    };
    auto hs = ode::rk4(ho, 0.0, 2*std::acos(-1), Vector{1.0, 0.0}, 0.01);
    double energy = hs.y.back()[0]*hs.y.back()[0] + hs.y.back()[1]*hs.y.back()[1];
    TEST("harmonic oscillator energy", NEAR(energy, 1.0, 1e-3));
}

// ── Optim tests ───────────────────────────────────────────────────────────────
void test_optim() {
    std::cout << "\n[Optim]\n";

    // Rosenbrock: f(x,y) = (1-x)^2 + 100*(y-x^2)^2, min at (1,1)
    auto rosen = [](const Vector& v) -> double {
        double x=v[0], y=v[1];
        return (1-x)*(1-x) + 100*(y-x*x)*(y-x*x);
    };
    auto rosen_g = [](const Vector& v) -> Vector {
        double x=v[0], y=v[1];
        return Vector{-2*(1-x) - 400*x*(y-x*x),
                       200*(y-x*x)};
    };

    // Newton finds minimum of a simpler quadratic: f = x^2 + y^2
    auto quad   = [](const Vector& v) { return v[0]*v[0] + v[1]*v[1]; };
    auto res_n  = optim::newton(quad, Vector{3.0, 4.0});
    TEST("newton quad", NEAR(res_n.x[0], 0.0, 1e-6) && NEAR(res_n.x[1], 0.0, 1e-6));

    // BFGS on Rosenbrock
    auto res_bfgs = optim::bfgs(rosen, rosen_g, Vector{0.0, 0.0});
    TEST("bfgs rosenbrock", NEAR(res_bfgs.x[0], 1.0, 1e-3) && NEAR(res_bfgs.x[1], 1.0, 1e-3));

    // Adam on quadratic
    auto quad_g = [](const Vector& v) { return v * 2.0; };
    auto res_a = optim::adam(quad, quad_g, Vector{5.0, -3.0});
    TEST("adam quad", NEAR(res_a.x[0], 0.0, 1e-3) && NEAR(res_a.x[1], 0.0, 1e-3));
}

int main() {
    std::cout << "=== numcpp C++ unit tests ===\n";
    test_matrix();
    test_vector();
    test_linalg();
    test_ode();
    test_optim();
    std::cout << "\nResults: " << passed << " passed, " << failed << " failed\n";
    return failed == 0 ? 0 : 1;
}



