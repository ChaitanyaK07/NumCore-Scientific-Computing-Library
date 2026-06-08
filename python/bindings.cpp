#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>

#include "NumCore/NumCore.hpp"


namespace py = pybind11;
using namespace NumCore;


Matrix numpy_to_matrix(py::array_t<double> arr){

    auto buf = arr.request();

    if (buf.ndim != 2) throw std::invalid_argument("2d array expected");
    size_t rows = buf.shape[0], cols = buf.shape[1];
    std::vector<double> data(rows * cols);

    auto ptr = static_cast<double*>(buf.ptr);

    for(size_t i = 0;i<rows;++i){
        for(size_t j = 0;j<cols;++j){
            data[i*cols + j] = ptr[i*(buf.strides[0]/sizeof(double)) +j*(buf.strides[1]/sizeof(double))];
        }
    }

    return Matrix(rows, cols, std::move(data));
}

py::array_t<double> matrix_to_numpy(const Matrix& m){

    py::array_t<double> arr({m.rows(), m.cols()});
    auto buf = arr.mutable_unchecked<2>();

    for (size_t i = 0; i < m.rows(); ++i) {
        for (size_t j = 0; j < m.cols(); ++j) {
            buf(i, j) = m(i, j);
        }
    }

    return arr;
}

Vector numpy_to_vector(py::array_t<double> arr){
    auto buf = arr.request();

    if(buf.ndim != 1) throw std::invalid_argument("expected 1d array");

    auto ptr = static_cast<double*>(buf.ptr);
    return Vector(std::vector<double>(ptr, ptr + buf.shape[0]));
}


py::array_t<double> vector_to_numpy(const Vector& v){
    return py::array_t<double>({v.size()}, {sizeof(double)}, v.data().data());
}



PYBIND11_MODULE(_NumCore, m){
    m.doc() = "NumCore: C++ numerical computing library with python bindings";



    py::class_<Matrix>(m, "Matrix")
        .def(py::init<size_t, size_t, double>(),
            py::arg("rows"), py::arg("cols"), py::arg("fill") = 0.0 )
        .def(py::init([](size_t rows, size_t cols, std::vector<double> data){
            return Matrix(rows, cols, std::move(data));
        }), py::arg("rows"), py::arg("cols"), py::arg("data"), "construct from flat data list")
        .def(py::init([](py::array_t<double> arr) {
                return numpy_to_matrix(arr);
             }), "Construct from 2D numpy array")
        .def_static("zeros", &Matrix::zeroes)
        .def_static("ones", &Matrix::ones)
        .def_static("eye", &Matrix::eye)


        .def("rows", &Matrix::rows)
        .def("cols", &Matrix::cols)
        .def("shape", &Matrix::shape)

        .def("__getitem__", [](const Matrix& m, std::pair<size_t, size_t> idx){
            return m(idx.first, idx.second);
        })


        .def("__setitem__", [](Matrix& m, std::pair<size_t, size_t> idx, double v){
            m(idx.first, idx.second) = v;
        })

        .def("__add__",  [](const Matrix& a, const Matrix& b) { return a + b; })
        .def("__sub__",  [](const Matrix& a, const Matrix& b) { return a - b; })
        .def("__mul__",  [](const Matrix& m, double s) { return m * s; })
        .def("__rmul__", [](const Matrix& m, double s) { return m * s; })
        .def("__truediv__", [](const Matrix& m, double s) { return m / s; })
        .def("__neg__",  [](const Matrix& m) { return -m; })
        .def("matmul",   &Matrix::matmul)
        .def("__matmul__", &Matrix::matmul)
        .def("T",        &Matrix::T)
        .def("apply",    &Matrix::apply)
        .def("sum",      &Matrix::sum)
        .def("norm_frobenius", &Matrix::norm_frobenius)
        .def("max",      &Matrix::max)
        .def("min",      &Matrix::min)
        .def("row",      &Matrix::row)
        .def("col",      &Matrix::col)
        .def("numpy",    matrix_to_numpy, "Return as numpy array")
        .def("__repr__", &Matrix::repr)
        .def("__str__",  &Matrix::repr);


    py::class_<Vector>(m, "Vector")
        .def(py::init<size_t, double>(), py::arg("n"), py::arg("fill")=0.0)
        .def(py::init([](py::array_t<double> arr) {
                return numpy_to_vector(arr);
             }), "Construct from 1D numpy array")
        .def(py::init([](std::vector<double> v) { return Vector(std::move(v)); }))

        .def("size",  &Vector::size)
        .def("__len__", &Vector::size)
        .def("__getitem__", [](const Vector& v, size_t i) { return v[i]; })
        .def("__setitem__", [](Vector& v, size_t i, double val) { v[i] = val; })

        .def("__add__",  [](const Vector& a, const Vector& b) { return a + b; })
        .def("__sub__",  [](const Vector& a, const Vector& b) { return a - b; })
        .def("__mul__",  [](const Vector& v, double s) { return v * s; })
        .def("__rmul__", [](const Vector& v, double s) { return v * s; })
        .def("__truediv__", [](const Vector& v, double s) { return v / s; })
        .def("__neg__",  [](const Vector& v) { return -v; })

        .def("dot",        &Vector::dot)
        .def("norm",       &Vector::norm)
        .def("normalized", &Vector::normalize)
        .def("cross",      &Vector::cross)
        .def("sum",        &Vector::sum)
        .def("mean",       &Vector::mean)
        .def("max",        &Vector::max)
        .def("min",        &Vector::min)
        .def("apply",      &Vector::apply)
        .def("as_column",  &Vector::as_col)
        .def("as_row",     &Vector::as_row)
        .def("numpy",      vector_to_numpy, "Return as numpy array")
        .def("__repr__",   &Vector::repr)
        .def("__str__",    &Vector::repr);

        m.def("matvec", &NumCore::linalg::matvec, "Matrix-vector multiply");

    // ═══════════════════════════════════════════════════════════════════════
    // linalg sub-module
    // ═══════════════════════════════════════════════════════════════════════
    auto linalg_m = m.def_submodule("linalg", "Linear algebra routines");

    py::class_<linalg::LUResult>(linalg_m, "LUResult")
        .def_readonly("L",    &linalg::LUResult::L)
        .def_readonly("U",    &linalg::LUResult::U)
        .def_readonly("perm", &linalg::LUResult::perm)
        .def_readonly("sign", &linalg::LUResult::sign);

    linalg_m.def("lu",     &linalg::lu,     "LU decomposition with partial pivoting");
    linalg_m.def("solve",  &linalg::solve,  "Solve A @ x = b");
    linalg_m.def("det",    &linalg::det,    "Determinant");
    linalg_m.def("inv",    &linalg::inv,    "Matrix inverse");
    linalg_m.def("lstsq",  &linalg::lstsq,  "Least-squares: argmin ||Ax - b||");
    linalg_m.def("gauss_jordan", &linalg::gauss_jordan, "Gauss-Jordan solver");
    linalg_m.def("norm",   [](const Vector& v) { return linalg::norm(v); });
    linalg_m.def("norm_frobenius", [](const Matrix& A) { return linalg::frobenius_norm(A); });

    // ═══════════════════════════════════════════════════════════════════════
    // ode sub-module
    // ═══════════════════════════════════════════════════════════════════════
    auto ode_m = m.def_submodule("ode", "ODE solvers");

    py::class_<ode::Odesoln>(ode_m, "ODESolution")
        .def_readonly("t", &ode::Odesoln::t)
        .def_readonly("y", &ode::Odesoln::y)
        .def("t_array", [](const ode::Odesoln& s) {
                return py::array_t<double>({s.t.size()}, {sizeof(double)}, s.t.data());
             })
        .def("y_array", [](const ode::Odesoln& s) {
                if (s.y.empty()) return py::array_t<double>();
                size_t nsteps = s.y.size(), ny = s.y[0].size();
                std::vector<double> flat(nsteps * ny);
                for (size_t i = 0; i < nsteps; ++i)
                    for (size_t j = 0; j < ny; ++j)
                        flat[i*ny+j] = s.y[i][j];
                return py::array_t<double>({nsteps, ny},
                    {ny*sizeof(double), sizeof(double)}, flat.data());
             });

    py::class_<ode::RK45Option>(ode_m, "RK45Options")
        .def(py::init<>())
        .def_readwrite("rtol",      &ode::RK45Option::rtol)
        .def_readwrite("atol",      &ode::RK45Option::atol)
        .def_readwrite("dt_init",   &ode::RK45Option::dt_init)
        .def_readwrite("dt_min",    &ode::RK45Option::dt_min)
        .def_readwrite("dt_max",    &ode::RK45Option::dt_max)
        .def_readwrite("max_steps", &ode::RK45Option::max_steps);

    ode_m.def("euler", &ode::euler, py::arg("f"), py::arg("t0"), py::arg("t_end"),
              py::arg("y0"), py::arg("dt"), "Euler method");
    ode_m.def("rk4",   &ode::rk4,   py::arg("f"), py::arg("t0"), py::arg("t_end"),
              py::arg("y0"), py::arg("dt"), "Classic 4th-order Runge-Kutta");
    ode_m.def("rk45",  &ode::rk45,  py::arg("f"), py::arg("t0"), py::arg("t_end"),
              py::arg("y0"), py::arg("opts")=ode::RK45Option{},
              "Dormand-Prince RK45 (adaptive step)");
    ode_m.def("extract_component", &ode::extract_component);

    // ═══════════════════════════════════════════════════════════════════════
    // optim sub-module
    // ═══════════════════════════════════════════════════════════════════════
    auto opt = m.def_submodule("optim", "Optimization algorithms");

    py::class_<optim::OptimResult>(opt, "OptimResult")
        .def_readonly("x",          &optim::OptimResult::x)
        .def_readonly("f_val",      &optim::OptimResult::f_val)
        .def_readonly("iterations", &optim::OptimResult::iterations)
        .def_readonly("converged",  &optim::OptimResult::converged)
        .def_readonly("message",    &optim::OptimResult::message);

    py::class_<optim::GDOption>(opt, "GDOptions")
        .def(py::init<>())
        .def_readwrite("lr",       &optim::GDOption::lr)
        .def_readwrite("tol",      &optim::GDOption::tol)
        .def_readwrite("max_iter", &optim::GDOption::max_iter);

    py::class_<optim::AdamOptim>(opt, "AdamOptions")
        .def(py::init<>())
        .def_readwrite("lr",       &optim::AdamOptim::lr)
        .def_readwrite("beta1",    &optim::AdamOptim::beta1)
        .def_readwrite("beta2",    &optim::AdamOptim::beta2)
        .def_readwrite("eps",      &optim::AdamOptim::eps)
        .def_readwrite("tol",      &optim::AdamOptim::tol)
        .def_readwrite("max_iter", &optim::AdamOptim::max_iter);

    py::class_<optim::NewtonOptim>(opt, "NewtonOptions")
        .def(py::init<>())
        .def_readwrite("tol",      &optim::NewtonOptim::tol)
        .def_readwrite("max_iter", &optim::NewtonOptim::max_iter)
        .def_readwrite("h",        &optim::NewtonOptim::h)
        .def_readwrite("lambda_",  &optim::NewtonOptim::lambda);

    py::class_<optim::BFGSOptions>(opt, "BFGSOptions")
        .def(py::init<>())
        .def_readwrite("tol",      &optim::BFGSOptions::tol)
        .def_readwrite("max_iter", &optim::BFGSOptions::max_iter);

    opt.def("numerical_grad", &optim::numerical_grad,
            py::arg("f"), py::arg("x"), py::arg("eps")=1e-7);
    opt.def("gradient_descent", &optim::grad_descent,
            py::arg("f"), py::arg("grad"), py::arg("x0"),
            py::arg("opts")=optim::GDOption{});
    opt.def("adam",   &optim::adam,
            py::arg("f"), py::arg("grad"), py::arg("x0"),
            py::arg("opts")=optim::AdamOptim{});
    opt.def("newton", &optim::newton,
            py::arg("f"), py::arg("x0"),
            py::arg("opts")=optim::NewtonOptim{});
    opt.def("bfgs",   &optim::bfgs,
            py::arg("f"), py::arg("grad"), py::arg("x0"),
            py::arg("opts")=optim::BFGSOptions{});
        
}     