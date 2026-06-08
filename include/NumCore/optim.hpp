#pragma once
#include "vector.hpp"
#include <functional>
#include <cmath>
#include <string>
#include <vector>
#include <stdexcept>
#include "linalg.hpp"

namespace NumCore{

namespace optim{

    using ScalarFunc = std::function<double(const Vector&)>;
    using GradFunc = std::function<Vector(const Vector&)>;


    struct OptimResult{
        Vector x;
        double f_val;
        size_t iterations;
        bool converged;
        std::string message;
    };

    inline Vector numerical_grad(ScalarFunc f, const Vector& x, double eps = 1e-7){
        Vector g(x.size());
        for (size_t i = 0;i<x.size();++i){
            Vector xp = x, xm = x;

            xp[i] += eps;
            xm[i] -= eps;

            g[i] = (f(xp) - f(xm)) / (2.0 * eps);
        }
            return g;
    }


    struct GDOption{

        double lr = 1e-3;
        double tol = 1e-6;
        size_t max_iter = 1000;
    };

    inline OptimResult grad_descent(ScalarFunc f, GradFunc grad, Vector x0, GDOption opts = {}){

        Vector x = std::move(x0);

        for(size_t itr = 0; itr < opts.max_iter ; ++itr){
                Vector g = grad(x);
                double gnorm = g.norm();
                
                if(gnorm < opts.tol){
                    return {x, f(x), itr, true, "converged : |grad| < tol"};
                }
                x = x - opts.lr * g;
        }

        return {x, f(x),opts.max_iter, false, "max iterations reached"};
    }

    struct AdamOptim{
        double lr = 1e-3;
        double beta1 = 0.9;
        double beta2 = 0.999;
        double eps = 1e-8;
        double tol = 1e-6;
        size_t max_iter = 10000;
    };

    inline OptimResult adam(ScalarFunc f, GradFunc grad, Vector x0, AdamOptim opts = {}){

        Vector x = std::move(x0);
        Vector m(x.size(), 0);
        Vector v(x.size(), 0);

        for(size_t it = 1; it<=opts.max_iter;++it){
            Vector g = grad(x);
            if(g.norm() < opts.tol){
                return {x, f(x), it, true, "converged!"};
            }


            for(size_t i = 0;i<x.size();++i){
                m[i] = opts.beta1 * m[i] + (1 - opts.beta1) * g[i];
                v[i] = opts.beta2 * v[i] + (1 - opts.beta2) * g[i] * g[i];
            }


            double bc1 = 1 - std::pow(opts.beta1, it);
            double bc2 = 1 - std::pow(opts.beta2, it);

            for(size_t i = 0;i<x.size();++i){
                x[i] -= (opts.lr*(m[i]/bc1)) / (std::sqrt(v[i]/bc2) + opts.eps); 
            }    
        }
        return {x, f(x), opts.max_iter, false, "maximum iterations reached"};
    }




    struct NewtonOptim{
        double tol = 1e-8;
        size_t max_iter = 100;
        double h = 1e-5;
        double lambda = 1e-6;
    };


    inline Matrix newton_hessian(ScalarFunc f, const Vector& x, double h){
        size_t n = x.size();
        Matrix H(n,n);

        double fx = f(x);

        for(size_t i = 0;i<n;++i){
            for(size_t j = i;j<n;++j){

                if(i==j){
                    Vector xp = x;
                    Vector xm = x;

                    xp[i] += h;
                    xm[i] -= h;


                    double val = (f(xp) - 2 * fx + f(xm))/(h*h);
                    H(i,i) = val;
                }
                else{
                    Vector xpp = x;
                    Vector xpm = x;
                    Vector xmp = x;
                    Vector xmm = x;


                    xpp[i] += h; xpp[j] += h;
                    xpm[i] += h; xpm[j] -= h;
                    xmp[i] -= h; xmp[j] += h;
                    xmm[i] -= h; xmm[j] -= h;


                    double val = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4.0*h*h);

                    H(i,j) = val;
                    H(j,i) = val;


                }
            }
        }
        return H;
    }


    inline OptimResult newton(ScalarFunc f, Vector x0, NewtonOptim opts = {}){
            Vector x = std::move(x0);

            for(size_t it = 0;it<opts.max_iter;++it){
                Vector g = numerical_grad(f, x, opts.h);

                if(g.norm() < opts.tol){
                    return {x, f(x), it, true, "converged : ||grad|| < tolerance"};
                }

                Matrix H = newton_hessian(f, x, opts.h);

                for(size_t i = 0;i<x.size(); ++i) H(i,i) += opts.lambda;


                try{
                    Vector dx = linalg::solve(H, -g);
                    x = x+dx;
                }
                catch(...){
                    return {x, f(x), it, false, "Hessian is singular"};
                }
            }
            return {x, f(x), opts.max_iter, false, "max iterations reached"};
    }


inline double armijo_line_search(ScalarFunc f, const Vector& x, const Vector& d, double alpha0 = 0.1, double c = 1e-4, double rho = 0.5){


    Vector g = numerical_grad(f, x);
    double slope = g.dot(d);

    double f0 = f(x);

    double alpha = alpha0;

    for(int i = 0;i<50;++i){
        if(f(x+d*alpha) <= f0 + c*alpha*slope) break;
        alpha *= rho;
    }
return alpha;
}



struct BFGSOptions{

    double tol = 1e-6;
    size_t max_iter = 1000;
};


inline OptimResult bfgs(ScalarFunc f, GradFunc grad, Vector x0, BFGSOptions opts = {}){

    size_t n = x0.size();
    Vector x = std::move(x0);
    Matrix H = Matrix::eye(n);

    Vector g = grad(x);


    for(size_t it = 0;it<opts.max_iter;++it){
        if(g.norm() < opts.tol){
            return {x, f(x), it, true, "converged"};
        }
        Vector d = linalg::matvec(H, -g);  

        double alpha = armijo_line_search(f, x, d);
        Vector s = d*alpha;
        Vector x_new = x+s;
        Vector g_new = grad(x_new);
        Vector yk = g_new - g;

        double sy = s.dot(yk);
        if(std::abs(sy) > 1e-15){
            double rho = 1/sy;
            Matrix sy_outer(n,n), ys_outer(n,n), ss_outer(n,n);

            for(size_t i = 0;i<n;++i){
                for(size_t j = 0;j<n;++j){
                    sy_outer(i,j) = s[i]*yk[j];
                    ys_outer(i,j) = yk[i]*s[j];
                    ss_outer(i,j) = s[i]*s[j];
                }
            }

            Matrix I = Matrix::eye(n);
            Matrix A = I - sy_outer * rho;
            Matrix B = I - ys_outer * rho;
            H = A.matmul(H).matmul(B) + ss_outer * rho;
        }


        x = x_new;
        g = g_new;
    }
    return {x, f(x), opts.max_iter, false, "max iterations reached"};
}



    
}
}