#pragma once
#include "vector.hpp"
#include <functional>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <algorithm>



namespace NumCore{
    namespace ode{


using OdeFunc = std::function<Vector(double, const Vector&)>;


struct Odesoln{
    std::vector<double> t;
    std::vector<Vector> y;

};

inline Odesoln euler(OdeFunc f, double t0, double t_end, Vector y0, double dt){

    if (dt<=0){
        throw std::invalid_argument("dt must be positive");
    }
    Odesoln sol;
    double t = t0;
    Vector y = std::move(y0);
    sol.t.push_back(t);
    sol.y.push_back(y);

    while(t < t_end - 1e-12){
        double step = std::min(dt, t_end - t);
        y = y + f(t , y) * step;
        t += step;
        sol.t.push_back(t);
        sol.y.push_back(y);
    }
    return sol;
}


inline Odesoln rk4(OdeFunc f, double t0, double t_end, Vector y0, double dt){
    if(dt<=0){
        throw std::invalid_argument("dt must be positive");
    }
    Odesoln sol;
    double t = t0;
    Vector y = std::move(y0);
    sol.t.push_back(t);
    sol.y.push_back(y);

    while(t < t_end - 1e-12){
        double h = std::min(dt, t_end - t);
        Vector k1 = f(t,y)*h;
        Vector k2 = f(t+ h/2, y+k1*0.5)*h;
        Vector k3 = f(t+h/2, y+k2*0.5)*h;
        Vector k4 = f(t+h, y+k3)*h;
        y = y + (k1 + k2*2.0 + k3*2.0 + k4) * (1.0/6.0);
        t += h;
        sol.t.push_back(t);
        sol.y.push_back(y);
    }

    return sol;
}



struct RK45Option{

    double rtol = 1e-6;
    double atol = 1e-8;
    double dt_init = 1e-3;
    double dt_min = 1e-12;
    double dt_max = 1;
    size_t max_steps = 1'000'000;
};


inline Odesoln rk45(OdeFunc f, double t0, double t_end, Vector y0, RK45Option opts = {}){

    static const double a21 = 1.0/5.0;
    static const double a31 = 3.0/40.0, a32 = 9.0/40.0;
    static const double a41 = 44.0/45.0,  a42 =-56.0/15.0,  a43 = 32.0/9.0;
    static const double a51 = 19372.0/6561.0, a52=-25360.0/2187.0,
                        a53 = 64448.0/6561.0, a54=-212.0/729.0;

    static const double a61 = 9017.0/3168.0,  a62=-355.0/33.0,
                        a63 = 46732.0/5247.0, a64= 49.0/176.0,
                        a65 =-5103.0/18656.0;

    static const double b1=35.0/384.0,  b3=500.0/1113.0, b4=125.0/192.0,
                        b5=-2187.0/6784.0, b6=11.0/84.0;

    static const double e1= 71.0/57600.0, e3=-71.0/16695.0, e4=71.0/1920.0,
                        e5=-17253.0/339200.0, e6=22.0/525.0, e7=-1.0/40.0;

                    

    Odesoln sol;
    double t = t0;
    Vector y = std::move(y0);
    double h = opts.dt_init;

    sol.t.push_back(t);
    sol.y.push_back(y);

    for(size_t step = 0; step < opts.max_steps && t < t_end - 1e-12; ++step){

        h = std::min(h, t_end - t);

        Vector k1 = f(t, y) * h;
        Vector k2 = f(t + h/5, y + k1*a21) * h;
        Vector k3 = f(t + 3*h/10, y+k1*a31 +k2*a32) * h;
        Vector k4 = f(t + 4.0*h/5.0, y + k1*a41 + k2*a42 + k3*a43) * h;
        Vector k5 = f(t + 8*h/9,   y + k1*a51 + k2*a52 + k3*a53 + k4*a54) * h;
        Vector k6 = f(t + h,       y + k1*a61 + k2*a62 + k3*a63 + k4*a64 + k5*a65) * h;

        Vector y_new = y + k1*b1 + k3*b3 + k4*b4 + k5*b5 + k6*b6;

        Vector k7 = f(t+h, y_new) * h;

        Vector err = k1*e1 + k3*e3 + k4*e4 + k5*e5 + k6*e6 + k7*e7;


        double err_norm = 0;


        for(size_t i = 0;i<err.size();++i){
            double sc = opts.atol + opts.rtol * std::max(std::abs(y[i]), std::abs(y_new[i]));
            err_norm += (err[i]/sc) * (err[i]/sc);
        }

        err_norm = std::sqrt(err_norm / err.size());

        if(err_norm <= 1){
            t += h;
            y = y_new;
            sol.t.push_back(t);
            sol.y.push_back(y);
        }

        double factor;
        if (err_norm == 0.0) {
            factor = 10.0;
        } else {
        factor = 0.9 * std::pow(err_norm, -0.2);
        }

        factor = std::max(0.1, std::min(factor, 10.0));
        h = std::min(std::max(h*factor, opts.dt_min), opts.dt_max);

    }

    return sol;
}   


inline std::vector<double> extract_component(const Odesoln& sol, size_t i){

    std::vector<double> out;

    out.reserve(sol.y.size());
    for(const auto& v : sol.y){
        if(i>= v.size()) throw std::out_of_range("component index out of range");
        out.push_back(v[i]);

    }
    return out;
}
};
}