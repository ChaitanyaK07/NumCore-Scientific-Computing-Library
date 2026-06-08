#pragma once
#include "vector.hpp"
#include "matrix.hpp"
#include <cmath>
#include <tuple>
#include <stdexcept>

namespace NumCore{

namespace linalg{
    struct LUResult{
        Matrix L, U;
        std::vector<size_t> perm;
        int sign;
    };

    inline LUResult lu(const Matrix& A){
        if(A.rows() != A.cols()){
            throw std::invalid_argument("LU decomposition requires square matrices");
        }

        size_t n = A.rows();
        Matrix U = A;
        Matrix L = Matrix::eye(n);

        std::vector<size_t> perm(n);
        std::iota(perm.begin(), perm.end(), 0);
        int sign = 1;
        

        for(size_t k = 0;k<n;++k){
            size_t pivot = k;
            double max_val = std::abs(U(k,k));

            for(size_t i = k+1;i<n;++i){
                if(std::abs(U(i,k)) > max_val){
                    max_val = std::abs(U(i,k));
                    pivot = i;
                }
            }

            if(max_val < 1e-15){
                throw std::runtime_error("matrix is singular (pivot ~ 0)");
            }

            if(pivot!=k){
                for(size_t j = 0;j<n;++j){
                    std::swap(U.data()[k*n +j], U.data()[pivot*n+j]);
                }
                for(size_t j = 0;j<k;++j){
                    std::swap(L.data()[k*n+j], L.data()[pivot*n+j]);
                }
                std::swap(perm[k], perm[pivot]);
                sign = -sign;
            }

            for(size_t i = k+1; i<n;++i){

                double factor = U(i,k)/U(k,k);
                L(i,k) = factor;
                U(i, k) = 0;
                for(size_t j = k+1 ;j<n;++j){
                    U(i,j) -= factor * U(k,j);
                }
            }
        }
        return {L, U, perm, sign};
    }

    inline Vector forward(const Matrix& L, const Vector& b){
        size_t n = L.rows();
        Vector y(n);
        for (size_t i = 0;i<n;++i){
            double s = b[i];
            for (size_t j = 0;j<i;++j){
                s -= L(i,j)*y[j];    
            }
            y[i] = s / L(i,i);
        }
        return y;
    }

    inline Vector backward(const Matrix& U, const Vector& y){
        size_t n = U.rows();
        Vector x(n);

        for(int i = static_cast<int>(n) - 1; i>=0; --i){
            double s = y[i];
            for(size_t j = i+1;j<n;++j){
                s -= U(i,j) * x[j];
            }
            x[i] = s / U(i,i);
        }
        return x;
    }

    inline Vector solve(const Matrix& A, const Vector& b){
        if(A.rows() != A.cols()){
            throw std::invalid_argument("solving requires square matrices");
        }
        if(A.rows() != b.size()){
            throw std::invalid_argument("argument size mismatch");
        }

        
        LUResult result = lu(A);

        Matrix L = result.L;
        Matrix U = result.U;

        std::vector<size_t> perm = result.perm;


        Vector pb(b.size());
        for(size_t i = 0;i<perm.size();++i){
            pb[i] = b[perm[i]];
        }
        Vector y = forward(L, pb);
        return backward(U, y);

    }


    inline double det(const Matrix& A){
        if(A.rows() != A.cols()){
            throw std::invalid_argument("argument requires sqaure matrix");
        }
        try{
            LUResult result = lu(A);

            Matrix L = result.L;
            Matrix U = result.U;

            std::vector<size_t> perm = result.perm;
            double d = result.sign;
            for (size_t i = 0;i<U.rows();++i){
                d *= U(i,i);
            }
            return d;
        }
        catch(const std::runtime_error){
            return 0;
        }
    }

    inline Matrix inv(const Matrix& A){
        if(A.rows() != A.cols()){
            throw std::invalid_argument("inverse required square matrix");
        }
        size_t n = A.rows();
        LUResult res = lu(A);

        Matrix L = res.L;
        Matrix U = res.U;
        std::vector<size_t> perm = res.perm;

        int sign  = res.sign;

        Matrix result(n, n);
        Vector e(n, 0);

        for(size_t col = 0;col<n;col++){
            Vector pe(n, 0.0);
            for (size_t i = 0; i < n; ++i) if (perm[i] == col) { pe[i] = 1.0; break; }
            Vector y = forward(L, pe);
            Vector x = backward(U, y);
            for (size_t i = 0; i < n; ++i) result(i, col) = x[i];
        }

        return result;
    }



    inline Vector gauss_jordan(Matrix A, Vector b){

        size_t n = A.rows();
    if (n != A.cols() || n != b.size())
        throw std::invalid_argument("gauss_jordan: bad shape");

    // Augmented matrix [A | b]
    Matrix aug(n, n+1);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) aug(i,j) = A(i,j);
        aug(i,n) = b[i];
    }

    for (size_t k = 0; k < n; ++k) {
        // partial pivot
        size_t pivot = k;
        for (size_t i = k+1; i < n; ++i)
            if (std::abs(aug(i,k)) > std::abs(aug(pivot,k))) pivot = i;
        if (std::abs(aug(pivot,k)) < 1e-15)
            throw std::runtime_error("Singular matrix in Gauss-Jordan");
        for (size_t j = 0; j <= n; ++j) std::swap(aug.data()[k*(n+1)+j], aug.data()[pivot*(n+1)+j]);

        // normalize pivot row
        double piv = aug(k,k);
        for (size_t j = k; j <= n; ++j) aug(k,j) /= piv;

        // eliminate column
        for (size_t i = 0; i < n; ++i) {
            if (i == k) continue;
            double factor = aug(i,k);
            for (size_t j = k; j <= n; ++j) aug(i,j) -= factor * aug(k,j);
        }
    }

        Vector x(n);
        for (size_t i = 0; i < n; ++i) x[i] = aug(i,n);
        return x;
    }


inline double norm(const Vector& v){
    return v.norm();
}

inline double frobenius_norm(const Matrix& A){
    return A.norm_frobenius();
}


inline Vector lstsq(const Matrix& A, const Vector& b){

    if (A.rows() != b.size()) {
        throw std::invalid_argument("lstsq: A rows must match b size");
    }

    Matrix AtA = A.T().matmul(A);
    Vector Atb_vec(A.cols(), 0);

    for(size_t j = 0;j<A.cols();++j){
        for(size_t i = 0;i<A.rows();++i){
            Atb_vec[j] += A(i,j)*b[i];
        }
    }
    return solve(AtA, Atb_vec);
}

inline Vector matvec(const Matrix& A, const Vector& v){
    if(A.cols() != v.size()) throw std::invalid_argument("mat-vec shape argument mismatch");

    Vector result(A.rows(), 0);

    for(size_t i = 0;i<A.rows();++i){
        for(size_t j = 0;j<A.cols();++j){
            result[i] += A(i,j) *v[j];
        }
    }
    return result;

}



}
}