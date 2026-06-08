#pragma once
#include <cmath>
#include <string>
#include <sstream>
#include <vector>
#include <stdexcept>
#include <iomanip>
#include <functional>
#include <numeric>
#include <algorithm>
#include "matrix.hpp"


namespace NumCore{

class Vector{


public :

    Vector() = default;
    
    explicit Vector(size_t n, double fill = 0) : data_(n, fill) {}
    Vector(std::initializer_list<double> il) : data_(il){}
    explicit Vector(std::vector<double> v) :data_(std::move(v)) {}

    size_t size() const {return data_.size();}

    double& operator[] (size_t i){
        if(i>= data_.size()) throw std::out_of_range("vector index " + std::to_string(i) + "is out of range");
        return data_[i];
    }

    double operator[] (size_t i) const{
        if(i>= data_.size()) throw std::out_of_range("vector index " + std::to_string(i) + "is out of range");
        return data_[i];
    }

    const std::vector<double>& data() const {
        return data_;
    }

    std::vector<double> data(){
        return data_;
    }


    Vector operator+ (const Vector& o) const{
        check_size(o);
        Vector r(data_.size());

        for(size_t i = 0;i<data_.size();++i){
            r.data_[i] = data_[i] + o.data_[i];
        }
        return r;
    }

    Vector operator- (const Vector& o) const{
        check_size(o);
        Vector r(data_.size());

        for(size_t i = 0;i<data_.size();++i){
            r.data_[i] = data_[i] - o.data_[i];
        }
        return r;
    }


    Vector operator*(double s) const{
        Vector r(data_.size());
        for(size_t i = 0;i<data_.size();++i){
            r.data_[i] = data_[i] * s;
        }

        return r;
    }

    Vector operator/(double s) const{

        if(s == 0){
            throw std::invalid_argument("division by zero");
        }

        return (*this) * (1/s);
    }

    friend Vector operator*(double s, const Vector& v){
        return v*s;
    }


    Vector& operator+=(const Vector& o){
        *this = *this + o;
        return *this;
    }

    Vector& operator-=(const Vector& o){
        *this = *this - o;
        return *this;
    }

    Vector& operator*=(double s){
        for (auto&v : data_){
            v *= s;
        }
        return *this;
    }

    Vector operator-() const {
        Vector r(data_.size());
        for (size_t i = 0; i < data_.size(); ++i) r.data_[i] = -data_[i];
        return r;
    }

    double dot(const Vector& o) const{
        check_size(o);
        double s = 0;
        for(size_t i = 0;i<data_.size();++i){
            s += data_[i] * o.data_[i];
        }
        return s;
    }


    double norm() const {return std::sqrt(dot(*this));}

    Vector normalize() const{
        double n = norm();
        if(n<1e-18) throw std::runtime_error("Cannot normalize zero vector");

        return (*this)/n;
    }

    double cross2D(const Vector& o) const {
    if (data_.size() != 2 || o.data_.size() != 2) {
        throw std::invalid_argument("2D cross product requires 2D vectors");
    }

    return data_[0] * o.data_[1] - data_[1] * o.data_[0];
}

    Vector cross(const Vector& o) const {
        if (data_.size() != 3 || o.data_.size() != 3) {
            throw std::invalid_argument("3D cross product requires 3D vectors");
        }

        return Vector{
            data_[1] * o.data_[2] - data_[2] * o.data_[1],
            data_[2] * o.data_[0] - data_[0] * o.data_[2],
            data_[0] * o.data_[1] - data_[1] * o.data_[0]
        };
    }

    Matrix as_row() const{
        Matrix m(1, data_.size());
        for(size_t i = 0;i<data_.size();++i){
            m(0, i) = data_[i];
        }
        return m;
    }

    Matrix as_col() const{
        Matrix m(data_.size(), 1);
        for(size_t i = 0;i<data_.size();++i){
            m(i,0) = data_[i];
        }
        return m;
    }




    double sum() const {
        return std::accumulate(data_.begin(), data_.end(), 0.0);
    }

    double max() const{

        if (data_.empty()) {
        throw std::runtime_error("Cannot take max of empty vector");
    }

        return *std::max_element(data_.begin(), data_.end());
    }

    double min() const{

        if (data_.empty()) {
        throw std::runtime_error("Cannot take min of empty vector");
    }
        return *std::min_element(data_.begin(), data_.end());
    }

    double mean() const{

        if (data_.empty()) {
        throw std::runtime_error("Cannot take mean of empty vector");
    }
        return sum() / static_cast<double>(data_.size());
    }

    Vector apply(std::function<double(double)> f) const{
        Vector r(data_.size());
        for(size_t i = 0;i<data_.size();++i){
            r.data_[i] = f(data_[i]);
        }
        return r;
    }

    std::string repr() const {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(4) << "Vector([";
        for (size_t i = 0; i < data_.size(); ++i) {
            if (i) ss << ", ";
            ss << data_[i];
        }
        ss << "])";
        return ss.str();
    }

    

private:
    std::vector<double> data_;
    void check_size(const Vector& o) const{
        if (data_.size() != o.data_.size()){
            throw std::invalid_argument("vector size mismatch: " + std::to_string(data_.size()) + " vs " + std::to_string(o.data_.size()));
        }
    }

};

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




