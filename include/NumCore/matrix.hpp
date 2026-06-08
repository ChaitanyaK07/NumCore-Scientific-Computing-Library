#pragma once
#include <vector>
#include <stdexcept>
#include <string>
#include <cmath>
#include <functional>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>

namespace NumCore{

class Matrix{   
    public : 

        Matrix () : rows_(0), cols_(0) {}

        Matrix (size_t rows, size_t cols, double fill = 0.0) : rows_(rows), cols_(cols), data_(rows*cols, fill) {}

        Matrix(size_t rows, size_t cols, std::vector<double> data) : rows_(rows), cols_(cols), data_(std::move(data)) {


            if (data_.size() != rows_ * cols_) {
        throw std::invalid_argument(
            "data size mismatch expected : " +
            std::to_string(rows_ * cols_) +
            " got " +
            std::to_string(data_.size())
        );
    }
        }


        Matrix(std::initializer_list<std::initializer_list<double>> rows){

            rows_ = rows.size();

            if(rows_ == 0){
                cols_ = 0;
                return;
            }

            cols_ = rows.begin()->size();
            data_.reserve(rows_ * cols_);

            for(const auto& row : rows){
                if (row.size() != cols_){
                    throw std::invalid_argument("Ragged initializer list");
                }
                for(double v : row){
                    data_.push_back(v);
                }
            }
        }


        static Matrix zeroes ( size_t r, size_t c){
            return Matrix(r, c, 0);
        }

        static Matrix ones(size_t r, size_t c){
            return Matrix(r, c, 1);
        }

        static Matrix eye(size_t n){
            Matrix m(n, n, 0);
            for(size_t i = 0; i< n ;++i){
                m(i,i) = 1;
            }
            return m;
        }


        size_t rows() const {return rows_ ;}
        size_t cols() const {return cols_ ;}
        size_t size() const {return data_.size() ; }
        std::pair<size_t, size_t> shape() const {return {rows_, cols_} ;}



        double& operator() (size_t r, size_t c){
            check_bounds(r, c);
            return data_[r*cols_ + c];
        }

        double operator()(size_t r, size_t c) const{
            check_bounds(r, c);
            return data_[r*cols_ + c];
        }

        const std::vector<double>& data() const {    
            return data_ ;
        
        }
        std::vector<double>& data(){
                return data_ ;
        }


        std::vector<double> row(size_t r) const {
            if (r>=rows_) throw std::out_of_range("row index out of range");

            return std::vector<double>(data_.begin() + r * cols_, data_.begin() + (r+1) * cols_);
        }


        std::vector<double> col(size_t c) const{
            if (c>=cols_) throw std::out_of_range("col index out of range");

            std::vector<double> result(rows_);

            for(size_t i = 0;i<rows_;++i){
                result[i] = data_[i*cols_+c];
            
            }
            return result;
        }


        Matrix operator+(const Matrix& o) const {
            check_same_shape(o);
            Matrix r(rows_, cols_);
            for (size_t i = 0;i<data_.size(); ++i){
                r.data_[i] = data_[i] + o.data_[i];
            }
            return r;
        }

        Matrix operator-(const Matrix& o) const {
            check_same_shape(o);
            Matrix r(rows_, cols_);
            for (size_t i = 0;i<data_.size(); ++i){
                r.data_[i] = data_[i] - o.data_[i];
            }
            return r;
        }

        Matrix operator*(double s) const{
            Matrix r(rows_, cols_);

            for (size_t i = 0;i<data_.size(); ++i){
                r.data_[i] = data_[i] * s;
            }
            return r;
        }


        Matrix operator/ (double s) const {
            return (*this) * (1.0/s);
        }


        friend Matrix operator*(double s, const Matrix& M){
            return M*s;
        }

        Matrix& operator+=(const Matrix& o) { *this = *this + o; return *this; }
        Matrix& operator-=(const Matrix& o) { *this = *this - o; return *this; }
        Matrix& operator*=(double s)        { for (auto& v : data_) v *= s; return *this; }

        Matrix operator-() const {
            Matrix r(rows_, cols_);
            for (size_t i = 0; i < data_.size(); ++i) r.data_[i] = -data_[i];
            return r;
        }


        Matrix matmul(const Matrix& o) const {

            if (cols_ != o.rows_){
                throw std::invalid_argument("matmul shape mismatch: (" +
                std::to_string(rows_) + "," + std::to_string(cols_) + ") @ (" +
                std::to_string(o.rows_) + "," + std::to_string(o.cols_) + ")");
            }

            Matrix r(rows_, o.cols_, 0);

            for(size_t i = 0; i<rows_; ++i){
                for(size_t k = 0;k<cols_;++k){
                    double aik = data_[i*cols_ + k];
                    for (size_t j = 0; j<o.cols_;j++){
                        r.data_[i*o.cols_+j] += aik * o.data_[k*o.cols_+j];
                    }
                }

            }
            return r;
        }

        Matrix T() const{
            Matrix r(cols_, rows_);

            for(size_t i = 0;i<rows_;++i){
                for (size_t j = 0; j<cols_;++j){
                    r.data_[j*rows_+i] = data_[i*cols_+j];
                }
            }
        
            return r;
        
        }


        Matrix apply(std::function<double(double)> f) const{
            Matrix r(rows_, cols_);
            for (size_t i = 0; i<data_.size();++i){
                r.data_[i] = f(data_[i]);
            }

            return r;
        }


        double sum() const {
            double s = 0;
            for(auto v:data_){
                s+=v;
            }
            return s;
        }

        double norm_frobenius() const{
            double s = 0;
            for(auto v:data_){
                s += v*v;
            }
            return std::sqrt(s);
        }
        double max() const {return *std::max_element(data_.begin(), data_.end()); }
        double min() const {return *std::min_element(data_.begin(), data_.end()); }



        std::string repr() const{
            std::ostringstream ss;
        ss << std::fixed << std::setprecision(4);
        ss << "Matrix(" << rows_ << "x" << cols_ << "):\n";
        for (size_t i = 0; i < rows_; ++i) {
            ss << "  [";
            for (size_t j = 0; j < cols_; ++j) {
                if (j) ss << ", ";
                ss << std::setw(9) << data_[i*cols_+j];
            }
            ss << "]\n";
        }
        return ss.str();
        }
private : 

        size_t rows_, cols_;
        std::vector<double> data_;

        void check_bounds(size_t r, size_t c) const{
            if (r>= rows_ || c>=cols_){
                throw std::out_of_range("Index (" + std::to_string(r) + "," +
                std::to_string(c) + ") out of range for " +
                std::to_string(rows_) + "x" + std::to_string(cols_));
            }
        }

        void check_same_shape(const Matrix& o) const{
            if (rows_ != o.rows_ || cols_ != o.cols_){
                throw std::invalid_argument("Shape mismatch: (" +
                std::to_string(rows_) + "," + std::to_string(cols_) + ") vs (" +
                std::to_string(o.rows_) + "," + std::to_string(o.cols_) + ")");
            }
        }





};
}