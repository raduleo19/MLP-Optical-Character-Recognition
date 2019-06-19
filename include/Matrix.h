// Copyright 2019 Ciobanu Bogdan-Calin

#pragma once

#include <vector>

template <class T>
class matrix {
public:
    matrix () {}
    
    matrix (int n, int m) {}
    
    matrix (const matrix &target) {}
    
    matrix (matrix &&target) {}
    
    matrix &operator = (const matrix &target) {}
    
    matrix &operator = (matrix &&target) {}
    
    matrix &operator + (const matrix &target) {}
    
    matrix &operator *= (const matrix &target) {}
    
    matrix &operator * (const matrix &target) {}
    
    matrix &operator * (const int &scalar) {}
    
    void transpose() {}
    
    matrix &hadamard_multiplication(const matrix &target) {}
    
    matrix &kronecker_multiplication(const matrix &target) {}
    
    matrix &horizontal_concatenation(const matrix &target) {}
    
    ~matrix() {}
    
private:
    std::vector<std::vector<T>> container;
};

