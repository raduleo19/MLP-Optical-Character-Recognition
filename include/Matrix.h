// Copyright 2019 Ciobanu Bogdan-Calin
// Copyright 2019 Rica Radu Leonard

#pragma once

#include <vector>

template <class T>
class Matrix {
   public:
    Matrix() {}

    Matrix(int numRows, int numColumns) {
        container.resize(numRows, numColumns);
    }

    Matrix(int numRows, int numColumns, int value)
        : Matrix(numRows, numColumns) {
        container.resize(numRows, numColumns);
        container.fill(value);
    }

    Matrix(const Matrix &target) {}

    Matrix(Matrix &&target) {}

    Matrix &operator=(const Matrix &target) {}

    Matrix &operator=(Matrix &&target) {}

    Matrix &operator+(const Matrix &target) {}

    Matrix &operator*=(const Matrix &target) {}

    Matrix &operator*(const Matrix &target) {}

    Matrix &operator*(const int &scalar) {}

    Matrix &transpose() const {}

    Matrix &hadamard_multiplication(const Matrix &target) const {}

    Matrix &kronecker_multiplication(const Matrix &target) const {}

    Matrix &horizontal_concatenation(const Matrix &target) const {}

    ~Matrix() {}

   private:
    std::vector<std::vector<T>> container;
};
