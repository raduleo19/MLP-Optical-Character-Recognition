// Copyright 2019 Ciobanu Bogdan-Calin
// Copyright 2019 Rica Radu Leonard

#pragma once

#include <vector>

template <class T>
class Matrix {
   public:
    Matrix() {}

    Matrix(int numRows, int numColumns) {
        container =
            std::vector<std::vector<T>>(numRows, std::vector<T>(numColumns));
    }

    Matrix(const Matrix &target) {}

    Matrix(Matrix &&target) {}

    Matrix &operator=(const Matrix &target) {}

    Matrix &operator=(Matrix &&target) {}

    Matrix &operator+(const Matrix &target) {}

    Matrix &operator*=(const Matrix &target) {}

    Matrix &operator*(const Matrix &target) {}

    Matrix &operator*(const int &scalar) {}

    Matrix &Transpose() const {}

    Matrix &HadamardMultiplication(const Matrix &target) const {}

    Matrix &KroneckerMultiplication(const Matrix &target) const {}

    Matrix &HorizontalConcatenation(const Matrix &target) const {}

    ~Matrix() {}

   private:
    std::vector<std::vector<T>> container;
};
