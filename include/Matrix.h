// Copyright 2019 Ciobanu Bogdan-Calin
// Copyright 2019 Rica Radu Leonard

#pragma once

#include <iostream>
#include <vector>

template <class T>
class Matrix {
   public:
    Matrix() {}

    Matrix(const std::vector<std::vector<T>> &target) {
        container = target;
        numRows = target.size();
        numColumns = target[0].size();
    }

    Matrix(std::vector<std::vector<T>> &&target) {
        numRows = target.size();
        numColumns = target[0].size();
        container = std::move(target);
    }

    Matrix(int numRows, int numColumns)
        : numRows(numRows), numColumns(numColumns) {
        container =
            std::vector<std::vector<T>>(numRows, std::vector<T>(numColumns));
    }

    Matrix(const Matrix &target) { *this = target; }

    Matrix(Matrix &&target) { *this = target; }

    Matrix &operator=(const Matrix &target) {
        numColumns = target.numColumns;
        numRows = target.numRows;
        container = target.container;

        return *this;
    }

    Matrix &operator=(Matrix &&target) {
        numColumns = target.numColumns;
        numRows = target.numRows;
        container = target.container;

        return *this;
    }

    Matrix &operator+(auto target) {
        static Matrix<T> newMatrix(numRows, numColumns);
        for (size_t i = 0; i < numRows; i++) {
            for (size_t j = 0; j < numColumns; j++) {
                newMatrix.data(i, j) = data(i, j) + target.data(i, j);
            }
        }
        return newMatrix;
    }

    void operator+=(const Matrix &target) { *this = (*this) + target; }

    Matrix &operator*(const Matrix &target) {
        static Matrix<T> newMatrix(numRows, numColumns);
        for (size_t i = 0; i < numRows; i++) {
            for (size_t j = 0; j < target.numColumns; j++) {
                T sum = 0;
                for (size_t k = 0; k < numColumns; ++k) {
                    sum += container[i][k] * target.container[k][j];
                }
                newMatrix.container[i][j] = sum;
            }
        }
        return newMatrix;
    }

    void operator*=(const Matrix &target) { return (*this) * target; }

    Matrix &operator*(const T &target) {
        static Matrix<T> newMatrix(numRows, numColumns);
        for (size_t i = 0; i < numRows; i++) {
            for (size_t j = 0; j < numColumns; j++) {
                newMatrix.container[i][j] = container[i][j] * target;
            }
        }
        return newMatrix;
    }

    void operator*=(const T &target) { return (*this) * target; }

    Matrix &operator-(auto target) {
        static Matrix<T> newMatrix(numRows, numColumns);
        for (size_t i = 0; i < numRows; i++) {
            for (size_t j = 0; j < numColumns; j++) {
                newMatrix.data(i, j) = data(i, j) - target.data(i, j);
            }
        }
        return newMatrix;
    }

    void operator-=(const Matrix &target) { *this = (*this) - target; }

    Matrix &Transpose() const {
        static Matrix<T> newMatrix(numRows, numColumns);
        for (size_t i = 0; i < numRows; i++) {
            for (size_t j = 0; j < numColumns; j++) {
                newMatrix.container[i][j] = container[j][i];
            }
        }
        return newMatrix;
    }

    Matrix &HadamardMultiply(const Matrix &target) {
        static Matrix<T> newMatrix(numRows, numColumns);
        for (size_t i = 0; i < numRows; i++) {
            for (size_t j = 0; j < numColumns; j++) {
                newMatrix.container[i][j] =
                    container[i][j] * target.container[i][j];
            }
        }
        return newMatrix;
    }

    T &data(int row, int col) { return container[row][col]; }

    std::pair<int, int> size() { return {numRows, numColumns}; }

    ~Matrix() {}

    template <class F>
    void applyFunction() {
        F function;
        for (auto &row : container) {
            for (auto &column : row) {
                column = function(column);
            }
        }
    }

   private:
    std::vector<std::vector<T>> container;
    size_t numRows;
    size_t numColumns;
};
