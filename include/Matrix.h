// Copyright 2019 Ciobanu Bogdan-Calin
// Copyright 2019 Rica Radu Leonard

#pragma once
#include <ctype.h>

class Matrix {
   public:
    Matrix() {}

    Matrix(unsigned numRows, unsigned numColumns)
        : numRows(numRows), numColumns(numColumns) {
        container = std::vector<std::vector<long double>>(
            numRows, std::vector<long double>(numColumns));
    }

    Matrix(const std::vector<std::vector<long double>> &container) {
        numRows = container.size();
        numColumns = container[0].size();
        this->container = container;
    }

    Matrix(const Matrix &other) {
        this->numRows = other.numRows;
        this->numColumns = other.numColumns;
        this->container = other.container;
    }

    ~Matrix() {}

    // Matrix Operations
    Matrix &operator=(const Matrix &other) {
        this->numRows = other.numRows;
        this->numColumns = other.numColumns;
        this->container = other.container;
        return *this;
    }

    Matrix operator+(const Matrix &other) const {
        Matrix newMatrix(numRows, numColumns);
        for (size_t i = 0; i < numRows; ++i) {
            for (size_t j = 0; j < numColumns; ++j) {
                newMatrix.container[i][j] =
                    this->container[i][j] + other.container[i][j];
            }
        }
        return newMatrix;
    }

    Matrix &operator+=(const Matrix &other) {
        *this = (*this) + other;
        return *this;
    }

    Matrix operator-(const Matrix &other) const {
        Matrix newMatrix(numRows, numColumns);
        for (size_t i = 0; i < numRows; ++i) {
            for (size_t j = 0; j < numColumns; ++j) {
                newMatrix.container[i][j] =
                    this->container[i][j] - other.container[i][j];
            }
        }
        return newMatrix;
    }

    Matrix &operator-=(const Matrix &other) {
        *this = (*this) - other;
        return *this;
    }

    Matrix operator*(const Matrix &other) const {
        Matrix newMatrix(numRows, numColumns);
        for (size_t i = 0; i < numRows; i++) {
            for (size_t j = 0; j < other.numColumns; j++) {
                long double sum = 0;
                for (size_t k = 0; k < numColumns; ++k) {
                    sum += container[i][k] * other.container[k][j];
                }
                newMatrix.container[i][j] = sum;
            }
        }
        return newMatrix;
    }

    Matrix &operator*=(const Matrix &other) {
        *this = (*this) * other;
        return *this;
    }

    Matrix Transpose() const {
        Matrix newMatrix(numRows, numColumns);
        for (size_t i = 0; i < numRows; ++i) {
            for (size_t j = 0; j < numColumns; ++j) {
                newMatrix.container[i][j] = this->container[j][i];
            }
        }
        return newMatrix;
    }

    Matrix HadamardMultiply(const Matrix &other) const {
        Matrix newMatrix(numRows, numColumns);
        for (size_t i = 0; i < numRows; ++i) {
            for (size_t j = 0; j < numColumns; ++j) {
                newMatrix.container[i][j] =
                    this->container[i][j] * other.container[i][j];
            }
        }
        return newMatrix;
    }

    template <class Func>
    Matrix ApplyFunction() const {
        Func function;
        Matrix newMatrix(numRows, numColumns);
        for (size_t i = 0; i < numRows; ++i) {
            for (size_t j = 0; j < numColumns; ++j) {
                newMatrix.container[i][j] = function(this->container[j][i]);
            }
        }
        return newMatrix;
    }

    // Scalar Operations
    Matrix operator*(const long double &other) const {
        Matrix newMatrix(numRows, numColumns);
        for (size_t i = 0; i < numRows; ++i) {
            for (size_t j = 0; j < numColumns; ++j) {
                newMatrix.container[i][j] = this->container[i][j] * other;
            }
        }
        return newMatrix;
    }

    Matrix &operator*=(const long double &other) {
        *this = (*this) * other;
        return *this;
    }

    auto GetContainer() const {
        return container;
    }

    long double &data(int row, int col) { return container[row][col]; }

    std::vector<std::vector<long double>> container;
    size_t numRows;
    size_t numColumns;
};
