// Copyright 2019 Ciobanu Bogdan-Calin
// Copyright 2019 Rica Radu Leonard

#pragma once
#include <cassert>
#include <vector>
#include "../include/Utils.h"

//#define UNSAFE_MODE

template <class T>
class Matrix {
   public:
    Matrix() {}

    Matrix(unsigned numRows, unsigned numColumns)
        : numRows(numRows), numColumns(numColumns) {
        container =
            std::vector<std::vector<T>>(numRows, std::vector<T>(numColumns));
        RandomEngine randomizer;
        for (auto &row : container) {
            for (auto &column : row) {
                column = (T)randomizer.getNumber();
            }
        }
    }

    Matrix(unsigned numRows, unsigned numColumns, T value)
        : numRows(numRows), numColumns(numColumns) {
        container = std::vector<std::vector<T>>(
            numRows, std::vector<T>(numColumns, value));
    }

    Matrix(const std::vector<std::vector<T>> &container) {
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
#ifndef UNSAFE_MODE
        assert(numRows == other.numRows && numColumns == other.numColumns);
#endif
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
#ifndef UNSAFE_MODE
        assert(numRows == other.numRows && numColumns == other.numColumns);
#endif
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
#ifndef UNSAFE_MODE
        assert(numColumns == other.numRows);
#endif
        Matrix newMatrix(numRows, other.numColumns);
        for (size_t i = 0; i < numRows; i++) {
            for (size_t j = 0; j < other.numColumns; j++) {
                T sum = 0;
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

    Matrix transpose() const {
        Matrix newMatrix(numColumns, numRows);
        for (size_t i = 0; i < numRows; ++i) {
            for (size_t j = 0; j < numColumns; ++j) {
                newMatrix.container[j][i] = this->container[i][j];
            }
        }
        return newMatrix;
    }

    Matrix hadamardMultiply(const Matrix &other) const {
#ifndef UNSAFE_MODE
        assert(numRows == other.numRows && numColumns == other.numColumns);
#endif
        Matrix newMatrix(numRows, numColumns);
        for (size_t i = 0; i < numRows; ++i) {
            for (size_t j = 0; j < numColumns; ++j) {
                newMatrix.container[i][j] =
                    this->container[i][j] * other.container[i][j];
            }
        }
        return newMatrix;
    }

    // Function Operations
    template <class Func>
    Matrix applyFunction() const {
        Func function;
        Matrix newMatrix(numRows, numColumns);
        for (size_t i = 0; i < numRows; ++i) {
            for (size_t j = 0; j < numColumns; ++j) {
                newMatrix.container[i][j] = function(this->container[i][j]);
            }
        }
        return newMatrix;
    }

    // Scalar Operations
    Matrix operator*(const T &other) const {
        Matrix newMatrix(numRows, numColumns);
        for (size_t i = 0; i < numRows; ++i) {
            for (size_t j = 0; j < numColumns; ++j) {
                newMatrix.container[i][j] = this->container[i][j] * other;
            }
        }
        return newMatrix;
    }

    Matrix &operator*=(const T &other) {
        *this = (*this) * other;
        return *this;
    }

    std::pair<size_t, size_t> size() {
        return {numRows, numColumns};
    }

    // Export Operations
    auto GetContainer() const { return container; }

    // Elements access
    T &data(int row, int col) { return container[row][col]; }

    std::vector<std::vector<T>> container;
    size_t numRows;
    size_t numColumns;
};
