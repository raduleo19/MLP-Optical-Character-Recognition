// Copyright 2019 Rica Radu Leonard
// Copyright 2019 Ciobanu Bogdan-Calin
#pragma once

#include <vector>
#include "../include/Matrix.h"

// Gradient Descent Backpropagate
template <class Derivative>
class Backpropagate {
   public:
    void backpropagate(std::vector<Matrix<long double>> &weights,
                       std::vector<Matrix<long double>> &biases,
                       std::vector<Matrix<long double>> &activations,
                       const size_t &layersCount,
                       const Matrix<long double> &desiredOutput,
                       const long double &learningRate) {
    

    };
};
