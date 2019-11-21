// Copyright 2019 Rica Radu Leonard
// Copyright 2019 Ciobanu Bogdan-Calin

#pragma once

#include <vector>
#include "../include/Matrix.h"

using std::cout;
using std::endl;

// Gradient Descent Backpropagate
template <class Derivative>
class Backpropagate {
 public:
    void takeStep(std::vector<Matrix<long double>> &weights,
                       std::vector<Matrix<long double>> &biases,
                       std::vector<Matrix<long double>> &activations,
                       const size_t &layersCount,
                       Matrix<long double> &desiredOutput,
                       const long double &learningRate) { // REWORK
        std::vector<Matrix<long double>> dCdW(layersCount - 1);
        std::vector<Matrix<long double>> dCdB(layersCount - 1);

        dCdB[layersCount - 2] = (activations[layersCount - 1] - desiredOutput).hadamardMultiply
                                (activations[layersCount - 2] * weights[layersCount - 2]  + biases[layersCount - 2]).applyFunction<Derivative>();

        for (int i = layersCount - 2; i > 0; --i) {
            auto temp = (activations[i - 1] * weights[i - 1] + biases[i - 1]).applyFunction<Derivative>();
            dCdB[i - 1] = (dCdB[i] * weights[i].transpose()).hadamardMultiply(temp);
        }

        for (unsigned i = 0; i < layersCount - 1; ++i) {
            dCdW[i] = activations[i].transpose() * dCdB[i];
            weights[i] = weights[i] - (dCdW[i] * learningRate);
            biases[i] = biases[i] - (dCdB[i] * learningRate);
        }
    };
};
