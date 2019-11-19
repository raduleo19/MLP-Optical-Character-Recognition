// Copyright 2019 Rica Radu Leonard
// Copyright 2019 Ciobanu Bogdan-Calin
#pragma once

#include <vector>
#include "../include/Matrix.h"

// Gradient Descent Backpropagate
template <class Derivative>
class Backpropagate {
   public:
    void takeStep(std::vector<Matrix<long double>> &weights,
                       std::vector<Matrix<long double>> &biases,
                       std::vector<Matrix<long double>> &activations,
                       const size_t &layersCount,
                       Matrix<long double> &desiredOutput,
                       const long double &learningRate) {
        /*
        std::vector<Matrix<long double>> dCdW(layersCount);
        std::vector<Matrix<long double>> dCdB(layersCount);

        dCdB[layersCount - 1] = (activations[layersCount - 1] - desiredOutput).hadamardMultiply(weights[layersCount - 1] * activations[layersCount - 2] + biases[layersCount - 1]).applyFunction<Derivative>();
        return;
        for (int i = layersCount - 2; i > 0; --i) {
            auto temp = (weights[i] * activations[i - 1] + biases[i]).applyFunction<Derivative>();
            dCdB[i] = (weights[i + 1].transpose() * dCdB[i + 1]).hadamardMultiply(temp);
        }

        for (unsigned i = 1; i < layersCount; ++i) {
            dCdW[i] = dCdB[i] * activations[i - 1].transpose();
            weights[i] = weights[i] - (dCdW[i] * learningRate);
            biases[i] = biases[i] - (dCdB[i] * learningRate);
        }
        */
    };
};
