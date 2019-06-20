// Copyright 2019 Rica Radu Leonard
// Copyright 2019 Ciobanu Bogdan-Calin
#pragma once

#include <vector>
#include "../include/ActivationFunction.h"

class Backpropagate {
   public:
    void backpropagate(std::vector<Matrix<long double>> &weights,
                       std::vector<Matrix<long double>> &biases,
                       std::vector<Matrix<long double>> &activations,
                       Matrix<long double> desiredOutput, double learningRate) {
        // TODO: Solve simple matrix system
        size_t layersCount = weights.size();

        std::vector<Matrix<long double>> dCdW(layersCount - 1);
        std::vector<Matrix<long double>> dCdB(layersCount - 1);
        dCdB[layersCount - 2] =
            (activations[layersCount - 1] - desiredOutput)
                .HadamardMultiply(activations[layersCount - 2] *
                                      weights[layersCount - 2] +
                                  biases[layersCount - 2])
                .applyFunction<DerivativeActivationFunction>();
    };
};
