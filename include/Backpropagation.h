// Copyright 2019 Rica Radu Leonard
// Copyright 2019 Ciobanu Bogdan-Calin
#pragma once

#include <vector>
#include "../include/ActivationFunction.h"

// Gradient Descent Backpropagate
class Backpropagate {
   public:
    void backpropagate(std::vector<Matrix> &weights,
                       std::vector<Matrix> &biases,
                       std::vector<Matrix> &activations,
                       Matrix desiredOutput,
                       long double learningRate) {

        size_t layersCount = weights.size();

        std::vector<Matrix> dCdW(layersCount - 1);
        std::vector<Matrix> dCdB(layersCount - 1);
        dCdB[layersCount - 2] =
            (activations[layersCount - 1] - desiredOutput)
                .HadamardMultiply(activations[layersCount - 2] *
                                      weights[layersCount - 2] +
                                  biases[layersCount - 2]).ApplyFunction<DerivativeActivationFunction>();

        for (int i = layersCount - 3; i >= 0; i--) {
            auto temp = activations[i] * weights[i] + biases[i].ApplyFunction<DerivativeActivationFunction>();
            dCdB[i] = (dCdB[i + 1] * weights[i + 1].Transpose())
                          .HadamardMultiply(temp);
        }

        for (int i = 0; i < layersCount - 1; i++) {
            dCdW[i] = activations[i].Transpose() * dCdB[i];
            weights[i] = weights[i] - (dCdW[i] * learningRate);
            biases[i] = biases[i] - (dCdB[i] * learningRate);
        }
    };
};
