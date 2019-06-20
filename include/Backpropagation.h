// Copyright 2019 Rica Radu Leonard
// Copyright 2019 Ciobanu Bogdan-Calin
#pragma once

#include <vector>
#include "../include/Matrix.h"

// Gradient Descent Backpropagate
template <class Derivative>
class Backpropagate {
   public:
    void backpropagate(std::vector<Matrix> &weights,
                       std::vector<Matrix> &biases,
                       std::vector<Matrix> &activations,
                       const size_t &layersCount,
                       const Matrix &desiredOutput,
                       const long double &learningRate) {
        std::vector<Matrix> dCdW(layersCount - 1);
        std::vector<Matrix> dCdB(layersCount - 1);

        dCdB[layersCount - 2] =
            (activations[layersCount - 1] - desiredOutput)
                .HadamardMultiply(activations[layersCount - 2] *
                                      weights[layersCount - 2] +
                                  biases[layersCount - 2])
                .ApplyFunction<Derivative>();

        for (int i = layersCount - 3; i >= 0; i--) {
            auto temp = activations[i] * weights[i] +
                        biases[i].ApplyFunction<Derivative>();
            dCdB[i] = (dCdB[i + 1] * weights[i + 1].Transpose())
                          .HadamardMultiply(temp);
        }

        for (unsigned i = 0; i < layersCount - 1; i++) {
            dCdW[i] = activations[i].Transpose() * dCdB[i];
            weights[i] = weights[i] - (dCdW[i] * learningRate);
            biases[i] = biases[i] - (dCdB[i] * learningRate);
        }
    };
};
