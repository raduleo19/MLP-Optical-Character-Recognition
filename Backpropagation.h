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
        std::vector<Matrix<long double>> dCdW(layersCount - 1);
        std::vector<Matrix<long double>> dCdB(layersCount - 1);

         dCdB[layersCount - 2] =
             (activations[layersCount - 1] - desiredOutput)
                 .HadamardMultiply(weights[layersCount - 1] *
                                   activations[layersCount - 2] +
                                   biases[layersCount - 1])
                 .ApplyFunction<Derivative>();

         for (int i = layersCount - 3; i >  0; i--) {
             auto temp = weights[i] * activations[i - 1] + biases[i].ApplyFunction<Derivative>();
             dCdB[i] = (weights[i + 2].Transpose() * dCdB[i + 1]).HadamardMultiply(temp);
         }
exit(1);
         for (unsigned i = 1; i < layersCount - 1; i++) {
             dCdW[i] = activations[i].Transpose() * dCdB[i];
             weights[i] = weights[i] - (dCdW[i] * learningRate);
             biases[i] = biases[i] - (dCdB[i] * learningRate);
         }
    };
};
