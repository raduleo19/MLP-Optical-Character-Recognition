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
        std::vector<Matrix<long double>> dCdW(layersCount);
        std::vector<Matrix<long double>> dCdB(layersCount);
        static std::ofstream t ("/home/administrator/projects/OpticalCharacterRecognition/MLP-Optical-Character-Recognition/logs.txt");
        
        t << "\nNew gen\n\n";
        for (auto i : weights[3].container) {
            for (auto j : i)
                t << j << " ";
            t << std::endl;
        }

        dCdB[layersCount - 1] =
            (activations[layersCount - 1] - desiredOutput).HadamardMultiply(weights[layersCount - 1] * activations[layersCount - 2] + biases[layersCount - 1]).ApplyFunction<Derivative>();

        for (int i = layersCount - 2; i > 0; --i) {
            auto temp = (weights[i] * activations[i - 1] + biases[i]).ApplyFunction<Derivative>();
            dCdB[i] = (weights[i + 1].Transpose() * dCdB[i + 1]).HadamardMultiply(temp);
        }

        for (unsigned i = 1; i < layersCount; ++i) {
            dCdW[i] = dCdB[i] * activations[i - 1].Transpose();
            weights[i] = weights[i] - (dCdW[i] * learningRate);
            biases[i] = biases[i] - (dCdB[i] * learningRate);
        }

        for (auto i : weights[3].container) {
            for (auto j : i)
                t << j << " ";
            t << std::endl;
        }

    };
};
