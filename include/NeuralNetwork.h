// Copyright 2019 Ciobanu Bogdan-Calin
// Copyright 2019 Rica Radu Leonard
#pragma once

#include <iostream>
#include <vector>
#include "../include/ActivationFunction.h"
#include "../include/Diagnostics.h"
#include "../include/Matrix.h"
#include "../include/Utils.h"

template <class ActivationFunction, class Backpropagator>
class NeuralNetwork {
   public:
    NeuralNetwork(const std::vector<int> &sizes, long double learningRate)
        : learningRate(learningRate) {
        layersCount = sizes.size();
        activations = std::vector<Matrix<long double>>(layersCount);
        weights = std::vector<Matrix<long double>>(layersCount - 1);
        biases = std::vector<Matrix<long double>>(layersCount - 1);

        for (int i = 0; i < layersCount - 1; i++) {
            weights[i] = Matrix<long double>(sizes[i], sizes[i + 1]);
            biases[i] = Matrix<long double>(1, sizes[i + 1]);
        }
    };

    void Train(const std::vector<long double> &input, int correctValue){};

    int Classify(const std::vector<long double> &input) {
        ForwardPropagate(input);
        long double best = 0;

        auto results = activations[layersCount - 1].container.front();
        for (int i = 1; i < results.size(); ++i) {
            if (results[i] > results[best]) {
                best = i;
            }
        }

        return best;
    };

    void ForwardPropagate(const std::vector<long double> &input) {
        activations[0] = Matrix<long double>({input});

        for (int i = 1; i < layersCount; ++i) {
            activations[i] =
                (activations[i - 1] * weights[i - 1] + biases[i - 1]);
            activations[i] = activations[i].ApplyFunction<ActivationFunction>();
        }
    }

    long double learningRate;
    std::vector<Matrix<long double>> weights;
    std::vector<Matrix<long double>> biases;
    std::vector<Matrix<long double>> activations;
    size_t layersCount;
};
