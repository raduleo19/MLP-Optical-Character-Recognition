// Copyright 2019 Ciobanu Bogdan-Calin
// Copyright 2019 Rica Radu Leonard
#pragma once

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
        neuronCount = sizes.size();
        activations = std::vector<Matrix<long double>>(neuronCount);
        weights = std::vector<Matrix<long double>>(neuronCount - 1);
        biases = std::vector<Matrix<long double>>(neuronCount - 1);

        for (int i = 0; i < sizes.size() - 1; i++) {
            weights[i] = Matrix<long double>(sizes[i], sizes[i + 1]);
            biases[i] = Matrix<long double>(1, sizes[i + 1]);
        }
    };

    void Train(const std::vector<long double> &input, int correctValue){};

    int Classify(const std::vector<long double> &input){};

    void ForwardPropagate(const std::vector<long double> &input) {}

    long double learningRate;
    std::vector<Matrix<long double>> weights;
    std::vector<Matrix<long double>> biases;
    std::vector<Matrix<long double>> activations;
    size_t neuronCount;
};
