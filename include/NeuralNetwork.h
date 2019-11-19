// Copyright 2019 Ciobanu Bogdan-Calin
// Copyright 2019 Rica Radu Leonard
#pragma once

#include <vector>
#include "../include/ActivationFunction.h"
#include "../include/Diagnostics.h"
#include "../include/Matrix.h"
#include "../include/Utils.h"

using matrix = Matrix<long double>;
using layers = std::vector<matrix>;
using std::cout;
using std::endl;

template <class ActivationFunction, class CorrectionFunction>
class NeuralNetwork {
   public:
    NeuralNetwork(const std::vector<int> &sizes, long double learningRate)
        : learningRate(learningRate) {
        layersCount = sizes.size();
        activations = layers(layersCount);
        weights = layers(layersCount - 1);
        biases = layers(layersCount - 1);

        for (int i = 0; i < layersCount - 1; ++i) {
            activations[i] = matrix(1, sizes[i]);
            weights[i] = matrix(sizes[i], sizes[i + 1]);
            biases[i] = matrix(1, sizes[i + 1]);
        }

        activations[layersCount - 1] = matrix(1, sizes[layersCount - 1]);

        cout << "Initialization complete" << endl;
    };

    void train(const std::vector<long double> &input, const int correctValue) {
        matrix desiredOutput(1, 10, 0.0);
        static auto coordinator = CorrectionFunction();
        
        forwardPropagate(input);
        desiredOutput.data(0, correctValue) = 1.0;

        coordinator.takeStep(weights, biases, activations, layersCount, desiredOutput, learningRate);
    };

    int classify(const std::vector<long double> &input) {
        forwardPropagate(input);
        long double best = 0;

        auto results = activations[layersCount - 1].container.front();
        for (int i = 1; i < results.size(); ++i) {
            if (results[i] > results[best]) {
                best = i;
            }
        }

        return best;
    };

    void forwardPropagate(const std::vector<long double> &input) {
        activations[0] = matrix({input});

        for (int i = 1; i < layersCount; ++i) {
            activations[i] = activations[i - 1] * weights[i - 1] + biases[i - 1];
            activations[i] = activations[i].applyFunction<ActivationFunction>();  // e complet contraproductiv, putea fi in linia de mai sus aplicat pe rvalue
        }
    }

    long double learningRate;
    layers weights;
    layers biases;
    layers activations;
    size_t layersCount;
};
