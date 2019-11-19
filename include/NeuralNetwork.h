// Copyright 2019 Ciobanu Bogdan-Calin
// Copyright 2019 Rica Radu Leonard
#pragma once

#include <iostream>
#include <vector>
#include "../include/ActivationFunction.h"
#include "../include/Diagnostics.h"
#include "../include/Matrix.h"
#include "../include/Utils.h"

using matrix = Matrix<long double>;
using layers = std::vector<matrix>;

template <class ActivationFunction, class Backpropagator>
class NeuralNetwork {
   public:
    NeuralNetwork(const std::vector<int> &sizes, long double learningRate)
        : learningRate(learningRate) {
        layersCount = sizes.size();
        activations = layers(layersCount);
        weights = layers(layersCount - 1);
        biases = layers(layersCount - 1);

        for (int i = 0; i < layersCount - 1; i++) {
            weights[i] = matrix(sizes[i], sizes[i + 1]);
            biases[i] = matrix(1, sizes[i + 1]);
        }
    };

    void Train(const std::vector<long double> &input, int correctValue) {
        /* OLD VERSION
        Matrix<long double> desiredOutput(outputNeuronCount, 1, 0.0);

        forwardPropagate(input);
        desiredOutput.data(correctValue, 0) = 1.0;

        for (auto it : neuralLayers)
            weights.push_back(it.weights), biases.push_back(it.bias),
            activations.push_back(it.activations);

        auto backpropagator = Backpropagator();
        backpropagator.backpropagate(weights, biases, activations,
            neuralNetworkSize, desiredOutput,
            learningRate);

        for (auto it = neuralLayers.begin(); it != neuralLayers.end(); ++it)
            it->weights = weights[it - neuralLayers.begin()],
            it->bias = biases[it - neuralLayers.begin()];
        */
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
            activations[i] =
                (activations[i - 1] * weights[i - 1] + biases[i - 1]);
            activations[i] = activations[i].applyFunction<ActivationFunction>();
        }
    }

    long double learningRate;
    layers weights;
    layers biases;
    layers activations;
    size_t layersCount;
};
