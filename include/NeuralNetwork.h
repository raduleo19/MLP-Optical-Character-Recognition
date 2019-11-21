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

template <class ActivationFunction, class Optimizer>
class NeuralNetwork {
public:
    NeuralNetwork(const std::vector<int>& sizes, long double learningRate) :
        learningRate(learningRate) {
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

    void train(const std::vector<long double>& input, const int correctValue) {
        matrix desiredOutput(activations[layersCount - 1].size().first, activations[layersCount - 1].size().second, 0.0);
        static auto coordinator = Optimizer();

        forwardPropagate(input);
        desiredOutput.container[0][correctValue] = 1.0;

        auto costFunction = [&, this]() -> double {
            double retval = 0;
            auto temp = activations[layersCount - 1] - desiredOutput;
            temp = temp.hadamardMultiply(temp);
            for (int i = 0; i < 10; ++i)
                retval += temp.container[0][i];
            return retval;
        };

        cout << "Cost function = " << costFunction() << endl;

        coordinator.takeStep(weights, biases, activations, layersCount, desiredOutput, learningRate);
    };

    int classify(const std::vector<long double>& input) {
        forwardPropagate(input);
        long double best = 0;

        auto results = activations[layersCount - 1].container.front();
        for (int i = 0; i < results.size(); ++i) {
            if (results[i] > results[best]) {
                best = i;
            }
        }

        return best;
    };

    void forwardPropagate(const std::vector<long double>& input) {
        activations[0] = matrix({ input });

        for (int i = 1; i < layersCount; ++i) {
            activations[i] = (activations[i - 1] * weights[i - 1] + biases[i - 1]).applyFunction<ActivationFunction>();
        }
    }

    long double learningRate;
    layers weights, biases, activations;
    size_t layersCount;
};
