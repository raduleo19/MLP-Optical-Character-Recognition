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
#ifdef NNDIAG
    friend class NeuralDiagnostics;
#endif  // NNDIAG

   public:
    NeuralNetwork(int _inputNeuronCount, int _hiddenLayersCount,
                  std::vector<int> _hiddenLayersSizes, int _outputNeuronCount,
                  double _learningRate) {
        int hiddenLayersCount;
        std::vector<int> hiddenLayersSizes;

        inputNeuronCount = _inputNeuronCount;
        hiddenLayersCount = _hiddenLayersCount;
        hiddenLayersSizes = _hiddenLayersSizes;
        outputNeuronCount = _outputNeuronCount;
        learningRate = _learningRate;
        neuralNetworkSize = 2 + hiddenLayersCount;

        neuralLayers.push_back(Layer(inputNeuronCount));
        if (hiddenLayersCount) {
            neuralLayers.push_back(
                Layer(inputNeuronCount, hiddenLayersSizes[0]));
            for (int i = 1; i < hiddenLayersCount; ++i)
                neuralLayers.push_back(
                    Layer(hiddenLayersSizes[i - 1], hiddenLayersSizes[i]));
            neuralLayers.push_back(
                Layer(hiddenLayersSizes.back(), outputNeuronCount));
        } else {
            neuralLayers.push_back(Layer(inputNeuronCount, outputNeuronCount));
        }

        setRandomStartingPoint();
    };

    void Train(const std::vector<long double> &input, int correctValue) {
        Matrix<long double> desiredOutput(outputNeuronCount, 1, 0.0);
        std::vector<Matrix<long double>> weights;
        std::vector<Matrix<long double>> biases;
        std::vector<Matrix<long double>> activations;

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
    };

    int Classify(const std::vector<long double> &input) {
        long double max = -1.0;
        int retval = -1;

        forwardPropagate(input);

        for (int i = 0; i < outputNeuronCount; ++i)
            if (neuralLayers.back().activations.data(i, 0) > max)
                max = neuralLayers.back().activations.data(i, 0), retval = i;

        return retval;
    };

   protected:
    class Layer {
        friend class NeuralNetwork<ActivationFunction, Backpropagator>;

       protected:
        Matrix<long double> activations, bias, weights;

        Layer() {}

        Layer(const size_t &size) {
            activations = Matrix<long double>(size, 1);
        }

        Layer(const size_t &previousLayerSize, const size_t &size) {
            activations = Matrix<long double>(size, 1);
            bias = Matrix<long double>(size, 1);
            weights = Matrix<long double>(size, previousLayerSize);
        }

        Layer &operator=(const Layer &target) {
            activations = target.activations;
            bias = target.bias;
            weights = target.weights;

            return *this;
        }
    };

    void forwardPropagate(const std::vector<long double> &input) {
        std::vector<std::vector<long double>> container;
        for (auto it : input) {
            container.push_back({it});
        }
        neuralLayers.front().activations =
            Matrix<long double>(container).ApplyFunction<ActivationFunction>();

        auto sigma = [=](Matrix<long double> target) {
            return target.ApplyFunction<ActivationFunction>();
        };

        for (auto it = neuralLayers.begin() + 1; it != neuralLayers.end(); ++it)
            it -> activations = sigma(it -> weights * (it - 1) -> activations + it -> bias);
    }

    void setRandomStartingPoint() {
        randomEngine randomizer;

        auto randomizeMatrix = [&randomizer](Matrix<long double> &target) {
            auto container = target.GetContainer();
            for (auto &row : container) {
                for (auto &column : row) {
                    column = randomizer.getNumber();
                }
            }
            return Matrix<long double>(container);
        };

        for (auto it = neuralLayers.begin() + 1; it != neuralLayers.end(); ++it)
            randomizeMatrix(it->weights), randomizeMatrix(it->bias);
    }

    int inputNeuronCount, outputNeuronCount;
    double learningRate;
    std::vector<Layer> neuralLayers;
    size_t neuralNetworkSize;
};
