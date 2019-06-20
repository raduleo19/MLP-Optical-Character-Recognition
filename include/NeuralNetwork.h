// Copyright 2019 Ciobanu Bogdan-Calin
// Copyright 2019 Rica Radu Leonard
#pragma once

#include <vector>
#include "../include/ActivationFunction.h"
#include "../include/Diagnostics.h"
#include "../include/Matrix.h"
#include "../include/Utils.h"

template <class T, class Sigmoid, class Backpropagator>
class NeuralNetwork {
#ifdef NNDIAG
    friend class NeuralDiagnostics;
#endif  // NNDIAG

   public:
    NeuralNetwork(int _inputNeuronCount, int _hiddenLayersCount,
                  std::vector<int> _hiddenLayersSizes, int _outputNeuronCount,
                  double _learningRate) {
        inputNeuronCount = _inputNeuronCount;
        hiddenLayersCount = _hiddenLayersCount;
        hiddenLayersSizes = std::move(_hiddenLayersSizes);
        outputNeuronCount = _outputNeuronCount;
        learningRate = _learningRate;
        neuralNetworkSize = 2 + hiddenLayersCount;

        neuralLayers.push_back(Layer(inputNeuronCount));
        if (hiddenLayersCount) {
            neuralLayers.push_back(Layer(inputNeuronCount, hiddenLayersSizes[0]));
            for (int i = 1; i < hiddenLayersCount; ++i)
                neuralLayers.push_back(Layer(hiddenLayersSizes[i - 1], hiddenLayersSizes[i]));
            neuralLayers.push_back(Layer(hiddenLayersSizes.back(), outputNeuronCount));
        } else {
            neuralLayers.push_back(Layer(inputNeuronCount, outputNeuronCount));
        }

        setRandomStartingPoint();
    };

    void Train(const std::vector<int> &input, int correctValue) {
        Matrix<long double> desiredOutput(outputNeuronCount, 1);
        std::vector<Matrix<long double>> weights;
        std::vector<Matrix<long double>> biases;
        std::vector<Matrix<long double>> activations;
        
        forwardPropagate(input);
        
        for (int i = 0; i < outputNeuronCount; ++i)
            desiredOutput.data(i, 1) = 0.0;
        desiredOutput.data(correctValue, 1) = 1.0;

        for (auto i : neuralLayers)
            weights.push_back(i.weights),
            biases.push_back(i.bias),
            activations.push_back(i.activations);
        
        auto backpropagator = Backpropagator();
        backpropagator.backpropagate(weights, biases, activations,
                                     desiredOutput, learningRate);
        
        for (auto it = neuralLayers.begin(); it != neuralLayers.end(); ++it)
            it -> weights = weights[it - neuralLayers.begin()],
            it -> bias = biases[it - neuralLayers.begin()];
    };

    int Classify(const std::vector<T> &input) {
        Layer outputLayer = neuralLayers.back();
        long double max = -1.0;
        int retval = -1;

        forwardPropagate(input);

        for (int i = 0; i < outputNeuronCount; ++i)
            if (outputLayer.activations.data(i, 1) > max)
                max = outputLayer.activations.data(i, 1), retval = i;

        return retval;
    };

   protected:
    class Layer {
    public:
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

        Layer &operator = (const Layer &target) {
            activations = target.activations;
            bias = target.bias;
            weights = target.weights;
        }
    };

    void forwardPropagate(const std::vector<T> &input) {
        auto sigma = [=](Matrix<long double> &target) {
            target.applyFunction<Sigmoid>();

            return target;
        };

        for (int i = 0; i < inputNeuronCount; ++i)
            neuralLayers.front().activations.data(i, 1) = input[i];

        neuralLayers.front().activations = sigma(neuralLayers.front().activations);

        for (auto it = neuralLayers.begin() + 1; it != neuralLayers.end(); ++it) {
            it -> activations = sigma(it -> weights * (it - 1) -> activations + it -> bias);
        }
    }

    void setRandomStartingPoint() {
        randomEngine randomizer;

        auto randomizeMatrix = [&randomizer](Matrix<auto> &target) {
            std::pair<int, int> size;
            size = target.size();
            for (int i = 0; i < size.first; ++i)
                for (int j = 0; j < size.second; ++j)
                    target.data(i, j) = randomizer.getNumber();
        };

        for (auto it = neuralLayers.begin() + 1; it != neuralLayers.end(); ++it)
            randomizeMatrix(it -> weights),
            randomizeMatrix(it -> bias);
    }

    int inputNeuronCount, outputNeuronCount, hiddenLayersCount;
    std::vector<int> hiddenLayersSizes;
    double learningRate;
    std::vector<double> fitnessRecord;
    std::vector<Layer> neuralLayers;
    size_t neuralNetworkSize;
};
