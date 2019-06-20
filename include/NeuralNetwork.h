// Copyright 2019 Ciobanu Bogdan-Calin
// Copyright 2019 Rica Radu Leonard
#pragma once

#include <vector>
#include "../include/ActivationFunction.h"
#include "../include/Diagnostics.h"
#include "../include/Matrix.h"
#include "../include/Utils.h"

template <class T, class NormalizationFunction, class Backpropagator>
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

        inputLayer =
            std::move(InputLayer<NormalizationFunction>(inputNeuronCount));

        outputLayer = std::move(OutputLayer<NormalizationFunction>(
            hiddenLayersCount ? hiddenLayersSizes[hiddenLayersCount - 1]
                              : inputNeuronCount,
            outputNeuronCount));

        hiddenLayers.push_back(std::move(HiddenLayer<NormalizationFunction>(
            inputNeuronCount, hiddenLayersSizes[0])));

        for (int i = 1; i < hiddenLayersCount; ++i)
            hiddenLayers.push_back(std::move(HiddenLayer<NormalizationFunction>(
                hiddenLayersSizes[i - 1], hiddenLayersSizes[i])));

        setRandomStartingPoint();
    };

    void Train(const std::vector<int> &input, int correctValue) {
        forwardPropagate(input);
        Matrix desiredOutput(outputNeuronCount, 1);

        for (int i = 0; i < outputNeuronCount; ++i)
            desiredOutput.data(i, 1) = 0.0;
        desiredOutput.data(correctValue, 1) = 1.0;

        std::vector<Matrix> weights;
        std::vector<Matrix> biases;
        std::vector<Matrix> activations;

        weights.push_back(inputLayer.GetWeights());
        biases.push_back(inputLayer.GetActivations());
        activations.push_back(inputLayer.GetActivations());
        for (auto it : hiddenLayers) {
            weights.push_back(it.GetWeights());
            biases.push_back(it.GetActivations());
            activations.push_back(it.GetActivations());
        }
        weights.push_back(outputLayer.GetWeights());
        biases.push_back(outputLayer.GetActivations());
        activations.push_back(outputLayer.GetActivations());

        auto backpropagator = Backpropagator();
        backpropagator.backpropagate(weights, biases, activations,
                                     desiredOutput, learningRate);

        inputLayer.SetWeights(std::move(weights[0]));
        inputLayer.SetBias(std::move(biases[0]));
        for (unsigned i = 1; i < weights.size() - 1; ++i) {
            hiddenLayers[i - 1].SetWeights(std::move(weights[i]));
            hiddenLayers[i - 1].SetBias(std::move(biases[i]));
        }
        outputLayer.SetWeights(std::move(weights[weights.size() - 1]));
        outputLayer.SetBias(std::move(biases[weights.size() - 1]));
    };

    int Classify(const std::vector<T> &input) {
        forwardPropagate(input);

        long double max = -1.0;
        int retval = -1;

        for (int i = 0; i < outputNeuronCount; ++i)
            if (outputLayer.activations.data(i, 1) > max)
                max = outputLayer.activations.data(i, 1), retval = i;

        return retval;
    };

   protected:
    template <class sigmoid, class biasType = long double>
    class Layer {
       public:
        Layer() {}

        Layer(const size_t &_size) {
            activations = Matrix(_size, 1);
        }

        Layer(const size_t &_previousLayerSize, const size_t &_size) {
            activations = Matrix(_size, 1);
            bias = Matrix(_size, 1);
            weights = Matrix(_size, _previousLayerSize);
        }

        Layer(const Layer &target) { *this = target; }

        Layer(Layer &&target) { *this = target; }

        Layer &operator=(const Layer &target) {
            size = target.size;
            nextLayerSize = target.nextLayerSize;
            activations = target.activations;
            bias = target.bias;
            weights = target.weights;

            return *this;
        }

        Layer &operator=(Layer &&target) {
            size = std::move(target.size);
            nextLayerSize = std::move(target.nextLayerSize);
            activations = std::move(target.activations);
            bias = std::move(target.bias);
            weights = std::move(target.weights);

            return *this;
        }

        Matrix &GetActivations() { return activations; }

        Matrix &GetBias() { return bias; }

        Matrix &GetWeights() { return weights; }

        Matrix &&ComputeNextLayer(Layer<sigmoid> next) {
            Matrix retval;

            retval = (next.weights * activations + next.bias).ApplyFunction<NormalizationFunction>();

            return std::move(retval);
        }

        void SetWeights(Matrix &&target) {
            weights = std::move(target);
        }

        void SetBias(Matrix &&target) { bias = std::move(target); }

        void SetActivations(Matrix &&target) {
            activations = std::move(target);
        }

        ~Layer() {}

       protected:
        size_t size;
        Matrix activations;

       private:
        size_t nextLayerSize;
        Matrix weights;
        Matrix bias;
    };

    template <class sigmoid, class biasType = long double>
    class InputLayer : Layer<sigmoid, biasType> {
        using Layer<sigmoid, biasType>::Layer;
        friend class NeuralNetwork<T, NormalizationFunction, Backpropagator>;
        friend class Backpropagate;

       protected:
        size_t nextLayerSize;
    };

    template <class sigmoid, class biasType = long double>
    class HiddenLayer : Layer<sigmoid, biasType> {
        using Layer<sigmoid, biasType>::Layer;
        friend class NeuralNetwork<T, NormalizationFunction, Backpropagator>;
        friend class Backpropagate;

       protected:
        size_t nextLayerSize;
        Matrix weights;
        Matrix bias;
    };

    template <class sigmoid, class biasType = long double>
    class OutputLayer : Layer<sigmoid, biasType> {
        using Layer<sigmoid, biasType>::Layer;
        friend class NeuralNetwork<T, NormalizationFunction, Backpropagator>;
        friend class Backpropagate;

       protected:
        Matrix weights;
        Matrix bias;
    };

    void forwardPropagate(const std::vector<T> &input) {
        for (int i = 0; i < inputNeuronCount; ++i)
            inputLayer.activations.data(i, 1) = sigmoidFunction(input[i]);

        if (hiddenLayersCount) {
            hiddenLayers[0].activations =
                std::move(inputLayer.ComputeNextLayer(hiddenLayers[0]));

            for (int i = 1; i < hiddenLayersCount - 1; ++i)
                hiddenLayers[i].activations = std::move(
                    hiddenLayers[i - 1].ComputeNextLayer(hiddenLayers[i]));

            outputLayer.activations =
                std::move(hiddenLayers[hiddenLayersCount - 1].ComputeNextLayer(
                    outputLayer));
        } else {
            outputLayer.activations =
                std::move(inputLayer.ComputeNextLayer(outputLayer));
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

        randomizeMatrix(outputLayer.weights);
        randomizeMatrix(outputLayer.bias);

        for (int i = 0; i < hiddenLayersCount; ++i)
            randomizeMatrix(hiddenLayers[i].weights),
                randomizeMatrix(hiddenLayers[i].bias);
    }

    int inputNeuronCount, outputNeuronCount, hiddenLayersCount;
    InputLayer<NormalizationFunction> inputLayer;
    OutputLayer<NormalizationFunction> outputLayer;
    std::vector<int> hiddenLayersSizes;
    std::vector<HiddenLayer<NormalizationFunction>> hiddenLayers;
    NormalizationFunction sigmoidFunction;
    double learningRate;
    std::vector<double> fitnessRecord;
};
