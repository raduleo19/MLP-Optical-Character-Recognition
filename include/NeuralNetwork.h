// Copyright 2019 Rica Radu Leonard
// Copyright 2019 Ciobanu Bogdan-Calin
#pragma once

#include <vector>
#include "../include/Utils.h"
#include "../include/ActivationFunction.h"
#include "../include/Matrix.h"
#include "../include/Diagnostics.h"

#define NNDIAG

template <class T, class NormalizationFunction, class TakeStep>
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

        inputLayer = std::move(InputLayer<NormalizationFunction>
                              (inputNeuronCount, hiddenLayersCount ?
                               hiddenLayersSizes[0] : outputNeuronCount));

        outputLayer = std::move(OutputLayer<NormalizationFunction>
                               (outputNeuronCount));

        for (int i = 0; i < hiddenLayersCount - 1; ++i)
            hiddenLayers.push_back(std::move(HiddenLayer<NormalizationFunction>(
                hiddenLayersSizes[i], hiddenLayersSizes[1 + i])));

        hiddenLayers.push_back(std::move(HiddenLayer<NormalizationFunction>(
                hiddenLayersSizes[hiddenLayersCount - 1], outputNeuronCount)));

        setRandomStartingPoint();
    };

    void Train(const std::vector<int> &input, int correctValue) {
        Matrix<long double> desiredOutput(outputNeuronCount, 1);

        for (int i = 0; i < outputNeuronCount; ++i)
            desiredOutput.data(i, 1) = 0.0;
        desiredOutput.data(correctValue, 1) = 1.0;

        auto fitnessFunction = [&]() {
            long double delta = 0;

            for (int i = 0; i < outputNeuronCount; ++i) {
                long double epsilon = outputLayer.activations.data(i, 1) - desiredOutput.data(i, 1);
                delta += epsilon * epsilon;
            }

            return delta;
        };

        forwardPropagate(input);

        fitnessRecord.push_back(fitnessFunction());
        
        /// TODO interface with TakeStep class (backpropagation)
    };

    int Classify(const std::vector<T> &input) {
        forwardPropagate(input);
        
        long double max = -1.0;
        int retval = -1;

        for (int i = 0; i < outputNeuronCount; ++i)
            if (outputLayer.activations.data(i, 1) > max)
                max = outputLayer.activations.data(i, 1),
                retval = i;

        return retval;
    };

   protected:
    template <class sigmoid, class biasType = long double>
    class Layer {
       public:
        Layer() {}

        Layer(const size_t &_size) {
            activations = Matrix<long double>(_size, 1);
        }

        Layer(const size_t &_size, const size_t &_nextLayerSize) {
            activations = Matrix<long double>(_size, 1);
            bias = Matrix<long double>(_size, 1);
            weights = Matrix<long double>(_nextLayerSize, _size);
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

        Matrix<long double> &GetActivations() { return activations; }

        Matrix<biasType> &GetBias() { return bias; }

        Matrix<long double> &GetWeights() { return weights; }

        Matrix<long double> &&ComputeNextLayer() {
            Matrix<long double> retval;
            auto sigma = [&, retval](Matrix<long double> &target) {
                target.applyFunction<sigmoid>();
                
                return target;
            };

            retval = sigma(weights * activations + bias);

            return std::move(retval);
        }

        void SetWeights(Matrix<long double> &&target) {
            weights = std::move(target);
        }

        void SetBias(Matrix<biasType> &&target) {
            bias = std::move(target);
        }

        void SetActivations(Matrix<long double> &&target) {
            activations = std::move(target);
        }

        ~Layer() {}

       protected:
        size_t size;
        Matrix<long double> activations;

       private:
        size_t nextLayerSize;
        Matrix<long double> weights;
        Matrix<biasType> bias;
    };

    template <class sigmoid, class biasType = long double>
    class InputLayer : Layer<sigmoid, biasType> {
        using Layer<sigmoid, biasType>::Layer;
        friend class NeuralNetwork<T, NormalizationFunction, TakeStep>;
        size_t nextLayerSize;
        Matrix<long double> weights;
        Matrix<biasType> bias;
    };

    template <class sigmoid, class biasType = long double>
    class HiddenLayer : Layer<sigmoid, biasType> {
        using Layer<sigmoid, biasType>::Layer;
        friend class NeuralNetwork<T, NormalizationFunction, TakeStep>;
        size_t nextLayerSize;
        Matrix<long double> weights;
        Matrix<biasType> bias;
    };

    template <class sigmoid, class biasType = long double>
    class OutputLayer : Layer<sigmoid, biasType> {
        using Layer<sigmoid, biasType>::Layer;
        friend class NeuralNetwork<T, NormalizationFunction, TakeStep>;
    };

    void forwardPropagate(const std::vector<T> &input) {
        for (int i = 0; i < inputNeuronCount; ++i)
            inputLayer.activations.data(i, 1) = sigmoidFunction(input[i]);

        if (hiddenLayersCount) {
            hiddenLayers[0].activations = std::move(inputLayer.ComputeNextLayer());
            
            for (int i = 1; i < hiddenLayersCount - 1; ++i)
                hiddenLayers[i].activations = std::move(hiddenLayers[i - 1].ComputeNextLayer());
            
            outputLayer.activations = std::move(hiddenLayers[hiddenLayersCount - 1].ComputeNextLayer());
        } else {
            outputLayer.activations = std::move(inputLayer.ComputeNextLayer());
        }
    }

    void setRandomStartingPoint() {
        randomEngine randomizer;
        
        auto randomizeMatrix = [&randomizer](Matrix<long double> &target) {
            std::pair<int, int> size;
            size = target.size();
            for (int i = 0; i < size.first; ++i)
                for (int j = 0; j < size.second; ++j)
                    target.data(i, j) = randomizer.getNumber();
        };

        randomizeMatrix(inputLayer.weights);
        randomizeMatrix(inputLayer.bias);
        
        for (int i = 0; i < hiddenLayersCount; ++i)
            randomizeMatrix(hiddenLayers[i].weights),
            randomizeMatrix(hiddenLayers[i].bias);
    }

    int inputNeuronCount, outputNeuronCount, hiddenLayersCount;
    InputLayer<NormalizationFunction> inputLayer;
    OutputLayer<NormalizationFunction> outputLayer;
    std::vector<int> hiddenLayersSizes;
    std::vector<HiddenLayer<NormalizationFunction> > hiddenLayers;
    NormalizationFunction sigmoidFunction;
    double learningRate;
    std::vector<double> fitnessRecord;
};
