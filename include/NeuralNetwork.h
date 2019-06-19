// Copyright 2019 Rica Radu Leonard
// Copyright 2019 Ciobanu Bogdan-Calin
#pragma once

#include <vector>
#include "../include/ActivationFunction.h"
#include "../include/Matrix.h"

template <class T, class NormalizationFunction>
class NeuralNetwork {
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
    };

    void Train(const std::vector<int> &input, int correctValue) {
        /// TODO now
    };

    int Classify(const std::vector<int> &input) const {
        /// TODO now
    };

   private:
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
            auto sigma = [=, &retval](const Matrix<long double> &target) {
                target.applyFunction<sigmoid>();
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
        size_t nextLayerSize;
        Matrix<long double> weights;
        Matrix<biasType> bias;
    };

    template <class sigmoid, class biasType = long double>
    class HiddenLayer : Layer<sigmoid, biasType> {
        using Layer<sigmoid, biasType>::Layer;
        size_t nextLayerSize;
        Matrix<long double> weights;
        Matrix<biasType> bias;
    };

    template <class sigmoid, class biasType = long double>
    class OutputLayer : Layer<sigmoid, biasType> {
        using Layer<sigmoid, biasType>::Layer;
    };

    int inputNeuronCount;
    int hiddenLayersCount;
    InputLayer<NormalizationFunction> inputLayer;
    OutputLayer<NormalizationFunction> outputLayer;
    std::vector<int> hiddenLayersSizes;
    std::vector<HiddenLayer<NormalizationFunction> > hiddenLayers;
    int outputNeuronCount;
    double learningRate;
};
