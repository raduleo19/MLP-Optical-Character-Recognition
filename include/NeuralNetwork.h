// Copyright 2019 Rica Radu Leonard
// Copyright 2019 Ciobanu Bogdan-Calin
#pragma once

#include <vector>
#include "../include/Matrix.h"
#include "../include/ActivationFunction.h"

template <class T>
class NeuralNetwork {
   public:
    NeuralNetwork(int _inputNeurons, int _hiddenLayers,
                  std::vector<int> _hiddenLayersSizes, int _outputNeurons,
                  double _learningRate) {};
                  
    void Train(const std::vector<int> &input, int correctValue) {};

    int Classify(const std::vector<int> &input) const {};

   private:
    template <class sigmoid, class biasType = long double>
    class HiddenLayer {
    public:
        HiddenLayer() {}
        
        HiddenLayer(const size_t &_size, const size_t &_nextLayerSize) {}
        
        HiddenLayer(const HiddenLayer &target) {}
        
        HiddenLayer(HiddenLayer &&target) {}
        
        HiddenLayer &operator=(const HiddenLayer &target) {}
        
        HiddenLayer &operator=(HiddenLayer &&target) {}
        
        Matrix<long double> &GetActivations() {}
        
        Matrix<biasType> &GetBias() {}
        
        Matrix<long double> &GetWeights() {}
        
        Matrix<long double> &&ComputeNextLayer() {}
        
        Matrix<long double> &&SetWeights(Matrix<long double> &&target) {}
        
        Matrix<biasType> &&SetBias(Matrix<biasType> &&target) {}
        
        Matrix<long double> &&SetNextLayer(Matrix<long double> &&target) {}
        
        ~HiddenLayer() {}

    private:
        size_t size, nextLayerSize;
        Matrix<long double> Activations, Weights;
        Matrix<biasType> Bias;
    };
       
    int inputNeurons;
    int hiddenLayers;
    std::vector<int> hiddenLayersSizes;
    int ouputNeurons;
    double learningRate;
};
