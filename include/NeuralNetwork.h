// Copyright 2019 Rica Radu Leonard
// Copyright 2019 Ciobanu Bogdan-Calin
#pragma once

#include <vector>
#include "../include/Matrix.h"
#include "../include/ActivationFunction.h"

template <class T>
class NeuralNetwork {
   public:
    NeuralNetwork(int inputNeurons, int hiddenLayers, int hiddenNeuronsPerLayer,
                  int ouputNeurons, double learningRate) {};
                  
    void Train(const std::vector<int> &input, int correctValue) {};

    int Classify(const std::vector<int> &input) const {};

   private:
    int inputNeurons;
    int hiddenLayers;
    int hiddenNeuronsPerLayer;
    int ouputNeurons;
    double learningRate;
};
