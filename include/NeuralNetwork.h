// Copyright 2019 Rica Radu Leonard
#pragma once

#include <vector>

template <class T = int>
class NeuralNetwork {
   public:
    NeuralNetwork(int inputNeurons, int hiddenLayers, int hiddenNeuronsPerLayer,
                  int ouputNeurons, double learningRate){};

    void Train(const std::vector<int> &input){};

    int Classify(const std::vector<int> &input) {};

   private:
    int inputNeurons;
    int hiddenLayers;
    int hiddenNeuronsPerLayer;
    int ouputNeurons;
    double learningRate;
};
