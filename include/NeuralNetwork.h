// Copyright 2019 Rica Radu Leonard
#pragma once

#include <vector>

class NeuralNetwork {
   public:
    NeuralNetwork(int inputNeurons, int hiddenLayers, int hiddenNeuronsPerLayer,
                  int ouputNeurons, double learningRate){};

    void Train(const std::vector<int> &input, int correctValue){};

    int Classify(const std::vector<int> &input) const{};

   private:
    int inputNeurons;
    int hiddenLayers;
    int hiddenNeuronsPerLayer;
    int ouputNeurons;
    double learningRate;
};
