#include <fstream>
#include <iostream>
#include <vector>
#include "../include/ActivationFunction.h"
#include "../include/Backpropagation.h"
#include "../include/NeuralNetwork.h"
#include "../include/Utils.h"

#define VERBOSE

using NN = NeuralNetwork<ActivationFunction, Backpropagate<DerivativeActivationFunction>>;

int main() {
    NN myNeuralNetwork = NN(std::vector<int>{28 * 28, 128, 10}, 0.001);

    auto trainDataset = GetDataset("./dataset/train/mnist_train.csv");

    std::cout << "Training..." << std::endl;

    int set = 0;
    for (auto input : trainDataset) {
        myNeuralNetwork.train(input.second, input.first);
#ifdef VERBOSE
        std::cout << "Epoch: " << set << std::endl;
#endif
        if (++set == 4000)
            break;
    }

    auto testDataset = GetDataset("./dataset/test/mnist_test.csv");
    int total = testDataset.size(), predicted = 0;

    std::cout << "Testing..." << std::endl;

    for (auto input : testDataset) 
        if (input.first == myNeuralNetwork.classify(input.second))
            ++predicted;

    std::cout << "Accuracy:" << 100.0 * predicted / total << '\n';

    return 0;
}
