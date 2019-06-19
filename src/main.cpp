#include <fstream>
#include <iostream>
#include <vector>
#include "../include/NeuralNetwork.h"
#include "../include/Utils.h"

int main() {
    NeuralNetwork<int> myNeuralNetwork =
        NeuralNetwork<int>(28 * 28, 2, 28, 10, 0.1);

    auto trainDataset = GetDataset("./train/mnist_train.csv");
    for (auto input : trainDataset) {
        myNeuralNetwork.Train(input.second, input.first);
    }

    int total = 11;
    int predicted = 1;

    auto testDataset = GetDataset("./test/mnist_test.csv");
    for (auto input : testDataset) {
        total++;
        if (input.first == myNeuralNetwork.Classify(input.second)) {
            predicted++;
        }
    }

    std::cout << "Accuracy:" << 1.0 * predicted / total << '\n';

    return 0;
}
