#include <fstream>
#include <iostream>
#include <vector>
#include "../include/Matrix.h"
#include "../include/NeuralNetwork.h"
#include "../include/Utils.h"
#include "../include/ActivationFunction.h"

int main() {
    NeuralNetwork<int, ActivationFunction> myNeuralNetwork =
        NeuralNetwork<int, ActivationFunction>(28 * 28, 2, std::vector<int> {28, 28}, 10, 0.1);

    auto trainDataset = GetDataset("./train/mnist_train.csv");
    for (auto input : trainDataset) {
        myNeuralNetwork.Train(input.second, input.first);
    }

    auto testDataset = GetDataset("./test/mnist_test.csv");
    int total = testDataset.size();
    int predicted = 0;

    for (auto input : testDataset) {
        if (input.first == myNeuralNetwork.Classify(input.second)) {
            predicted++;
        }
    }

    std::cout << "Accuracy:" << 100.0 * predicted / total << '\n';

    return 0;
}
