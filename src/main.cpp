#include <fstream>
#include <iostream>
#include <vector>
#include "../include/ActivationFunction.h"
#include "../include/Backpropagation.h"
#include "../include/NeuralNetwork.h"
#include "../include/Utils.h"

using NN = NeuralNetwork<ActivationFunction, Backpropagate<DerivativeActivationFunction>>;

int main() {
    NN myNeuralNetwork = NN(std::vector<int>{28 * 28, 128, 10}, 0.1);

    auto trainDataset = GetDataset("./dataset/train/mnist_train.csv");

    std::cout << "Training..." << std::endl;
    for (int i = 1; i <= 50; ++i) {
#ifdef VERBOSE
        std::cout << "Epoch: " << i << '\n';
#endif
        int set = 1;
        for (auto input : trainDataset) {
            // std::cout << "Set: " << set << ' ';
            set++;
            myNeuralNetwork.train(input.second, input.first);
            if (set == 70) break;
        }
        // std::cout << "\n";
    }

    auto testDataset = GetDataset("./dataset/test/mnist_test.csv");
    int total = testDataset.size();
    int predicted = 0;
    std::cout << "Testing..." << std::endl;

    int id = 1;
    for (auto input : testDataset) {
#ifdef VERBOSE
        std::cout << "Testing image:" << id << ' ';
#endif
        if (input.first == myNeuralNetwork.classify(input.second)) {
            ++predicted;
#ifdef VERBOSE
            std::cout << "Predicted\n";
#endif
        } else {
#ifdef VERBOSE
            std::cout << "Not Predicted\n";
#endif
        }
        
        ++id;
    }

    std::cout << "Accuracy:" << 100.0 * predicted / total << '\n';

    return 0;
}
