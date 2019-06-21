#include <fstream>
#include <iostream>
#include <vector>
#include "../include/ActivationFunction.h"
#include "../include/Backpropagation.h"
#include "../include/NeuralNetwork.h"
#include "../include/Utils.h"

int main() {
    NeuralNetwork<int, ActivationFunction,
                  Backpropagate<DerivativeActivationFunction>>
        myNeuralNetwork =
            NeuralNetwork<int, ActivationFunction,
                          Backpropagate<DerivativeActivationFunction>>(
                28 * 28, 2, std::vector<int>{28, 28}, 10, 0.1);

    auto trainDataset = GetDataset("./train/mnist_train.csv");

    std::cout << "Training...\n";
    for (int i = 1; i <= 50; ++i) {
        std::cout << "Epoch: " << i << '\n';
        int set = 1;
        for (auto input : trainDataset) {
            // std::cout << "Set: " << set << ' ';
            set++;
            myNeuralNetwork.Train(input.second, input.first);
            if (set == 70) break;
        }
        // std::cout << "\n";
    }

    auto testDataset = GetDataset("./test/mnist_test.csv");
    int total = testDataset.size();
    int predicted = 0;

    int id = 1;
    for (auto input : testDataset) {
        std::cout << "Testing image:" << id << ' ';
        if (input.first == myNeuralNetwork.Classify(input.second)) {
            ++predicted;
            std::cout << "Predicted"; 
        } else {
            std::cout << "Not Predicted"; 
        }
        std::cout << '\n';
        id++;
    }

    std::cout << "Accuracy:" << 100.0 * predicted / total << '\n';

    return 0;
}
