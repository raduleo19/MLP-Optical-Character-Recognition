// Copyright 2019 Rica Radu Leonard
// Copyright 2019 Ciobanu Bogdan-Calin
#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <utility>
#include <random>
#include <chrono>

std::vector<int> GetRow(std::ifstream &inputFile);
std::vector<std::pair<int, std::vector<int>>> GetDataset(std::string inputFilename);

class randomEngine {
public:
    randomEngine() {
        seed = std::chrono::system_clock::now().time_since_epoch().count();
        generator = std::ranlux24(seed);
    }

    long double getNumber() {
        long double retval = (generator() % 2 ? (-1) : 1) * 6.0 *
                              generator() / granularityConstant;
        return retval > 1.0 ? 1.0 : retval;
    }
    
private:
    double seed;
    std::ranlux24 generator;
    const int granularityConstant = 1.0 * 1e8;
};
