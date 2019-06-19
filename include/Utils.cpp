#include "Utils.h"

std::vector<int> GetRow(std::ifstream &inputFile) {
    std::vector<int> buffer;

    std::string line;
    std::getline(inputFile, line);
    std::istringstream inputStream(line);

    std::string numString;
    while (std::getline(inputStream, numString, ',')) {
        buffer.push_back(atoi(numString.c_str()));
    }

    return buffer;
}

std::vector<std::vector<int>> GetDataset(std::string inputFilename) {
    std::vector<std::vector<int>> trainDataset;
    std::ifstream trainingFile(inputFilename);

    while (true) {
        auto row = GetRow(trainingFile);
        if (row.empty()) {
            break;
        }
        trainDataset.push_back(row);
    }
}