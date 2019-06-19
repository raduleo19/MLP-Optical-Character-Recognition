// Copyright 2019 Rica Radu Leonard
#pragma once

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

std::vector<int> GetRow(std::ifstream &inputFile);
std::vector<std::vector<int>> GetDataset(std::string inputFilename);