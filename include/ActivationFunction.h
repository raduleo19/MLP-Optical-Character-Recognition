// Copyright 2019 Ciobanu Bogdan-Calin

#pragma once

#include <cmath>

class ActivationFunction {
 public:
    long double operator()(const long double &target) {
        return (1 / (1 + exp(-target)));
    }
};

class DerivativeActivationFunction {
 private:
    ActivationFunction auxiliary = ActivationFunction();
 public:
    long double operator()(const long double &target) {
        return auxiliary(target) * (1 - auxiliary(target));
    }
};
