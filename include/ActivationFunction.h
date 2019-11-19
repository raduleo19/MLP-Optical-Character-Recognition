// Copyright 2019 Ciobanu Bogdan-Calin

#pragma once

#include <cmath>

// SoftSign
class ActivationFunction {
   public:
    long double operator()(const long double &target) {
        return (1 / (1 + exp(-target)));
    }
};

class DerivativeActivationFunction {
   public:
    long double operator()(const long double &target) {
        return exp(-target) / ((1 + exp(-target) * (1 + exp(-target))));
    }
};
