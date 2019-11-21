// Copyright 2019 Ciobanu Bogdan-Calin

#pragma once

#define SOFTSIGN
//#define NORMAL

#include <cmath>
#ifdef NORMAL
class ActivationFunction {
 public:
    inline long double operator()(const long double &target) {
        return (1 / (1 + exp(-target)));
    }
};

class DerivativeActivationFunction {
 private:
    ActivationFunction auxiliary = ActivationFunction();
 public:
    inline long double operator()(const long double &target) {
        return auxiliary(target) * (1 - auxiliary(target));
    }
};
#endif

#ifdef SOFTSIGN
class ActivationFunction {
public:
    inline long double operator()(const long double& target) {
        return (1 + target / (1 + abs(target))) / 2;
    }
private:
    inline long double abs(const long double& target) {
        return target < 0 ? -target : target;
    }
};

class DerivativeActivationFunction {
public:
    inline long double operator()(const long double& target) {
        return 1 / (2 * (1 + abs(target)) * (1 + abs(target)));
    }
private:
    inline long double abs(const long double& target) {
        return target < 0 ? -target : target;
    }
};
#endif
