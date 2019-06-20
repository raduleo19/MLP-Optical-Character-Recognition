// Copyright 2019 Ciobanu Bogdan-Calin

#pragma once

// SoftSign
class ActivationFunction {
   public:
    long double operator()(const long double &target) {
        return ((target / (1 + abs(target))) + 1) / 2;
    }

    long double derivative(const long double &target) {
        return 1 / (2 * (1 + abs(target)) * (1 + abs(target)));
    }

   private:
    long double abs(const long double &target) {
        return target < 0.0 ? -target : target;
    }
};

class DerivativeActivationFunction {
   public:
    long double operator()(const long double &target) {
        return 1 / (2 * (1 + abs(target)) * (1 + abs(target)));
    }

   private:
    long double abs(const long double &target) {
        return target < 0.0 ? -target : target;
    }
};
