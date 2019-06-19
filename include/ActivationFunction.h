// Copyright 2019 Ciobanu Bogdan-Calin

#pragma once

class ActivationFunction {
public:
    long double operator () (const long double &target) {
        /// Putem experimenta cu mai multe tipuri de functii de normalizare
        return target / (1 + abs(target));
    }
private:
    long double abs(const long double &target) {
        return target < 0.0 ? -target : target;
    }
};
