#ifndef POISSON1D_HPP
#define POISSON1D_HPP

/*
 * poisson1d.hpp
 *
 * Definition of the 1D Poisson operator.
 */

#include "../core/vector.hpp"

class Poisson1D {
public:
    // Problem size
    int n;

    // Constructor
    explicit Poisson1D(int n_);

    // Apply y = A*x
    void matvec(const Vector& x, Vector& y) const;

    // Diagonal entry
    double diagonal() const;
};

#endif
