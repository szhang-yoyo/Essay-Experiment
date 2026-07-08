/*
 * poisson1d.cpp
 *
 * Implementation of the 1D Poisson operator.
 */

#include "poisson1d.hpp"

// Construct a 1D Poisson problem
Poisson1D::Poisson1D(int n_) : n(n_) {}

// Compute y = A * x
void Poisson1D::matvec(const Vector& x, Vector& y) const {

    // Reset the output vector
    y.set_zero();

    for (int i = 0; i < n; ++i) {

        // Main diagonal
        y[i] = 2.0 * x[i];

        // Left neighbour
        if (i > 0) {
            y[i] -= x[i - 1];
        }

        // Right neighbour
        if (i < n - 1) {
            y[i] -= x[i + 1];
        }
    }
}

// Return the diagonal value
double Poisson1D::diagonal() const {
    return 2.0;
}
