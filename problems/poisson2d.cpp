/*
 * poisson2d.cpp
 *
 * Implementation of the 2D Poisson operator.
 */

#include "poisson2d.hpp"
#include <cmath>
// Construct a 2D Poisson problem
Poisson2D::Poisson2D(int n_) : n(n_), size(n_ * n_) {}

// Convert a 2D grid position to a 1D vector position
int Poisson2D::index(int row, int col) const {
    return row * n + col;
}

// Compute y = A * x
void Poisson2D::matvec(const Vector& x, Vector& y) const {

    // Reset the output vector
    y.set_zero();

    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {

            int centre = index(row, col);

            // Main diagonal
            y[centre] = 4.0 * x[centre];

            // Left neighbour
            if (col > 0) {
                int left = index(row, col - 1);
                y[centre] -= x[left];
            }

            // Right neighbour
            if (col < n - 1) {
                int right = index(row, col + 1);
                y[centre] -= x[right];
            }

            // Bottom neighbour
            if (row > 0) {
                int bottom = index(row - 1, col);
                y[centre] -= x[bottom];
            }

            // Top neighbour
            if (row < n - 1) {
                int top = index(row + 1, col);
                y[centre] -= x[top];
            }
        }
    }
}

// Return the diagonal value
double Poisson2D::diagonal() const {
    return 4.0;
}


// Build the exact solution
void Poisson2D::b_ex_sol(Vector& u) const {

    // Grid spacing
    double h = 1.0 / (n + 1);

    double pi = std::acos(-1.0);

    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {

            // Grid coordinates
            double x = (col + 1) * h;
            double y = (row + 1) * h;

            int id = index(row, col);

            // u(x,y) = sin(pi*x) sin(pi*y)
            u[id] = std::sin(pi * x) * std::sin(pi * y);
        }
    }
}

// Build the right-hand side
void Poisson2D::build_rhs(Vector& b) const {

    // Grid spacing
    double h = 1.0 / (n + 1);
 
    double pi = std::acos(-1.0);

    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {

            // Grid coordinates
            double x = (col + 1) * h;
            double y = (row + 1) * h;

            int id = index(row, col);

            // f(x,y) 
            b[id] =
                2.0 * pi * pi * h * h
                * std::sin(pi * x)
                * std::sin(pi * y);
        }
    }
}


// Compute the relative L2 error
double Poisson2D::comp_re_err(
    const Vector& num,
    const Vector& ex
) const {

    double diff_sum = 0.0;
    double exact_sum = 0.0;

    for (int i = 0; i < size; ++i) {

        double diff = num[i] - ex[i];

        diff_sum += diff * diff;
        exact_sum += ex[i] * ex[i];
    }

    return std::sqrt(diff_sum / exact_sum);
}
