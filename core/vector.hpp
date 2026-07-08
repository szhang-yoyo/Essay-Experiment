#ifndef VECTOR_HPP
#define VECTOR_HPP

/*
 * vector.hpp
 *
 * A lightweight vector class used throughout the serial CG and PCG solvers.
 * The goal is to avoid introducing external linear algebra libraries during
 * the first development stage.
 */

#include <vector>
#include <cmath>
#include <cassert>

class Vector {
public:
    // Underlying storage of the vector entries
    std::vector<double> data;

    // Construct an empty vector
    Vector() = default;

    // Construct a vector of length n with all entries initialized to value
    explicit Vector(int n, double value = 0.0)
        : data(n, value) {}

    // Return the vector size
    int size() const {
        return static_cast<int>(data.size());
    }

    // Access an element (modifiable)
    double& operator[](int i) {
        return data[i];
    }

    // Access an element (read-only)
    const double& operator[](int i) const {
        return data[i];
    }

    // Set every entry of the vector to zero
    void set_zero() {
        for (double& value : data) {
            value = 0.0;
        }
    }

    /*
     * Perform the AXPY operation:
     *
     * this = this + alpha * x
     *
     * This operation appears frequently in CG/PCG iterations when updating
     * the solution, residual, and search direction.
     */
    void axpy(double alpha, const Vector& x) {
        assert(size() == x.size());

        for (int i = 0; i < size(); ++i) {
            data[i] += alpha * x[i];
        }
    }

    /*
     * Compute the Euclidean inner product:
     *
     * return this^T * other
     *
     * Used to evaluate step sizes, update coefficients,
     * and compute residual norms.
     */
    double dot(const Vector& other) const {
        assert(size() == other.size());

        double sum = 0.0;

        for (int i = 0; i < size(); ++i) {
            sum += data[i] * other[i];
        }

        return sum;
    }

    /*
     * Compute the Euclidean (L2) norm:
     *
     * ||x|| = sqrt(x^T x)
     *
     * This is typically used as the convergence criterion
     * in iterative solvers.
     */
    double norm() const {
        return std::sqrt(dot(*this));
    }
};

#endif
