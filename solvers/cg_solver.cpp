/*
 * cg_solver.cpp
 *
 * Implementation of the serial Conjugate Gradient solver.
 */

#include "cg_solver.hpp"

#include <chrono>
#include <cmath>
#include <stdexcept>

// Solve A*x = b using the CG method
SolverResult conjugate_gradient(
    const Poisson1D& A,
    const Vector& b,
    Vector& x,
    int max_iter,
    double tol
) {
    auto t_start = std::chrono::high_resolution_clock::now();

    int n = A.n;

    Vector r(n), p(n), Ap(n);

    // Compute ||b||
    double b_norm = b.norm();

    if (b_norm == 0.0) {
        throw std::runtime_error("CG error: norm of b is zero.");
    }

    // Initial residual
    A.matvec(x, Ap);

    for (int i = 0; i < n; ++i) {
        r[i] = b[i] - Ap[i];
        p[i] = r[i];
    }

    double rr_old = r.dot(r);
    double relative_residual = std::sqrt(rr_old) / b_norm;

    int iter = 0;

    for (iter = 0; iter < max_iter; ++iter) {

        // Matrix-vector product
        A.matvec(p, Ap);

        double pAp = p.dot(Ap);

        // Avoid division by zero
        if (std::abs(pAp) < 1e-30) {
            break;
        }

        double alpha = rr_old / pAp;

        // Update solution
        x.axpy(alpha, p);

        // Update residual
        r.axpy(-alpha, Ap);

        double rr_new = r.dot(r);
        relative_residual = std::sqrt(rr_new) / b_norm;

        // Check convergence
        if (relative_residual < tol) {
            break;
        }

        double beta = rr_new / rr_old;

        // Update search direction
        for (int i = 0; i < n; ++i) {
            p[i] = r[i] + beta * p[i];
        }

        rr_old = rr_new;
    }

    auto t_end = std::chrono::high_resolution_clock::now();

    double runtime =
        std::chrono::duration<double>(t_end - t_start).count();

    return {iter + 1, relative_residual, runtime};
}
