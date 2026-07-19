/*
 * pcg_solver.cpp
 *
 * Implementation of the serial PCG solver.
 */

#include "pcg_solver.hpp"

#include <chrono>
#include <cmath>
#include <stdexcept>

// Solve A*x = b using PCG with Block Jacobi
SolverResult pcg(
    const Poisson1D& A,
    const BJ& M,
    const Vector& b,
    Vector& x,
    int max_iter,
    double tol
) {
    auto t_start = std::chrono::high_resolution_clock::now();

    int n = A.n;

    Vector r(n), z(n), p(n), Ap(n);

    double b_norm = b.norm();

    if (b_norm == 0.0) {
        throw std::runtime_error("PCG error: norm of b is zero.");
    }

    // Initial residual
    A.matvec(x, Ap);

    for (int i = 0; i < n; ++i) {
        r[i] = b[i] - Ap[i];
    }

    // Apply preconditioner
    M.apply(A, r, z);

    for (int i = 0; i < n; ++i) {
        p[i] = z[i];
    }

    double rz_old = r.dot(z);
    double relative_residual = r.norm() / b_norm;

    int iter = 0;

    for (iter = 0; iter < max_iter; ++iter) {

        // Matrix-vector product
        A.matvec(p, Ap);

        double pAp = p.dot(Ap);

        if (std::abs(pAp) < 1e-30) {
            break;
        }

        double alpha = rz_old / pAp;

        // Update solution and residual
        x.axpy(alpha, p);
        r.axpy(-alpha, Ap);

        relative_residual = r.norm() / b_norm;

        if (relative_residual < tol) {
            break;
        }

        // Update preconditioned residual
        M.apply(A, r, z);

        double rz_new = r.dot(z);

        if (std::abs(rz_old) < 1e-30) {
            break;
        }

        double beta = rz_new / rz_old;

        // Update search direction
        for (int i = 0; i < n; ++i) {
            p[i] = z[i] + beta * p[i];
        }

        rz_old = rz_new;
    }

    auto t_end = std::chrono::high_resolution_clock::now();

    double runtime =
        std::chrono::duration<double>(t_end - t_start).count();

    return {iter + 1, relative_residual, runtime};
}


// Solve a 2D Poisson problem using PCG with Block Jacobi
SolverResult pcg(
    const Poisson2D& A,
    const BJ& M,
    const Vector& b,
    Vector& x,
    int max_iter,
    double tol
) {
    auto t_start = std::chrono::high_resolution_clock::now();

    // Total number of unknowns
    int size = A.size;

    Vector r(size);
    Vector z(size);
    Vector p(size);
    Vector Ap(size);

    double b_norm = b.norm();

    if (b_norm == 0.0) {
        throw std::runtime_error("PCG error: norm of b is zero.");
    }

    // Initial residual r = b - A*x
    A.matvec(x, Ap);

    for (int i = 0; i < size; ++i) {
        r[i] = b[i] - Ap[i];
    }

    // Apply the Block Jacobi preconditioner
    M.apply(A, r, z);

    // Initial search direction
    for (int i = 0; i < size; ++i) {
        p[i] = z[i];
    }

    double rz_old = r.dot(z);
    double relative_residual = r.norm() / b_norm;

    int iter = 0;

    for (iter = 0; iter < max_iter; ++iter) {

        // Matrix-vector product
        A.matvec(p, Ap);

        double pAp = p.dot(Ap);

        if (std::abs(pAp) < 1.0e-30) {
            break;
        }

        double alpha = rz_old / pAp;

        // Update solution
        x.axpy(alpha, p);

        // Update residual
        r.axpy(-alpha, Ap);

        relative_residual = r.norm() / b_norm;

        if (relative_residual < tol) {
            break;
        }

        // Apply the preconditioner
        M.apply(A, r, z);

        double rz_new = r.dot(z);

        if (std::abs(rz_old) < 1.0e-30) {
            break;
        }

        double beta = rz_new / rz_old;

        // Update search direction
        for (int i = 0; i < size; ++i) {
            p[i] = z[i] + beta * p[i];
        }

        rz_old = rz_new;
    }

    auto t_end = std::chrono::high_resolution_clock::now();

    double runtime =
        std::chrono::duration<double>(t_end - t_start).count();

    return {iter + 1, relative_residual, runtime};
}
