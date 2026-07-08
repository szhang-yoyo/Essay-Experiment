#ifndef CG_SOLVER_HPP
#define CG_SOLVER_HPP

/*
 * cg_solver.hpp
 *
 * Interface for the serial Conjugate Gradient solver.
 */

#include "../core/vector.hpp"
#include "../problems/poisson1d.hpp"

struct SolverResult {
    int iterations;              // Number of iterations used
    double relative_residual;    // Final relative residual
    double runtime;              // Runtime in seconds
};

// Solve A*x = b using CG
SolverResult conjugate_gradient(
    const Poisson1D& A,
    const Vector& b,
    Vector& x,
    int max_iter,
    double tol
);

#endif
