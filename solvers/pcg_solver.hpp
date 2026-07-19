#ifndef PCG_SOLVER_HPP
#define PCG_SOLVER_HPP

/*
 * pcg_solver.hpp
 *
 * Interface for the serial PCG solver.
 */

#include "../core/vector.hpp"
#include "../problems/poisson1d.hpp"
#include "../preconditioners/block_jacobi.hpp"
#include "cg_solver.hpp"
#include "../problems/poisson2d.hpp"

// Solve A*x = b using PCG with Block Jacobi for 1D
SolverResult pcg(
    const Poisson1D& A,
    const BJ& M,
    const Vector& b,
    Vector& x,
    int max_iter,
    double tol
);


// Solve A*x = b using PCG with Block Jacobi for 2D
SolverResult pcg(
    const Poisson2D& A,
    const BJ& M,
    const Vector& b,
    Vector& x,
    int max_iter,
    double tol
);

#endif
