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

// Solve A*x = b using PCG with Block Jacobi
SolverResult pcg(
    const Poisson1D& A,
    const BlockJacobi& M,
    const Vector& b,
    Vector& x,
    int max_iter,
    double tol
);

#endif
