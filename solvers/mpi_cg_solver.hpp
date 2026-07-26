#ifndef MPI_CG_SOLVER_HPP
#define MPI_CG_SOLVER_HPP

#include <mpi.h>

#include "../core/vector.hpp"
#include "../problems/poisson2d_mpi.hpp"

int mpi_cg(const Poisson2DMPI& A,
           const Vector& b,
           Vector& x,
           int max_iter,
           double tol,
           double& rel_res,
           MPI_Comm comm);

#endif
