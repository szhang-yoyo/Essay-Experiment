#ifndef MPI_PCG_SOLVER_HPP
#define MPI_PCG_SOLVER_HPP

#include <mpi.h>

#include "../core/mpi_times.hpp"
#include "../core/vector.hpp"
#include "../preconditioners/additive_schwarz.hpp"
#include "../preconditioners/mpi_bj.hpp"
#include "../problems/poisson2d_mpi.hpp"

// Solve with Block Jacobi
int mpi_pcg(const Poisson2DMPI& A,
            const MPIBJ& M,
            const Vector& b,
            Vector& x,
            int max_iter,
            double tol,
            double& rel_res,
            MPI_Comm comm,
            MPITimes* times = nullptr);

// Solve with Additive Schwarz
int mpi_pcg(const Poisson2DMPI& A,
            const AS& M,
            const Vector& b,
            Vector& x,
            int max_iter,
            double tol,
            double& rel_res,
            MPI_Comm comm,
            MPITimes* times = nullptr);

#endif
