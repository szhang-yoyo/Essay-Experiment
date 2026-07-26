#ifndef MPI_VECTOR_HPP
#define MPI_VECTOR_HPP

#include <mpi.h>

#include "../core/mpi_times.hpp"
#include "../core/vector.hpp"

// Compute the global dot product
double global_dot(const Vector& x,
                  const Vector& y,
                  MPI_Comm comm,
                  MPITimes* times = nullptr);

// Compute the global Euclidean norm
double global_norm(const Vector& x,
                   MPI_Comm comm,
                   MPITimes* times = nullptr);

#endif
