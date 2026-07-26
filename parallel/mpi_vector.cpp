#include "mpi_vector.hpp"

#include <cmath>

// Compute the global dot product
double global_dot(const Vector& x,
                  const Vector& y,
                  MPI_Comm comm,
                  MPITimes* times)
{
    // Compute the local dot product
    double lo_sum = 0.0;

    for (int i = 0; i < x.size(); ++i)
    {
        lo_sum += x[i] * y[i];
    }

    // Sum the local results
    double gl_sum = 0.0;

    double t0 = MPI_Wtime();

    MPI_Allreduce(&lo_sum,
                  &gl_sum,
                  1,
                  MPI_DOUBLE,
                  MPI_SUM,
                  comm);

    if (times != nullptr)
    {
        times->reduction_time += MPI_Wtime() - t0;
    }

    return gl_sum;
}

// Compute the global Euclidean norm
double global_norm(const Vector& x,
                   MPI_Comm comm,
                   MPITimes* times)
{
    double val = global_dot(x, x, comm, times);

    return std::sqrt(val);
}
