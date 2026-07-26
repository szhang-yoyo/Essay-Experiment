#ifndef MPI_TIMES_HPP
#define MPI_TIMES_HPP

struct MPITimes
{
    double total_time = 0.0;
    double spmv_time = 0.0;
    double halo_time = 0.0;
    double preconditioner_time = 0.0;
    double reduction_time = 0.0;
    double vector_time = 0.0;
};

#endif
