#include <mpi.h>

#include <cmath>
#include <iostream>

#include "../core/vector.hpp"
#include "../parallel/mpi_vector.hpp"

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank;
    int np;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    const int lo_size = 4;

    Vector x(lo_size);
    Vector y(lo_size);

    // Each process stores four identical values in x
    for (int i = 0; i < lo_size; ++i)
    {
        x[i] = rank + 1.0;
        y[i] = i + 1.0;
    }

    double dot_val = global_dot(x, y, MPI_COMM_WORLD);
    double norm_val = global_norm(x, MPI_COMM_WORLD);

    // Expected global dot product
    double exp_dot = 5.0 * np * (np + 1);

    // Expected global norm
    double sum_sq = np * (np + 1) * (2.0 * np + 1) / 6.0;
    double exp_norm = std::sqrt(4.0 * sum_sq);

    double dot_err = std::abs(dot_val - exp_dot);
    double norm_err = std::abs(norm_val - exp_norm);

    if (rank == 0)
    {
        std::cout << "Global dot product: " << dot_val << '\n';
        std::cout << "Expected dot product: " << exp_dot << '\n';

        std::cout << "Global norm: " << norm_val << '\n';
        std::cout << "Expected norm: " << exp_norm << '\n';

        if (dot_err < 1.0e-12 && norm_err < 1.0e-12)
        {
            std::cout << "MPI vector test: PASS\n";
        }
        else
        {
            std::cout << "MPI vector test: FAIL\n";
        }
    }

    MPI_Finalize();

    return 0;
}
