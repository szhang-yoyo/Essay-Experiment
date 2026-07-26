#include <mpi.h>

#include <cmath>
#include <iostream>

#include "../core/vector.hpp"
#include "../preconditioners/mpi_bj.hpp"

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank;
    int np;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    const int n = 4;
    const int lo_rows = 2;
    const int lo_size = lo_rows * n;

    Vector z_ex(lo_size);
    Vector r(lo_size);
    Vector z(lo_size);

    // Build a known solution
    for (int i = 0; i < lo_size; ++i)
    {
        z_ex[i] = rank * lo_size + i + 1.0;
    }

    // Compute r for each tridiagonal row block
    for (int lo_row = 0; lo_row < lo_rows; ++lo_row)
    {
        int start = lo_row * n;

        for (int j = 0; j < n; ++j)
        {
            int i = start + j;

            r[i] = 4.0 * z_ex[i];

            if (j > 0)
            {
                r[i] -= z_ex[i - 1];
            }

            if (j < n - 1)
            {
                r[i] -= z_ex[i + 1];
            }
        }
    }

    MPIBJ M(n);

    // Solve the local block systems
    M.apply(r, z);

    double lo_err = 0.0;

    for (int i = 0; i < lo_size; ++i)
    {
        double err = std::abs(z[i] - z_ex[i]);

        if (err > lo_err)
        {
            lo_err = err;
        }
    }

    double gl_err = 0.0;

    // Find the maximum error over all processes
    MPI_Allreduce(&lo_err,
                  &gl_err,
                  1,
                  MPI_DOUBLE,
                  MPI_MAX,
                  MPI_COMM_WORLD);

    if (rank == 0)
    {
        std::cout << "MPI Block Jacobi test\n";
        std::cout << "MPI processes: " << np << '\n';
        std::cout << "Maximum error: " << gl_err << '\n';

        if (gl_err < 1.0e-12)
        {
            std::cout << "MPI Block Jacobi test: PASS\n";
        }
        else
        {
            std::cout << "MPI Block Jacobi test: FAIL\n";
        }
    }

    MPI_Finalize();

    return 0;
}
