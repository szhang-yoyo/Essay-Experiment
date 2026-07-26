#include <mpi.h>

#include <cmath>
#include <cstdlib>
#include <iostream>

#include "../core/vector.hpp"
#include "../parallel/mpi_vector.hpp"
#include "../problems/poisson2d_mpi.hpp"
#include "../solvers/mpi_cg_solver.hpp"

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank;
    int np;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    if (argc != 2)
    {
        if (rank == 0)
        {
            std::cout << "Usage: ./apps/mpi_cg <grid_size>\n";
        }

        MPI_Finalize();
        return 1;
    }

    int n = std::atoi(argv[1]);

    if (n <= 0 || np > n)
    {
        if (rank == 0)
        {
            std::cout << "Invalid grid size or process count\n";
        }

        MPI_Finalize();
        return 1;
    }

    const int max_iter = 10000;
    const double tol = 1.0e-8;

    Poisson2DMPI A(n, rank, np);

    int lo_size = A.size();

    Vector b(lo_size);
    Vector x(lo_size);

    // Build a fixed right-hand side
    for (int i = 0; i < lo_size; ++i)
    {
        b[i] =  1.0;
    }

    // Use the zero initial guess
    x.set_zero();

    double rel_res = 0.0;

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    // Solve the distributed system
    int iter = mpi_cg(A,
                      b,
                      x,
                      max_iter,
                      tol,
                      rel_res,
                      MPI_COMM_WORLD);

    double lo_time = MPI_Wtime() - start;
    double gl_time = 0.0;

    // Use the maximum process runtime
    MPI_Reduce(&lo_time,
               &gl_time,
               1,
               MPI_DOUBLE,
               MPI_MAX,
               0,
               MPI_COMM_WORLD);



    if (rank == 0)
    {
        std::cout << "2D Poisson MPI-CG\n";
        std::cout << "Grid size: " << n << " x " << n << '\n';
        std::cout << "Unknowns: " << n * n << '\n';
        std::cout << "MPI processes: " << np << '\n';
        std::cout << "Iterations: " << iter << '\n';
        std::cout << "Relative residual: " << rel_res << '\n';
        std::cout << "Runtime: " << gl_time << '\n';
    }

    MPI_Finalize();

    return 0;
}
