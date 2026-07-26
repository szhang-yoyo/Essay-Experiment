#include <mpi.h>

#include <cmath>
#include <iostream>

#include "../core/vector.hpp"
#include "../parallel/mpi_vector.hpp"
#include "../preconditioners/mpi_bj.hpp"
#include "../problems/poisson2d_mpi.hpp"
#include "../solvers/mpi_pcg_solver.hpp"

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank;
    int np;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    const int n = 16;
    const int max_iter = 1000;
    const double tol = 1.0e-10;

    Poisson2DMPI A(n, rank, np);
    MPIBJ M(n);

    int lo_size = A.size();

    Vector x_ex(lo_size);
    Vector b(lo_size);
    Vector x(lo_size);
    Vector err(lo_size);

    // Build a known exact solution
    for (int lo_i = 0; lo_i < lo_size; ++lo_i)
    {
        int lo_row = lo_i / n;
        int col = lo_i % n;
        int gl_row = A.fst_row() + lo_row;
        int gl_i = gl_row * n + col;

        x_ex[lo_i] =
            std::sin(0.1 * (gl_i + 1)) +
            0.001 * (gl_i + 1);
    }

    // Build the right-hand side b = A * x_ex
    A.matvec(x_ex, b);

    // Use the zero initial guess
    x.set_zero();

    double rel_res = 0.0;

    // Solve Ax = b
    int iter = mpi_pcg(A,
                       M,
                       b,
                       x,
                       max_iter,
                       tol,
                       rel_res,
                       MPI_COMM_WORLD);

    // Compute the solution error
    for (int i = 0; i < lo_size; ++i)
    {
        err[i] = x[i] - x_ex[i];
    }

    double err_norm = global_norm(err, MPI_COMM_WORLD);
    double ex_norm = global_norm(x_ex, MPI_COMM_WORLD);
    double rel_err = err_norm / ex_norm;

    if (rank == 0)
    {
        std::cout << "MPI-PCG Block Jacobi test\n";
        std::cout << "Grid size: " << n << " x " << n << '\n';
        std::cout << "MPI processes: " << np << '\n';
        std::cout << "Iterations: " << iter << '\n';
        std::cout << "Relative residual: " << rel_res << '\n';
        std::cout << "Relative solution error: " << rel_err << '\n';

        if (rel_res < tol && rel_err < 1.0e-8)
        {
            std::cout << "MPI-PCG test: PASS\n";
        }
        else
        {
            std::cout << "MPI-PCG test: FAIL\n";
        }
    }

    MPI_Finalize();

    return 0;
}
