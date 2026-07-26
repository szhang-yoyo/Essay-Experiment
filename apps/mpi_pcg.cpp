#include <mpi.h>
#include <vector>
#include <cstdlib>
#include <iostream>
#include "../core/mpi_times.hpp"
#include "../core/vector.hpp"
#include "../preconditioners/additive_schwarz.hpp"
#include "../preconditioners/mpi_bj.hpp"
#include "../problems/poisson2d_mpi.hpp"
#include "../solvers/mpi_pcg_solver.hpp"

int main(int argc, char** argv)
{
    // Start MPI
    MPI_Init(&argc, &argv);

    int rank;
    int np;

    // Get process information
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    // Check command-line arguments
    if (argc < 3 || argc > 4)
    {
        if (rank == 0)
        {
            std::cout
                << "Usage:\n"
                << "  ./apps/mpi_pcg <grid_size> bj\n"
                << "  ./apps/mpi_pcg <grid_size> as <overlap>\n";
        }

        MPI_Finalize();
        return 1;
    }

    int n = std::atoi(argv[1]);
    std::string prec = argv[2];

    // Check grid size and process count
    if (n <= 0 || np > n)
    {
        if (rank == 0)
        {
            std::cout << "Invalid grid size or process count\n";
        }

        MPI_Finalize();
        return 1;
    }

    int delta = 0;

    // Read the overlap value for AS
    if (prec == "as")
    {
        if (argc != 4)
        {
            if (rank == 0)
            {
                std::cout << "AS requires an overlap value\n";
            }

            MPI_Finalize();
            return 1;
        }

        delta = std::atoi(argv[3]);

        if (delta < 0)
        {
            if (rank == 0)
            {
                std::cout << "Overlap must be non-negative\n";
            }

            MPI_Finalize();
            return 1;
        }
    }
    else if (prec != "bj")
    {
        if (rank == 0)
        {
            std::cout << "Preconditioner must be bj or as\n";
        }

        MPI_Finalize();
        return 1;
    }

    const int max_iter = 10000;
    const double tol = 1.0e-8;

    // Build the distributed Poisson problem
    Poisson2DMPI A(n, rank, np);

    int lo_size = A.size();

    Vector b(lo_size);
    Vector x(lo_size);

    // Build a fixed right-hand side
    for (int i = 0; i < lo_size; ++i)
    {
        b[i] = 1.0;
    }

    // Use the zero initial guess
    x.set_zero();

    double rel_res = 0.0;
    int iter = 0;

    MPITimes times;

    // Solve with Block Jacobi
    if (prec == "bj")
    {
        MPIBJ M(n);

       MPI_Barrier(MPI_COMM_WORLD);
       double start = MPI_Wtime();

        iter = mpi_pcg(
            A,
            M,
            b,
            x,
            max_iter,
            tol,
            rel_res,
            MPI_COMM_WORLD,
            &times
        );
        times.total_time = MPI_Wtime() - start;
    }
    // Solve with Additive Schwarz
    else
    {
        AS M(n, rank, np, delta, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();

        iter = mpi_pcg(
            A,
            M,
            b,
            x,
            max_iter,
            tol,
            rel_res,
            MPI_COMM_WORLD,
            &times
        );
        times.total_time = MPI_Wtime() - start;
    }

    //double lo_time = MPI_Wtime() - start;
    //double gl_time = 0.0;

    // Use the maximum process runtime
   // MPI_Reduce(
     //   &lo_time,
       // &gl_time,
 //       1,
   //     MPI_DOUBLE,
       // MPI_MAX,
     //   0,
       // MPI_COMM_WORLD
 //   );
    // Store local timing results

    double lo_times[6] = {
        times.total_time,
        times.spmv_time,
        times.halo_time,
        times.preconditioner_time,
        times.reduction_time,
        times.vector_time
    };

    // Gather the complete timing vector from every process
    std::vector<double> all_times;

    if (rank == 0)
    {
        all_times.resize(6 * np);
    }

    MPI_Gather(
        lo_times,
        6,
        MPI_DOUBLE,
        rank == 0 ? all_times.data() : nullptr,
        6,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    // Use all timing components from the process
    // with the largest total runtime
    double gl_times[6] = {0.0};

    if (rank == 0)
    {
        int critical_rank = 0;

        for (int r = 1; r < np; ++r)
        {
            if (all_times[6 * r] >
                all_times[6 * critical_rank])
            {
                critical_rank = r;
            }
        }

        for (int k = 0; k < 6; ++k)
        {
            gl_times[k] =
                all_times[6 * critical_rank + k];
        }
    }


    // Print results on the root process
    if (rank == 0)
    {
        std::cout << "2D Poisson MPI-PCG\n";

        if (prec == "bj")
        {
            std::cout << "Preconditioner: Block Jacobi\n";
        }
        else
        {
            std::cout << "Preconditioner: Additive Schwarz\n";
            std::cout << "Overlap: " << delta << '\n';
        }

        std::cout << "Grid size: " << n << " x " << n << '\n';
        std::cout << "Unknowns: " << n * n << '\n';
        std::cout << "MPI processes: " << np << '\n';
        std::cout << "Iterations: " << iter << '\n';
        std::cout << "Relative residual: " << rel_res << '\n';
    //    std::cout << "Runtime: " << gl_time << '\n';
        std::cout << "Total time: " << gl_times[0] << '\n';
        std::cout << "SpMV time: " << gl_times[1] << '\n';
        std::cout << "Halo time: " << gl_times[2] << '\n';
        std::cout << "Preconditioner time: " << gl_times[3] << '\n';
        std::cout << "Reduction time: " << gl_times[4] << '\n';
        std::cout << "Vector time: " << gl_times[5] << '\n';



    }

    MPI_Finalize();

    return 0;
}
