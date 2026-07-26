#include <mpi.h>

#include <cmath>
#include <iostream>
#include <vector>

#include "../parallel/partition.hpp"
#include "../problems/poisson2d_mpi.hpp"

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank;
    int np;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    const int n = 4;

    // One process must own at least one row
    if (np > n)
    {
        if (rank == 0)
        {
            std::cout << "Too many MPI processes" << std::endl;
        }

        MPI_Finalize();
        return 1;
    }

    // Create the distributed Poisson problem
    Poisson2DMPI A(n, rank, np);

    int fst = A.fst_row();
    int nr = A.lo_rows();
    int lo_n = A.size();

    Vector x(lo_n);
    Vector y(lo_n);

    x.set_zero();
    y.set_zero();

    // Set global x values to 1, 2, ..., 16
    for (int lo_i = 0; lo_i < nr; lo_i++)
    {
        int gl_i = fst + lo_i;

        for (int j = 0; j < n; j++)
        {
            int lo_id = lo_i * n + j;
            int gl_id = gl_i * n + j;

            x[lo_id] = static_cast<double>(gl_id + 1);
        }
    }

    // Compute y = A * x
    A.matvec(x, y);

    // Copy the local result to an MPI buffer
    std::vector<double> send(lo_n);

    for (int i = 0; i < lo_n; i++)
    {
        send[i] = y[i];
    }

    // Build receive counts and positions
    std::vector<int> cnt(np);
    std::vector<int> dsp(np);

    for (int p = 0; p < np; p++)
    {
        Partition p_part(n, p, np);

        cnt[p] = p_part.get_local_size();
        dsp[p] = p_part.get_first_row() * n;
    }

    std::vector<double> recv;

    if (rank == 0)
    {
        recv.resize(n * n);
    }

    // Gather all local results on rank 0
    MPI_Gatherv(send.data(),
                lo_n,
                MPI_DOUBLE,
                recv.data(),
                cnt.data(),
                dsp.data(),
                MPI_DOUBLE,
                0,
                MPI_COMM_WORLD);

    if (rank == 0)
    {
        // Correct five-point SpMV result
        const double exp[16] =
        {
            -3.0, -2.0, -1.0,  5.0,
             4.0,  0.0,  0.0,  9.0,
             8.0,  0.0,  0.0, 13.0,
            29.0, 18.0, 19.0, 37.0
        };

        std::cout << "Global SpMV result" << std::endl;

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                std::cout << recv[i * n + j] << " ";
            }

            std::cout << std::endl;
        }

        // Find the largest error
        double max_err = 0.0;

        for (int i = 0; i < n * n; i++)
        {
            double err = std::fabs(recv[i] - exp[i]);

            if (err > max_err)
            {
                max_err = err;
            }
        }

        std::cout << "Maximum error: "
                  << max_err << std::endl;

        if (max_err < 1.0e-12)
        {
            std::cout << "Five-point SpMV test: PASS"
                      << std::endl;
        }
        else
        {
            std::cout << "Five-point SpMV test: FAIL"
                      << std::endl;
        }
    }

    MPI_Finalize();

    return 0;
}
