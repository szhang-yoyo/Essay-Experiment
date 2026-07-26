#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <iostream>

#include "../core/vector.hpp"
#include "../parallel/partition.hpp"
#include "../preconditioners/additive_schwarz.hpp"

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank;
    int np;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    const int n = 8;

    if (np != 4)
    {
        if (rank == 0)
        {
            std::cout << "Run this test with 4 MPI processes\n";
        }

        MPI_Finalize();
        return 1;
    }

    Partition part(n, rank, np);

    int fst_row = part.get_first_row();
    int lo_rows = part.get_local_rows();
    int lo_size = lo_rows * n;

    Vector x(lo_size);
    Vector y(lo_size);
    Vector zx(lo_size);
    Vector zy(lo_size);

    // Build two distributed test vectors
    for (int i = 0; i < lo_rows; i++)
    {
        int gl_i = fst_row + i;

        for (int j = 0; j < n; j++)
        {
            int k = i * n + j;
            int gl_k = gl_i * n + j;

            x[k] = 1.0 + 0.01 * gl_k;
            y[k] = 2.0 - 0.005 * gl_k;
        }
    }

    int all_pass = 1;

    // Test several overlap widths
    for (int delta = 0; delta <= 2; delta++)
    {
        AS precond(n, rank, np, delta);

        precond.apply(x, zx);
        precond.apply(y, zy);

        double lo_xy = 0.0;
        double lo_yx = 0.0;
        double lo_pos = 0.0;

        for (int i = 0; i < lo_size; i++)
        {
            lo_xy += x[i] * zy[i];
            lo_yx += y[i] * zx[i];
            lo_pos += x[i] * zx[i];
        }

        double gl_xy = 0.0;
        double gl_yx = 0.0;
        double gl_pos = 0.0;

        MPI_Allreduce(
            &lo_xy,
            &gl_xy,
            1,
            MPI_DOUBLE,
            MPI_SUM,
            MPI_COMM_WORLD
        );

        MPI_Allreduce(
            &lo_yx,
            &gl_yx,
            1,
            MPI_DOUBLE,
            MPI_SUM,
            MPI_COMM_WORLD
        );

        MPI_Allreduce(
            &lo_pos,
            &gl_pos,
            1,
            MPI_DOUBLE,
            MPI_SUM,
            MPI_COMM_WORLD
        );

        double scale = std::max(
            1.0,
            std::max(std::abs(gl_xy), std::abs(gl_yx))
        );

        double sym_err = std::abs(gl_xy - gl_yx) / scale;

        bool pass =
            std::isfinite(sym_err) &&
            std::isfinite(gl_pos) &&
            sym_err < 1.0e-10 &&
            gl_pos > 0.0;

        if (!pass)
        {
            all_pass = 0;
        }

        if (rank == 0)
        {
            std::cout
                << "Delta: " << delta
                << ", symmetry error: " << sym_err
                << ", positive value: " << gl_pos
                << "\n";
        }
    }

    int gl_pass = 0;

    MPI_Allreduce(
        &all_pass,
        &gl_pass,
        1,
        MPI_INT,
        MPI_MIN,
        MPI_COMM_WORLD
    );

    if (rank == 0)
    {
        if (gl_pass == 1)
        {
            std::cout << "Additive Schwarz apply test: PASS\n";
        }
        else
        {
            std::cout << "Additive Schwarz apply test: FAIL\n";
        }
    }

    MPI_Finalize();
    return gl_pass == 1 ? 0 : 1;
}
