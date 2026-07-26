#include <mpi.h>

#include <iostream>

#include "../preconditioners/additive_schwarz.hpp"

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank;
    int np;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    const int n = 8;
    const int delta = 1;

    if (np != 4)
    {
        if (rank == 0)
        {
            std::cout << "Run this test with 4 MPI processes\n";
        }

        MPI_Finalize();
        return 1;
    }

    AS precond(n, rank, np, delta);

    int exp_fst[4] = {0, 1, 3, 5};
    int exp_lst[4] = {2, 4, 6, 7};

    bool local_pass =
        precond.ov_fst_row() == exp_fst[rank] &&
        precond.ov_lst_row() == exp_lst[rank];

    int local_value = local_pass ? 1 : 0;
    int global_value = 0;

    MPI_Allreduce(
        &local_value,
        &global_value,
        1,
        MPI_INT,
        MPI_MIN,
        MPI_COMM_WORLD
    );

    // Print ranks in order
    for (int p = 0; p < np; p++)
    {
        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == p)
        {
            std::cout
                << "Rank " << rank
                << ": overlap rows "
                << precond.ov_fst_row()
                << " to "
                << precond.ov_lst_row()
                << ", rows = "
                << precond.ov_rows()
                << ", size = "
                << precond.ov_size()
                << "\n";
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0)
    {
        if (global_value == 1)
        {
            std::cout << "Overlap range test: PASS\n";
        }
        else
        {
            std::cout << "Overlap range test: FAIL\n";
        }
    }

    MPI_Finalize();
    return global_value == 1 ? 0 : 1;
}
