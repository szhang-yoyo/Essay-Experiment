#include <mpi.h>
#include <iostream>

#include "../parallel/partition.hpp"

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank;
    int size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 10;

    Partition part(n, rank, size);

    for (int p = 0; p < size; p++)
    {
        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == p)
        {
            std::cout
                << "Rank " << rank
                << ": rows "
                << part.get_first_row()
                << " to "
                << part.get_last_row()
                << ", local rows = "
                << part.get_local_rows()
                << ", local unknowns = "
                << part.get_local_size()
                << std::endl;
        }
    }

    MPI_Finalize();

    return 0;
}
