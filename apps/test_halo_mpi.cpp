#include <mpi.h>

#include <iostream>
#include <vector>

#include "../parallel/partition.hpp"
#include "../parallel/halo.hpp"

void print_row(const std::vector<double>& row)
{
    for (int j = 0; j < static_cast<int>(row.size()); j++)
    {
        std::cout << row[j] << " ";
    }

    std::cout << std::endl;
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank;
    int size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 8;

    Partition part(n, rank, size);

    int local_rows = part.get_local_rows();
    int local_size = part.get_local_size();
    int first_row = part.get_first_row();

    std::vector<double> x(local_size);

    for (int i = 0; i < local_rows; i++)
    {
        int global_row = first_row + i;

        for (int j = 0; j < n; j++)
        {
            int k = i * n + j;
            x[k] = global_row * 100 + j;
        }
    }

    std::vector<double> top_row(n);
    std::vector<double> bottom_row(n);

    Halo halo(n, rank, size, local_rows);
    halo.exchange(x, top_row, bottom_row);

    for (int p = 0; p < size; p++)
    {
        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == p)
        {
            std::cout << "Rank " << rank << std::endl;

            std::cout << "Top halo:    ";
            print_row(top_row);

            std::cout << "Bottom halo: ";
            print_row(bottom_row);

            std::cout << std::endl;
        }
    }

    MPI_Finalize();

    return 0;
}
