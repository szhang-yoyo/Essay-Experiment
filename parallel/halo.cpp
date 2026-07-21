#include "halo.hpp"

#include <algorithm>

Halo::Halo(
    int n_value,
    int rank_value,
    int size_value,
    int local_rows_value)
{
    n = n_value;
    rank = rank_value;
    size = size_value;
    local_rows = local_rows_value;
}

void Halo::exchange(
    const std::vector<double>& x,
    std::vector<double>& top_row,
    std::vector<double>& bottom_row) const
{
    std::fill(top_row.begin(), top_row.end(), 0.0);
    std::fill(bottom_row.begin(), bottom_row.end(), 0.0);

    int top_rank = MPI_PROC_NULL;
    int bottom_rank = MPI_PROC_NULL;

    if (rank > 0)
    {
        top_rank = rank - 1;
    }

    if (rank < size - 1)
    {
        bottom_rank = rank + 1;
    }

    // Send the first local row upward.
    // Receive the first row of the lower process.
    MPI_Sendrecv(
        x.data(),
        n,
        MPI_DOUBLE,
        top_rank,
        0,
        bottom_row.data(),
        n,
        MPI_DOUBLE,
        bottom_rank,
        0,
        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE);

    // Send the last local row downward.
    // Receive the last row of the upper process.
    MPI_Sendrecv(
        x.data() + (local_rows - 1) * n,
        n,
        MPI_DOUBLE,
        bottom_rank,
        1,
        top_row.data(),
        n,
        MPI_DOUBLE,
        top_rank,
        1,
        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE);
}
