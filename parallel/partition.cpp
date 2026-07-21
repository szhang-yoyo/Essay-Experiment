#include "partition.hpp"

#include <algorithm>

Partition::Partition(int n_value, int rank_value, int size_value)
{
    n = n_value;
    rank = rank_value;
    size = size_value;

    int base_rows = n / size;
    int extra_rows = n % size;

    local_rows = base_rows;

    if (rank < extra_rows)
    {
        local_rows = local_rows + 1;
    }

    first_row =
        rank * base_rows
        + std::min(rank, extra_rows);
}

int Partition::get_first_row() const
{
    return first_row;
}

int Partition::get_last_row() const
{
    return first_row + local_rows - 1;
}

int Partition::get_local_rows() const
{
    return local_rows;
}

int Partition::get_local_size() const
{
    return local_rows * n;
}
