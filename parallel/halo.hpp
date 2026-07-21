#ifndef HALO_HPP
#define HALO_HPP

#include <mpi.h>
#include <vector>

class Halo
{
private:
    int n;
    int rank;
    int size;
    int local_rows;

public:
    Halo(int n_value, int rank_value, int size_value, int local_rows_value);

    void exchange(
        const std::vector<double>& x,
        std::vector<double>& top_row,
        std::vector<double>& bottom_row) const;
};

#endif
