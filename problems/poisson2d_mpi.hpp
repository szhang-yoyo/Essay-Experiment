#ifndef POISSON2D_MPI_HPP
#define POISSON2D_MPI_HPP
#include "../core/mpi_times.hpp"
#include "../core/vector.hpp"
#include "../parallel/partition.hpp"
#include "../parallel/halo.hpp"

class Poisson2DMPI
{
private:
    int n;

    Partition part;
    Halo halo;

public:
    Poisson2DMPI(int n_in, int rank, int np);

    int fst_row() const;
    int lst_row() const;
    int lo_rows() const;
    int size() const;

    double diag() const;

    void matvec(const Vector& x, Vector& y, MPITimes* times = nullptr) const;
};

#endif
