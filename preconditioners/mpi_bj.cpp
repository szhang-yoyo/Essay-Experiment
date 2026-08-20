#include "mpi_bj.hpp"
#include "local_sgs.hpp"

MPIBJ::MPIBJ(int n)
    : n_(n)
{
}


void MPIBJ::apply(const Vector& r, Vector& z) const
{
    int lo_rows =  r.size() / n_;
    local_sgs(r, z, lo_rows, n_, 4);
}
