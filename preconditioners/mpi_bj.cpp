#include "mpi_bj.hpp"
#include "local_sgs.hpp"
#include <cstdlib>

MPIBJ::MPIBJ(int n)
    : n_(n)
{
    const char* env = std::getenv("SGS_SWEEPS");

    if (env != nullptr)
    {
        sgs_sweeps_ = std::atoi(env);
    }
    else
    {
        sgs_sweeps_ = 4;
    }
}


void MPIBJ::apply(const Vector& r, Vector& z) const
{
    int lo_rows =  r.size() / n_;
    local_sgs(r, z, lo_rows, n_, sgs_sweeps_);
}
