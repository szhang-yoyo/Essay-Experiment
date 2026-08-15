#ifndef MPI_BJ_HPP
#define MPI_BJ_HPP

#include "../core/vector.hpp"

class MPIBJ
{
private:
    int n_;
    int sgs_sweeps_; 

public:
    MPIBJ(int n);

    void apply(const Vector& r,
               Vector& z) const;
};

#endif
