#ifndef BLOCK_JACOBI_HPP
#define BLOCK_JACOBI_HPP

/*
 * block_jacobi.hpp
 *
 * Definition of the Block Jacobi preconditioner.
 */

#include "../core/vector.hpp"
#include "../problems/poisson1d.hpp"

class BlockJacobi {
public:
    // Local block size used by the preconditioner
    int block_size;

    // Build a Block Jacobi preconditioner
    explicit BlockJacobi(int block_size_);

    // Apply z = M^{-1}r
    void apply(const Poisson1D& A, const Vector& r, Vector& z) const;
};

#endif
