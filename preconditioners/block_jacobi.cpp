/*
 * block_jacobi.cpp
 *
 * Implementation of the Block Jacobi preconditioner.
 */

#include "block_jacobi.hpp"

#include <algorithm>
#include <vector>

// Solve one local tridiagonal block
static void solve_tridiagonal_block(
    int m,
    const double* rhs,
    double* sol
) {
    std::vector<double> a(m, -1.0);
    std::vector<double> b(m,  2.0);
    std::vector<double> c(m, -1.0);
    std::vector<double> d(m);

    // Copy right-hand side
    for (int i = 0; i < m; ++i) {
        d[i] = rhs[i];
    }

    // Forward sweep
    for (int i = 1; i < m; ++i) {
        double factor = a[i] / b[i - 1];
        b[i] -= factor * c[i - 1];
        d[i] -= factor * d[i - 1];
    }

    // Backward sweep
    sol[m - 1] = d[m - 1] / b[m - 1];

    for (int i = m - 2; i >= 0; --i) {
        sol[i] = (d[i] - c[i] * sol[i + 1]) / b[i];
    }
}

// Store the block size
BlockJacobi::BlockJacobi(int block_size_)
    : block_size(block_size_) {}

// Apply z = M^{-1}r block by block
void BlockJacobi::apply(const Poisson1D& A, const Vector& r, Vector& z) const {
    int n = A.n;

    z.set_zero();

    for (int start = 0; start < n; start += block_size) {
        int end = std::min(start + block_size, n);
        int m = end - start;

        solve_tridiagonal_block(
            m,
            &r.data[start],
            &z.data[start]
        );
    }
}
