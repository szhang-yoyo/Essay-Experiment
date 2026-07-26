#ifndef LOCAL_SGS_HPP
#define LOCAL_SGS_HPP

#include "../core/vector.hpp"

// Approximately solve a local 2D Poisson system
// using a fixed number of symmetric Gauss-Seidel sweeps.
void local_sgs(const Vector& rhs,
               Vector& x,
               int rows,
               int cols,
               int sweeps);

#endif
