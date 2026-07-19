#ifndef POISSON2D_HPP
#define POISSON2D_HPP

/*
 * poisson2d.hpp
 *
 * Definition of the 2D Poisson operator.
 */

#include "../core/vector.hpp"

class Poisson2D {
public:
    //Number of interior grid points in one direction
    int n;

    // Total number of unknowns
    int size;

    // Constructor
    explicit Poisson2D(int n_);

    // Convert grid position (i, j) to vector position
    int index(int i, int j) const;

    // Apply y = A*x
    void matvec(const Vector& x, Vector& y) const;

    // Build the right-hand side vector
     void build_rhs(Vector& b) const;

     // Build the exact solution
     void b_ex_sol(Vector& u) const;

    // Compute the relative L2 error
    double comp_re_err(
    const Vector& num,
    const Vector& ex
    ) const;
 
    // Diagonal entry
    double diagonal() const;
};

#endif
