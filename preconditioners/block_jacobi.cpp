/*
 * block_jacobi.cpp
 *
 * Implementation of the Block Jacobi preconditioner.
 */

#include "block_jacobi.hpp"

//#include <algorithm>
#include <vector>

// Solve one local tridiagonal block
//static void solve_tridiagonal_block(
  //  int m,
    //double diagonal,
   // const double* rhs,
   // double* sol,
 //   std::vector<double>& work_diag,
   // std::vector<double>& work_rhs
//) {
    // Copy the diagonal and right-hand side
  //  for (int i = 0; i < m; ++i) {
    //    work_diag[i] = diagonal;
      //  work_rhs[i] = rhs[i];
  //  }

    // Forward sweep
   // for (int i = 1; i < m; ++i) {

     //   double factor =
       //     -1.0 / work_diag[i - 1];

       // work_diag[i] -= factor * (-1.0);
       // work_rhs[i] -= factor * work_rhs[i - 1];
   // }

    // Backward sweep
   // sol[m - 1] =
     //   work_rhs[m - 1] / work_diag[m - 1];

   // for (int i = m - 2; i >= 0; --i) {
     //   sol[i] =
       //     (work_rhs[i] + sol[i + 1])
         //   / work_diag[i];
    //}
//}

// Store the block size
BJ::BJ(int block_size_)
     : block_size(block_size_)
{ 

     if (block_size < 1)
    {
        block_size = 1;
    }

    // Build the 1D LU factors
    build_factor(
        2.0,
        lower_1d,
        inv_upper_1d);

    // Build the 2D LU factors
    build_factor(
        4.0,
        lower_2d,
        inv_upper_2d);

    // Create the work space once
    work.resize(block_size);
}


void BJ::build_factor(
    double diag,
    std::vector<double>& lower,
    std::vector<double>& inv_upper)
{
    lower.resize(block_size);
    inv_upper.resize(block_size);

    // First row
    lower[0] = 0.0;
    inv_upper[0] = 1.0 / diag;

    for (int i = 1; i < block_size; i++)
    {
        // Lower factor
        lower[i] = -inv_upper[i - 1];

        // Upper diag
        double upper_diag =
            diag + lower[i];

        // Store its inverse
        inv_upper[i] =
            1.0 / upper_diag;
    }
}


void BJ::solve_factored_block(
    int start,
    int size,
    const Vector& r,
    Vector& z,
    const std::vector<double>& lower,
    const std::vector<double>& inv_upper) const
{
    if (size <= 0)
    {
        return;
    }

    // Forward solve
    work[0] = r[start];

    for (int i = 1; i < size; i++)
    {
        work[i] =
            r[start + i]
            - lower[i] * work[i - 1];
    }

    // Backward solve
    z[start + size - 1] =
        work[size - 1]
        * inv_upper[size - 1];

    for (int i = size - 2; i >= 0; i--)
    {
        z[start + i] =
            (work[i] + z[start + i + 1])
            * inv_upper[i];
    }
}


// Apply z = M^{-1}r block by block
void BJ::apply(const Poisson1D& A, const Vector& r, Vector& z) const {
   
      (void)A;

      int n = r.size();

    z.set_zero();

    for (int start = 0; start < n; start += block_size) {
       // int end = std::min(start + block_size, n);
       // int m = end - start;

       // solve_tridiagonal_block(
         //   m,
           // 2.0,
        //    &r.data[start],
          //  &z.data[start],
         //   work_diag,
        //    work_rhs
      //  );



        int current_size = block_size;

        // Handle the last short block
        if (start + current_size > n)
        {
            current_size = n - start;
        }

        solve_factored_block(
            start,
            current_size,
            r,
            z,
            lower_1d,
            inv_upper_1d);
    }
}


// Apply BJ to the 2D problem
void BJ::apply(
    const Poisson2D& A,
    const Vector& r,
    Vector& z
) const
{
    int n = A.n;

    z.set_zero();

    // Process each row
    for (int row = 0; row < n; row++)
    {
        for (int col = 0; col < n; col += block_size)
        {
            int size = block_size;

            // Handle the last short block
            if (col + size > n)
            {
                size = n - col;
            }

            int start = row * n + col;

            solve_factored_block(
                start,
                size,
                r,
                z,
                lower_2d,
                inv_upper_2d);
        }
    }
}

  
