#include "mpi_cg_solver.hpp"

#include <cmath>

#include "../parallel/mpi_vector.hpp"

int mpi_cg(const Poisson2DMPI& A,
           const Vector& b,
           Vector& x,
           int max_iter,
           double tol,
           double& rel_res,
           MPI_Comm comm)
{
    int lo_size = b.size();

    Vector r(lo_size);
    Vector p(lo_size);
    Vector Ap(lo_size);

    // Compute the initial residual r = b - Ax
    A.matvec(x, Ap);

    for (int i = 0; i < lo_size; ++i)
    {
        r[i] = b[i] - Ap[i];
        p[i] = r[i];
    }

    // Compute the norm of the right-hand side
    double b_norm = global_norm(b, comm);

    if (b_norm == 0.0)
    {
        b_norm = 1.0;
    }

    // Compute the initial residual norm
    double rr = global_dot(r, r, comm);

    rel_res = std::sqrt(rr) / b_norm;

    if (rel_res < tol)
    {
        return 0;
    }

    for (int iter = 0; iter < max_iter; ++iter)
    {
        // Compute Ap
        A.matvec(p, Ap);

        // Compute the step length
        double pAp = global_dot(p, Ap, comm);
        double alpha = rr / pAp;

        // Update x and r
        for (int i = 0; i < lo_size; ++i)
        {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }

        // Compute the new residual norm
        double rr_new = global_dot(r, r, comm);

        rel_res = std::sqrt(rr_new) / b_norm;

        if (rel_res < tol)
        {
            return iter + 1;
        }

        // Update the search direction
        double beta = rr_new / rr;

        for (int i = 0; i < lo_size; ++i)
        {
            p[i] = r[i] + beta * p[i];
        }

        rr = rr_new;
    }

    return max_iter;
}
