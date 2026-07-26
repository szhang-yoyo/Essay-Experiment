#include "mpi_pcg_solver.hpp"

#include <cmath>

#include "../parallel/mpi_vector.hpp"

int mpi_pcg(const Poisson2DMPI& A,
            const MPIBJ& M,
            const Vector& b,
            Vector& x,
            int max_iter,
            double tol,
            double& rel_res,
            MPI_Comm comm,
            MPITimes* times)
{
    int lo_size = b.size();

    Vector r(lo_size);
    Vector z(lo_size);
    Vector p(lo_size);
    Vector Ap(lo_size);

    // Compute the initial residual
    A.matvec(x, Ap, times);

    double t0 = MPI_Wtime();

    for (int i = 0; i < lo_size; ++i)
    {
        r[i] = b[i] - Ap[i];
    }

    if (times != nullptr)
    {
        times->vector_time += MPI_Wtime() - t0;
    }



    // Apply the preconditioner
    t0 = MPI_Wtime();
    M.apply(r, z);

   if (times != nullptr)
   {
       times->preconditioner_time += MPI_Wtime() - t0;

   }
        // Copy z to p
       t0 = MPI_Wtime();
        for (int i = 0; i < lo_size; ++i)
        {
            p[i] = z[i];
        }

     if (times != nullptr)
     {
         times->vector_time += MPI_Wtime() - t0;
     }

    double b_norm = global_norm(b, comm, times);

    if (b_norm == 0.0)
    {
        b_norm = 1.0;
    }

    double rz = global_dot(r, z, comm, times);

    rel_res = global_norm(r, comm, times) / b_norm;

    if (rel_res < tol)
    {
        return 0;
    }

    for (int iter = 0; iter < max_iter; ++iter)
    {
        // Compute Ap
        A.matvec(p, Ap, times);

        // Compute the step length
        double pAp = global_dot(p, Ap, comm, times);
        double alpha = rz / pAp;

        // Update x and r
        t0 = MPI_Wtime();
        for (int i = 0; i < lo_size; ++i)
        {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }

        if (times != nullptr)
        {
            times->vector_time +=  MPI_Wtime() - t0;
        }

        rel_res = global_norm(r, comm, times) / b_norm;

        if (rel_res < tol)
        {
            return iter + 1;
        }

        // Apply the preconditioner
        t0 = MPI_Wtime();
        M.apply(r, z);

        if (times != nullptr)
        {
            times->preconditioner_time +=  MPI_Wtime() - t0;
        }



        double rz_new = global_dot(r, z, comm, times);
        double beta = rz_new / rz;

        // Update the search direction
        t0 = MPI_Wtime();

        for (int i = 0; i < lo_size; ++i)
        {
            p[i] = z[i] + beta * p[i];
        }

         if (times != nullptr)
        {
            times->vector_time +=  MPI_Wtime() - t0;
        }


        rz = rz_new;
    }

    return max_iter;
}


int mpi_pcg(const Poisson2DMPI& A,
            const AS& M,
            const Vector& b,
            Vector& x,
            int max_iter,
            double tol,
            double& rel_res,
            MPI_Comm comm,
            MPITimes* times)
{
    int lo_size = b.size();

    Vector r(lo_size);
    Vector z(lo_size);
    Vector p(lo_size);
    Vector Ap(lo_size);

    // Compute the initial residual
    A.matvec(x, Ap,times);

    double t0 = MPI_Wtime();
    for (int i = 0; i < lo_size; ++i)
    {
        r[i] = b[i] - Ap[i];
    }

     if (times != nullptr)
     {
         times->vector_time +=  MPI_Wtime() - t0;
     }




    // Apply the preconditioner
    t0 = MPI_Wtime();
    M.apply(r, z);

     if (times != nullptr)
     {
         times->preconditioner_time +=  MPI_Wtime() - t0;
     }

    // Copy z to p
    t0 = MPI_Wtime();

    for (int i = 0; i < lo_size; ++i)
    {
        p[i] = z[i];
    }

     if (times != nullptr)
     {
         times->vector_time +=  MPI_Wtime() - t0;
     }


    double b_norm = global_norm(b, comm, times);

    if (b_norm == 0.0)
    {
        b_norm = 1.0;
    }

    double rz = global_dot(r, z, comm, times);

    rel_res = global_norm(r, comm, times) / b_norm;

    if (rel_res < tol)
    {
        return 0;
    }

    for (int iter = 0; iter < max_iter; ++iter)
    {
        // Compute Ap
        A.matvec(p, Ap, times);

        // Compute the step length
        double pAp = global_dot(p, Ap, comm, times);
        double alpha = rz / pAp;

        // Update x and r
        t0 = MPI_Wtime();

        for (int i = 0; i < lo_size; ++i)
        {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }

        if (times != nullptr)
        {
            times->vector_time +=  MPI_Wtime() - t0;
        }


        rel_res = global_norm(r, comm, times) / b_norm;

        if (rel_res < tol)
        {
            return iter + 1;
        }

        // Apply the preconditioner
        t0 = MPI_Wtime();

        M.apply(r, z);


        if (times != nullptr)
        {
            times->preconditioner_time +=  MPI_Wtime() - t0;
        }


        double rz_new = global_dot(r, z, comm, times);
        double beta = rz_new / rz;

        // Update the search direction
        t0 = MPI_Wtime();

        for (int i = 0; i < lo_size; ++i)
        {
            p[i] = z[i] + beta * p[i];
        }

        if (times != nullptr)
        {
            times->vector_time +=  MPI_Wtime() - t0;
        }

        rz = rz_new;
    }

    return max_iter;
}
