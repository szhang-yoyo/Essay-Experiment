#include "poisson2d_mpi.hpp"
#include <mpi.h>
#include <vector>

Poisson2DMPI::Poisson2DMPI(int n_in, int rank, int np)
    : n(n_in),
      part(n_in, rank, np),
      halo(n_in, rank, np, part.get_local_rows())
{
}

int Poisson2DMPI::fst_row() const
{
    return part.get_first_row();
}

int Poisson2DMPI::lst_row() const
{
    return part.get_last_row();
}

int Poisson2DMPI::lo_rows() const
{
    return part.get_local_rows();
}

int Poisson2DMPI::size() const
{
    return part.get_local_size();
}

double Poisson2DMPI::diag() const
{
    return 4.0;
}

void Poisson2DMPI::matvec(const Vector& x, Vector& y, MPITimes* times) const
{
    double spmv_start = MPI_Wtime();

    // Local partition information
    int fst = part.get_first_row();
    int nr = part.get_local_rows();
    int lo_n = part.get_local_size();

    // Convert Vector to std::vector for halo exchange
    std::vector<double> x_vec(lo_n);

    for (int i = 0; i < lo_n; i++)
    {
        x_vec[i] = x[i];
    }

    // Halo rows from neighbouring processes
    std::vector<double> top_vec(n, 0.0);
    std::vector<double> bot_vec(n, 0.0);

    y.set_zero();

    // Exchange boundary rows
    double halo_start = MPI_Wtime();
    halo.exchange(x_vec, top_vec, bot_vec);


    if (times != nullptr)
    {
        times->halo_time += MPI_Wtime() - halo_start;
    }

    // Loop over local rows
     for (int lo_i = 0; lo_i < nr; lo_i++)
    {
        int gl_i = fst + lo_i;

        // Loop over columns
        for (int j = 0; j < n; j++)
        {
            int id = lo_i * n + j;

            // Centre value
            double val = 4.0 * x[id];

            // Left neighbour
            if (j > 0)
            {
                val -= x[id - 1];
            }

            // Right neighbour
            if (j < n - 1)
            {
                val -= x[id + 1];
            }

            // Top neighbour
            if (gl_i > 0)
            {
                if (lo_i > 0)
                {
                    val -= x[id - n];
                }
                else
                {
                    val -= top_vec[j];
                }
            }

            // Bottom neighbour
            if (gl_i < n - 1)
            {
                if (lo_i < nr - 1)
                {
                    val -= x[id + n];
                }
                else
                {
                    val -= bot_vec[j];
                }
            }

            y[id] = val;
        }
    }


    if (times != nullptr)
    {
        times->spmv_time += MPI_Wtime() - spmv_start;
    }

}
