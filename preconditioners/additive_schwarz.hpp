#ifndef ADDITIVE_SCHWARZ_HPP
#define ADDITIVE_SCHWARZ_HPP

#include <mpi.h>

#include "../core/vector.hpp"
#include "../parallel/partition.hpp"

class AS
{
private:
    int n_;
    int rank_;
    int np_;
    int delta_;
    int sgs_sweeps_;

    MPI_Comm comm_;
    Partition part_;

    int fst_row_;
    int lst_row_;
    int lo_rows_;

    int ov_fst_row_;
    int ov_lst_row_;
    int ov_rows_;
    int ov_size_;

    // Exchange residual values in the overlap
    void exch_res(const Vector& r, Vector& ov_r) const;

    // Solve the local overlapping problem
    void solve_local(const Vector& ov_r, Vector& ov_z) const;

    // Add corrections on shared rows
    void add_corr(const Vector& ov_z, Vector& z) const;

public:
    // Create an Additive Schwarz preconditioner
    AS(
        int n,
        int rank,
        int np,
        int delta,
        MPI_Comm comm = MPI_COMM_WORLD
    );

    // Apply the preconditioner
    void apply(const Vector& r, Vector& z) const;

    // Return overlap information
    int delta() const;
    int ov_fst_row() const;
    int ov_lst_row() const;
    int ov_rows() const;
    int ov_size() const;
};

#endif
