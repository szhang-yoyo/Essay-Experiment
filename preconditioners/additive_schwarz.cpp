#include "additive_schwarz.hpp"
#include "local_sgs.hpp"
#include <algorithm>
#include <vector>

AS::AS(
    int n,
    int rank,
    int np,
    int delta,
    MPI_Comm comm
)
    : n_(n),
      rank_(rank),
      np_(np),
      delta_(std::max(0, delta)),
      comm_(comm),
      part_(n, rank, np)
{
    fst_row_ = part_.get_first_row();
    lst_row_ = part_.get_last_row();
    lo_rows_ = part_.get_local_rows();

    // Extend the local row range
    ov_fst_row_ = std::max(0, fst_row_ - delta_);
    ov_lst_row_ = std::min(n_ - 1, lst_row_ + delta_);

    ov_rows_ = ov_lst_row_ - ov_fst_row_ + 1;
    ov_size_ = ov_rows_ * n_;
}

void AS::exch_res(const Vector& r, Vector& ov_r) const
{
    ov_r.set_zero();

    // Copy locally owned residual rows
    int lo_offset = (fst_row_ - ov_fst_row_) * n_;

    for (int i = 0; i < lo_rows_ * n_; i++)
    {
        ov_r[lo_offset + i] = r[i];
    }

    std::vector<MPI_Request> reqs;

    // Exchange residual rows with overlapping processes
    for (int peer = 0; peer < np_; peer++)
    {
        if (peer == rank_)
        {
            continue;
        }

        Partition peer_part(n_, peer, np_);

        int peer_fst = peer_part.get_first_row();
        int peer_lst = peer_part.get_last_row();

        int peer_ov_fst = std::max(0, peer_fst - delta_);
        int peer_ov_lst = std::min(n_ - 1, peer_lst + delta_);

        // Receive rows owned by peer
        int recv_fst = std::max(ov_fst_row_, peer_fst);
        int recv_lst = std::min(ov_lst_row_, peer_lst);

        if (recv_fst <= recv_lst)
        {
            int recv_rows = recv_lst - recv_fst + 1;
            int recv_pos = (recv_fst - ov_fst_row_) * n_;

            MPI_Request req;

            MPI_Irecv(
                &ov_r[recv_pos],
                recv_rows * n_,
                MPI_DOUBLE,
                peer,
                100,
                comm_,
                &req
            );

            reqs.push_back(req);
        }

        // Send rows needed by peer
        int send_fst = std::max(fst_row_, peer_ov_fst);
        int send_lst = std::min(lst_row_, peer_ov_lst);

        if (send_fst <= send_lst)
        {
            int send_rows = send_lst - send_fst + 1;
            int send_pos = (send_fst - fst_row_) * n_;

            MPI_Request req;

            MPI_Isend(
                &r[send_pos],
                send_rows * n_,
                MPI_DOUBLE,
                peer,
                100,
                comm_,
                &req
            );

            reqs.push_back(req);
        }
    }

    if (!reqs.empty())
    {
        MPI_Waitall(
            static_cast<int>(reqs.size()),
            reqs.data(),
            MPI_STATUSES_IGNORE
        );
    }
}

void AS::solve_local(
    const Vector& ov_r,
    Vector& ov_z
) const
{
       local_sgs(ov_r, ov_z, ov_rows_, n_, 4);
}

void AS::add_corr(
    const Vector& ov_z,
    Vector& z
) const
{
    z.set_zero();

    // Add the local correction
    int ov_offset = (fst_row_ - ov_fst_row_) * n_;

    for (int i = 0; i < lo_rows_ * n_; i++)
    {
        z[i] += ov_z[ov_offset + i];
    }

    std::vector<std::vector<double>> recv_buf(np_);
    std::vector<int> recv_fst_row(np_, 0);
    std::vector<MPI_Request> reqs;

    // Exchange corrections on shared rows
    for (int peer = 0; peer < np_; peer++)
    {
        if (peer == rank_)
        {
            continue;
        }

        Partition peer_part(n_, peer, np_);

        int peer_fst = peer_part.get_first_row();
        int peer_lst = peer_part.get_last_row();

        int peer_ov_fst = std::max(0, peer_fst - delta_);
        int peer_ov_lst = std::min(n_ - 1, peer_lst + delta_);

        // Receive corrections for locally owned rows
        int recv_fst = std::max(fst_row_, peer_ov_fst);
        int recv_lst = std::min(lst_row_, peer_ov_lst);

        if (recv_fst <= recv_lst)
        {
            int recv_rows = recv_lst - recv_fst + 1;

            recv_fst_row[peer] = recv_fst;
            recv_buf[peer].resize(recv_rows * n_);

            MPI_Request req;

            MPI_Irecv(
                recv_buf[peer].data(),
                recv_rows * n_,
                MPI_DOUBLE,
                peer,
                200,
                comm_,
                &req
            );

            reqs.push_back(req);
        }

        // Send corrections for peer-owned rows
        int send_fst = std::max(ov_fst_row_, peer_fst);
        int send_lst = std::min(ov_lst_row_, peer_lst);

        if (send_fst <= send_lst)
        {
            int send_rows = send_lst - send_fst + 1;
            int send_pos = (send_fst - ov_fst_row_) * n_;

            MPI_Request req;

            MPI_Isend(
                &ov_z[send_pos],
                send_rows * n_,
                MPI_DOUBLE,
                peer,
                200,
                comm_,
                &req
            );

            reqs.push_back(req);
        }
    }

    if (!reqs.empty())
    {
        MPI_Waitall(
            static_cast<int>(reqs.size()),
            reqs.data(),
            MPI_STATUSES_IGNORE
        );
    }

    // Accumulate received corrections
    for (int peer = 0; peer < np_; peer++)
    {
        if (recv_buf[peer].empty())
        {
            continue;
        }

        int start = (recv_fst_row[peer] - fst_row_) * n_;

        for (int i = 0;
             i < static_cast<int>(recv_buf[peer].size());
             i++)
        {
            z[start + i] += recv_buf[peer][i];
        }
    }
}

void AS::apply(
    const Vector& r,
    Vector& z
) const
{
    Vector ov_r(ov_size_);
    Vector ov_z(ov_size_);

    // Apply the overlapping local correction
    exch_res(r, ov_r);
    solve_local(ov_r, ov_z);
    add_corr(ov_z, z);
}

int AS::delta() const
{
    return delta_;
}

int AS::ov_fst_row() const
{
    return ov_fst_row_;
}

int AS::ov_lst_row() const
{
    return ov_lst_row_;
}

int AS::ov_rows() const
{
    return ov_rows_;
}

int AS::ov_size() const
{
    return ov_size_;
}
