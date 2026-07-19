#ifndef BLOCK_JACOBI_HPP
#define BLOCK_JACOBI_HPP

/*
 * block_jacobi.hpp
 *
 * Definition of the Block Jacobi preconditioner.
 */

#include <vector>
#include "../core/vector.hpp"
#include "../problems/poisson1d.hpp"
#include "../problems/poisson2d.hpp"

class BJ {
public:
    // Local block size used by the preconditioner
    int block_size;

    // Build a Block Jacobi preconditioner
    explicit BJ(int block_size_);

    // Apply the preconditioner to a 1D Poi
    void apply(const Poisson1D& A, const Vector& r, Vector& z) const;

    // Apply the preconditioner to a 2D Poi
    void apply(const Poisson2D& A, const Vector& r, Vector& z) const;

private:
    // 1D Poisson 局部矩阵的 LU 分解
    std::vector<double> lower_1d;
    std::vector<double> inv_upper_1d;

    // 2D 行内块局部矩阵的 LU 
    std::vector<double> lower_2d;
    std::vector<double> inv_upper_2d;

    // 每次 apply() 重复使用
    mutable std::vector<double> work;

    // 预计算三对角矩阵的 LU 分解
    void build_factor(
        double diagonal,
        std::vector<double>& lower,
        std::vector<double>& inv_upper);

    // 使用已经计算好的 LU 分解求解局部系统
    void solve_factored_block(
        int start,
        int size,
        const Vector& r,
        Vector& z,
        const std::vector<double>& lower,
        const std::vector<double>& inv_upper) const;
};

#endif
