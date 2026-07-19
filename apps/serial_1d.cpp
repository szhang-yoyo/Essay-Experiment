/*
 * serial_1d.cpp
 *
 * Phase 1 driver for serial CG and PCG-BJ tests.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>

#include "../core/vector.hpp"
#include "../problems/poisson1d.hpp"
#include "../preconditioners/block_jacobi.hpp"
#include "../solvers/cg_solver.hpp"
#include "../solvers/pcg_solver.hpp"

// Load problem sizes from a plain text file
std::vector<int> read_problem_sizes(const std::string& filename) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Cannot open problem size file: " + filename);
    }

    std::vector<int> sizes;
    int n;

    while (file >> n) {
        if (n > 0) {
            sizes.push_back(n);
        }
    }

    if (sizes.empty()) {
        throw std::runtime_error("No valid problem sizes found.");
    }

    return sizes;
}

int main() {
    const std::string size_file = "config/problem_sizes.txt";
    const std::string output_file = "results/phase1/serial_1d_results.csv";

    const int max_iter = 100000;
    const double tol = 1e-8;
    const int block_size = 20;
    const int repeats = 5;

    std::vector<int> sizes = read_problem_sizes(size_file);

    std::ofstream file(output_file);

    if (!file.is_open()) {
        throw std::runtime_error("Cannot open output file: " + output_file);
    }

    file << "method,n,block_size,tol,max_iter,repeats,"
         << "iterations_avg,runtime_avg,relative_residual\n";

    for (int n : sizes) {
        Poisson1D A(n);
        Vector b(n, 1.0);

        // ---- CG runs ----
        double cg_runtime_sum = 0.0;
        double cg_residual = 0.0;
        int cg_iterations = 0;

        for (int rep = 0; rep < repeats; ++rep) {
            Vector x(n, 0.0);

            SolverResult result =
                conjugate_gradient(A, b, x, max_iter, tol);

            cg_runtime_sum += result.runtime;
            cg_residual = result.relative_residual;
            cg_iterations = result.iterations;
        }

        double cg_runtime_avg = cg_runtime_sum / repeats;

        file << "CG,"
             << n << ","
             << 0 << ","
             << tol << ","
             << max_iter << ","
             << repeats << ","
             << cg_iterations << ","
             << cg_runtime_avg << ","
             << cg_residual << "\n";

        std::cout << "CG      n=" << n
                  << "  iter=" << cg_iterations
                  << "  time=" << cg_runtime_avg
                  << "s  relres=" << cg_residual << "\n";

        // ---- PCG-BJ runs ----
        BJ M(block_size);

        double pcg_runtime_sum = 0.0;
        double pcg_residual = 0.0;
        int pcg_iterations = 0;

        for (int rep = 0; rep < repeats; ++rep) {
            Vector x(n, 0.0);

            SolverResult result =
                pcg(A, M, b, x, max_iter, tol);

            pcg_runtime_sum += result.runtime;
            pcg_residual = result.relative_residual;
            pcg_iterations = result.iterations;
        }

        double pcg_runtime_avg = pcg_runtime_sum / repeats;

        file << "PCG-BJ,"
             << n << ","
             << block_size << ","
             << tol << ","
             << max_iter << ","
             << repeats << ","
             << pcg_iterations << ","
             << pcg_runtime_avg << ","
             << pcg_residual << "\n";

        std::cout << "PCG-BJ  n=" << n
                  << "  block=" << block_size
                  << "  iter=" << pcg_iterations
                  << "  time=" << pcg_runtime_avg
                  << "s  relres=" << pcg_residual << "\n";
    }

    file.close();

    std::cout << "\nWrote results to " << output_file << "\n";

    return 0;
}
