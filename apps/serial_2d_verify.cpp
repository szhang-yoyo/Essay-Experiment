/*
 * serial_2d.cpp
 *
 * Serial CG solver for the 2D Poisson problem.
 */

#include <iostream>
#include <cstdlib>

#include "../problems/poisson2d.hpp"
#include "../solvers/cg_solver.hpp"
#include "../preconditioners/block_jacobi.hpp"
#include "../solvers/pcg_solver.hpp"


int main(int argc, char* argv[])
{
    // Check the command line
    if (argc != 2) {
        std::cout << "Usage: ./serial_2d <grid_size>" << std::endl;
        return 1;
    }

    // Read the grid size
    int n = std::atoi(argv[1]);

    // Create the 2D Poisson problem
    Poisson2D problem(n);

    // Create the right-hand side vector
    Vector b(problem.size);

    // Create the initial guess
    Vector x(problem.size);

    // Create the PCG solution vector
    Vector x_pcg(problem.size);

    // Create the exact solution vector
    Vector exact(problem.size);

    // Initial guess x = 0
    x.set_zero();
    x_pcg.set_zero();
    // Build the right-hand side
    problem.build_rhs(b);

    // Build the exact solution
    problem.b_ex_sol(exact);

    // Solver parameters
    int max_iter = 1000;
    double tol = 1.0e-8;
    // Use one grid row as one block
    int block_size = n;
    // Create the Block Jacobi preconditioner
    BlockJacobi preconditioner(block_size);


    // Solve the linear system
    SolverResult result =
        conjugate_gradient(problem, b, x, max_iter, tol);

    // Solve using PCG with Block Jacobi
    SolverResult pcg_result =
        pcg(problem, preconditioner, b, x_pcg, max_iter, tol);
    
    // Compute the relative L2 error
    double relative_error =
        problem.comp_re_err(x, exact);

    // Compute the PCG relative L2 error
    double pcg_error =
        problem.comp_re_err(x_pcg, exact);

    std::cout << "2D Poisson" << std::endl;
    std::cout << "Grid size: "
              << problem.n << " x " << problem.n << std::endl;
    std::cout << "Unknowns: "
              << problem.size << std::endl;

    std::cout << "CG Solver Results" << std::endl;
    std::cout << "Iterations: "
              << result.iterations << std::endl;

    std::cout << "Relative residual: "
              << result.relative_residual << std::endl;

    std::cout << "Runtime: "
              << result.runtime << std::endl;

    std::cout << "Relative L2 error: "
              << relative_error << std::endl;

    //PCG:
    std::cout << std::endl;
    std::cout << "PCG-Block Jacobi Results" << std::endl;

    std::cout << "Block size: "
              << block_size << std::endl;

    std::cout << "Iterations: "
              << pcg_result.iterations << std::endl;

    std::cout << "Relative residual: "
              << pcg_result.relative_residual << std::endl;

    std::cout << "Runtime: "
              << pcg_result.runtime << std::endl;

    std::cout << "Relative L2 error: "
              << pcg_error << std::endl;


    std::cout << "First 10 values of x" << std::endl;

    int print_size = 10;

    if (problem.size < print_size) 
    {
        print_size = problem.size;
    }

    for (int i = 0; i < print_size; ++i) 
    {
        std::cout << "x[" << i << "] = "
                  << x[i]
                  << std::endl;
    }

    return 0;
}
