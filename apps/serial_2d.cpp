/*
 * serial_2d.cpp
 *
 * Serial CG solver for the 2D Poisson problem.
 */

#include <fstream>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include "../problems/poisson2d.hpp"
#include "../solvers/cg_solver.hpp"
#include "../preconditioners/block_jacobi.hpp"
#include "../solvers/pcg_solver.hpp"


int main(int argc, char* argv[])
{
    // Check the command line
    if (argc != 4) {
        std::cout << "Usage: ./serial_2d <grid_size> <block_size>" << std::endl;
        return 1;
    }

    // Read the grid size
    //int n = std::atoi(argv[1]);
   // int block_size = std::atoi(argv[2]);

    // Read the command-line parameters
    int n = std::atoi(argv[1]);
    int block_size = std::atoi(argv[2]);
    int repeat = std::atoi(argv[3]);

    // Create the 2D Poisson problem
    Poisson2D problem(n);

    // Create the right-hand side vector
    Vector b(problem.size);

    // Create the initial guess
    Vector x(problem.size);

    // Create the PCG solution vector
    Vector x_pcg(problem.size);

    // Create the exact solution vector
   // Vector exact(problem.size);

    // Initial guess x = 0
    x.set_zero();
    x_pcg.set_zero();
    
    // Set the right-hand side b = 1 
   for (int i = 0; i < problem.size; ++i) {
       b[i] = 1.0;
   }

    // Solver parameters
    int max_iter = 5000;
    double tol = 1.0e-8;
    // Use one grid row as one block
    //int block_size = n;
    // Create the Block Jacobi preconditioner
    BJ preconditioner(block_size);


    // Solve the linear system
    SolverResult result =
        conjugate_gradient(problem, b, x, max_iter, tol);

    // Solve using PCG with Block Jacobi
    SolverResult pcg_result =
        pcg(problem, preconditioner, b, x_pcg, max_iter, tol);
    
    // Compute the relative L2 error
   // double relative_error =
    //    problem.comp_re_err(x, exact);

    // Compute the PCG relative L2 error
   // double pcg_error =
  //        problem.comp_re_err(x_pcg, exact);
    
    // Compare the CG and PCG solutions
    double diff_sum = 0.0;
    double solution_sum = 0.0;

    for (int i = 0; i < problem.size; ++i) {
        double diff = x[i] - x_pcg[i];

        diff_sum += diff * diff;
        solution_sum += x[i] * x[i];
    }

    double solution_difference =
        std::sqrt(diff_sum / solution_sum);

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

    //std::cout << "Relative L2 error: "
      //        << relative_error << std::endl;

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

   // std::cout << "Relative L2 error: "
     //         << pcg_error << std::endl;

    std::cout << "CG-PCG solution difference: "
              << solution_difference << std::endl;
    
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

    // CSV output file
    const char* output_file =
        "results/phase2/serial_2d_results.csv";

    // Check
    std::ifstream check_file(output_file);
    bool file_has_data =
        check_file.good() &&
        check_file.peek() != std::ifstream::traits_type::eof();

    check_file.close();

    // Open the file 
    std::ofstream csv(output_file, std::ios::app);

    if (!csv) {
        std::cerr << "Cannot open output file." << std::endl;
        return 1;
    }

    // Write the header once
    if (!file_has_data) {
        csv
            << "grid_size,"
            << "unknowns,"
            << "block_size,"
            << "repeat,"
            << "cg_iterations,"
        << "cg_residual,"
        << "cg_runtime,"
        << "pcg_iterations,"
        << "pcg_residual,"
        << "pcg_runtime,"
        << "solution_difference"
        << std::endl;
    }

    // Write one result row
    csv
    << n << ","
    << problem.size << ","
    << block_size << ","
    << repeat << ","
    << result.iterations << ","
    << result.relative_residual << ","
    << result.runtime << ","
    << pcg_result.iterations << ","
    << pcg_result.relative_residual << ","
    << pcg_result.runtime << ","
    << solution_difference
    << std::endl;

    std::cout
        << "\nWrote result to "
        << output_file
        << std::endl;



    return 0;
}
