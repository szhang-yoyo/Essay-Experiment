/*
 * test_poisson2d.cpp
 *
 * Test the 2D Poisson operator.
 */

#include <iostream>

#include "../problems/poisson2d.hpp"

int main()
{
    Poisson2D problem(3);

    Vector x(problem.size);
    Vector y(problem.size);

    // Fill x with ones
    for (int i = 0; i < problem.size; i++) {
        x[i] = 1.0;
    }

    // Compute y = A * x
    problem.matvec(x, y);

    // Print the result
    std::cout << "Result:" << std::endl;

    for (int i = 0; i < problem.size; i++) {
        std::cout << y[i] << std::endl;
    }

    return 0;
}
