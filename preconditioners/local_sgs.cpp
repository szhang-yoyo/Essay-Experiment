#include "local_sgs.hpp"

void local_sgs(const Vector& rhs,
               Vector& x,
               int rows,
               int cols,
               int sweeps)
{
    x.set_zero();

    for (int s = 0; s < sweeps; ++s)
    {
        // Forward sweep
        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < cols; ++j)
            {
                int k = i * cols + j;
                double sum = rhs[k];

                if (i > 0)
                {
                    sum += x[k - cols];
                }

                if (i < rows - 1)
                {
                    sum += x[k + cols];
                }

                if (j > 0)
                {
                    sum += x[k - 1];
                }

                if (j < cols - 1)
                {
                    sum += x[k + 1];
                }

                x[k] = sum / 4.0;
            }
        }

        // Backward sweep
        for (int i = rows - 1; i >= 0; --i)
        {
            for (int j = cols - 1; j >= 0; --j)
            {
                int k = i * cols + j;
                double sum = rhs[k];

                if (i > 0)
                {
                    sum += x[k - cols];
                }

                if (i < rows - 1)
                {
                    sum += x[k + cols];
                }

                if (j > 0)
                {
                    sum += x[k - 1];
                }

                if (j < cols - 1)
                {
                    sum += x[k + 1];
                }

                x[k] = sum / 4.0;
            }
        }
    }
}
