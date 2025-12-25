#include "../defines.h"

__device__ void gaussSolveFunction(double * mat, double * x, double * rhs,
                                   unsigned int N, unsigned int dim, 
                                   unsigned int n, unsigned int row)
{
    for (unsigned int r = 0; r < dim; ++r)
    {
        __syncthreads();

        if (n < N)
        {
            if (row != r)
            {
                double m = mat[row + dim * r] / mat[r + dim * r];

                rhs[row] -= m * rhs[r];

                for (unsigned int col = r; col < dim; ++col)
                    mat[row + dim * col] -= m * mat[r + dim * col];
            }
        }
    }

    if (n < N)
        x[row + dim * n] = rhs[row] / mat[row + dim * row];
}
