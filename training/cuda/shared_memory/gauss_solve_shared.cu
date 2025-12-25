#include "../defines.h"

constexpr unsigned int N = 10000000;
constexpr unsigned int dim = 8;
constexpr unsigned int block_size = 128;

__global__ void gaussSolve(double * A, double * x, double * b)
{
    auto row = threadIdx.x;
    auto n = threadIdx.y + blockDim.y * blockIdx.x;

    __shared__ double shmem[dim * (dim + 1) * (block_size / dim)];

    double * mat = shmem + dim * (dim + 1) * threadIdx.y;
    double * rhs = mat + dim * dim;

    if (n < N)
    {
        for (unsigned int col = 0; col < dim; ++col)
            mat[row + dim * col] = A[row + dim * (col + dim * n)];
        
        rhs[row] = b[row + dim * n];
    }

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

int main(int argc, char * argv[])
{
    cudaCheckError(cudaSetDevice(0));

    double * A;
    double * x;
    double * b;

    vector<double> A_host(dim * dim * N, 0);
    vector<double> b_host(dim * N, 0);
    vector<double> x_host(dim * N, 0);

    for (unsigned int n = 0; n < N; ++n)
    {
        b_host[n * dim] = 1;
        b_host[dim - 1 + n * dim] = 1;

        for (unsigned int i = 0; i < dim; ++i)
        {
            A_host[i + dim * (i + dim * n)] = 2;

            if (i != 0)
                A_host[i - 1 + dim * (i + dim * n)] = -1;
            if (i != dim - 1)
                A_host[i + 1 + dim * (i + dim * n)] = -1;
        }
    }

    cudaCheckError(cudaMalloc(&A, dim * dim * N * sizeof(double)));
    cudaCheckError(cudaMalloc(&x, dim * N * sizeof(double)));
    cudaCheckError(cudaMalloc(&b, dim * N * sizeof(double)));
    cudaCheckError(cudaMemcpy(A, A_host.data(), dim * dim * N * sizeof(double), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(b, b_host.data(), dim * N * sizeof(double), cudaMemcpyHostToDevice));

    dim3 blocks = dim3(dim * N / block_size + 1, 1, 1);
    dim3 threads = dim3(dim, (block_size / dim), 1);

    const auto start{chrono::steady_clock::now()};

    for (unsigned int i = 0; i < 100; ++i)
    {
        gaussSolve <<< blocks, threads >>> (A, x, b);
        cudaCheckError(cudaDeviceSynchronize());
    }

    const auto finish{chrono::steady_clock::now()};
    const chrono::duration<double> elapsed_seconds{finish - start};

    cout << "Elapsed : " << elapsed_seconds.count() << " seconds" << endl;

    cudaCheckError(cudaMemcpy(x_host.data(), x, dim * N * sizeof(double), cudaMemcpyDeviceToHost));

    if (all_of(x_host.begin(), x_host.end(), [](double val) { return abs(val - 1) < 1.0e-8; }))
        cout << "All solutions are unity" << endl;
}
