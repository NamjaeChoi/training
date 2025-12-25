#include "../defines.h"

constexpr unsigned int num_threads = 100000000;
constexpr unsigned int loop_count = 10;

__global__ void increment(double * result, unsigned int size)
{
    uint64_t tid = threadIdx.x + blockDim.x * blockIdx.x;

    if (tid >= num_threads)
        return;
    
    double sum[10] = { 0 };

    for (unsigned int i = 0; i < size; ++i)
        for (unsigned int j = 0; j < 10; ++j)
            sum[j] += i * j;
    
    for (unsigned int j = 0; j < 10; ++j)
        result[tid] += sum[j];
}

int main(int argc, char * argv[])
{
    cudaCheckError(cudaSetDevice(0));

    double * result;

    cudaCheckError(cudaMalloc(&result, num_threads * sizeof(double)));

    dim3 blocks = dim3(num_threads / 128 + 1, 1, 1);
    dim3 threads = dim3(128, 1, 1);

    const auto start{chrono::steady_clock::now()};

    increment <<< blocks, threads >>> (result, loop_count);

    cudaCheckError(cudaDeviceSynchronize());

    const auto finish{chrono::steady_clock::now()};
    const chrono::duration<double> elapsed_seconds{finish - start};

    cout << "Elapsed : " << elapsed_seconds.count() << " seconds" << endl;
}
