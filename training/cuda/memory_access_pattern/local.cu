#include "../defines.h"

constexpr unsigned int num_threads = 100000000;
constexpr unsigned int loop_count = 1000;

__global__ void powerOfTwo(double * result, double * input, unsigned int size)
{
    uint64_t tid = threadIdx.x + blockDim.x * blockIdx.x;

    if (tid >= num_threads)
        return;
    
    double val = input[tid];
    double sum = 0;

    for (unsigned int i = 0; i < size; ++i)
    {
        sum += val;
        val = sum;
    }

    result[tid] = sum;
}

int main(int argc, char * argv[])
{
    cudaCheckError(cudaSetDevice(0));

    double * result;
    double * input;

    vector<double> host_input(num_threads, 1);

    cudaCheckError(cudaMalloc(&result, num_threads * sizeof(double)));
    cudaCheckError(cudaMalloc(&input, num_threads * sizeof(double)));
    cudaCheckError(cudaMemcpy(input, host_input.data(), num_threads * sizeof(double), cudaMemcpyHostToDevice));

    dim3 blocks = dim3(num_threads / 128 + 1, 1, 1);
    dim3 threads = dim3(128, 1, 1);

    const auto start{chrono::steady_clock::now()};

    powerOfTwo <<< blocks, threads >>> (result, input, loop_count);

    cudaCheckError(cudaDeviceSynchronize());

    const auto finish{chrono::steady_clock::now()};
    const chrono::duration<double> elapsed_seconds{finish - start};

    cout << "Elapsed : " << elapsed_seconds.count() << " seconds" << endl;
}
