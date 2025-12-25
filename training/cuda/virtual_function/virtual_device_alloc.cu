#include "../defines.h"

constexpr unsigned int num_threads = 100000000;
constexpr unsigned int loop_count = 1000;

class Base
{
public:
    __host__ __device__ Base() {}
    __device__ virtual double get() { return 1; }
};

class Derived : public Base
{
public:
    __host__ __device__ Derived() {}
    __device__ virtual double get() override { return 2; }
};

__global__ void alloc(Base * value)
{
    new (value) Derived;
}

__global__ void increment(Base * value, double * result, unsigned int size)
{
    uint64_t tid = threadIdx.x + blockDim.x * blockIdx.x;

    if (tid >= num_threads)
        return;
    
    double sum = 0;

    for (unsigned int i = 0; i < size; ++i)
        sum += value->get();
    
    result[tid] = sum;
}

int main(int argc, char * argv[])
{
    cudaCheckError(cudaSetDevice(0));

    double * result;
    Base * value;

    cudaCheckError(cudaMalloc(&result, num_threads * sizeof(double)));
    cudaCheckError(cudaMalloc(&value, sizeof(Derived)));

    alloc <<< 1, 1 >>> (value);

    cudaCheckError(cudaDeviceSynchronize());

    dim3 blocks = dim3(num_threads / 128 + 1, 1, 1);
    dim3 threads = dim3(128, 1, 1);

    const auto start{chrono::steady_clock::now()};

    increment <<< blocks, threads >>> (value, result, loop_count);

    cudaCheckError(cudaDeviceSynchronize());

    const auto finish{chrono::steady_clock::now()};
    const chrono::duration<double> elapsed_seconds{finish - start};

    cout << "Elapsed : " << elapsed_seconds.count() << " seconds" << endl;
}
