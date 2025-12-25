#include <cstdio>
#include <iostream>
#include <chrono>

#include "Kokkos_Core.hpp"

#ifdef KOKKOS_ENABLE_CUDA
#define MemSpace Kokkos::CudaSpace
#else
#define MemSpace Kokkos::HostSpace
#endif

class Functor
{
public:
    struct Add {};
    struct Multiply {};
    struct Divide {};

    Functor(Kokkos::View<double *, MemSpace> input) : result(input) {}

    KOKKOS_FUNCTION void operator()(Add, const size_t i) const
    {
        result[i] += 1;
    }
    KOKKOS_FUNCTION void operator()(Multiply, const size_t i) const
    {
        double val = result[i];

        for (unsigned int iter = 0; iter < 100; ++iter)
            val *= 2;

        result[i] = val;
    }
    KOKKOS_FUNCTION void operator()(Divide, const size_t i) const
    {
        double val = result[i];

        for (unsigned int iter = 0; iter < 100; ++iter)
            val /= 2;

        result[i] = val;
    }

    Kokkos::View<double *, MemSpace> result;
};

int main(int argc, char * argv[])
{
    Kokkos::InitializationSettings settings;
    settings.set_disable_warnings(true);
    Kokkos::initialize(settings);
    std::atexit(Kokkos::finalize);

    unsigned int N = atoi(argv[1]);

    Kokkos::View<double *, MemSpace> input("input", N);

    Functor functor(input);

    const auto start{std::chrono::steady_clock::now()};

    Kokkos::parallel_for(Kokkos::RangePolicy<Functor::Add, Kokkos::IndexType<size_t>>(0, N), functor);
    Kokkos::parallel_for(Kokkos::RangePolicy<Functor::Multiply, Kokkos::IndexType<size_t>>(0, N), functor);
    Kokkos::parallel_for(Kokkos::RangePolicy<Functor::Divide, Kokkos::IndexType<size_t>>(0, N), functor);
    Kokkos::fence();

    const auto finish{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> elapsed_seconds{finish - start};

    std::cout << "Elapsed : " << elapsed_seconds.count() << " seconds" << std::endl;

    auto verify = Kokkos::create_mirror(input);
    Kokkos::deep_copy(verify, input);

    for (unsigned int i = 0; i < N; ++i)
        if (verify[i] != 1)
            return 0;

    std::cout << "All solutions are unity" << std::endl;

    return 0;
}
