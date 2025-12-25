#include <cstdio>
#include <iostream>

#include "Kokkos_Core.hpp"

class HelloFunctor
{
public:
    HelloFunctor() = default;
    HelloFunctor(const HelloFunctor & functor)
    {
        std::cout << "Functor copied" << std::endl;
    }

    KOKKOS_FUNCTION void operator()(const unsigned int i) const
    {
        Kokkos::printf("Hello from iteration %i\n",i);
    }
};

int main(int argc, char * argv[])
{
    Kokkos::InitializationSettings settings;
    settings.set_disable_warnings(true);
    Kokkos::initialize(settings);

    unsigned int N = atoi(argv[1]);

    HelloFunctor functor;

    Kokkos::parallel_for(N, functor);

    Kokkos::finalize();

    return 0;
}
