#include <cstdio>

#include "Kokkos_Core.hpp"

int main(int argc, char * argv[])
{
    Kokkos::InitializationSettings settings;
    settings.set_disable_warnings(true);
    Kokkos::initialize(settings);

    unsigned int N = atoi(argv[1]);

    Kokkos::parallel_for(N, KOKKOS_LAMBDA (const unsigned int i) {
        Kokkos::printf("Hello from iteration %i\n",i);
    });

    Kokkos::finalize();

    return 0;
}
