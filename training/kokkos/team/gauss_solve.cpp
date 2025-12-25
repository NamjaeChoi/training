#include <cstdio>
#include <iostream>
#include <chrono>

#include "Kokkos_Core.hpp"

constexpr unsigned int N = 10000000;
constexpr unsigned int dim = 8;
constexpr unsigned int block_size = 128;

#define Layout Kokkos::LayoutLeft

class GaussSolve
{
public:
    GaussSolve(Kokkos::View<double ***, Layout> A_in,
               Kokkos::View<double **, Layout> x_in,
               Kokkos::View<double **, Layout> b_in)
        : A(A_in), x(x_in), b(b_in)
    {
    }
    
    KOKKOS_FUNCTION void operator()(Kokkos::TeamPolicy<>::member_type team) const
    {
        auto row = team.team_rank() % dim;
        auto n = (team.team_rank() / dim) + (team.team_size() / dim) * team.league_rank();

        double * shmem = reinterpret_cast<double *>(
            team.team_shmem().get_shmem(dim * (dim + 1) * (block_size / dim)));

        double * mat = shmem + dim * (dim + 1) * (team.team_rank() / dim);
        double * rhs = mat + dim * dim;

        if (n < N)
        {
            for (unsigned int col = 0; col < dim; ++col)
                mat[row + dim * col] = A(row, col, n);
        
            rhs[row] = b(row, n);
        }

        for (unsigned int r = 0; r < dim; ++r)
        {
            team.team_barrier();

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
            x(row, n) = rhs[row] / mat[row + dim * row];
    }

private:
    Kokkos::View<double ***, Layout> A;
    Kokkos::View<double **, Layout> x;
    Kokkos::View<double **, Layout> b;
};

int main(int argc, char * argv[])
{
    Kokkos::InitializationSettings settings;
    settings.set_disable_warnings(true);
    Kokkos::initialize(settings);
    std::atexit(Kokkos::finalize);

    Kokkos::View<double ***, Layout> A("A", dim, dim, N);
    Kokkos::View<double **, Layout> x("x", dim, N);
    Kokkos::View<double **, Layout> b("b", dim, N);

    auto A_host = Kokkos::create_mirror(A);
    auto x_host = Kokkos::create_mirror(x);
    auto b_host = Kokkos::create_mirror(b);

    for (unsigned int n = 0; n < N; ++n)
    {
        b_host(0, n) = 1;
        b_host(dim - 1, n) = 1;

        for (unsigned int i = 0; i < dim; ++i)
        {
            A_host(i, i, n) = 2;

            if (i != 0)
                A_host(i - 1, i, n) = -1;
            if (i != dim - 1)
                A_host(i + 1, i, n) = -1;
        }
    }

    Kokkos::deep_copy(A, A_host);
    Kokkos::deep_copy(b, b_host);

    GaussSolve solve(A, x, b);

    const auto start{std::chrono::steady_clock::now()};

    Kokkos::TeamPolicy<> policy(N * dim / block_size + 1, block_size);
    policy.set_scratch_size(0, Kokkos::PerTeam(dim * (dim + 1) * (block_size / dim) * sizeof(double)));

    for (unsigned int i = 0; i < 100; ++i)
    {
        Kokkos::parallel_for(policy, solve);
        Kokkos::fence();
    }

    const auto finish{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> elapsed_seconds{finish - start};

    std::cout << "Elapsed : " << elapsed_seconds.count() << " seconds" << std::endl;

    Kokkos::deep_copy(x_host, x);

    for (unsigned int n = 0; n < N; ++n)
        for (unsigned int i = 0; i < dim; ++i)
            if (abs(x_host(i, n) - 1) > 1.0e-8)
                return 0;

    std::cout << "All solutions are unity" << std::endl;

    return 0;
}
