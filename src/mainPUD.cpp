
#include <omp.h>
#include "isingPUD.h"
#include "randutils.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <omp.h>

int main(){
    int L = 30;
    std::ofstream config("lattice");
    std::ofstream data("data");
    data.setf(std::ios::fixed);
    data.precision(5);

    std::uniform_real_distribution<> JDist(-3.0, 1.0);
    std::uniform_real_distribution<> TDist(0.0002, 3.0);

    static uint64_t seed1 = 32857595895136289;
    // alt seed 328575958954136219

    omp_set_num_threads(6);
    omp_set_dynamic(0);
    double J, T, T0, beta;

    randutils::seed_seq_fe128 seeder{uint32_t(seed1),uint32_t(seed1 >> 32)};
    std::vector<std::uint32_t> thread_seeds(omp_get_max_threads());
    seeder.generate(thread_seeds.begin(), thread_seeds.end());
    std::vector<std::mt19937> mt19937Engines(omp_get_max_threads());
    for (int i = 0; i < omp_get_max_threads(); ++i)
    {
        mt19937Engines[i] = std::mt19937(thread_seeds[i]);
    }

    static std::mt19937 *mt19937Engine;
    static isingLattice2D *lattice;
#pragma omp threadprivate(mt19937Engine, lattice)

#pragma omp parallel
{
    mt19937Engine = &mt19937Engines[omp_get_thread_num()];
    lattice = new isingLattice2D(L, *mt19937Engine);
}

#pragma omp parallel for private(J, T, T0, beta) schedule(dynamic)
    for (size_t i = 0; i < 400; i++)
    {
#pragma omp critical
{
        J = JDist(*mt19937Engine); T = TDist(*mt19937Engine);
}
        T0 = 5.0;
        lattice->initialise(0.0);
        while(T < T0){
            beta = 1.0/T0;
            lattice->metropolis2DimPUDSweep(beta, J);
            T0 -= 0.0002;
        }
        beta = 1.0/T;

        for (size_t k = 0; k < 3000; k++)
        {
            lattice->metropolis2DimPUDSweep(beta, J);
        }

#pragma omp critical
{
        lattice->writeConfig(config);
        data << std::showpos << J << " " << std::showpos << T << "\n";
}
    }
    return 0;
}
