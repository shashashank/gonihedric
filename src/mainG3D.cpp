
#include "isingGonihedric.h"
#include "randutils.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <omp.h>

int main(int argc, char **argv){
    InputParser input(argc, argv);
    if(input.cmdOptionExists("-h")){
        // To do: write help
    }

    int L;
    const std::string &lstring = input.getCmdOption("-L");
    if (!lstring.empty()){
        L = std::stoi(lstring);
    }else{
        std::cerr << "Error: L not specified\n";
        exit(0);
    }
    std::cout << "L = " << L << "\n";

    double k;
    const std::string &kstring = input.getCmdOption("-k");
    if (!kstring.empty()){
        k = std::stod(kstring);
    }else{
        std::cerr << "Error: k not specified\n";
        exit(0);
    }
    std::cout << "k = " << k << "\n";

    std::ofstream data("dataL"+std::to_string(L)+"k"+std::to_string(k));
    data.setf(std::ios::fixed);
    data.precision(5);
    // alt seed 328575958951598690
    static uint64_t seed1;
    const std::string &seedstring = input.getCmdOption("-s");
    if (!seedstring.empty()){
        seed1 = std::stoull(seedstring);
    }else{
        seed1 = 328575958954136219;
    }

    omp_set_num_threads(10);
    omp_set_dynamic(0);
    randutils::seed_seq_fe128 seeder{uint32_t(seed1),uint32_t(seed1 >> 32)};
    std::vector<std::uint32_t> thread_seeds(omp_get_max_threads());
    seeder.generate(thread_seeds.begin(), thread_seeds.end());
    std::vector<std::mt19937> mt19937Engines(omp_get_max_threads());
    for (int i = 0; i < omp_get_max_threads(); ++i)
    {
        mt19937Engines[i] = std::mt19937(thread_seeds[i]);
    }

    static std::mt19937 *mt19937Engine;
    static isingLattice *lattice;
#pragma omp threadprivate(mt19937Engine, lattice)

#pragma omp parallel
{
    mt19937Engine = &mt19937Engines[omp_get_thread_num()];
    lattice = new isingLattice(L, mt19937Engine);
}

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < 1000; i++){
        double beta = 0.0;
        while(beta > 1.05){
            lattice->initialise(0.5);

            for (size_t i = 0; i < 10000; i++)
                lattice->metropolis3DimSweep(beta, k);
            
            for (int j = 0; j < 1000; j++){
                lattice->metropolis3DimSweep(beta, k);

#pragma omp critical
            lattice->writeConfig(data);
            
            }
            beta+=.05;
        }
    }

#pragma omp parallel
    delete lattice;

    return 0;
}
