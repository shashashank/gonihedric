
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

    // std::cout << "no. of trials t = " << t << "\n";


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

    double beta;
    const std::string &betastring = input.getCmdOption("-B");
    if (!betastring.empty()){
        beta = std::stod(betastring);
    }else{
        std::cerr << "Error: beta not specified\n";
        exit(0);
    }
    std::cout << "beta = " << beta << "\n";

    // alt seed 328575958951598690
    static uint64_t seed1;
    const std::string &seedstring = input.getCmdOption("-s");
    if (!seedstring.empty()){
        seed1 = std::stoull(seedstring);
    }else{
        seed1 = 328575958413625279;
    }
    // std::cout << "seed = " << seed << "\n";

    std::ofstream data("dataL"+std::to_string(L)+"k"+std::to_string(k));
    data.setf(std::ios::fixed);
    data.precision(5);

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
    double mAbsx, mSqrx, mAbsy, mSqry, mAbsz, mSqrz;
    double e;
    int pTrials = 100, lTrials = 1000;

    for (int i = 0; i < 200; i++){
        mAbsx = 0.0; mSqrx = 0.0; mAbsy = 0.0; mSqry = 0.0; mAbsz = 0.0; mSqrz = 0.0, e=0.0;
#pragma omp parallel for schedule(dynamic) reduction(+:mAbsx, mSqrx, mAbsy, mSqry, mAbsz, mSqrz, e)
        for (size_t p = 0; p < pTrials; p++){
            std::pair<double, double> mFNx, mFNy, mFNz;
            lattice->initialise(0.0);
            // Equilibrate
            for (size_t j = 0; j < 5000; j++)
                lattice->metropolis3DimSweep(beta, k);
            for (size_t j = 0; j < lTrials; j++){
                lattice->metropolis3DimSweep(beta, k);
                lattice->magFukiNukeX(mFNx.first, mFNx.second);
                lattice->magFukiNukeY(mFNy.first, mFNy.second);
                lattice->magFukiNukeZ(mFNz.first, mFNz.second);
                mAbsx += mFNx.first; mAbsy += mFNy.first; mAbsz += mFNz.first;
                mSqrx += mFNx.second; mSqry += mFNy.second; mSqrz += mFNz.second;
                e += lattice->energy3D(k);
            }
        }
        data << beta << " " << mAbsx/(pTrials*lTrials) << " " << mSqrx/(pTrials*lTrials) << " " << mAbsy/(pTrials*lTrials) << " " 
        << mSqry/(pTrials*lTrials) << " " << mAbsz/(pTrials*lTrials) << " " << mSqrz/(pTrials*lTrials) << " "<< e/(pTrials*lTrials) << "\n";
        beta+=0.001;
    }

#pragma omp parallel
    delete lattice;

    return 0;
}
