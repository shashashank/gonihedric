
#include "isingGonihedric.h"
#include "randutils.hpp"
#include <iostream>
#include <fstream>
#include <string>

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
    // std::string kstring2 = std::format("{}",std::to_string(k));

    std::ofstream data("testL"+std::to_string(L)+"k"+kstring+"configs");
    std::ofstream temps("testL"+std::to_string(L)+"k"+kstring+"temps");

    // std::ofstream data("testL"+std::to_string(L)+"configs");
    // std::ofstream temps("testL"+std::to_string(L)+"params");

    temps.setf(std::ios::fixed);
    temps.precision(4);
    // alt seed 328575958951598690
    static uint64_t seed1;
    const std::string &seedstring = input.getCmdOption("-s");
    if (!seedstring.empty()){
        seed1 = std::stoull(seedstring);
    }else{
        seed1 = 328575958954136219;
    }

    randutils::seed_seq_fe128 seeder{uint32_t(seed1),uint32_t(seed1 >> 32)};
    std::mt19937 mt19937Engine(seeder);
    isingLattice lattice(L, &mt19937Engine);
    double tempDelta = 0.8/(114-1),
    tStart = 2/(std::log(1+std::sqrt(2))) + 0.4, 
    tEnd = 2/(std::log(1+std::sqrt(2))) - 0.4;
    int tau = std::ceil(std::pow(L, 2));
    double T = tStart;
    lattice.initialise(0.0);

    while(T > tEnd){
        for (int k = 0; k < 15*tau; k++){
                lattice.metropolis3DimSweep(1/T);
                lattice.metropolis3DimSweepTyp(1/T);
        }
        for (size_t j = 0; j < 1500; j++){
            for (int k = 0; k < tau; k++){
                lattice.metropolis3DimSweep(1/T);
                lattice.metropolis3DimSweepTyp(1/T);
            }
            lattice.writeConfig(data);
            data << T << "\n";
        }
        T -= tempDelta;
    }
    return 0;
}
