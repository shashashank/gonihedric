
#include <omp.h>
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

    std::string label;
    const std::string &labelstring = input.getCmdOption("-lbl");
    if (!labelstring.empty()){
        label = labelstring;
    }else{
        label = "";
    }

    std::cout << "L = " << L << "\n";
    std::ofstream config(label+"L"+lstring+"configs");
    std::ofstream data(label+"L"+lstring+"temps");
    data.setf(std::ios::fixed);
    data.precision(4);

    // alt seed 328575958954136219
    static uint64_t seed1;
    const std::string &seedstring = input.getCmdOption("-s");
    if (!seedstring.empty()){
        seed1 = std::stoull(seedstring);
    }else{
        seed1 = 328575958951598690;
    }
    // std::cout << "seed = " << seed << "\n";

    randutils::seed_seq_fe128 seeder{uint32_t(seed1),uint32_t(seed1 >> 32)};
    std::mt19937 mt19937Engine(seeder);
    isingLattice lattice(L, &mt19937Engine);

    double tempDelta = 0.8/(114-1),
    tStart = 2/(std::log(1+std::sqrt(2))) + 0.4, 
    tEnd = 2/(std::log(1+std::sqrt(2))) - 0.4;
    int tau = std::ceil(std::pow(L, 1.4));
    double T = tStart;
    lattice.initialise(0.5);
    while(T > tEnd){
        for (int k = 0; k < 10*tau; k++){
                lattice.metro3DIsingSweep(1/T);
                lattice.metro3DIsingTyp(1/T);
        }
        for (size_t j = 0; j < 1500; j++){
            for (int k = 0; k < tau; k++){
                lattice.metro3DIsingSweep(1/T);
                lattice.metro3DIsingTyp(1/T);
            }
            lattice.writeConfig(config);
            data << T << "\n";
        }
        T -= tempDelta;
    }
    return 0;
}
