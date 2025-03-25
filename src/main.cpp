// Gonihedric Ising model 3D

#include"ising.h"
#include<chrono>
#include<iomanip>
#include<omp.h>
#include<format>
// #include<gsl_randist.h>

int main(int argc, char **argv){
    
    int x, k = 0;
    std::cout << "Enter the size of the lattice: ";
    std::cin >> x;
    isingLattice lattice(x);
    lattice.initialise(1); // initialising the lattice
    lattice.writeConfig(std::cout);
    double energy1 = lattice.energy3D(k), energy2;

    while(x != -1){
        lattice.initialise(1); // initialising the lattice
        std::cout << "Enter the site to flip: ";
        std::cin >> x;
        lattice.flipSeriesOfSites(x, 1);
        lattice.writeConfig(std::cout);
        energy2 = lattice.energy3D(k);

        std::cout << "Initial energy: " << energy1 << std::endl;
        std::cout << "Energy after flipping: " << energy2 << std::endl;
    }
    
}