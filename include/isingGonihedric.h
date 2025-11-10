#ifndef ISING_H
#define ISING_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include"processing.h"
#include"randutils.hpp"

// static uint64_t seed = 328575958951598690;
// randutils::seed_seq_fe128 seeder{uint32_t(seed),uint32_t(seed >> 32)};
// std::mt19937 mt19937Engine(seeder);
// std::uniform_real_distribution<> rDist(0.0, 1.0);

class isingLattice{
    private:
        const int N, L, L2;
        int *lattice; // lattice
        int *nn, *nnn; // neighbours
        std::uniform_int_distribution<> lDist;
        std::uniform_real_distribution<> rDist;
        std::mt19937 *mt19937Engine;
        void neighboutList(void){
            for (int i = 0; i < N; ++i)
            {
                nn[6*i+0] = (mod(i,L)==0) ?  i+L-1 : i-1;
                nn[6*i+1] = (mod(i+1,L)==0) ? i+1-L : i+1;
                nn[6*i+2] = (mod(i,L2)<L) ? (i-mod(i,L2)) + L2 - (L-mod(i,L)) : i-L;
                nn[6*i+3] = (mod(i,L2)>=L2-L) ? (i-mod(i,L2)) + mod(i,L) : i+L;
                nn[6*i+4] = mod(i-L2,N);
                nn[6*i+5] = mod(i+L2,N);
            }
            for (int i = 0; i < N; i++)
            {
                nnn[12*i+0] = nn[6*nn[6*i+0]+2];
                nnn[12*i+1] = nn[6*nn[6*i+1]+2];
                nnn[12*i+2] = nn[6*nn[6*i+0]+3];
                nnn[12*i+3] = nn[6*nn[6*i+1]+3];
                nnn[12*i+4] = nn[6*nn[6*i+0]+4];
                nnn[12*i+5] = nn[6*nn[6*i+1]+4];
                nnn[12*i+6] = nn[6*nn[6*i+0]+5];
                nnn[12*i+7] = nn[6*nn[6*i+1]+5];
                nnn[12*i+8] = nn[6*nn[6*i+3]+5];
                nnn[12*i+9] = nn[6*nn[6*i+2]+5];
                nnn[12*i+10] = nn[6*nn[6*i+2]+4];
                nnn[12*i+11] = nn[6*nn[6*i+3]+4];
            }
        };
    public:
        isingLattice(int l, std::mt19937 *rng): N(l*l*l), L(l), L2(l*l), mt19937Engine(rng){
            lDist = std::uniform_int_distribution<>(0, N-1);
            lattice = new int[N];
            nn = new int[6*N];
            nnn = new int[12*N];
            neighboutList();
        };

        void initialise(double m0){
            for(int i=0; i<N; ++i)
                lattice[i] = (rDist(*mt19937Engine) < m0)? 1 : -1;
        }
        void metropolis3DimSweep(double, double);
        void printLattice(std::ostream &data);
        double magnetisation(void) const;
        double energy3D(double) const;
        double siteEnergy3D(int, double) const;
        void writeConfig(std::ostream &data) const;
        void flipSeriesOfSites(int, int);
        ~isingLattice() {
            delete[] lattice;
            delete[] nn;
            delete[] nnn;
        };
};

void isingLattice::writeConfig(std::ostream &data) const{
    for(int i=0; i<N; ++i){
        switch (lattice[i]){
            case 1:
                data.write("1", 1);
                break;
            case -1:
                data.write("0", 1);
                break;
            default:
                std::cerr << "Error: lattice value not 1 or -1" << std::endl;
                exit(0);
        }
    }
    data.write("\n", 1);
}
double isingLattice::energy3D(double k=0.0) const{
    double e = 0.0;
    for(int x=0; x< N; ++x)
        e += siteEnergy3D(x,k);
    return -e/((double) 2.0*N);
}

double isingLattice::magnetisation(void) const{
    double m=0.0;
    for (int i = 0; i < N; ++i) m += lattice[i];
    return m/N;
}

void isingLattice::printLattice(std::ostream &data){
    for(int i=0; i<N; ++i){
        switch (lattice[i]){
            case 1:
                data.write("1", 1);
                break;
            case -1:
                data.write("0", 1);
                break;
            default:
                std::cerr << "Error: lattice value not 1 or -1" << std::endl;
                exit(0);
        }
    }
    data.write("\n", 1);
}

double isingLattice::siteEnergy3D(int x, double k) const{
    double energy = 0.0;
    energy = -2.0*k*(lattice[nn[6*x+0]] +lattice[nn[6*x+1]] +lattice[nn[6*x+2]] 
                    + lattice[nn[6*x+3]] +lattice[nn[6*x+4]] +lattice[nn[6*x+5]]);
    energy += k/2.0*(lattice[nnn[12*x+0]] + lattice[nnn[12*x+1]] + lattice[nnn[12*x+2]] + lattice[nnn[12*x+3]]
                    + lattice[nnn[12*x+4]] + lattice[nnn[12*x+5]] + lattice[nnn[12*x+6]] + lattice[nnn[12*x+7]]
                    + lattice[nnn[12*x+8]] + lattice[nnn[12*x+9]] + lattice[nnn[12*x+10]] + lattice[nnn[12*x+11]]);
    energy += -(1.0-k)/2.0*(lattice[nn[6*x+0]]*lattice[nn[6*x+2]]*lattice[nnn[12*x+0]]
                        + lattice[nn[6*x+1]]*lattice[nn[6*x+2]]*lattice[nnn[12*x+1]]
                        + lattice[nn[6*x+0]]*lattice[nn[6*x+3]]*lattice[nnn[12*x+2]]
                        + lattice[nn[6*x+1]]*lattice[nn[6*x+3]]*lattice[nnn[12*x+3]]
                        + lattice[nn[6*x+0]]*lattice[nn[6*x+4]]*lattice[nnn[12*x+4]]
                        + lattice[nn[6*x+1]]*lattice[nn[6*x+4]]*lattice[nnn[12*x+5]]
                        + lattice[nn[6*x+0]]*lattice[nn[6*x+5]]*lattice[nnn[12*x+6]]
                        + lattice[nn[6*x+1]]*lattice[nn[6*x+5]]*lattice[nnn[12*x+7]]
                        + lattice[nn[6*x+4]]*lattice[nn[6*x+3]]*lattice[nnn[12*x+11]]
                        + lattice[nn[6*x+2]]*lattice[nn[6*x+4]]*lattice[nnn[12*x+10]]
                        + lattice[nn[6*x+3]]*lattice[nn[6*x+5]]*lattice[nnn[12*x+8]]
                        + lattice[nn[6*x+2]]*lattice[nn[6*x+5]]*lattice[nnn[12*x+9]]);
    return energy*lattice[x];
}

void isingLattice::metropolis3DimSweep(double beta, double k=0.0){
    double ide;
    for(int i=0; i<N; ++i){
        int x = lDist(*mt19937Engine);
        ide = -siteEnergy3D(x, k);
        if (ide <=0 || rDist(*mt19937Engine) < exp(beta*ide)){
            lattice[x] = -lattice[x];
        }
    }
}

void isingLattice::flipSeriesOfSites(int x, int d){
    x = mod(x,L);
    if (d==0){
        x = x*L;
        for (int i = 0; i < L; ++i){
            for (int j = 0; j < L; j++)
            {
                lattice[x+j] = -lattice[x+j];
            }
            x += L2;
        }
    } else {
        x = x*L2;
        for (int i = 0; i < L2; ++i){
            lattice[x+i] = -lattice[x+i];
        }
    }


    // std::cout << "Next Nearest Neighbours: ";
    // for(int i=0; i<12; i++){
    //     lattice[nnn[12*x+i]] = -lattice[nnn[12*x+i]];
    //     std::cout << nnn[12*x+i] << ", ";
    // }
    // std::cout << std::endl;

    // lattice[x] = 10;

}



class isingLattice2D{
        private:
        const int N, L, L2;
        int *lattice; // lattice
        int *nn, *nnn; // neighbours
        std::uniform_int_distribution<> lDist;
        std::uniform_real_distribution<> rDist;
        std::mt19937 *mt19937Engine;
        void neighboutList(void){
            for (int i = 0; i < N; ++i)
            {
                nn[4*i+0] = (mod(i,L)==0) ?  i+L-1 : i-1;
                nn[4*i+1] = (mod(i+1,L)==0) ? i+1-L : i+1;
                nn[(4*i)+2] = mod(i-L,N);
                nn[(4*i)+3] = mod(i+L,N);
            }
            for (int i = 0; i < N; i++)
            {
                nnn[4*i+0] = nn[4*nn[4*i+0]+2];
                nnn[4*i+1] = nn[4*nn[4*i+1]+2];
                nnn[4*i+2] = nn[4*nn[4*i+0]+3];
                nnn[4*i+3] = nn[4*nn[4*i+1]+3];
            }
        };
    public:
        isingLattice2D(int l, std::mt19937 *rng): N(l*l), L(l), L2(l*l), mt19937Engine(rng){
            lDist = std::uniform_int_distribution<>(0, N-1);
            lattice = new int[N];
            nn = new int[4*N];
            nnn = new int[4*N];
            neighboutList();
        };

        void initialise(double m0){
            for(int i=0; i<N; ++i)
                lattice[i] = (rDist(*mt19937Engine) < m0) ? 1 : -1;
        }
        void metropolis2DimSweep(double, double);
        void printLattice(std::ostream &data);
        double magnetisation(void) const;
        double energy2D(double) const;
        double siteEnergy2D(int, double) const;
        void writeConfig(std::ostream &data) const;
        void flipSeriesOfSites(int, int);
        ~isingLattice2D() {
            delete[] lattice;
            delete[] nn;
            delete[] nnn;
        };
};

void isingLattice2D::writeConfig(std::ostream &data) const{
    for(int i=0; i<N; ++i){
        switch (lattice[i]){
            case 1:
                data.write("1", 1);
                break;
            case -1:
                data.write("0", 1);
                break;
            default:
                std::cerr << "Error: lattice value not 1 or -1" << std::endl;
                exit(0);
        }
    }
    data.write("\n", 1);
}

double isingLattice2D::energy2D(double k=0.0) const{
    double e = 0.0;
    for(int x=0; x< N; ++x)
        e += siteEnergy2D(x,k);
    return -e/((double) 2.0*N);
}

double isingLattice2D::magnetisation(void) const{
    double m=0.0;
    for (int i = 0; i < N; ++i) m += lattice[i];
    return m/N;
}

void isingLattice2D::printLattice(std::ostream &data){
    for(int i=0; i<N; ++i){
        switch (lattice[i]){
            case 1:
                data.write("1", 1);
                break;
            case -1:
                data.write("0", 1);
                break;
            default:
                std::cerr << "Error: lattice value not 1 or -1" << std::endl;
                exit(0);
        }
    }
    data.write("\n", 1);
}

double isingLattice2D::siteEnergy2D(int x, double k) const{
    double energy = 0.0;
    energy = -2.0*k*(lattice[nn[4*x+0]]+lattice[nn[4*x+1]]+lattice[nn[4*x+2]]+lattice[nn[4*x+3]]);
    energy += k/2.0*(lattice[nnn[4*x+0]] + lattice[nnn[4*x+1]] + lattice[nnn[4*x+2]] + lattice[nnn[4*x+3]]);
    energy += -(1.0-k)/2.0*(lattice[nn[4*x+0]]*lattice[nn[4*x+2]]*lattice[nnn[4*x+0]]
                        + lattice[nn[4*x+1]]*lattice[nn[4*x+2]]*lattice[nnn[4*x+1]]
                        + lattice[nn[4*x+0]]*lattice[nn[4*x+3]]*lattice[nnn[4*x+2]]
                        + lattice[nn[4*x+1]]*lattice[nn[4*x+3]]*lattice[nnn[4*x+3]]);
    return energy*lattice[x];
}

void isingLattice2D::metropolis2DimSweep(double beta, double k=0.0){
    double ide;
    for(int i=0; i<N; ++i){
        int x = lDist(*mt19937Engine);
        ide = -siteEnergy2D(x, k);
        if (ide <=0 || rDist(*mt19937Engine) < exp(beta*ide)){
            lattice[x] = -lattice[x];
        }
    }
}

void isingLattice2D::flipSeriesOfSites(int x, int d){
    x = mod(x,L);
    if (d==0){
        x = x*L;
        for (int i = 0; i < L; ++i){
            for (int j = 0; j < L; j++)
            {
                lattice[x+j] = -lattice[x+j];
            }
            x += L2;
        }
    } else {
        x = x*L2;
        for (int i = 0; i < L2; ++i){
            lattice[x+i] = -lattice[x+i];
        }
    }
}

#endif