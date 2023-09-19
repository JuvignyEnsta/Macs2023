#include <cmath>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <mpi.h>
#include "legendre.hpp"

double f( double x )
{
    return std::abs(std::sin(x*x))*std::exp(-x*x);
}

int main(int nargs, char* argv[])
{
    constexpr double a = -100.;
    constexpr double b = +100.;
    constexpr std::int32_t nbSubIntervals = 144'000;
    constexpr double h = (b-a)/nbSubIntervals;
    constexpr int order = 64;

    constexpr auto quadrature = get_quadrature<order>();

    MPI_Comm global;
    int rank, nbp;
    MPI_Init(&nargs, &argv);
    MPI_Comm_dup(MPI_COMM_WORLD, &global);
    MPI_Comm_size(global, &nbp);
    MPI_Comm_rank(global, &rank);

    char bufferFileName[1024];
    sprintf(bufferFileName, "output%03d.txt", rank);
    std::ofstream out(bufferFileName);

    
    auto beg = std::chrono::high_resolution_clock::now();
    double sumLoc = 0.;
    std::int32_t nbSubLoc = nbSubIntervals/nbp;
    std::int32_t reste    = nbSubIntervals%nbp;
    if (reste > rank) nbSubLoc += 1;
    std::int32_t begSub   = rank * nbSubLoc;
    if (reste < rank) begSub += reste;
    for ( std::int32_t s=begSub; s< begSub + nbSubLoc; ++s) {
        double ai = a + h*s;
        double bi = ai + h;
        double mi = 0.5*(ai+bi);
        double si = 0.;
        // Gauss quadrature :
        for ( std::size_t iGauss=0; iGauss < quadrature.size(); ++iGauss ) {
            double gi = mi + 0.5*h*quadrature[iGauss][0];
            si += quadrature[iGauss][1] * f(gi);
        }
        si = 0.5*h*si;
        sumLoc += si;
    }
    double sum;
    MPI_Reduce(&sumLoc, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, global);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duree = end - beg;
    if (rank == 0)
        out << "Integral_[-100.;100;] sin(x*x) * std::exp(-x*x) = " << std::setprecision(14)  << sum << std::endl;
    out << "Temps calcul : " << duree.count() << "s" << std::endl;

    out.close();
    MPI_Finalize();
    
    return EXIT_SUCCESS;
}
