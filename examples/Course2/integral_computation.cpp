#include <cmath>
#include <iostream>
#include <chrono>
#include <iomanip>
#include "legendre.hpp"

double f( double x )
{
    return std::abs(std::sin(x*x))*std::exp(-x*x);
}

int main()
{
    constexpr double a = -100.;
    constexpr double b = +100.;
    constexpr std::int32_t nbSubIntervals = 144'000;
    constexpr double h = (b-a)/nbSubIntervals;
    constexpr int order = 64;

    constexpr auto quadrature = get_quadrature<order>();

    auto beg = std::chrono::high_resolution_clock::now();
    double sum = 0.;
    for ( std::int32_t s=0; s<nbSubIntervals; ++s) {
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
        sum += si;
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duree = end - beg;
    std::cout << "Integral_[-100.;100;] sin(x*x) * std::exp(-x*x) = " << std::setprecision(14)  << sum << std::endl;
    std::cout << "Temps calcul : " << duree.count() << "s" << std::endl;
                                                   
    return EXIT_SUCCESS;
}
