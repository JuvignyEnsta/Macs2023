#include <iostream>
#include <execution>
#include <vector>
#include <complex>
#include <algorithm>
#include <numeric>
#include <chrono>

int main()
{
    constexpr std::size_t nbDistances = 4'000'000;
    constexpr double      k           = 0.5;
    std::vector<double> sqrDistances(nbDistances);
    for (std::size_t iDist=0; iDist<nbDistances; ++iDist)
    {
        sqrDistances[iDist] = 1'000./(1. + iDist);
    }
    // iota is a function which fill a container with incrementing values (the third parameter is the initial value):
    std::vector<std::size_t> indices(nbDistances);
    std::iota(indices.begin(),indices.end(), 0);
    std::vector<std::complex<double>> green(nbDistances);
    // Compute exp(i*k*r)/r² where i=sqrt(-1), k a constant and r the distance:
    {
        // Sequential version
        auto beg = std::chrono::high_resolution_clock::now();
        std::for_each(indices.begin(), indices.end(), 
                    [&sqrDistances, &green](const std::size_t& index) {
                        double dist = std::sqrt(sqrDistances[index]);
                        green[index] = (1./sqrDistances[k*dist])*std::complex<double>{std::cos(k*dist),std::sin(k*dist)};
        });
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duree = end - beg;
        std::cout << "Time for sequential computing of Green kernel : " << duree.count() << "s" << std::endl;
    }
    {
        // Parallel sequenced version
        auto beg = std::chrono::high_resolution_clock::now();
        std::for_each(std::execution::par, indices.begin(), indices.end(), 
                    [&sqrDistances, &green](const std::size_t& index) {
                        double dist = std::sqrt(sqrDistances[index]);
                        green[index] = (1./sqrDistances[k*dist])*std::complex<double>{std::cos(k*dist),std::sin(k*dist)};
        });
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duree = end - beg;
        std::cout << "Time for sequential computing of Green kernel : " << duree.count() << "s" << std::endl;
    }

    {
        // Parallel sequenced version
        auto beg = std::chrono::high_resolution_clock::now();
        std::for_each(std::execution::par_unseq, indices.begin(), indices.end(), 
                    [&sqrDistances, &green](const std::size_t& index) {
                        double dist = std::sqrt(sqrDistances[index]);
                        green[index] = (1./sqrDistances[k*dist])*std::complex<double>{std::cos(k*dist),std::sin(k*dist)};
        });
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duree = end - beg;
        std::cout << "Time for sequential computing of Green kernel : " << duree.count() << "s" << std::endl;
    }

}