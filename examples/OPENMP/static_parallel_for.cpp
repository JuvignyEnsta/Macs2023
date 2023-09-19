#include <cstdlib>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <omp.h>

int main()
{
    constexpr double pi = 3.141592653589793;
    constexpr std::size_t nbValues = 1'000'000;

    for (int i=1; i<=8; ++i)
    {
        std::vector<double> values(nbValues);
        std::cout << "Test avec " << i << " threads." << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        auto beg = std::chrono::high_resolution_clock::now();
        //_______________________________________________________________________________________________________________
        // Parallelization of the loop with i threads (if num_threads not provided, omp take a default number of threads)
#       pragma omp parallel for num_threads(i)
        for (std::size_t iValue=0; iValue<nbValues; ++iValue)
            values[iValue] = (std::cos(2*iValue*pi/nbValues) + 0.25*std::sin(2*iValue*pi/nbValues))/100.;
        //_______________________________________________________________________________________________________________
        double sum = 0.;
        // Loop with reduction (necessary to have right summation in sum variable)
#       pragma omp parallel for num_threads(i) reduction(+:sum) 
        for (std::size_t iValue=0; iValue<nbValues; ++iValue)
            sum += values[iValue]*values[iValue];
        //_______________________________________________________________________________________________________________
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "sum : " << sum << std::endl;
        std::chrono::duration<double> duree = end - beg;
        std::cout << "Temps du calcul : " << duree.count() << "s." << std::endl;
        std::cout << "########################################" << std::endl;
    }
    return EXIT_SUCCESS;
}