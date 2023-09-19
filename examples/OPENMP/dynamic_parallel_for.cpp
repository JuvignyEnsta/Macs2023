#include <cstdlib>
#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <chrono>

bool isPrime( std::int64_t t_value )
{
    if ((t_value==2) || (t_value==3)) return true;
    if (t_value%2==0) return false;
    if (t_value%3==0) return false;
    for (std::int64_t i=1; 6*i-1<=std::sqrt(t_value); ++i)
    {
        if (t_value%(6*i-1) ==0) return false;
        if (t_value%(6*i+1) ==0) return false;
    }
    return true;
}

int main()
{
    constexpr int nbValues = 10'000'000;
    std::vector<std::int64_t> primeCandidates(nbValues);
    // Sequential loop :
    {
        std::cout << "############  Sequential algorithm ##########" << std::endl;
        primeCandidates[0] = 2;
        primeCandidates[1] = 3;
        for (std::size_t iCandidates=0; iCandidates<=nbValues/2-1; ++iCandidates)
        {
            primeCandidates[2*iCandidates  ] = 6*(iCandidates+1)-1;
            primeCandidates[2*iCandidates+1] = 6*(iCandidates+1)+1;
        }
        auto beg = std::chrono::high_resolution_clock::now();
        for (std::size_t iCandidates=0; iCandidates<nbValues; ++iCandidates)
        {            
            if (!isPrime(primeCandidates[iCandidates])) 
            {
                primeCandidates[iCandidates] = -1;
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Number of found primes : ";
        std::size_t nbPrimes = 0;
        for (auto val : primeCandidates)
            if (val > 0) nbPrimes++;

        std::cout << nbPrimes << std::endl;
        auto duree = std::chrono::duration<double>(end-beg);
        std::cout << "Primes founded in " << duree.count() << " seconds." << std::endl;
    }
    // Static parallel loop :
    {
        std::cout << "############  Static parallel algorithm ##########" << std::endl;
        primeCandidates[0] = 2;
        primeCandidates[1] = 3;
        for (std::size_t iCandidates=0; iCandidates<=nbValues/2-1; ++iCandidates)
        {
            primeCandidates[2*iCandidates  ] = 6*(iCandidates+1)-1;
            primeCandidates[2*iCandidates+1] = 6*(iCandidates+1)+1;
        }
        auto beg = std::chrono::high_resolution_clock::now();
#       pragma omp parallel for
        for (std::size_t iCandidates=0; iCandidates<nbValues; ++iCandidates)
        {            
            if (!isPrime(primeCandidates[iCandidates])) 
            {
                primeCandidates[iCandidates] = -1;
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Number of found primes : ";
        std::size_t nbPrimes = 0;
        for (auto val : primeCandidates)
            if (val > 0) nbPrimes++;

        std::cout << nbPrimes << std::endl;
        auto duree = std::chrono::duration<double>(end-beg);
        std::cout << "Primes founded in " << duree.count() << " seconds." << std::endl;
    }

    // Dynamic parallel loop :
    {
        std::cout << "############  Dynamic parallel algorithm ##########" << std::endl;
        primeCandidates[0] = 2;
        primeCandidates[1] = 3;
        for (std::size_t iCandidates=0; iCandidates<=nbValues/2-1; ++iCandidates)
        {
            primeCandidates[2*iCandidates  ] = 6*(iCandidates+1)-1;
            primeCandidates[2*iCandidates+1] = 6*(iCandidates+1)+1;
        }
        auto beg = std::chrono::high_resolution_clock::now();
#       pragma omp parallel for schedule(dynamic,1'000)
        for (std::size_t iCandidates=0; iCandidates<nbValues; ++iCandidates)
        {            
            if (!isPrime(primeCandidates[iCandidates])) 
            {
                primeCandidates[iCandidates] = -1;
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Number of found primes : ";
        std::size_t nbPrimes = 0;
        for (auto val : primeCandidates)
            if (val > 0) nbPrimes++;

        std::cout << nbPrimes << std::endl;
        auto duree = std::chrono::duration<double>(end-beg);
        std::cout << "Primes founded in " << duree.count() << " seconds." << std::endl;
    }

    // Guided parallel loop :
    {
        std::cout << "############  Guided parallel algorithm ##########" << std::endl;
        primeCandidates[0] = 2;
        primeCandidates[1] = 3;
        for (std::size_t iCandidates=0; iCandidates<=nbValues/2-1; ++iCandidates)
        {
            primeCandidates[2*iCandidates  ] = 6*(iCandidates+1)-1;
            primeCandidates[2*iCandidates+1] = 6*(iCandidates+1)+1;
        }
        auto beg = std::chrono::high_resolution_clock::now();
#       pragma omp parallel for schedule(guided,1000)
        for (std::size_t iCandidates=0; iCandidates<nbValues; ++iCandidates)
        {            
            if (!isPrime(primeCandidates[iCandidates])) 
            {
                primeCandidates[iCandidates] = -1;
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Number of found primes : ";
        std::size_t nbPrimes = 0;
        for (auto val : primeCandidates)
            if (val > 0) nbPrimes++;

        std::cout << nbPrimes << std::endl;
        auto duree = std::chrono::duration<double>(end-beg);
        std::cout << "Primes founded in " << duree.count() << " seconds." << std::endl;
    }


    return EXIT_SUCCESS;
}