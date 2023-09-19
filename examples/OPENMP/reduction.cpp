#include <cstdlib>
#include <iostream>
#include <cmath>
#include <chrono>

int main()
{
    constexpr std::size_t dim = 10'000'000;
    {
    double sum = 0.;
    double prod= 1.;
    auto beg = std::chrono::high_resolution_clock::now();
    for (std::size_t i=1; i<=dim; ++i)
    {
        double value1 = 1./std::pow(1.*i, 1.5);
        double value2 = std::pow(i+1., 0.66)/std::pow(1.*i, 0.66);
        sum  += value1;
        prod *= value2;
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duree = end - beg;
    std::cout << "Sequential time : " << duree.count() << ", sum = " << sum << ", prod = " << prod << std::endl;
    }
    {
    double sum = 0.;
    double prod= 1.;
    auto beg = std::chrono::high_resolution_clock::now();
#   pragma omp parallel for reduction( + : sum) reduction( * : prod)
    for (std::size_t i=1; i<=dim; ++i)
    {
        double value1 = 1./std::pow(1.*i, 1.5);
        double value2 = std::pow(i+1., 0.66)/std::pow(1.*i, 0.66);
        sum  += value1;
        prod *= value2;
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duree = end - beg;
    std::cout << "parallel time : " << duree.count() << ", sum = " << sum << ", prod = " << prod << std::endl;
    }
    return EXIT_SUCCESS;
}