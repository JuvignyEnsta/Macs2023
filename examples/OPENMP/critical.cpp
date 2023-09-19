#include <cstdlib>
#include <iostream>
#include <vector>
#include <chrono>
#include <complex>

bool isConvergent( std::complex<double> const& value )
{
    constexpr std::size_t maxIter = 8'000;
    std::size_t iter = 0;
    std::complex<double> xn = value;
    do
    {
        xn = xn*xn + value;
        ++iter;
    } while ( (std::norm(xn) < 2.) && (iter < maxIter) );
    return (iter == maxIter);
}

int main()
{
    constexpr std::size_t dim = 1'500;
    {
        auto beg = std::chrono::high_resolution_clock::now();
        double deltax = 2.25/dim;
        double deltay = 4./dim;
        std::vector<std::complex<double>> result;
        for (std::size_t i=0; i<dim; ++i)
        {
            for (std::size_t j=0; j<dim; ++j)
            {
                std::complex<double> val{-2. + deltax*i, -2. + deltay*i};
                if (!isConvergent(val))
                    result.push_back(val);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duree = end - beg;
        std::cout << "number of values divergent : " << result.size() << std::endl;
        std::cout << "Sequential time : " << duree.count() << " seconds." << std::endl;
    }
    {
        auto beg = std::chrono::high_resolution_clock::now();
        double deltax = 2.25/dim;
        double deltay = 4./dim;
        std::vector<std::complex<double>> result;
#       pragma omp parallel for schedule(dynamic)
        for (std::size_t i=0; i<dim; ++i)
        {
            for (std::size_t j=0; j<dim; ++j)
            {
                std::complex<double> val{-2. + deltax*i, -2. + deltay*i};
                if (!isConvergent(val))
#                   pragma omp critical
                    result.push_back(val);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duree = end - beg;
        std::cout << "number of values divergent : " << result.size() << std::endl;
        std::cout << "Parallel time : " << duree.count() << " seconds." << std::endl;
    }


    return EXIT_SUCCESS;
}