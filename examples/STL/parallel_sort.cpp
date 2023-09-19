#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <chrono>
#include <execution>

int main()
{
    constexpr std::size_t dim = 100'000'000;
    std::vector<double> values(dim);
    std::vector<double> copyValues(dim);
    for (std::size_t iValue=0; iValue<dim; ++iValue)
    {
        values[iValue] = std::rand()*1'000;
    }

    {
        std::copy(values.begin(), values.end(), copyValues.begin());
        auto beg = std::chrono::high_resolution_clock::now();
        std::sort(copyValues.begin(), copyValues.end());
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duree = end - beg;
        for ( std::size_t i=1; i<copyValues.size(); ++i)
            if (copyValues[i-1] > copyValues[i])
                std::cout << "Error, copyValues is not sorted !" << std::endl;
        std::cout << "Time for sequential sort : " << duree.count() << " seconds." << std::endl;
    }

    {   // Parallel sequenced version
        std::copy(values.begin(), values.end(), copyValues.begin());
        auto beg = std::chrono::high_resolution_clock::now();
        std::sort(std::execution::par, copyValues.begin(), copyValues.end());
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duree = end - beg;
        for ( std::size_t i=1; i<copyValues.size(); ++i)
            if (copyValues[i-1] > copyValues[i])
                std::cout << "Error, copyValues is not sorted !" << std::endl;
        std::cout << "Time for parallel sort : " << duree.count() << " seconds." << std::endl;
    }

    {   // Parallel unsequenced version
        std::copy(values.begin(), values.end(), copyValues.begin());
        auto beg = std::chrono::high_resolution_clock::now();
        std::sort(std::execution::par_unseq, copyValues.begin(), copyValues.end());
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duree = end - beg;
        for ( std::size_t i=1; i<copyValues.size(); ++i)
            if (copyValues[i-1] > copyValues[i])
                std::cout << "Error, copyValues is not sorted !" << std::endl;
        std::cout << "Time for parallel sort : " << duree.count() << " seconds." << std::endl;
    }


    return EXIT_SUCCESS;
}