#include <cstdlib>
#include <cstdint>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <vector>

std::vector<std::int32_t>
generateValues(std::int32_t nbValues)
{
    std::vector<std::int32_t> values; values.reserve(nbValues);
    for (std::size_t iVal=0; iVal<nbValues; ++iVal) {
        values.emplace_back(iVal+1);
    }
    std::random_shuffle(values.begin(),values.end());
    return values;
}

std::vector<std::int32_t>
generateSortedValues( std::vector<std::int32_t> const& t_values)
{
    std::vector<std::int32_t> sorted(t_values.size());

#   pragma omp parallel for
    for ( std::size_t iVal=0; iVal<t_values.size(); ++iVal) {
        std::int32_t iPos=0;
        for ( auto const& val : t_values ) {
            if (t_values[iVal] > val) ++iPos;
        }
        sorted[iPos] = t_values[iVal];
    }
    return sorted;
}

int main(int nargs, char* argv[])
{
    std::int32_t nbVals = 100'000;
    auto values = generateValues(nbVals);
    if (nbVals<100) {
        for (auto const& val : values)
            std::cout << val << " ";
        std::cout << std::endl;
    }
    auto beg = std::chrono::high_resolution_clock::now();
    auto sorted = generateSortedValues( values );
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duree = end - beg;
    std::cout << "Temps effectif pour le tri : " << duree.count() << "s" << std::endl;
    if (nbVals<100) {
        for (auto const& val : sorted)
            std::cout << val << " ";
        std::cout << std::endl;
    }
    if (std::is_sorted(sorted.begin(), sorted.end())) {
        std::cout << "Success !" << std::endl;
    } else {
        std::cout << "Fail to sort !" << std::endl;
    }
    return EXIT_SUCCESS;
}
