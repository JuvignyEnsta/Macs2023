#include <cstdint>
#include <iostream>
#include <utility>
#include <vector>
#include <chrono>
#include <array>
#include <tuple>

struct SyracuseResult
{
    std::int64_t length, height;
};

SyracuseResult computeSerie( std::int64_t u0 )
{
    SyracuseResult result;
    result.length = 0;
    result.height = u0;
    std::int64_t un = u0;
    while (un != 1)
    {
        un = (un%2==0 ? un/2 : 3*un+1);
        result.height = std::max(result.height, un);
        result.length += 1;
    }
    return result;
}

std::vector<std::int64_t> packSyracuse( std::int64_t begIndex, std::int64_t nbSeries )
{
    std::int64_t maxLength=0, maxHeight=0, u0MaxLength, u0MaxHeight;
    for ( std::int64_t index=begIndex; index < begIndex + nbSeries; index ++)
    {
        std::int64_t u0 = 2*index + 1;

        auto result = computeSerie(u0);
        if (maxLength < result.length)
        {
            u0MaxLength = u0;
            maxLength   = result.length;
        }
        if (maxHeight < result.height)
        {
            u0MaxHeight = u0;
            maxHeight   = result.height;
        }
    }
    return {maxLength, u0MaxLength, maxHeight, u0MaxHeight};
}

int main()
{
    constexpr std::int64_t nbSeries = 24'948'000;
    constexpr std::int64_t szPack   = 100;

    std::int64_t maxLength=0, maxHeight=0, u0MaxLength, u0MaxHeight;

    auto beg = std::chrono::high_resolution_clock::now();
    std::int64_t nbPacks = nbSeries/szPack; 
#   pragma omp parallel for //schedule(dynamic)
    for ( std::int64_t index=0; index < nbPacks; index ++)
    {
        auto res = packSyracuse(index*szPack, szPack);
#       pragma omp critical 
        {
        u0MaxLength = maxLength < res[0] ? res[1] : u0MaxLength;
        maxLength   = std::max(maxLength, res[0]);
        u0MaxHeight = maxHeight < res[2] ? res[3] : u0MaxHeight;
        maxHeight   = std::max(maxHeight, res[2]);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duree = end - beg;
    std::cout << "Temps de calcul des series de Syracuse : " << duree.count() << std::endl;

    std::cout << "Hauteur maximale trouvee " << maxHeight << " avec u0 = " << u0MaxHeight << std::endl;
    std::cout << "Longueur maximale trouvee " << maxLength << " avec u0 = " << u0MaxLength << std::endl;

    return EXIT_SUCCESS;
}