#include <cstdint>
#include <iostream>
#include <utility>
#include <vector>
#include <chrono>

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

int main()
{
    constexpr std::int64_t nbSeries = 24'948'000;

    std::int64_t maxLength=0, maxHeight=0, u0MaxLength, u0MaxHeight;

    auto beg = std::chrono::high_resolution_clock::now();
    for ( std::int64_t index=0; index < nbSeries; index ++)
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
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duree = end - beg;
    std::cout << "Temps de calcul des series de Syracuse : " << duree.count() << std::endl;

    std::cout << "Hauteur maximale trouvee " << maxHeight << " avec u0 = " << u0MaxHeight << std::endl;
    std::cout << "Longueur maximale trouvee " << maxLength << " avec u0 = " << u0MaxLength << std::endl;

    return EXIT_SUCCESS;
}