#include <vector>
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <chrono>
#include <fstream>
#include <cassert>
#include <string>
#include <random>
#include <mpi.h>

std::vector<std::int32_t>
generateValues(int rank, std::int32_t nbLocalValues)
{
    std::mt19937 gen(rank * 2047);    
    std::uniform_int_distribution<std::int32_t> distribution(-327670L,327670L);

    std::vector<std::int32_t> values;
    values.reserve(nbLocalValues);
    for (std::int32_t iValue=0; iValue<nbLocalValues; ++iValue) {
        values.emplace_back( distribution(gen));
    }
    return values;
}
// ------------------------------------------------------------------------
std::vector<std::int32_t>
fusionSort(std::int32_t nbLoc, std::vector<std::int32_t> const& t_values1,
           std::vector<std::int32_t> const& t_values2, bool keepMax)
{
    auto itValues1 = t_values1.begin();
    auto itValues2 = t_values2.begin();

    std::size_t totSize = t_values1.size() + t_values2.size();
    std::vector<std::int32_t> sortedValues;
    sortedValues.reserve(totSize);
    while ( (itValues1 != t_values1.end()) && (itValues2 != t_values2.end()) ) {
        if ((*itValues1)>(*itValues2)) {
            sortedValues.emplace_back((*itValues2));
            ++ itValues2;
        } else {
            sortedValues.emplace_back((*itValues1));
            ++ itValues1;
        }
    }
    for ( ; itValues1 !=  t_values1.end(); ++itValues1) {
        sortedValues.emplace_back(*itValues1);
    }
    for ( ; itValues2 !=  t_values2.end(); ++itValues2) {
        sortedValues.emplace_back(*itValues2);
    }
    if (keepMax)
        return std::vector<std::int32_t>( sortedValues.begin()+totSize-nbLoc, sortedValues.end() );
    return std::vector<std::int32_t>( sortedValues.begin(), sortedValues.begin()+nbLoc);
}
// ------------------------------------------------------------------------
int
main(int nargs, char* argv[])
{
    constexpr bool keepMax = true;
    constexpr bool keepMin = false;
    std::size_t N = 360'000;
    MPI_Comm global;
    int rank, nbp;
    MPI_Init(&nargs, &argv);
    MPI_Comm_dup(MPI_COMM_WORLD, &global);
    MPI_Comm_size(global, &nbp);
    MPI_Comm_rank(global, &rank);

    if (nargs > 1) {
        N = std::stol(argv[1]);
    }
    
    char bufferFileName[1024];
    sprintf(bufferFileName, "output%03d.txt", rank);
    std::ofstream out(bufferFileName);

    std::int32_t NLoc = N/nbp;
    if (N%nbp > rank) NLoc += 1;
    out << "Nombre de valeurs locales : " << NLoc << std::endl;

    auto values = generateValues(rank, NLoc);

    auto beg = std::chrono::system_clock::now();
    std::sort(values.begin(), values.end());

    std::int32_t prevNbLoc = N/nbp, nextNbLoc = N/nbp;
    if (N%nbp > rank-1) prevNbLoc += 1;
    if (N%nbp > rank+1) nextNbLoc += 1;
    std::vector<std::int32_t> prevBuffer(prevNbLoc);
    std::vector<std::int32_t> nextBuffer(nextNbLoc);
    MPI_Status status;
    for (int it=0; it<nbp; ++it) {
        out << "it : " << it << std::endl;
        if (it&1) { // Si it est impair
            if (rank%2 == 0) {
                if (rank > 0) {
                    MPI_Recv(prevBuffer.data(), prevNbLoc, MPI_INT32_T, rank-1,
                             404, global, &status);
                    MPI_Ssend(values.data(), NLoc, MPI_INT32_T,
                              rank-1, 404, global);
                    values = fusionSort(NLoc, values, prevBuffer, keepMax);
                }
            } else if (rank < nbp-1) {
                MPI_Ssend(values.data(), NLoc, MPI_INT32_T,
                          rank+1, 404, global);
                MPI_Recv(nextBuffer.data(), nextNbLoc, MPI_INT32_T,
                         rank+1, 404, global, &status);
                values = fusionSort(NLoc, values, nextBuffer, keepMin);
            }
        } else {
            if (rank % 2 == 0) {
                if (rank < nbp-1) {
                    MPI_Recv(nextBuffer.data(), nextNbLoc, MPI_INT32_T, rank+1,
                             404, global, &status);
                    MPI_Ssend(values.data(), NLoc, MPI_INT32_T,rank+1,
                              404, global);
                    values = fusionSort(NLoc, values, nextBuffer, keepMin);
                }
            } else {
                MPI_Ssend(values.data(), NLoc, MPI_INT32_T,
                          rank-1, 404, global);
                MPI_Recv(prevBuffer.data(), prevNbLoc, MPI_INT32_T, rank-1,
                         404, global, &status);
                values = fusionSort(NLoc, values, prevBuffer, keepMax);
            }
        }
    }
    auto end = std::chrono::system_clock::now();
    assert(values.size() == NLoc);
    std::chrono::duration<double> duree = end - beg;
    out << "Temps local pour le tri : " << duree.count() << std::endl;
    out << "Premiere valeurs locale : " << values.front() << std::endl;
    out << "Derniere valeurs locale : " << values.back() << std::endl;
    if (values.size() < 100)
    {
        for ( auto const& v : values ) out << v << " ";
        out << std::endl;
    }
    out.close();
    MPI_Finalize();
    return EXIT_SUCCESS;
}
