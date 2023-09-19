#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <chrono>
#include <fstream>
#include <cassert>
#include <string>
#include <cmath>
#include <iostream>
#include <limits>
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
// ====================================================================================================================
std::vector<std::int32_t>
fusionSort(int nbValues1, std::int32_t const* t_values1, int nbValues2, std::int32_t const* t_values2)
{
    auto itValues1 = t_values1;
    auto itValues2 = t_values2;
    auto endItValues1 = t_values1 + nbValues1;
    auto endItValues2 = t_values2 + nbValues2;

    std::size_t totSize = nbValues1 + nbValues2;
    std::vector<std::int32_t> sortedValues;
    sortedValues.reserve(totSize);
    while ( (itValues1 != endItValues1) && (itValues2 != endItValues2) ) {
        if ((*itValues1)>(*itValues2)) {
            sortedValues.emplace_back((*itValues2));
            ++ itValues2;
        } else {
            sortedValues.emplace_back((*itValues1));
            ++ itValues1;
        }
    }
    for ( ; itValues1 !=  endItValues1; ++itValues1) {
        sortedValues.emplace_back(*itValues1);
    }
    for ( ; itValues2 !=  endItValues2; ++itValues2) {
        sortedValues.emplace_back(*itValues2);
    }
    return sortedValues;
}
// --------------------------------------------------------------------------------
int main(int nargs, char* argv[])
{
    std::int32_t nbValues = 36'000;

    MPI_Comm global, rowComm, colComm;
    int rank, nbp;
    MPI_Init(&nargs, &argv);
    MPI_Comm_dup(MPI_COMM_WORLD, &global);
    MPI_Comm_size(global, &nbp);
    MPI_Comm_rank(global, &rank);

    if (nargs > 1) {
        nbValues = std::stol(argv[1]);
    }

    char bufferFileName[1024];
    sprintf(bufferFileName, "output%03d.txt", rank);
    std::ofstream out(bufferFileName);

    std::int32_t nbLocalValues = nbValues/nbp;
    if (nbValues%nbp != 0) {
        std::cerr << "Le nombre de processeur doit aussi divisé le nombre d'élément à trier." << std::endl;
        MPI_Abort(global, EXIT_FAILURE);
    }

    auto localValues = generateValues(rank, nbLocalValues);

    if (localValues.size() < 100)
    {
        out << "Valeurs initiales : ";
        for ( auto const& v : localValues ) out << v << " ";
        out << std::endl;
    }
    // Calcul la dimension de l'hypercube :
    int dim = int(std::log(nbp)/std::log(2)+0.1);
    if ((1<<dim) != nbp)
    {
        std::cerr << "Le nombre de processeur doit être une puissance de deux !" << std::endl;
        MPI_Abort(global, EXIT_FAILURE);
    }
    out << "Dimension du cube : " << dim << std::endl;

    MPI_Status status;
    auto beg = std::chrono::high_resolution_clock::now();
    std::vector<std::int32_t> buffer(localValues.size());
    std::sort(localValues.begin(), localValues.end());
    for (int d=dim-1; d>=0; --d)
    {
#       ifdef DEBUG
        out << "Dimension " << d << std::endl;
#       endif
        MPI_Comm subcube;
        MPI_Comm_split(global, rank/(1<<(d+1)), rank, &subcube);
        int srank;
        MPI_Comm_rank(subcube, &srank);
        std::int32_t pivot;
        if (srank == 0)
        {
            pivot = localValues[localValues.size()/2-1];
#           ifdef DEBUG
            out << "Pivot a envoye : " << pivot << std::endl;
#           endif
        }
        MPI_Bcast(&pivot, 1, MPI_INT32_T, 0, subcube);
#       ifdef DEBUG
        out << "pivot recupere : " << pivot << std::endl << std::flush;
#       endif
        // Search first value greater than pivot value :
        auto itSup = std::upper_bound(localValues.begin(), localValues.end(), pivot);
        int ipivot = itSup - localValues.begin();

        if ((rank & (1<<d)) == 0)
        {
#           ifdef DEBUG
            out << "Pairing avec " << rank + (1<<d) << std::endl;
#           endif
            MPI_Ssend(localValues.data()+ipivot, localValues.size()-ipivot, MPI_INT32_T, rank + (1<<d), 404, global);
            MPI_Probe(rank + (1<<d), 404, global, &status);
            int recvSize;
            MPI_Get_count(&status, MPI_INT32_T, &recvSize);
            buffer.resize(recvSize);
            MPI_Recv(buffer.data(), recvSize, MPI_INT32_T, rank+(1<<d), 404, global, &status);
            localValues = fusionSort(ipivot, localValues.data(), recvSize, buffer.data());
        }
        else 
        {
#           ifdef DEBUG
            out << "Pairing avec " << rank - (1<<d) << std::endl;
#           endif
            MPI_Probe(rank - (1<<d), 404, global, &status);
            int recvSize, tag;
            MPI_Get_count(&status, MPI_INT32_T, &recvSize);
            buffer.resize(recvSize);
            MPI_Recv(buffer.data(), recvSize, MPI_INT32_T, rank-(1<<d), 404, global, &status);
            MPI_Ssend(localValues.data(), ipivot, MPI_INT32_T, rank - (1<<d), 404, global);
            localValues = fusionSort( localValues.size()-ipivot, localValues.data()+ipivot, recvSize, buffer.data());            
        }
#       ifdef DEBUG
        if (localValues.size() < 100)
        {
            for ( auto const& v : localValues ) out << v << " ";
            out << std::endl;
        }
        else 
        {
            out << "Range : " << localValues.front() << " to " << localValues.back() << std::endl;
        }
#       endif
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duree = end - beg;
    out << "Temps pour le tri : " << duree.count() << std::endl;
    out << "Premiere valeurs locale : " << localValues.front() << std::endl;
    out << "Derniere valeurs locale : " << localValues.back()  << std::endl;
    out << "Nombre de valeurs locales : " << localValues.size() << std::endl;
    if (localValues.size() < 100)
    {
        for ( auto const& v : localValues ) out << v << " ";
        out << std::endl;
    }
    out.close();
    MPI_Finalize();

    return EXIT_SUCCESS;
}