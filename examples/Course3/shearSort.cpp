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

constexpr bool keepMax = true;
constexpr bool keepMin = false;

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
// --------------------------------------------------------------------------------
struct ProcessGridCoordinate
{
    int I,J;
};
// ................................................................................
ProcessGridCoordinate getProcessCoordinate( int rank, int nbp )
{
    int nbProcessInRow = int(std::sqrt(nbp)+0.5);
    assert(nbProcessInRow * nbProcessInRow == nbp);
    int IProc = rank/nbProcessInRow;
    int JProc = IProc%2 == 0 ? rank%nbProcessInRow : nbProcessInRow-1-rank%nbProcessInRow;
    return { .I=IProc, .J=JProc};
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
// --------------------------------------------------------------------------------
std::vector<std::int32_t> oddEvenSort( std::vector<std::int32_t> locValues, MPI_Comm comm)
{
    MPI_Status status;
    int nbp, rank;
    MPI_Comm_size(comm, &nbp);
    MPI_Comm_rank(comm, &rank);
    int NLoc = int(locValues.size());
    std::vector<std::int32_t> buffer(locValues.size());
    for (int it=0; it<nbp; ++it) {
        if (it&1) { // Si it est impair
            if (rank%2 == 0) {
                if (rank > 0) {
                    MPI_Recv(buffer.data(), NLoc, MPI_INT32_T, rank-1, 404, comm, &status);
                    MPI_Ssend(locValues.data(), NLoc, MPI_INT32_T, rank-1, 404, comm);
                    locValues = fusionSort(NLoc, locValues, buffer, keepMax);
                }
            } else if (rank < nbp-1) {
                MPI_Ssend(locValues.data(), NLoc, MPI_INT32_T, rank+1, 404, comm);
                MPI_Recv(buffer.data(), NLoc, MPI_INT32_T, rank+1, 404, comm, &status);
                locValues = fusionSort(NLoc, locValues, buffer, keepMin);
            }
        } else {
            if (rank % 2 == 0) {
                if (rank < nbp-1) {
                    MPI_Recv(buffer.data(), NLoc, MPI_INT32_T, rank+1, 404, comm, &status);
                    MPI_Ssend(locValues.data(), NLoc, MPI_INT32_T, rank+1, 404, comm);
                    locValues = fusionSort(NLoc, locValues, buffer, keepMin);
                }
            } else {
                MPI_Ssend(locValues.data(), NLoc, MPI_INT32_T, rank-1, 404, comm);
                MPI_Recv(buffer.data(), NLoc, MPI_INT32_T, rank-1, 404, comm, &status);
                locValues = fusionSort(NLoc, locValues, buffer, keepMax);
            }
        }
    }
    return locValues;
}
// =================================================================================
int main( int nargs, char* argv[] )
{
    std::int32_t nbValues = 36'000;

    MPI_Comm global, rowComm, colComm;
    int rank, nbp;
    MPI_Init(&nargs, &argv);
    MPI_Comm_dup(MPI_COMM_WORLD, &global);
    MPI_Comm_size(global, &nbp);
    MPI_Comm_rank(global, &rank);

    int sqNbp = int(sqrt(nbp)+0.5);
    if (sqNbp*sqNbp != nbp)
    {
        std::cerr << "Nombre de processus doit etre le carre d'un entier !" << std::endl;
        MPI_Abort(global, EXIT_FAILURE);
    }

    if (nargs > 1) {
        nbValues = std::stol(argv[1]);
    }
    std::int32_t nbLocalValues = nbValues/nbp;
    if (nbValues%nbp != 0) {
        std::cerr << "Le nombre de processeur doit aussi divisé le nombre d'élément à trier." << std::endl;
        MPI_Abort(global, EXIT_FAILURE);
    }

    auto pGrdCrd = getProcessCoordinate(rank, nbp);
    MPI_Comm_split(global, pGrdCrd.I, rank, &rowComm);
    MPI_Comm_split(global, pGrdCrd.J, rank, &colComm);

    char bufferFileName[1024];
    sprintf(bufferFileName, "output%03d.txt", rank);
    std::ofstream out(bufferFileName);

    out << "Grid coordinate : " << pGrdCrd.I << " , " << pGrdCrd.J << std::endl;
    int nbpRow, rankRow;
    MPI_Comm_size(rowComm, &nbpRow);
    MPI_Comm_rank(rowComm, &rankRow);

    int nbpCol, rankCol;
    MPI_Comm_size(colComm, &nbpCol);
    MPI_Comm_rank(colComm, &rankCol);

    out << "Comm row : " << rankRow << "/" << nbpRow << std::endl;
    out << "Comm col : " << rankCol << "/" << nbpCol << std::endl;

    auto localValues = generateValues(rank, nbLocalValues);

    if (localValues.size() < 100)
    {
        out << "Valeurs initiales : ";
        for ( auto const& v : localValues ) out << v << " ";
        out << std::endl;
    }   

    auto beg = std::chrono::high_resolution_clock::now();
    std::int32_t nbIter = std::int32_t(std::log2(double(nbp))+1)/2;
    std::sort(localValues.begin(), localValues.end());
    if (nbp>1)
    for (std::int32_t iter=0; iter<=nbIter; ++iter){
        // Row sort :
        localValues = oddEvenSort( localValues, rowComm);
#       if defined(DEBUG)        
        if (localValues.size() < 100)
        {
            out << "iter " << iter+0.5 << " => ";
            for ( auto const& v : localValues ) out << v << " ";
            out << std::endl;
        }
#       endif
        // Colum sort :
        localValues = oddEvenSort( localValues, colComm);
#       if defined(DEBUG)
        if (localValues.size() < 100)
        {
            out << "iter " << iter << " => ";
            for ( auto const& v : localValues ) out << v << " ";
            out << std::endl;
        }
#       endif
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duree = end - beg;
    out << "Temps pour le tri : " << duree.count() << std::endl;
    out << "Premiere valeurs locale : " << localValues.front() << std::endl;
    out << "Derniere valeurs locale : " << localValues.back()  << std::endl;
    if (localValues.size() < 100)
    {
        for ( auto const& v : localValues ) out << v << " ";
        out << std::endl;
    }
    out.close();
    MPI_Finalize();

    return EXIT_SUCCESS;
}