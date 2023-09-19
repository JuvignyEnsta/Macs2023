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

static std::vector<MPI_Comm> commCubes;
static std::ofstream out; 

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
void sortBitonicSequence( std::size_t nbLocalVals, std::int32_t* bitonicSequence, bool increasingSort = true)
{
    if (nbLocalVals == 1) return;
    if (nbLocalVals == 2)
    {
        if (increasingSort)
        {
            if (bitonicSequence[0] > bitonicSequence[1]) std::swap(bitonicSequence[0],bitonicSequence[1]);
        }
        else 
        {
            if (bitonicSequence[0] < bitonicSequence[1]) std::swap(bitonicSequence[0],bitonicSequence[1]);
        }
        return;
    }
    std::size_t Nd2 = nbLocalVals/2;
    if (increasingSort)
    {
        for ( std::size_t i=0; i < Nd2; ++i )
        {
            if (bitonicSequence[i] > bitonicSequence[Nd2+i]) std::swap(bitonicSequence[i], bitonicSequence[Nd2+i]);
        }
    }
    else 
    {
        for ( std::size_t i=0; i < Nd2; ++i )
        {
            if (bitonicSequence[i] < bitonicSequence[Nd2+i]) std::swap(bitonicSequence[i], bitonicSequence[Nd2+i]);
        }
    }
    sortBitonicSequence(Nd2, bitonicSequence, increasingSort);
    sortBitonicSequence(Nd2, bitonicSequence+Nd2, increasingSort);
}
// ====================================================================================================================
void distributedSortBitonicSequence( std::size_t nbLocalVals, std::int32_t* bitonicSequence, int level, bool increasingSort = true)
{
    int rank, nbp;
    MPI_Comm_rank(commCubes[level], &rank);
    MPI_Comm_size(commCubes[level], &nbp );

#   ifdef DEBUG
    out << "-----------------------" << std::endl;
    out << "\t" << level << " : " << rank << " " << nbp << std::endl;
#   endif


    int exchgRank = 2*rank < nbp ?  rank + nbp/2 : rank - nbp/2;

    MPI_Status status;
    std::vector<std::int32_t> buffer(nbLocalVals);
    MPI_Sendrecv( bitonicSequence, nbLocalVals, MPI_INT32_T, exchgRank, 303, buffer.data(), nbLocalVals, MPI_INT32_T, exchgRank, 303, commCubes[level], &status);

    if (2*rank < nbp)
    {
        if (increasingSort)
            for (std::size_t i=0; i<nbLocalVals; ++i)
                bitonicSequence[i] = std::min(bitonicSequence[i], buffer[i]);
        else 
            for (std::size_t i=0; i<nbLocalVals; ++i)
                bitonicSequence[i] = std::max(bitonicSequence[i], buffer[i]);
    }
    else 
    {
        if (increasingSort)
            for (std::size_t i=0; i<nbLocalVals; ++i)
                bitonicSequence[i] = std::max(bitonicSequence[i], buffer[i]);
        else 
            for (std::size_t i=0; i<nbLocalVals; ++i)
                bitonicSequence[i] = std::min(bitonicSequence[i], buffer[i]);
    }
#   ifdef DEBUG
        out << "\tlevel " << level << " : ";
        if (nbLocalVals < 100)
        {
            for ( std::size_t i=0; i< nbLocalVals; ++i ) out << bitonicSequence[i] << " ";
            out << std::endl;
        }
        else 
        {
            out << "Range : " << bitonicSequence[0] << " to " << bitonicSequence[nbLocalVals-1] << std::endl;
        }
#       endif

    if (level > 1)
    {
        distributedSortBitonicSequence( nbLocalVals, bitonicSequence, level-1, increasingSort);       
    }
    else 
    {
        sortBitonicSequence( nbLocalVals, bitonicSequence, increasingSort);
    }
}
// ====================================================================================================================
int main( int nargs, char* argv[] )
{
    std::int32_t nbValues = 65'536;

    MPI_Comm global, rowComm, colComm;
    int rank, nbp;
    MPI_Init(&nargs, &argv);
    MPI_Comm_dup(MPI_COMM_WORLD, &global);
    MPI_Comm_size(global, &nbp);
    MPI_Comm_rank(global, &rank);

    char bufferFileName[1024];
    sprintf(bufferFileName, "output%03d.txt", rank);
    out = std::ofstream(bufferFileName);

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
    commCubes.resize(dim+1);
    MPI_Comm_dup(global, &commCubes[dim]);
    for ( int idim=1; idim < dim; ++idim )
    {
        MPI_Comm_split(global, rank/(1<<idim), rank, &commCubes[idim]);
    }// On n'utilise pas le comm zéro car représente un seul process
    std::vector<std::int32_t> buffer(localValues.size());
    if (rank%2 == 0)// On trie dans l'ordre croissant
        std::sort(localValues.begin(), localValues.end());
    else // On trie dans le sens inverse (decroissant) : 
        std::sort(localValues.rbegin(), localValues.rend() );
#   ifdef DEBUG
    if (localValues.size() < 100)
    {
        for ( auto const& v : localValues ) out << v << " ";
        out << std::endl;
    }
    else 
    {
        out << "Range : " << localValues.front() << " to " << localValues.back() << std::endl;
    }
#   endif
    for (int d=0; d<dim; ++d)
    {
#       ifdef DEBUG
        out << "Avant Iteration " << d << " : ";
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
        // Si rank%(2^(d+2)) < 2^{d+1} => increasing sinon decreasing
        distributedSortBitonicSequence( nbLocalValues, localValues.data(), d+1, ((rank%(1<<(d+2))) < (1<<(d+1)) ) );
#       ifdef DEBUG
        out << "Iteration " << d << " : ";
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

}
