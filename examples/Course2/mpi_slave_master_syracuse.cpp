#include <cstdint>
#include <iostream>
#include <utility>
#include <vector>
#include <chrono>
#include <array>
#include <mpi.h>
#include <fstream>

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

int main(int nargs, char* argv[])
{
    constexpr std::int64_t nbSeries = 24'948'000;
    constexpr std::int64_t szPack   = 100;

    MPI_Comm global;
    int rank, nbp;
    MPI_Init(&nargs, &argv);
    MPI_Comm_dup(MPI_COMM_WORLD, &global);
    MPI_Comm_size(global, &nbp);
    MPI_Comm_rank(global, &rank);

    char bufferFileName[1024];
    sprintf(bufferFileName, "output%03d.txt", rank);
    std::ofstream out(bufferFileName);


    auto beg = std::chrono::high_resolution_clock::now();
    std::int64_t nbPacks = nbSeries/szPack;
    if (0==rank)
    {
        std::int64_t maxLength=0, maxHeight=0, u0MaxLength, u0MaxHeight;
        int iPack;
        for (int i=1; i<nbp; ++i)
        {
            iPack = i-1;
            MPI_Send(&iPack, 1, MPI_INT, i, 101, global);
        }
        iPack = nbp;
        MPI_Status status;
        while (iPack < nbPacks)
        {
            std::int64_t res[4];
            MPI_Recv(res, 4, MPI_INT64_T, MPI_ANY_SOURCE, MPI_ANY_TAG, global, &status);
            u0MaxLength = maxLength < res[0] ? res[1] : u0MaxLength;
            maxLength   = std::max(maxLength, res[0]);
            u0MaxHeight = maxHeight < res[2] ? res[3] : u0MaxHeight;
            maxHeight   = std::max(maxHeight, res[2]);
            int slaveRk = status.MPI_SOURCE;
            MPI_Send(&iPack, 1, MPI_INT, slaveRk, 101, global);
            ++iPack;                       
        }
        iPack = -1;
        for (int i=1; i<nbp; ++i)
        {
            std::int64_t res[4];
            MPI_Recv(res, 4, MPI_INT64_T, MPI_ANY_SOURCE, MPI_ANY_TAG, global, &status);
            u0MaxLength = maxLength < res[0] ? res[1] : u0MaxLength;
            maxLength   = std::max(maxLength, res[0]);
            u0MaxHeight = maxHeight < res[2] ? res[3] : u0MaxHeight;
            maxHeight   = std::max(maxHeight, res[2]);
            int slaveRk = status.MPI_SOURCE;
            MPI_Send(&iPack, 1, MPI_INT, slaveRk, 101, global);
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duree = end - beg;
        std::cout << "Temps de calcul des series de Syracuse : " << duree.count() << std::endl;

        std::cout << "Hauteur maximale trouvee " << maxHeight << " avec u0 = " << u0MaxHeight << std::endl;
        std::cout << "Longueur maximale trouvee " << maxLength << " avec u0 = " << u0MaxLength << std::endl;

    }
    else // Slave
    {
        MPI_Status status;
        int iPack;
        do 
        {
            MPI_Recv(&iPack, 1, MPI_INT, 0, 101, global, &status);
            if (iPack != -1)
            {
                auto res = packSyracuse(iPack*szPack, szPack);
                MPI_Send(res.data(), 4, MPI_INT64_T, 0, 101, global);
            }
        } while (iPack != -1);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duree = end - beg;
        std::cout << "Temps de calcul des series de Syracuse : " << duree.count() << std::endl;
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}