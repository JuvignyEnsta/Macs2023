#include <iostream>
#include <vector>
#include <mpi.h>

int main(int nargs, char* argv[])
{
    MPI_Comm global;
    int rank, nbp;
    MPI_Init(&nargs, &argv);
    MPI_Comm_dup(MPI_COMM_WORLD, &global);
    MPI_Comm_size(global, &nbp);
    MPI_Comm_rank(global, &rank);

    int localSize  = 5;
    int globalSize = nbp*localSize;
    std::vector<int> scatteredData(localSize);
    std::vector<int> globalData;
    if (rank==0)
    {
        globalData.resize(globalSize);
        for (int i=0; i<globalSize; ++i)
        {
            globalData[i] = 6*i-1;
        }
    }
    MPI_Scatter(globalData.data(), localSize, MPI_INT, scatteredData.data(), localSize, MPI_INT, 0, global);

    std::cout << rank << " : Scattered data : ";
    for (auto v : scatteredData)
        std::cout << v << " ";
    std::cout << std::endl;

    MPI_Finalize();
    return EXIT_SUCCESS;
}