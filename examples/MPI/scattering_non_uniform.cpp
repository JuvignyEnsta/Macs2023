#include <cstdlib>
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

    int globalSize = 101;
    int localSize = globalSize/nbp;
    int root      = 0; // We can take other value if you want
    if (rank < globalSize%nbp) localSize += 1;
    std::vector<int> scatSizes; // A array which contains the size of received data for each processes (useful only for root)
    std::vector<int> displs   ; // For each process, tell where begin the data to receive in global array 
    std::vector<double> globalValues; // Values to scatter for root process
    std::vector<double> localValues;  // Buffer where store scattered values
    if (rank == root)
    {
        scatSizes.resize(nbp);
        displs.resize(nbp);
        displs[0] = 0;
        for ( int i=0; i<globalSize%nbp; ++i)
        {
            scatSizes[i] = globalSize/nbp+1;
            if (i<nbp)
                displs[i+1] = displs[i] + scatSizes[i];
        }
        for ( int i=globalSize%nbp; i<nbp; ++i)
        {
            scatSizes[i] = globalSize/nbp;
            if (i<nbp)
                displs[i+1] = displs[i] + scatSizes[i];
        }
        globalValues.resize(globalSize);
        for ( std::size_t i=0; i<globalValues.size(); ++i)
            globalValues[i] = 6*i+1;
    }
    localValues.resize(localSize);
    MPI_Scatterv(globalValues.data(), scatSizes.data(), displs.data(), MPI_DOUBLE, 
                 localValues.data(), localSize, MPI_DOUBLE, root, global);

    std::cout << rank << " : Scattered array => ";
    for (auto const& val : localValues)
        std::cout << val << " ";
    MPI_Finalize();
    return EXIT_SUCCESS;
}