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

    int localSize = 5; // Local size for each local array
    int root      = 0; // The process which receive the final result (can be changed)
    std::vector<float> localValues(localSize);
    for (int i=0; i<localSize; ++i)
    {
        localValues[i] = rank*localSize + i + 1;
    }
    std::vector<float> result;
    auto op = MPI_SUM; // We can change for other operations MPI_PROD, MPI_MAX, MPI_MIN, MPI_MAXLOC, MPI_MINLOC, and so.
    if (rank == root) result.resize(localSize);
    /** Store the reduction operation in result array */
    MPI_Reduce(localValues.data(), result.data(), localSize, MPI_FLOAT, 
               op, root, global);

    std::cout << rank << " : localValues => ";
    for (auto const& val : localValues)
        std::cout << val << " ";
    std::cout << std::endl;

    if (0==rank)
    {
        std::cout << rank << " : result of reduction : ";
        for (auto const& v : result)
        {
            std::cout << v << " ";
        }
    }
    MPI_Finalize();
    return EXIT_SUCCESS;
}
