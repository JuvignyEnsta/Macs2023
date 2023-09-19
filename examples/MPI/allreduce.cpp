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
    std::vector<float> localValues(localSize);
    for (int i=0; i<localSize; ++i)
    {
        localValues[i] = rank*localSize + i + 1;
    }
    std::vector<float> result;
    auto op = MPI_SUM; // We can change for other operations MPI_PROD, MPI_MAX, MPI_MIN, MPI_MAXLOC, MPI_MINLOC, and so.
    result.resize(localSize);
    /** Store the reduction operation in result array */
    MPI_Allreduce(localValues.data(), result.data(), localSize, MPI_FLOAT, 
               op, global);

    std::cout << rank << " : localValues => ";
    for (auto const& val : localValues)
        std::cout << val << " ";
    std::cout << std::endl;

    std::cout << rank << " : result of reduction : ";
    for (auto const& v : result)
    {
        std::cout << v << " ";
    }
    MPI_Finalize();
    return EXIT_SUCCESS;
}
