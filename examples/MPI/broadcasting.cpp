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

    std::vector<float> tab(6);
    if (rank==0)
    {
        tab = std::vector<float>{2.f, 3.f, 5.f, 7.f, 11.f, 13.f};
    }
    MPI_Bcast(tab.data(), tab.size(), MPI_FLOAT, 0, global);

    std::cout << rank << " : ";
    for (auto v : tab )
        std::cout << v << " ";

    MPI_Finalize();
}