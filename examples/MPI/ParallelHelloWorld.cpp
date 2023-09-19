#include <iostream>
#include <mpi.h>

int main(int nargs, char* argv[])
{
    int rank, nbp, lenres;
    MPI_Comm globCom;
    char name[4096];
    MPI_Init(&nargs, &argv);
    MPI_Comm_dup(MPI_COMM_WORLD, &globCom);
    MPI_Comm_size(globCom, &nbp);
    MPI_Comm_rank(globCom, &rank);
    MPI_Get_processor_name(name, &lenres);

    std::cout << "I'm the process " << rank << " on " << nbp << " processes" << std::endl;
    std::cout << "I'm running on " << name << " host." << std::endl;

    MPI_Finalize();
    return 0;
}
