#include <iostream>
#include <mpi.h>

int main(int nargs, char* argv[] )
{
    int token, rank, nbp;
    MPI_Comm globCom;

    MPI_Init(&nargs, &argv);
    MPI_Comm_dup(MPI_COMM_WORLD, &globCom);
    MPI_Comm_size(globCom, &nbp);
    MPI_Comm_rank(globCom, &rank);

    int tag = 404;
    if (rank == 0)
    {
        token = 42;// Token has universe's answer as value
        // Send the token (one int) to process 1 tagged with the tag variable
        MPI_Send(&token, 1, MPI_INT, 1, tag, globCom);
    }
    else if (rank == 1)
    {
        MPI_Status status;
        // Receive the token (one int) from process 0 with no tag to care.
        MPI_Recv(&token, 1, MPI_INT, 0, MPI_ANY_TAG, globCom, &status);
    }
    std::cout << "Process " << rank << " has token " << token << " in memory." << std::endl;

    MPI_Finalize();
}