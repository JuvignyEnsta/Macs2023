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

    MPI_Request request;
    MPI_Status  status ;
    if (rank==0)
    {
        std::vector<int> sendBuffer{ 2, 5, 11, 17};// Send buffer
        std::vector<int> recvBuffer(4);                         // Recv buffer
        // Non blocking receive
        MPI_Irecv(recvBuffer.data(), recvBuffer.size(), MPI_INT, 1, 405, global, &request);
        // Blocking send
        MPI_Send ( sendBuffer.data(), sendBuffer.size(), MPI_INT, 1, 404, global);
        // Wait end of the reception
        MPI_Wait(&request, &status);
        std::cout << rank << " : ";
        for (auto v : recvBuffer)
            std::cout << v << " ";
    }
    else if (rank==1)
    {
        std::vector<int> sendBuffer{ 3, 7, 13, 19 };// Send Buffer
        std::vector<int> recvBuffer(4);                          // Recv Buffer
        // Non blocking receive
        MPI_Irecv(recvBuffer.data(), recvBuffer.size(), MPI_INT, 0, 404, global, &request);
        // Blocking send
        MPI_Send( sendBuffer.data(), sendBuffer.size(), MPI_INT, 0, 405, global);
        // Wait end of the reception
        MPI_Wait(&request, &status);
        std::cout << rank << " : ";
        for (auto v : recvBuffer)
            std::cout << v << " ";
    }

    MPI_Finalize();
    return EXIT_FAILURE;
}
