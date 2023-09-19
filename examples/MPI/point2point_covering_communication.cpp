#include <iostream>
#include <mpi.h>
#include <vector>

int main(int nargs, char* argv[])
{
    int rank, nbp;
    MPI_Comm global;
    MPI_Init(&nargs, &argv);
    MPI_Comm_dup(MPI_COMM_WORLD, &global);
    MPI_Comm_size(global, &nbp);
    MPI_Comm_rank(global, &rank);

    MPI_Request request;// Structure using to synchronize non blocking messages
    if (rank==0)
    {
        MPI_Status status;// To have message state
        std::vector<double> tab{2.,3.,5.,7.,11.,13.,17.};// The data to send
        // Non blocking send to compute sum at the same time than data is sended.
        MPI_Isend(tab.data(), tab.size(), MPI_DOUBLE, 1, 404, global, &request);
        double accum=0;
        for (double val : tab)
            accum += 1./val;
        std::cout << rank << " : sum(tab) = " << accum << std::endl;
        MPI_Wait(&request,&status);// Wait the end of the sending.
    }
    else if (rank==1)
    {
        int isReceived=0;
        MPI_Status status;// Message state
        std::vector<double> tab(7);// Array where data are received
        // Non blocking receive
        MPI_Irecv(tab.data(), tab.size(), MPI_DOUBLE, 0, 404, global, &request);
        // While message is not yet received
        while (isReceived==0)
        {
            MPI_Test(&request, &isReceived, &status);// Is message received ? (one=yes, zero=no)
            std::cout << "Waiting message. Computing other data..." << std::endl;
        }// OK, at the exit of while loop, the message is received. We can use the data in tab.
        double accum=0;
        for (double val : tab)
            accum += 1./val;
        std::cout << rank << " : sum(tab) = " << accum << std::endl;
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}