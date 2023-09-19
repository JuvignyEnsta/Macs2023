#include <vector>
#include <utility>
#include <cassert>
#include <fstream>
#include <iostream> 
#include <mpi.h>

std::pair<std::vector<float>,std::vector<float>> 
assembleVectors( std::size_t iBeg, std::size_t iEnd)
{
    assert(iEnd > iBeg);
    std::int64_t dimLoc = iEnd - iBeg;
    std::vector<float> uLoc(dimLoc);
    std::vector<float> vLoc(dimLoc);
    for (std::int64_t iLocal=0; iLocal<dimLoc; ++iLocal)
    {
        std::size_t iGlob = iLocal + iBeg;
        uLoc[iLocal] = -0.49f*iGlob + 1.f;
        vLoc[iLocal] = 0.5f*iGlob - 1.f;
    }
    return {uLoc, vLoc};
}

std::vector<float>
addVectors( std::vector<float> const& u, std::vector<float>& v )
{
    assert(u.size() == v.size());
    std::vector<float> w(u.size());
    for (std::size_t i=0; i < u.size(); ++i )
        w[i] = u[i] + v[i];
    return w;
}

int main(int nargs, char* argv[])
{
    std::size_t N = 360;
    MPI_Comm global;
    int rank, nbp;
    MPI_Init(&nargs, &argv);
    MPI_Comm_dup(MPI_COMM_WORLD, &global);
    MPI_Comm_size(global, &nbp);
    MPI_Comm_rank(global, &rank);

    char bufferFileName[1024];
    sprintf(bufferFileName, "output%03d.txt", rank);
    std::ofstream out(bufferFileName);

    if (N%nbp != 0)
    {
        std::cerr << "Must have a number of processes which divides the dimension " << N << " of the vectors." << std::endl;
        MPI_Abort(global,-1);
    }
    std::size_t NLoc = N/nbp;
    std::size_t ibeg = rank * NLoc;
    std::size_t iend = (rank+1)*NLoc;

    auto [uLoc,vLoc] = assembleVectors(ibeg, iend);

    auto wLoc = addVectors(uLoc,vLoc);

    out << "wLoc = [ ";
    for (auto const& val : wLoc )
        out << val << "\t";
    out << "]" << std::endl;

    out.close();
    MPI_Finalize();
    return EXIT_SUCCESS;
}