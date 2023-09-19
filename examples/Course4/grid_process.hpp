#ifndef _PARALLEL_GRID_PROCESS_HPP_
#define _PARALLEL_GRID_PROCESS_HPP_
#include <cassert>
#include <array>

namespace parallel
{
    constexpr int X=0, Y=1, Z=2;
    constexpr int CURRENT=0, LEFT=-1, RIGHT=1, BELOW=-2, ABOVE=+2, BACK=-3, FRONT=+3;
    template<int dim>
    struct gridProcess
    {
        std::array<int,dim> gridDim;
        std::array<int,dim> indices;

        gridProcess( std::array<int,dim> const& nbBlocks, std::array<int,dim> const& indicesBlock )
            :     gridDim(nbBlocks), indices(indicesBlock)
        {}

        int getRank( int dir = CURRENT )
        {
            int rank;
            if constexpr (dim==3) {
                rank = indices[2] + indices[1]*gridDim[2] + indices[0]*gridDim[2]*gridDim[1];
            }
        }
        
    };
}

#endif
