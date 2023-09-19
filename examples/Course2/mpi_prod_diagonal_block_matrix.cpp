#include <vector>
#include <cstdint>
#include <iostream>
#include <cmath>
#include <cassert>
#include <cstdlib> 
#include <chrono>
#include <fstream>
#include <mpi.h>
#include <algorithm>
#include <cblas.h>

constexpr double twoPi = 6.283185307179586;

extern "C" void dgemm_( const char& trA, const char& trB, std::int32_t const& m, std::int32_t const &n, std::int32_t const& k,
                        const double& alpha, double const* A, std::int32_t const& ldA, double const* B, std::int32_t const& ldB,
                        double const& beta, double* C, std::int32_t const& ldC );

class DiagonalBlock
{
public:
    DiagonalBlock( ) = default; 
    DiagonalBlock( std::int32_t t_dimension )
        :   m_dimension(t_dimension),
            m_coefs(t_dimension*t_dimension)
    {}
    DiagonalBlock( std::int32_t t_dimension, double initValue )
        :   m_dimension(t_dimension),
            m_coefs(t_dimension*t_dimension, initValue)
    {}
    DiagonalBlock( std::int32_t t_dimension, double t_frequence, std::int32_t t_firstIndex );
    DiagonalBlock( DiagonalBlock const& ) = delete;
    DiagonalBlock( DiagonalBlock     && ) = default;
    ~DiagonalBlock() = default;

    DiagonalBlock& operator = ( DiagonalBlock const& ) = delete;
    DiagonalBlock& operator = ( DiagonalBlock     && ) = default;

    double& operator() ( std::int32_t i, std::int32_t j )
    {
        return m_coefs[i + j*m_dimension];
    }

    double operator() ( std::int32_t i, std::int32_t j ) const 
    {
        return m_coefs[i + j*m_dimension];
    }

    double const* data() const { return m_coefs.data(); }
    double      * data()       { return m_coefs.data(); }

    std::int32_t getDimension() const { return m_dimension; }

    DiagonalBlock operator * ( DiagonalBlock const& B ) const;

    std::ostream& print( std::ostream& out ) const;

private:
    std::int32_t        m_dimension;
    std::vector<double> m_coefs;
};

DiagonalBlock::DiagonalBlock( std::int32_t t_dimension, double t_frequence, 
                              std::int32_t t_firstIndex )
    :   m_dimension(t_dimension),
        m_coefs(t_dimension * t_dimension)
{
    std::vector<double> Ua(t_dimension);
    for (std::int32_t i=0; i<t_dimension; ++i)
    {
        std:int64_t iGlob = i + t_firstIndex;
        Ua[i] = std::sin(twoPi * t_frequence * iGlob);
    }
    // Remplissage des coefficients du bloc diagonal
    for (std::int32_t j=0; j<t_dimension; ++j)
    {
        std::int32_t jGlob = j + t_firstIndex;
        double jcos = std::cos(twoPi * t_frequence * jGlob);
        for (std::int32_t i=0; i<t_dimension; ++i)
        {
            m_coefs[i + j*t_dimension] = jcos * Ua[i];
        }
    }
}

DiagonalBlock DiagonalBlock::operator * ( DiagonalBlock const& B ) const
{
    DiagonalBlock const& A = *this;
    assert(B.getDimension() == A.getDimension());
    std::int32_t dim = A.getDimension();
    DiagonalBlock Result(dim, 0.);
    char tr = 'N';
    double alpha = 1., beta = 0.;
    openblas_set_num_threads(1);

    dgemm_( tr, tr, dim, dim, dim, alpha, A.data(), dim, B.data(), dim, beta, Result.data(), dim);

    return Result;
}

std::ostream& 
DiagonalBlock::print( std::ostream& out ) const
{
    for (std::int32_t i=0; i<m_dimension; ++i)
    {
        out << "[ ";
        for ( std::int32_t j=0; j<m_dimension; ++j)
            out << m_coefs[i + j*m_dimension] << "\t";
        out << " ]" << std::endl;
    }
    return out;
}

bool verifBlockOfC( std::int32_t t_dim, std::int32_t indFirstRow, double freqA, double freqB,
                    DiagonalBlock const& Cii )
{
    /* Calcul de s = Va^{T}.Ub */
    double prodScal=0;
    for ( std::size_t iLoc=0; iLoc < t_dim; iLoc++ )
    {
        std::int32_t iGlob = iLoc + indFirstRow;
        prodScal += std::cos(twoPi * freqA * iGlob) * std::sin(twoPi * freqB * iGlob);
    }
    std::vector<double> jcos(t_dim), isin(t_dim);
    for (std::int32_t i=0; i<t_dim; ++i)
    {
        std::int32_t iGlob = i + indFirstRow;
        jcos[i] = std::cos(twoPi * freqB * iGlob);
        isin[i] = std::sin(twoPi * freqA * iGlob);
    }
    for ( std::size_t jLoc=0; jLoc<t_dim; ++jLoc )
    {
        std::int32_t jGlob = jLoc + indFirstRow;
        for ( std::size_t iLoc=0; iLoc<t_dim; ++iLoc )
        {
            std::int32_t iGlob = iLoc + indFirstRow;
            // Calcul du coefficient i,j de Ua.Vb^{T}
            double coef = prodScal * jcos[jLoc] * isin[iLoc];
            if (std::abs(Cii(iLoc,jLoc) - coef) > 1.E-10)
            {
                std::cerr << "Erreur dans le produit matrice-matrice : ";
                std::cerr << "C[" << iLoc << "," << jLoc << "] = "
                          << Cii(iLoc,jLoc) << " et valeurs attendue : "
                          << coef << std::endl;
                return false;
            }
        }
    }
    return true;
}

std::vector<std::int32_t> distribBlocks( std::vector<std::int32_t> const& t_dimensions, int nbp, int rank )
{
    std::vector<double> weights(nbp);
    std::vector<std::size_t> sort_index(t_dimensions.size());
    std::vector<std::int32_t> indexBlock;
    for ( std::size_t i=0; i<t_dimensions.size(); ++i )
    {
        sort_index[i] = i;
    }
    // On fait un tri du tableau d'index de sorte de trier indirectement le tableau des dimensions
    // de façon décroissante
    std::sort(sort_index.begin(), sort_index.end(), [&t_dimensions] (std::size_t i, std::size_t j) { return t_dimensions[i] > t_dimensions[j]; });

    // Initialisation des poids en distribuant les plus grandes dimensions sur les processus :
    for ( int p=0; p<nbp; ++p)
    {
        std::int32_t dim = t_dimensions[sort_index[p]];
        weights[p] = dim;
    }
    indexBlock.push_back(sort_index[rank]);
    for (std::size_t p=nbp; p < t_dimensions.size(); ++p)
    {
        std::int32_t dim = t_dimensions[sort_index[p]];
        // Recherche du poids le plus petit :
        auto iterMin = std::min_element(weights.begin(), weights.end());
        *iterMin += dim;
        auto minRank = std::distance(weights.begin(), iterMin);
        if ( minRank == rank) indexBlock.push_back(sort_index[p]);
    }
    return indexBlock;
}

int main(int nargs, char* argv[])
{
    constexpr std::int32_t nbBlocks = 180;
    constexpr double freq1 = 0.125;
    constexpr double freq2 = 0.0134;

    MPI_Comm global;
    int rank, nbp;
    MPI_Init(&nargs, &argv);
    MPI_Comm_dup(MPI_COMM_WORLD, &global);
    MPI_Comm_size(global, &nbp);
    MPI_Comm_rank(global, &rank);

    char bufferFileName[1024];
    sprintf(bufferFileName, "output%03d.txt", rank);
    std::ofstream out(bufferFileName);

    std::vector<std::int32_t> dimensions(nbBlocks), begRows(nbBlocks);
    begRows[0] = 0;
    // Calcul des dimensions (et du début de chaque bloc)
    for (std::int32_t iBlock=0; iBlock<nbBlocks; ++iBlock)
    {
        dimensions[iBlock] = 10*(iBlock+1);
        if (iBlock < nbBlocks-1)
            begRows[iBlock+1] = begRows[iBlock] + dimensions[iBlock];
    }
    // Distributions des blocs pour optimiser l'équilibrage :
    auto indexLocalBlocks = distribBlocks( dimensions, nbp, rank );

    out << "Distribution : " << std::endl;
    for ( auto index : indexLocalBlocks )
        out << index << " ";
    out << std::endl;
    // Initialisation et distribution des blocs diagonaux de A et B :
    // --------------------------------------------------------------
    // A_ii est sous la forme : Ua.Va^{T} (avec Ua et Va des vecteurs)
    // B_ii est sous la forme : Ub.Vb^{T} (avec Ub et Vb des vecteurs)
    //
    auto beg = std::chrono::high_resolution_clock::now();
    auto nbBlocksLoc = indexLocalBlocks.size();
    std::vector<DiagonalBlock> A; A.reserve(nbBlocksLoc);
    std::vector<DiagonalBlock> B; B.reserve(nbBlocksLoc);
    for ( std::int32_t iBlock = 0; iBlock < nbBlocksLoc; ++iBlock )
    {
        std::int32_t index = indexLocalBlocks[iBlock];
        // On génère la dimension de chaque bloc avec un calcul pseudo-aléatoire
        A.emplace_back(dimensions[index], freq1, begRows[index]);
        B.emplace_back(dimensions[index], freq2, begRows[index]);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duree = end - beg;
    out << "Temps d'assemblage des blocs : " << duree.count() << " secondes" << std::endl;
    
    // Calcul des blocs diagonaux de C = A.B
    beg = std::chrono::high_resolution_clock::now();
    std::vector<DiagonalBlock> C(nbBlocksLoc);
    for ( std::int32_t iBlock=0; iBlock<nbBlocksLoc; ++iBlock)
    {
        C[iBlock] = std::move(A[iBlock]*B[iBlock]);
    }
    end = std::chrono::high_resolution_clock::now();
    duree = end - beg;
    out << "Temps produit des blocs diagonaux : " << duree.count() << " secondes" << std::endl;

    // Vérification des blocs diagonaux calculés pour C :
    // -------------------------------------------------
    // Un bloc de C se calcul en fait comme : C_{ii} = (Va^{T}.Ub) Ua.Vb^{T}
    beg = std::chrono::high_resolution_clock::now();
    for ( std::int32_t iBlock=0; iBlock<nbBlocksLoc; ++iBlock)
    {
        std::int32_t index = indexLocalBlocks[iBlock];
        if (!verifBlockOfC(C[iBlock].getDimension(), begRows[index], freq1, freq2, C[iBlock]))
        {
            std::cerr << "Erreur dans le calcul du bloc " << iBlock << std::endl;
        }
    }
    end = std::chrono::high_resolution_clock::now();
    duree = end - beg;
    out << "Temps verification des blocs diagonaux de C : " << duree.count() << " secondes" << std::endl;

    double mem = 0.;
    for (auto const& ABlk : A)
    {
        std::int32_t dim = ABlk.getDimension();
        mem += 3.*dim*dim;
    }
    out << "Mémoire prise (en nombre de double) : " << mem << std::endl;

    out.close();
    MPI_Finalize();
    return EXIT_SUCCESS;
}