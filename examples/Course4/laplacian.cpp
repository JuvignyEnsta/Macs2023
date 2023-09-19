#include <chrono>
#include <iostream>
#include "laplacian.hpp"
#include <Eigen/SparseCholesky>

algebra::Laplacian::Laplacian( int t_width, int t_height )
    :    m_width(t_width),
         m_height(t_height),
         m_sparseMatrix(t_width*t_height, t_width*t_height)
{
    auto width = t_width;
    auto height= t_height;
    using EltT =  Eigen::Triplet<double>;
    std::vector<EltT> tripletList;
    tripletList.reserve(5*t_width*t_height);

    // Laplacien première ligne :
    tripletList.emplace_back(0,0,4.);
    tripletList.emplace_back(0,1,-1.);
    tripletList.emplace_back(0,m_width,-1.);
    for ( unsigned iWidth=1; iWidth < width-1; ++iWidth ) {
        tripletList.emplace_back(iWidth, iWidth-1,-1.);
        tripletList.emplace_back(iWidth, iWidth  ,+4.);
        tripletList.emplace_back(iWidth, iWidth+1,-1.);
        tripletList.emplace_back(iWidth, iWidth+m_width,-1.);
    }
    tripletList.emplace_back(m_width-1, m_width-2,-1.);
    tripletList.emplace_back(m_width-1, m_width-1,+4.);
    tripletList.emplace_back(m_width-1, 2*m_width-1,-1.);
    
    // Laplacien sur les lignes 1 à height-2
    for (unsigned jHeight=1; jHeight < m_height-1; ++jHeight) {
        unsigned index = jHeight*m_width;
        tripletList.emplace_back(index,index-m_width,-1.);
        tripletList.emplace_back(index,index        ,+4.);
        tripletList.emplace_back(index,index+1      ,-1.);
        tripletList.emplace_back(index,index+m_width,-1.);

        for (unsigned iWidth=1; iWidth< width-1; ++iWidth) {
            unsigned index = jHeight*width + iWidth;
            tripletList.emplace_back(index, index-m_width,-1.);
            tripletList.emplace_back(index, index-1      ,-1.);
            tripletList.emplace_back(index, index        ,+4.);
            tripletList.emplace_back(index, index+1      ,-1.);
            tripletList.emplace_back(index, index+m_width,-1.);
        }
        index = jHeight*width + width-1;
        tripletList.emplace_back(index, index-m_width,-1.);
        tripletList.emplace_back(index, index-1      ,-1.);
        tripletList.emplace_back(index, index        ,+4.);
        tripletList.emplace_back(index, index+m_width,-1.);
    }// End for jHeight
    
    // Laplacien dernière ligne :
    unsigned index = (height-1)*width;
    tripletList.emplace_back(index, index-m_width,-1.);
    tripletList.emplace_back(index, index        ,+4.);
    tripletList.emplace_back(index, index+1      ,-1.);
    for (unsigned iWidth=1; iWidth< width-1; ++iWidth) {
        unsigned index = (height-1)*width + iWidth;
        tripletList.emplace_back(index, index-m_width,-1.);
        tripletList.emplace_back(index, index-1      ,-1.);
        tripletList.emplace_back(index, index        ,+4.);
        tripletList.emplace_back(index, index+1      ,-1.);
    }// End for iwidth
    index = (height-1)*m_width + m_width-1;
    tripletList.emplace_back(index, index-m_width,-1.);
    tripletList.emplace_back(index, index-1      ,-1.);
    tripletList.emplace_back(index, index        ,+4.);
    m_sparseMatrix.setFromTriplets(tripletList.begin(),tripletList.end());
}
// ------------------------------------------------------------------------
auto
algebra::Laplacian::operator * ( Vecteur const& x ) const -> Vecteur
{
    return m_sparseMatrix * x;
}
// ------------------------------------------------------------------------
auto
algebra::Laplacian::solve( Vecteur const& b ) const -> Vecteur
{
    auto beg = std::chrono::high_resolution_clock::now();
    Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>> chol(m_sparseMatrix);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duree = end - beg;
    std::cout << "Temps factorisation : " << duree.count() << "s" << std::endl;
    beg = std::chrono::high_resolution_clock::now();
    auto res = chol.solve(b);
    end = std::chrono::high_resolution_clock::now();
    duree = end - beg;
    std::cout << "Temps resolution : " << duree.count() << "s" << std::endl;
    return res;
}
