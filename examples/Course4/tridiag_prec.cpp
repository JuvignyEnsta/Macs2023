#include <iostream>
#include "tridiag_prec.hpp"

algebra::TriDiagPrec::TriDiagPrec( int t_width, int t_height )
    :    m_width(t_width),
         m_height(t_height)
{
    auto width = t_width;
    auto height= t_height;
    using EltT =  Eigen::Triplet<double>;
    std::vector<EltT> tripletList;
    tripletList.reserve(3*t_width*t_height);

    // Laplacien première ligne :
    tripletList.emplace_back(0,0,4.);
    tripletList.emplace_back(0,1,-1.);
    for ( unsigned iWidth=1; iWidth < width-1; ++iWidth ) {
        tripletList.emplace_back(iWidth, iWidth-1,-1.);
        tripletList.emplace_back(iWidth, iWidth  ,+4.);
        tripletList.emplace_back(iWidth, iWidth+1,-1.);
    }
    tripletList.emplace_back(m_width-1, m_width-2,-1.);
    tripletList.emplace_back(m_width-1, m_width-1,+4.);
    
    // Laplacien sur les lignes 1 à height-2
    for (unsigned jHeight=1; jHeight < m_height-1; ++jHeight) {
        unsigned index = jHeight*m_width;
        tripletList.emplace_back(index,index        ,+4.);
        tripletList.emplace_back(index,index+1      ,-1.);

        for (unsigned iWidth=1; iWidth< width-1; ++iWidth) {
            unsigned index = jHeight*width + iWidth;
            tripletList.emplace_back(index, index-1      ,-1.);
            tripletList.emplace_back(index, index        ,+4.);
            tripletList.emplace_back(index, index+1      ,-1.);
        }
        index = jHeight*width + width-1;
        tripletList.emplace_back(index, index-1      ,-1.);
        tripletList.emplace_back(index, index        ,+4.);
    }// End for jHeight
    
    // Laplacien dernière ligne :
    unsigned index = (height-1)*width;
    tripletList.emplace_back(index, index        ,+4.);
    tripletList.emplace_back(index, index+1      ,-1.);
    for (unsigned iWidth=1; iWidth< width-1; ++iWidth) {
        unsigned index = (height-1)*width + iWidth;
        tripletList.emplace_back(index, index-1      ,-1.);
        tripletList.emplace_back(index, index        ,+4.);
        tripletList.emplace_back(index, index+1      ,-1.);
    }// End for iwidth
    index = (height-1)*m_width + m_width-1;
    tripletList.emplace_back(index, index-1      ,-1.);
    tripletList.emplace_back(index, index        ,+4.);
    Eigen::SparseMatrix<double> L(t_width*t_height, t_width*t_height);
    L.setFromTriplets(tripletList.begin(),tripletList.end());
    m_pt_llt = std::make_unique<Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>>>(L);
}
// ------------------------------------------------------------------------
