#ifndef _laplacian_hpp_
#define _laplacian_hpp_
#include <Eigen/SparseCore>
#include "vecteur.hpp"

namespace algebra {
class Laplacian
{
public:
    Laplacian( int t_width, int t_height );
    ~Laplacian() = default;

    std::int64_t dimension() const { return m_width*m_height; }
    Vecteur operator * ( Vecteur const& x ) const;
    Vecteur solve( Vecteur const& x ) const;
    Eigen::SparseMatrix<double> const& matrix() const { return m_sparseMatrix; }
    Eigen::SparseMatrix<double>& matrix() { return m_sparseMatrix; }
private:
    int m_width, m_height;
    Eigen::SparseMatrix<double> m_sparseMatrix;
};
}
#endif
