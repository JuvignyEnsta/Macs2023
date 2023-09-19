#ifndef _CHOLESKY_INCOMPLET_HPP_
#define _CHOLESKY_INCOMPLET_HPP_
#include <Eigen/SparseCore>
#include "vecteur.hpp"

namespace algebra
{
class CholeskyIncomplet
{
public:
    CholeskyIncomplet( Eigen::SparseMatrix<double> const& A );
    ~CholeskyIncomplet() = default;

    Vecteur solve ( Vecteur const& x ) const;
private:
    int m_width, m_height;
    Eigen::SparseMatrix<double> m_llt;
};
}

#endif
