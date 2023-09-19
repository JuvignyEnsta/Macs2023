#ifndef _TRIDIAG_PREC_HPP_
#define _TRIDIAG_PREC_HPP_
#include <memory>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>
#include "vecteur.hpp"

namespace algebra
{
class TriDiagPrec
{
public:
    TriDiagPrec( int t_width, int t_height );
    ~TriDiagPrec() = default;

    Vecteur solve ( Vecteur const& x ) const
    {
        return m_pt_llt->solve(x);
    }
private:
    int m_width, m_height;
    std::unique_ptr<Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>>> m_pt_llt{nullptr};
};
}

#endif
