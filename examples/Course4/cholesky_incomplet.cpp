#include <iostream>
#include <cmath>
#include "cholesky_incomplet.hpp"

algebra::CholeskyIncomplet::CholeskyIncomplet( Eigen::SparseMatrix<double> const& A )
{
    m_llt = A.triangularView<Eigen::Lower>();
    double pivot;
    for (int jcol=0; jcol<m_llt.outerSize(); ++jcol) { // Pour chaque colonne de la matrice A
        if ((jcol & 2047) == 0) std::cout << "jcol : " << jcol << std::endl;
        for (Eigen::SparseMatrix<double>::InnerIterator it(m_llt,jcol); it; ++it){
            double& val = it.valueRef();
            auto   row = it.row();
            auto col = it.col();
            assert(col == jcol);
            if (row == col) {
                pivot = std::sqrt(val);
                val   = pivot;
            } else {
                assert(row>col);
                val /= pivot;
            }
        }
        for (int j=jcol+1; j<m_llt.outerSize(); ++j) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(m_llt,j); it; ++it){
                double& val = it.valueRef();
                auto    row = it.row();
                auto   col2 = it.col();
                assert(col2 == j);
                assert(row >= j);
                double a[2];
                int ia = 0;
                for (Eigen::SparseMatrix<double>::InnerIterator it2(m_llt,jcol); it2 && (ia<2); ++it2) {
                    if (row == it2.row())
                        a[ia++] = it2.value();
                    if  (j == it2.row())
                        a[ia++] = it2.value();
                }
                if (ia==2) {
                    val -= a[0] * a[1];
                }
            }
        }
    }
}
// ------------------------------------------------------------------------
auto
algebra::CholeskyIncomplet::solve( Vecteur const& x ) const -> Vecteur
{
    Vecteur y(x);
    // Descente
    for (int jcol=0; jcol<m_llt.outerSize(); ++jcol) { // Pour chaque colonne de la matrice A
        for (Eigen::SparseMatrix<double>::InnerIterator it(m_llt,jcol); it; ++it){
            auto row = it.row();
            auto val = it.value();
            assert(row >= jcol);
            if (row == jcol) {
                y[row] /= val;
            }
            else y[row] -= val*y[jcol];
        }
    }
    // Remontée :
    double pivot;
    for (int jcol=m_llt.outerSize()-1; jcol>=0;--jcol) {
        pivot=0;
        for (Eigen::SparseMatrix<double>::InnerIterator it(m_llt,jcol); it; ++it){
            auto row = it.row();
            auto val = it.value();
            assert(row >= jcol);
            if (row == jcol) {
                assert(val != 0.);
                pivot = 1./val;
            }
            else y[jcol] -= val*y[row];
        }
        y[jcol] *= pivot;
    }
    return y;
}
