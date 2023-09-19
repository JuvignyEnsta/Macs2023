#include <stdexcept>
#include "sparse_matrix.hpp"

algebra::SparseMatrix::SparseMatrix( std::uint32_t t_dimension )
    :   m_begRows(std::size_t(t_dimension + 1), 0u)
{}
// ------------------------------------------------------------------------------------------------------------------------
void
algebra::SparseMatrix::reserveCoefs()
{
    if (m_begRows.back()==0)
        throw std::logic_error("Remplissez le tableau begRows avant d'appeler cette methode.");
    std::vector<std::uint32_t>(m_begRows.back()).swap(m_indCols);
    std::vector<std::uint32_t>(m_begRows.back()).swap(m_coefs);
}
// ------------------------------------------------------------------------------------------------------------------------
double
algebra::SparseMatrix::operator() ( std::uint32_t i, std::uint32_t j ) const
{
    assert(i<dimension());
    for ( std::uint32_t index=m_begRows[i]; index < m_begRows[i+1]; ++index ) {
        if (m_indCols[index] == j) return m_coefs[index];
    }
    return 0.;
}
// ------------------------------------------------------------------------------------------------------------------------
auto
algebra::SparseMatrix::operator * ( Vecteur const& u ) const -> Vecteur
{
    assert(u.dimension() == dimension());
    Vecteur v(dimension());
    for (std::uint32_t i=0. i<dimension(); ++i) {
        v[i] = 0.;
        for (std::uint32_t index=m_begRows[i]; index<m_begRows[i+1]; ++index) {
            auto j = m_indCols[index];
            v[i] += m_coefs[index]*u[j];
        }
    }
    return v;
}
