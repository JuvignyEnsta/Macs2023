#ifndef _SPARSE_MATRIX_HPP_
#define _SPARSE_MATRIX_HPP_
#include "vecteur.hpp"

namespace algebra
{
    class SparseMatrix
    {
    public:

        SparseMatrix( std::uint32_t t_dimension );
        SparseMatrix( SparseMatrix const& ) = delete;
        SparseMatrix( SparseMatrix     && ) = default;
        ~SparseMatrix()                     = default;

        std::uint32_t dimension() const
        { return std::uint32_t(m_begRows.size()-1); }
        std::vector<std::uint32_t>& begRows() { return m_begRows; }
        std::vector<std::uint32_t> const& begRows() const
        { return m_begRows; }
        void reserveCoefs();
        std::vector<std::uint32_t>& indCols() { return m_indCols; }
        std::vector<std::uint32_t> const& indCols() const { return m_indCols; }
        std::vector<double>& coefs() { return m_coefs; }
        std::vector<double> const& coefs() const { return m_coefs; }
        double operator () ( std::uint32_t i, std::uint32_t j) const;

        SparseMatrix& operator = ( SparseMatrix const& ) =  default;
        SparseMatrix& operator = ( SparseMatrix&& ) = default;
        
        Vecteur operator * ( Vecteur const& ) const;
        std::uint32_t nnz() const { return m_begRows.back(); }
    private:
        std::vector<std::uint32_t> m_begRows;
        std::vector<std::uint32_t> m_indCols{};
        std::vector<double>        m_coefs  {};
    };
}

#endif
