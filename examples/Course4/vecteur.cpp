#include <cmath>
#include <cassert>
#include "vecteur.hpp"

algebra::Vecteur::Vecteur( std::int32_t t_dimension )
    :    m_coefficients(t_dimension)
{}
// ------------------------------------------------------------------------------------------------------------------------
double
algebra::Vecteur::operator [] ( std::int32_t t_index ) const
{
    assert(t_index >= 0);
    assert(t_index < std::int32_t(m_coefficients.size()));
    return m_coefficients[t_index];
}
// ------------------------------------------------------------------------------------------------------------------------
double&
algebra::Vecteur::operator [] ( std::int32_t t_index )
{
    assert(t_index >= 0);
    assert(t_index < std::int32_t(m_coefficients.size()));
    return m_coefficients[t_index];
}
// ------------------------------------------------------------------------------------------------------------------------
auto
algebra::Vecteur::operator + ( Vecteur const & u ) const -> Vecteur
{
    assert(dimension() == u.dimension());
    Vecteur v(dimension());
    for ( std::int32_t i=0; i<dimension(); ++i )
        v[i] = m_coefficients[i] + u[i];
    return v;
}
// ------------------------------------------------------------------------------------------------------------------------
auto
algebra::Vecteur::operator += ( Vecteur const & u ) -> Vecteur&
{
    assert(dimension() == u.dimension());
    for ( std::int32_t i=0; i<dimension(); ++i )
        m_coefficients[i] += u[i];
    return *this;
}
// ------------------------------------------------------------------------------------------------------------------------
auto
algebra::Vecteur::operator -() const -> Vecteur
{
    Vecteur v(dimension());
    for (std::int32_t i=0; i<dimension(); ++i)
        v[i] = -m_coefficients[i];
    return v;
}
// ------------------------------------------------------------------------------------------------------------------------
auto
algebra::Vecteur::operator - ( Vecteur const& u ) const -> Vecteur
{
    assert(dimension() == u.dimension());
    Vecteur v(dimension());
    for ( std::int32_t i=0; i<dimension(); ++i )
        v[i] = m_coefficients[i] - u[i];
    return v;
}
// ------------------------------------------------------------------------------------------------------------------------
auto
algebra::Vecteur::operator -= ( Vecteur const & u ) -> Vecteur&
{
    assert(dimension() == u.dimension());
    for ( std::int32_t i=0; i<dimension(); ++i )
        m_coefficients[i] -= u[i];
    return *this;
}
// ------------------------------------------------------------------------------------------------------------------------
auto
algebra::Vecteur::homothetie( double t_scal ) const -> Vecteur
{
    Vecteur v(dimension());
    for ( std::int32_t i=0; i<dimension(); ++i)
        v[i] = t_scal * m_coefficients[i];
    return v;
}
// ------------------------------------------------------------------------------------------------------------------------
double
algebra::Vecteur::dot( Vecteur const& u ) const
{
    assert(dimension() == u.dimension());
    double s=0.;
    for ( std::int32_t i=0; i<dimension(); ++i)
        s += m_coefficients[i] * u[i];
    return s;
}
// ------------------------------------------------------------------------------------------------------------------------
double
algebra::Vecteur::nrmL2() const
{
    return std::sqrt(this->dot(*this));
}
