#ifndef _GHOSTCELLS_VECTEUR_HPP_
#define _GHOSTCELLS_VECTEUR_HPP_

namespace parallel::ghostcells::algebra
{
    class Vecteur
    {
    public:
        Vecteur();
        Vecteur( int nBli, int nBlj, int bI, int bJ, int ni, int nj, int layer=1 );
        Vecteur( Vecteur const& ) = default;
        Vecteur( Vecteur     && ) = default;
        ~Vecteur()                = default;

        Vecteur& operator = ( Vecteur const& ) = default;
        Vecteur& operator = ( Vecteur     && ) = default;
    };
}

#endif
