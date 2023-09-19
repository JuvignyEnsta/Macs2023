#ifndef _CONJUGATE_GRADIENT_HPP_
#define _CONJUGATE_GRADIENT_HPP_
#include <iostream>
#include <cassert>  

namespace algebra
{
    template<typename LhsT>
    Vecteur gradientConjugue( LhsT const& A, Vecteur const& b, Vecteur const& x0, std::int64_t niterMax, double epsilon, bool isVerbose=0 )
    {
        auto n = A.dimension();
        Vecteur g(n), h(n), Ah(n);
        g = A*x0 - b;
        h = -g;
        double g2 = g.dot(g);
        double g0 = g2;
        if (g2 < epsilon*epsilon) {
            std::cout << "Iteration 0 : ||r||^2 = " << g2 << std::endl;
            return x0;
        }
        Vecteur x = x0;
        double reps2 = epsilon*epsilon*g2;
        for ( std::int64_t iter=1; iter <= niterMax; ++iter ){
            Ah = A*h;
            double ro = -g.dot(h)/(h.dot(Ah));
            x += ro*h;
            g += ro*Ah;
            double g2p = g2;
            g2 = g.dot(g);
            if (isVerbose)
                std::cout << "Iteration " << iter
                          << " ||r||^2 : " << g2/g0 << std::endl;
            if (g2 < reps2) {
                std::cout << "Converge en " << iter
                          << " iterations." << std::endl;
                return x;
            }
            double gamma = g2/g2p;
            h = gamma*h - g;
        }        
        std::cout << "Attention, le gradient conjugue n'a pas converge" << std::endl;
        return x;
    }
// ------------------------------------------------------------------------------------------------------------------------
    template<typename LhsT, typename PrecT>
    Vecteur gradientConjugue( LhsT const& A, PrecT const& P, Vecteur const& b, Vecteur const& x0, std::int64_t niterMax, double epsilon, bool isVerbose=0 )
    {
        //assert(A.dimension() == b.dimension());
        auto n = A.dimension();
        double sqrNrmb = b.dot(b);
        Vecteur r(n), z(n), d(n), Ad(n);
        r = b - A*x0;
        z = P.solve(r);
        d = z;
        double sqrErr = r.dot(r);
        if (sqrErr < epsilon*epsilon*sqrNrmb) {
            std::cout << "Solution deja convergee!" << std::endl;
            return x0;
        }
        Vecteur x = x0;
        double reps2 = epsilon*epsilon*sqrNrmb;
        double rdz = r.dot(z);
        for ( std::int64_t iter=1; iter <= niterMax; ++iter ){
            Ad = A*d;
            double alpha = rdz/(d.dot(Ad));
            x += alpha*d;
            r -= alpha*Ad;
            z = P.solve(r);
            double rdzold = rdz;
            sqrErr = r.dot(r);
            if (isVerbose)
                std::cout << "Iteration " << iter
                          << " ||r||^2 : " << sqrErr/sqrNrmb << std::endl;
            if (sqrErr < reps2) {
                std::cout << "Converge en " << iter
                          << " iterations." << std::endl;
                return x;
            }
            rdz = r.dot(z);
            double gamma = rdz/rdzold;
            d = z + gamma*d;
        }        
        std::cout << "Attention, le gradient conjugue n'a pas converge" << std::endl;
        return x;
    }
}

#endif
