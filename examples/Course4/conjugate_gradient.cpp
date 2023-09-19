#include "conjugate_gradient.hpp"

auto
algebra::gradientConjugue( SparseMatrix const& A, Vecteur const& b, double epsilon ) -> Vecteur
{
    Vecteur r = b;
    Vecteur p = r;
    double sqrErr = r.dot(r);

    Vecteur x(b.dimension());
    for (auto& val : x ) val = 0.;
    
    while (sqrErr  > epsilon*epsilon) {
        Vecteur Apk = A * p;
        double alpha = sqrErr/(p.dot(Apk));
        x += alpha * p;
        r -= alpha * Apk;
        double sqrErrNew = r.dot(r);
        double beta = sqrErrNew/sqrErr;
        sqrErr = sqrErrNew;
        p = r + beta*p;
    }
    return x;
}
