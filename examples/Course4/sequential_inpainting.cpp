#include "lodepng.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include <cassert>
#include <chrono>
#include "vecteur.hpp"
#include "laplacian.hpp"
#include "conjugate_gradient.hpp"
#include "cholesky_incomplet.hpp"
#include "tridiag_prec.hpp"

using Vecteur=algebra::Vecteur;

// ------------------------------------------------------------------------
std::vector<unsigned char>
convertToRGBA( Vecteur const& intensity )
{
    std::vector<unsigned char> image(4*intensity.size());
    double minValue = intensity[0], maxValue = intensity[0];
    for(auto const& value : intensity) {
        minValue = std::min(minValue, value);
        maxValue = std::max(maxValue, value);
    }
    double scal = 255./(maxValue - minValue);
    for ( std::size_t i=0; i<intensity.size(); ++i) {
        image[4*i+0] = (unsigned char)(scal*(intensity[i]-minValue));
        image[4*i+1] = (unsigned char)(scal*(intensity[i]-minValue));
        image[4*i+2] = (unsigned char)(scal*(intensity[i]-minValue));
        image[4*i+3] = 255;
    }
    return image;
}
// ------------------------------------------------------------------------
void loadLaplacianData( unsigned& width, unsigned& height, std::vector<double>& coefs,
                        std::string const& filename )
{
    std::ifstream input(filename, std::ios::binary);
    input.read((char*)&width, sizeof(unsigned));
    input.read((char*)&height, sizeof(unsigned));
    coefs.resize(width*height);
    input.read((char*)coefs.data(), width*height*sizeof(double));
    input.close();
}
// ========================================================================================================================
Vecteur
solveDirectLaplacian( int t_width, int t_height, Vecteur const& filteredImg )
{
    std::cout << "Assembling" << std::endl;
    algebra::Laplacian A(t_width, t_height);
    std::cout << "Solving" << std::endl;
    return A.solve(filteredImg);
}
// ------------------------------------------------------------------------------------------------------------------------
Vecteur
solveIterLaplacian( int t_width, int t_height, Vecteur const& filteredImg,
                    std::int64_t maxIter, double eps, bool isVerbose)
{
    std::cout << "Assembling" << std::endl;
    algebra::Laplacian A(t_width, t_height);

    std::cout << "Preconditioning" << std::endl;
    auto deb = std::chrono::high_resolution_clock::now();
    //algebra::CholeskyIncomplet llt(A.matrix());
    //algebra::TriDiagPrec T(t_width,t_height);
    auto fin = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duree = fin -deb;
    std::cout << "Temps calcul préconditionneur : " << duree.count() << "s" << std::endl;  
    Vecteur x0(A.dimension());
    for (auto& val : x0) val = 0.;
    std::cout << "Solving" << std::endl;
    deb = std::chrono::high_resolution_clock::now();
    auto res = gradientConjugue(A, filteredImg, x0, maxIter, eps, isVerbose);
    //auto res = gradientConjugue(A, T, filteredImg, x0, maxIter, eps, isVerbose);
    //auto res = gradientConjugue(A, llt, filteredImg, x0, maxIter, eps, isVerbose);
    fin = std::chrono::high_resolution_clock::now();
    duree = fin - deb;
    std::cout << "Temps resolution GC : " << duree.count() << "s" << std::endl;
    return res;
}
// =======================================================================
int main( int nargs, char *argv[] )
{
    constexpr int WIDTH=0, HEIGHT=1;

    bool useIterativeSolver = false;
    bool isVerbose          = false;
    int maxiter = 100;
    double eps = 1.E-7;
    const char* filename = nargs > 1 ? argv[1] : "laplacian_lena.dat";

    if (nargs > 2) useIterativeSolver = std::stoi(argv[2])>0;
    if (nargs > 3) maxiter = std::stoi(argv[3]);
    if (nargs > 4) eps     = std::stod(argv[4]);
    if (nargs > 5) isVerbose = (std::string(argv[5]) == std::string("verbose"));

    unsigned dimensions[2];
    std::vector<double> pixels;
    loadLaplacianData(dimensions[WIDTH],dimensions[HEIGHT],pixels,filename);
    Vecteur global_values =  -Eigen::VectorXd::Map(pixels.data(),
                                                   pixels.size());
    Vecteur image;
    if (useIterativeSolver)
        image = solveIterLaplacian( dimensions[WIDTH], dimensions[HEIGHT],
                                    global_values, maxiter, eps, isVerbose );
    else
        image = solveDirectLaplacian(dimensions[WIDTH], dimensions[HEIGHT],
                                     global_values );

    std::cout << "Residu de b : " << global_values.norm() << std::endl;
    algebra::Laplacian L(dimensions[WIDTH], dimensions[HEIGHT]);
    auto r = L*image - global_values;
    std::cout << "Residu final : " << r.norm()/global_values.norm() << std::endl;
    auto result = convertToRGBA( image );
    std::string resultFileName = std::string("inpainting_") + std::string(filename) + std::string(".png");
    auto error = lodepng::encode(resultFileName, result, dimensions[WIDTH],
                                 dimensions[HEIGHT]);
    if (error) {
        std::cerr << "Error saving png image : " << error
                  << " : " << lodepng_error_text(error) << std::endl;
        exit(EXIT_FAILURE);
    }
    
    return EXIT_SUCCESS;
}
