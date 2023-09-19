#include "lodepng.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include <cassert>
#include <mpi.h>

static std::ofstream out; 

std::vector<double>
convertToIntensity( std::vector<unsigned char> const& image )
{
    std::vector<double> intensity(image.size()/4);
    for ( std::size_t i=0; i<intensity.size(); ++i ) {
        intensity[i] = std::max({image[4*i],image[4*i+1],image[4*i+2]})/255.;
    }
    return intensity;
}
// ------------------------------------------------------------------------
std::vector<unsigned char>
convertToRGBA( std::vector<double> const& intensity )
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
void
saveLaplacianData( unsigned width, unsigned height, std::vector<double> const& values, std::string const& filename)
{
    std::ofstream output(filename, std::ios::binary);
    output.write((const char*)&width, sizeof(unsigned));
    output.write((const char*)&height, sizeof(unsigned));
    output.write((const char*)values.data(), values.size()*sizeof(double));
    output.close();
}
// ========================================================================
/**
 * @brief Applique un filtre laplacien en considérant que les pixels aux bords sont noirs.
 */
std::vector<double>
laplacianFilter( unsigned width, unsigned height,
                 std::vector<double> const& image )
{
    std::vector<double> lapl(image.size());
    // Laplacien sur la première ligne :
    lapl[0] = -4*image[0] + image[1] + image[width];
    for ( unsigned iWidth=1; iWidth < width-1; ++iWidth ) {
        lapl[iWidth] = -4*image[iWidth] + image[iWidth-1] + image[iWidth+1] + image[iWidth+width];
    }
    lapl[width-1] = -4*image[width-1] + image[width-2] + image[2*width-1];
    // Laplacien sur les lignes 1 à height-2
    for (unsigned jHeight=1; jHeight < height-1; ++jHeight) {
        unsigned index = jHeight*width;
        lapl[index] = -4*image[index] + image[index+1] + image[index-width] + image[index+width];
        for (unsigned iWidth=1; iWidth< width-1; ++iWidth) {
            unsigned index = jHeight*width + iWidth;
            lapl[index] = -4*image[index] + image[index-1] + image[index+1] + image[index-width] + image[index+width];
        }// End for iwidth
        index = jHeight*width + width-1;
        lapl[index] = -4*image[index] + image[index-1] + image[index-width] + image[index+width];
    }// End for jHeight
    // Laplacien dernière ligne :
    unsigned index = (height-1)*width;
    lapl[index] = -4*image[index] + image[index+1] + image[index-width];
    for (unsigned iWidth=1; iWidth< width-1; ++iWidth) {
        unsigned index = (height-1)*width + iWidth;
        lapl[index] = -4*image[index] + image[index-1] + image[index+1] + image[index-width];
    }// End for iwidth
    index = (height-1)*width + width-1;
    lapl[index] = -4*image[index] + image[index-1] + image[index-width];
    return lapl;
}
// =======================================================================
int main( int nargs, char *argv[] )
{
    constexpr int WIDTH=0, HEIGHT=1;
    MPI_Comm global, rowComm, colComm;
    int rank, nbp;
    MPI_Init(&nargs, &argv);
    MPI_Comm_dup(MPI_COMM_WORLD, &global);
    MPI_Comm_size(global, &nbp);
    MPI_Comm_rank(global, &rank);

    char bufferFileName[1024];
    sprintf(bufferFileName, "output%03d.txt", rank);
    out = std::ofstream(bufferFileName);

    const char* filename = nargs > 1 ? argv[1] : "lena_gray.png";

    unsigned dimensions[2];
    std::vector<double> local_values, global_values;
    if (rank==0) {
        std::vector<unsigned char> globalImage;
        unsigned error = lodepng::decode(globalImage, dimensions[WIDTH], dimensions[HEIGHT], filename);
        if (error) {
            out << "Error loading png image : " << error
                << " : " << lodepng_error_text(error) << std::endl;
            MPI_Abort(global, EXIT_FAILURE);
        }
        global_values = convertToIntensity(globalImage);
    }
    MPI_Bcast(dimensions, 2, MPI_UNSIGNED, 0, global);
    out << "Global image : " << dimensions[WIDTH] << "x" << dimensions[HEIGHT]
        << std::endl;

    // Si il n'y a pas d'autres arguments pour l'exécutable, on découpe en tranche,
    // Sinon on va découper selon un certain nombre de blocs par direction
    int nbx, nby;
    if (nargs > 3) {
        nbx = std::stoi(argv[2]);
        nby = std::stoi(argv[3]);
        assert(nbx*nby == nbp);
    } else {
        nby = nbp;
        nbx = 1  ;
    }
    out << "Decomposition de l'image en " << nbx << " x " << nby << " blocs" << std::endl;
    // Calcul du bloc (I,J) dont s'occupe le processus courant (J variant le plus vite ) :
    int I = rank/nbx;
    int J = rank%nbx;
    out << "Je m'occupe du bloc (" << I << "," << J << ")" << std::endl; 
    // On crée un communicateur pour chaque groupe de processus ayant un I commun :
    MPI_Comm_split(global, I, J, &rowComm);
    // De même avec un J commun :
    MPI_Comm_split(global, J, I, &colComm);

    // Calcul de la taille de l'image locale (avec ghostcells)
    // On va supposer que le nombre de pixels de l'image divise
    // le nombre de blocs par ligne et colonne (pour la décomposition).
    int dimLocal[2];
    dimLocal[WIDTH]  = dimensions[WIDTH] /nbx + 2;// +2 pour les cellules fantômes
    dimLocal[HEIGHT] = dimensions[HEIGHT]/nby + 2;// +2 pour les cellules fantômes

    // Le processus 0 envoie les morceaux d'images (ghostcells compris) à chaque processus :
    // En terminant par lui à la fin :
    std::vector<double> buffer(dimLocal[WIDTH]*dimLocal[HEIGHT]);
    if (rank == 0) {
        for (int proc=1; proc < nbp; ++proc) {
            buffer.assign(buffer.size(), 0.);
            int iproc = proc/nbx;
            int istart = iproc*(dimLocal[HEIGHT]-2) - (iproc > 0 ? 1 : 0);

            if (nbx == 1) {// Si découpage en tranche
                // Premiere tranche locale, donc pas à envoyer ( et donc istart > 0)
                std::int32_t decalGlobImg = istart * dimensions[WIDTH];
                std::int32_t decalGlobLoc = 1;
#               if defined(DEBUG)
                out << "proc = " << proc << " => iproc : " << iproc << ", istart : " << istart
                    << ", decalGlobImg : " << decalGlobImg << ", decalGlobLoc : " << decalGlobLoc
                    << std::endl << std::flush;
#               endif
                for (std::int32_t iPixel=0; iPixel < dimLocal[HEIGHT]-1; ++iPixel ) {
                    std::copy(global_values.data()+decalGlobImg,
                              global_values.data()+decalGlobImg + dimensions[WIDTH],
                              buffer.data() + decalGlobLoc);
                    decalGlobImg += dimensions[WIDTH];
                    decalGlobLoc += dimLocal  [WIDTH];
                }
                if (iproc < nby-1) {
                    std::copy(global_values.data()+decalGlobImg,
                              global_values.data()+decalGlobImg + dimensions[WIDTH],
                              buffer.data() + decalGlobLoc);                    
                }
                MPI_Ssend(buffer.data(), buffer.size(), MPI_DOUBLE, proc, 404, global);
            }
            else { // Découpage en damier de l'image. Le proc zéro contiendra le bloc (0,0)
                int jproc = proc%nbx;
                int jstart  = jproc*(dimLocal[WIDTH]-2) - (jproc > 0 ? 1 : 0);
                std::int32_t lineWidth = dimLocal[WIDTH] - 2 + (jproc < nbx-1 ? 1 : 0) + (jproc>0 ? 1 : 0);
                std::int32_t decalGlobImg = istart * dimensions[WIDTH] + jstart;
                std::int32_t decalGlobLoc = (jproc == 0 ? 1 : 0) + (iproc==0 ? dimLocal[WIDTH] : 0);
#               if defined(DEBUG)
                out << "proc = " << proc << " => iproc : " << iproc << ", jproc : "
                    << jproc << ", istart : " << istart << "; jstart : "
                    << jstart << ", lineWidth : " << lineWidth << ", decalGlobImg : "
                    << decalGlobImg << ", decalGlobLoc : " << decalGlobLoc
                    << std::endl << std::flush;
#               endif
                // istart = 0
                for (std::int32_t irow= (iproc==0 ? 1 : 0); irow < dimLocal[HEIGHT] - (iproc==nby-1 ? 1 : 0); ++irow) { 
#                   if defined(DEBUG)
                    out << "irow : " << irow << " : decalGlobImg :" << decalGlobImg
                        << ", decalGlobLoc : " << decalGlobLoc << std::endl;
#                   endif
                    std::copy(global_values.data() + decalGlobImg, global_values.data() + decalGlobImg + lineWidth,
                              buffer.data() + decalGlobLoc);
                    decalGlobImg += dimensions[WIDTH];
                    decalGlobLoc += dimLocal[WIDTH];
                }
                // 
                MPI_Ssend(buffer.data(), buffer.size(), MPI_DOUBLE, proc, 404, global);
            }
        }// End for proc
        // On remplit maintenant le buffer du proc 0 avec son propre bout d'image :
        std::int32_t lineWidth = dimLocal[WIDTH] - 2 + ( nbx-1 > 0 ? 1 : 0);
        std::int32_t decalGlobImg = 0;
        std::int32_t decalGlobLoc = 1 + dimLocal[WIDTH];
        buffer.assign(buffer.size(), 0.);
#       if defined(DEBUG)
        out << "LineWidth : " << lineWidth << ", decalGlobImg : "
                    << decalGlobImg << ", decalGlobLoc : " << decalGlobLoc
                    << std::endl << std::flush;
#               endif
        for (std::int32_t irow=1; irow < dimLocal[HEIGHT] - (nby==1 ? 1 : 0); ++irow) { 
#           if defined(DEBUG)
            out << "irow : " << irow << ", decalGlobImg : "
                << decalGlobImg << ", decalGlobLoc : " << decalGlobLoc
                << std::endl << std::flush;
#           endif
            std::copy(global_values.data() + decalGlobImg, global_values.data() + decalGlobImg + lineWidth,
                      buffer.data() + decalGlobLoc);
            decalGlobImg += dimensions[WIDTH];
            decalGlobLoc += dimLocal[WIDTH];
        }
    } else {// Si le proc courant n'est pas le zéro :
        MPI_Status status;
        MPI_Recv(buffer.data(), dimLocal[WIDTH]*dimLocal[HEIGHT], MPI_DOUBLE, 0, 404, global, &status);
    }
#   if defined(DEBUG)
    {
        auto result = convertToRGBA( buffer );
        std::string resultFileName = std::string("debug_") + std::to_string(rank) + std::string("_") + std::string(filename);
        auto error = lodepng::encode(resultFileName, result,
                                     dimLocal[WIDTH], dimLocal[HEIGHT]);
        if (error) {
            std::cerr << "Error saving png image : " << error
                      << " : " << lodepng_error_text(error) << std::endl;
            MPI_Abort(global,EXIT_FAILURE);
        }
    }
#   endif
    // Chaque processus calcul le laplacien de son bout d'image :
    auto draw = laplacianFilter(dimLocal[WIDTH], dimLocal[HEIGHT], buffer);

    // On va tout ramener su le processus 0 en enlevant au fur et à mesure les
    // ghostcells :
    // Première étape : On recopie dans le "buffer" le bout d'image sans les ghostcells :
    for (std::int32_t irow=1; irow<dimLocal[HEIGHT]-1; ++irow) {
        std::int32_t start = irow * dimLocal[WIDTH] + 1;
        std::int32_t width = dimLocal[WIDTH] - 2;
        std::copy(draw.data() + start, draw.data() + start + width,
                  buffer.data() + (irow-1)*width );
    }
    if (nbx > 1) {
        // Si on a un découpage en damier, on effectue les étapes 2 et 3
        // sinon on passe directement à l'étape 4
        
        // Seconde étape : on collecte les bouts d'images d'une même ligne :
        std::vector<double> rowBuffer;
        if (J==0) rowBuffer.resize((dimLocal[HEIGHT]-2)*dimensions[WIDTH]);
        MPI_Gather(buffer.data(), (dimLocal[WIDTH]-2)*(dimLocal[HEIGHT]-2), MPI_DOUBLE,
                   rowBuffer.data(), (dimLocal[WIDTH]-2)*(dimLocal[HEIGHT]-2),
                   MPI_DOUBLE, 0, rowComm);

        // Troisième étape : On réordonne dans buffer les morceaux d'image sur le premier processus de chaque "ligne"
        if (J==0) {
            std::vector<double>((dimLocal[HEIGHT]-2)*dimensions[WIDTH]).swap(buffer);
            std::int32_t szLocalBuffer = (dimLocal[WIDTH]-2)*(dimLocal[HEIGHT]-2);
            for (std::int32_t irow=0; irow < dimLocal[HEIGHT]-2; ++irow)
                for (std::int32_t icol=0; icol<nbx; ++icol) {
                    std::int32_t sourceIndex = irow*(dimLocal[WIDTH]-2) + icol*szLocalBuffer;
                    std::int32_t targetIndex = irow*dimensions[WIDTH] + icol*(dimLocal[WIDTH]-2);
                    std::copy(rowBuffer.data()+sourceIndex, rowBuffer.data()+sourceIndex+(dimLocal[WIDTH]-2), buffer.data()+targetIndex);
                }
        }
    }
    // Quatrième étape : On reassemble les différentes tranches de l'image à l'aide des processus d'indices J=0 sur le proc de rang 0
    if (rank==0) {
        std::vector<double>(dimensions[WIDTH]*dimensions[HEIGHT]).swap(draw);
    }
    if (J==0) 
        MPI_Gather(buffer.data(), dimensions[WIDTH]*(dimLocal[HEIGHT]-2), MPI_DOUBLE,
                   draw.data(), dimensions[WIDTH]*(dimLocal[HEIGHT]-2), MPI_DOUBLE,
                   0, colComm);        
    if (rank==0) {
        saveLaplacianData( dimensions[WIDTH], dimensions[HEIGHT], draw, "laplacian.dat");
        auto result = convertToRGBA( draw );
        std::string resultFileName = std::string("lapl_") + std::string(filename);
        auto error = lodepng::encode(resultFileName, result,
                                     dimensions[WIDTH], dimensions[HEIGHT]);
        if (error) {
            std::cerr << "Error saving png image : " << error
                      << " : " << lodepng_error_text(error) << std::endl;
            MPI_Abort(global,EXIT_FAILURE);
        }
    }
    MPI_Finalize();
    
    return EXIT_SUCCESS;
}
