#include "lodepng.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>

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
    const char* filename = nargs > 1 ? argv[1] : "lena_gray.png";

    unsigned width, height;
    std::vector<unsigned char> image;
    unsigned error = lodepng::decode(image, width, height, filename);
    if (error) {
        std::cerr << "Error loading png image : " << error
                  << " : " << lodepng_error_text(error) << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "Image : " << width << "x" << height << " pixels and " << image.size()/(width*height) << " octets par pixel" << std::endl;

    auto valueImage = convertToIntensity(image);
    auto draw = laplacianFilter(width, height, valueImage);
    saveLaplacianData( width, height, draw, "laplacian.dat");
    auto result = convertToRGBA( draw );
    std::string resultFileName = std::string("lapl_") + std::string(filename);
    error = lodepng::encode(resultFileName, result, width, height);
    if (error) {
        std::cerr << "Error saving png image : " << error
                  << " : " << lodepng_error_text(error) << std::endl;
        exit(EXIT_FAILURE);
    }
    
    return EXIT_SUCCESS;
}
