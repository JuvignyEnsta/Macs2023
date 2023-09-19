// Test Logger with CPU chronometer
# include <cmath>
# include <vector>
# include <random>
# include <cassert>
# include <tuple>
# include <iostream>
# include <functional>
# include <algorithm>
# include <chrono>
# include <fstream>
# include <mpi.h>

std::ofstream out;

struct Complex
{
    float re, im;    
    Complex() : re(0.), im(0.)
    {}
    Complex(double r, double i) : re(r), im(i)
    {}
    double sqNorm() { return re*re + im*im; }
    Complex operator + ( const Complex& z )
    {
        return Complex(re + z.re, im + z.im );
    }
    Complex operator * ( const Complex& z )
    {
        return Complex(re*z.re-im*z.im, re*z.im+im*z.re);
    }
};

bool test_mandelbrot_divergent( int maxIter, const Complex& c)
{
    Complex z{0.,0.};
    // On vérifie dans un premier temps si le complexe
    // n'appartient pas à une zone de convergence connue :
    // Appartenance aux disques  C0{(0,0),1/4} et C1{(-1,0),1/4}
    if ( c.re*c.re+c.im*c.im < 0.0625 )
        return false;
    if ( (c.re+1)*(c.re+1)+c.im*c.im < 0.0625 )
        return false;
    // Appartenance à la cardioïde {(1/4,0),1/2(1-cos(theta))}    
    if ((c.re > -0.75) && (c.re < 0.5) ) {
        Complex ct{c.re-0.25f,c.im};
        double ctnrm2 = std::sqrt(ct.sqNorm());
        if (ctnrm2 < 0.5f*(1.f-ct.re/ctnrm2)) return false;
    }
    int niter = 0;
    while ((z.sqNorm() < 4.) && (niter < maxIter))
    {
        z = z*z + c;
        ++niter;
    }
    return (niter < maxIter);
}

void comp_mandelbrot_orbit( int maxIter, const Complex& c, unsigned width, unsigned height, 
                            std::vector<unsigned>& image )
{
     int i = 0;
     Complex z{0.,0.};
     Complex z2{0.,0.};
     while ( (z2.re+z2.im < float(4)) && (i<maxIter) )
     {
        z.im = float(2)*z.re*z.im + c.im;
        z.re = z2.re - z2.im + c.re;
        z2.re = z.re*z.re;
        z2.im = z.im*z.im;
        unsigned ip = unsigned((float(2)+z.im)*width/float(4));
        unsigned jp = unsigned((float(2)+z.re)*height/float(4));
        if ( (ip<width) && (jp<height) ) {
           int ind = ip+jp*width;
           image[ind] += 1;
        }
        i ++;
     }
}
// Bhuddabrot to test the chronometer
// ---------------------------------------------------------------------
unsigned
bhuddabrot_task ( unsigned long nbSamples, unsigned long maxIter, unsigned width, unsigned height,
		  std::vector<unsigned>& image)
{
    std::random_device rd;
    std::mt19937 generator1(rd());
    std::mt19937 generator2(rd());
    
    std::uniform_real_distribution<float> normDistrib(0.,2.);
    auto genNorm = std::bind( normDistrib, generator1 );
    std::uniform_real_distribution<float> angleDistrib(0.,
                                                     6.283185307179586);
    auto genAngle = std::bind( angleDistrib, generator2 );

    //std::vector<unsigned> image(width*height, 0U);
    for ( unsigned long iSample = 0; iSample < nbSamples; ) {
        float r = genNorm();
        float angle = genAngle();
        Complex c{ r * std::cos(angle), r * std::sin(angle) };
        Complex c0{c.re,c.im};
        if ( test_mandelbrot_divergent( maxIter, c0 ) == true ) {
            // Calcul de l'orbite si la suite diverge :
            comp_mandelbrot_orbit( maxIter, c0, width, height, image );
            iSample ++;
        }
    }
    return 1;
}
// ---------------------------------------------------------------------
std::vector<unsigned>
bhudda( unsigned long nbSamples, unsigned long maxIter, unsigned width, unsigned height,
	MPI_Comm& comm)
{
  int nbp, rank;
  constexpr unsigned long packSize = 64;
  std::vector<unsigned> image(width*height, 0U), image_loc(width*height, 0U);
  unsigned long nbPacks = (nbSamples+packSize-1)/packSize;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &nbp );
  if (0==rank) {
    unsigned long iPack = 0;
    for (int i=1; i<nbp; ++i, ++iPack)
      {
	MPI_Send(&iPack, 1, MPI_INT, i, 101, comm);
      }
    MPI_Status status;
    while (iPack < nbPacks)
      {
	int done;
	MPI_Recv(&done, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, comm, &status);
        int slaveRk = status.MPI_SOURCE;
        MPI_Send(&iPack, 1, MPI_INT, slaveRk, 101, comm);
        ++iPack;
      }
      iPack = -1;
      for (int i=1; i<nbp; ++i)
        {
	  int done;
	  MPI_Recv(&done, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, comm, &status);
	  int slaveRk = status.MPI_SOURCE;
	  MPI_Send(&iPack, 1, MPI_INT, slaveRk, 101, comm);	  
	}
      MPI_Reduce(image_loc.data(), image.data(), width*height, MPI_UNSIGNED, MPI_SUM, 0, comm);
  }
  else
    {
        MPI_Status status;
        int iPack, res=1;
        do 
        {
            MPI_Recv(&iPack, 1, MPI_INT, 0, 101, comm, &status);
            if (iPack != -1)
            {
	      bhuddabrot_task ( packSize, maxIter, width, height, image_loc);
              MPI_Send(&res, 1, MPI_INT, 0, 101, comm);
            }
        } while (iPack != -1);
      MPI_Reduce(image_loc.data(), image.data(), width*height, MPI_UNSIGNED, MPI_SUM, 0, comm);
    }
  return image;
}
// ---------------------------------------------------------------------
void save_image( const std::string &filename, unsigned width, unsigned height, const std::vector<unsigned char> &img )
{
    std::ofstream ofs( filename.c_str(), std::ios::out | std::ios::binary );
    ofs << "P6\n"
        << width << " " << height << "\n255\n";
    for ( unsigned i = 0; i < width * height; ++i ) {
        ofs << img[ 4 * i + 0 ] << img[ 4 * i + 1 ] << img[ 4 * i + 2 ];
    }
    ofs.close();
}


int main(int nargs, char* argv[])
{
    MPI_Comm global;
    int rank, nbp;
    MPI_Init(&nargs, &argv);
    MPI_Comm_dup(MPI_COMM_WORLD, &global);
    MPI_Comm_size(global, &nbp);
    MPI_Comm_rank(global, &rank);

    char bufferFileName[1024];
    sprintf(bufferFileName, "output%03d.txt", rank);
    out = std::ofstream(bufferFileName);

    unsigned width = 768U, height = 1024U;
    out << "Starting program\n";
    std::chrono::time_point<std::chrono::system_clock> start, end;
    std::chrono::duration<double> elapsed_seconds;
    
    //const unsigned long l1 = 20000, l2 = 100000, l3 = 1000000;
    const unsigned long l1 = 2000, l2 = 10000, l3 = 10000;
    out << "Bhudda 1\n";
    start = std::chrono::system_clock::now();
    auto buddha1 = bhudda( 15000000, l1, width, height, global);
    end = std::chrono::system_clock::now();
    elapsed_seconds = end-start;
    out << "Temps calcul Bhudda 1 : " << elapsed_seconds.count() 
        << std::endl;

    out << "Bhudda 2\n";
    start = std::chrono::system_clock::now();
    auto buddha2 = bhudda( 5000000, l2, width, height, global );
    end = std::chrono::system_clock::now();
    elapsed_seconds = end-start;
    out << "Temps calcul Bhudda 2 : " << elapsed_seconds.count() 
        << std::endl;
    out << "Bhudda 3\n";
    start = std::chrono::system_clock::now();
    auto buddha3 = bhudda( 300000, l3, width, height, global );
    end = std::chrono::system_clock::now();
    elapsed_seconds = end-start;
    out << "Temps calcul Bhudda 3 : " << elapsed_seconds.count() 
        << std::endl;
    if (rank == 0)
    {
    out << "Preparing the image\n";
    std::vector<unsigned char> image(4*width*height);
    start = std::chrono::system_clock::now();
    float b1, b2, b3;
    b1 = 0; b2 = 0; b3 = 0;
    for ( unsigned i = 0; i < height; ++i ) {
        for (unsigned j = 0; j < width; ++j ) {
            unsigned long ind = i*width+j;
            b1 += buddha1[ind];
            b2 += buddha2[ind];
            b3 += buddha3[ind];
        }
    }
    unsigned long stride = width*height;
    float scal1 = 16.*stride/b1;
    float scal2 = 16.*stride/b2;
    float scal3 = 16.*stride/b3;
    for ( unsigned i = 0; i < height; ++i ) {
        for (unsigned j = 0; j < width; ++j ) {
            unsigned long ind = i*width+j;
            image[ 4*ind   ] = std::min(unsigned(buddha1[ind]*scal1),255U);
            image[ 4*ind+1 ] = std::min(unsigned(buddha2[ind]*scal2),255U);
            image[ 4*ind+2 ] = std::min(unsigned(buddha3[ind]*scal3),255U);
            image[ 4*ind+3 ] = 255;
        }
    }    
    save_image("bhuddabrot.ppm", width, height, image);
    end = std::chrono::system_clock::now();
    elapsed_seconds = end-start;
    out << "Temps Sauvegarde image : " << elapsed_seconds.count() 
        << std::endl;
    }
    MPI_Finalize();
    return EXIT_SUCCESS;
}
