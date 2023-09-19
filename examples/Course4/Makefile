include Make_linux.inc
#include Make_msys2.inc
#include Make_osx.inc

CXXVERS = -std=c++17 -fPIC
CXXFLAGS17 = -std=c++17 -fPIC
ifdef DEBUG
CFLAGS      = -g -O0 -Wall -fbounds-check   -pedantic -fPIC # $(SANITIZER)
CXXFLAGS    = ${CFLAGS} -std=c++17 -D_GLIBCXX_DEBUG -DDEBUG $(INCPATH)
else
CFLAGS      = -O3 -march=native -fPIC
CXXFLAGS    = ${CFLAGS} -std=c++17 $(INCPATH)
endif

ALL= sequential_laplacian.exe ghostcells_laplacian.exe \
	sequential_inpainting.exe

default:	help

all: $(ALL)

clean:
	@rm -fr *.o *.exe *~ *.tga

sequential_laplacian.exe : sequential_laplacian.cpp lodepng.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

ghostcells_laplacian.exe : ghostcells_laplacian.cpp lodepng.cpp
	$(MPICXX) $(CXXFLAGS) -o $@ $^ $(LIBMPI)

sequential_inpainting.exe : sequential_inpainting.cpp laplacian.cpp cholesky_incomplet.cpp tridiag_prec.cpp lodepng.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

help:
	@echo "Available targets : "
	@echo "    all            : compile all executables"
	@echo "Add DEBUG=yes to compile in debug"
	@echo "Configuration :"
	@echo "    CXX      :    $(CXX)"
	@echo "    CXXFLAGS :    $(CXXFLAGS)"
