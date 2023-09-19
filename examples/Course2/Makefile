include Make_linux.inc
#include Make_msys2.inc
#include Make_osx.inc

CXXVERS = -std=c++17 -fPIC
CXXFLAGS17 = -std=c++17 -fPIC
ifdef DEBUG
CFLAGS      = -g -O0 -Wall -fbounds-check  -fsanitize=thread -pedantic -fPIC
CXXFLAGS    = ${CFLAGS} -std=c++17 -D_GLIBCXX_DEBUG $(INCPATH)
else
CFLAGS      = -O3 -march=native -fPIC
CXXFLAGS    = ${CFLAGS} -std=c++17 $(INCPATH)
endif

ALL= mpi_vector_addition_exact_scatter.exe omp_prod_diagonal_block_matrix.exe sequential_prod_diagonal_block_matrix.exe  stl_vector_addition.exe \
     mpi_vector_addition_non_uniform_scattering.exe  omp_vector_addition.exe sequential_vector_addition.exe tbb_vector_addition.exe \
	 mpi_naive_prod_diagonal_block_matrix.exe mpi_prod_diagonal_block_matrix.exe sequential_syracuse.exe omp_syracuse.exe mpi_static_syracuse.exe \
	 mpi_slave_master_syracuse.exe integral_computation.exe omp_integral_computation.exe \
    mpi_integral_computation.exe sequential_bhudda_set.exe mpi_bhudda_set.exe

default:	help

all: $(ALL)

clean:
	@rm -fr *.o *.exe *~ *.tga

mpi_vector_addition_exact_scatter.exe : mpi_vector_addition_exact_scatter.cpp
	$(MPICXX) $(CXXFLAGS) -o $@ $^ $(LIBMPI)

omp_prod_diagonal_block_matrix.exe : omp_prod_diagonal_block_matrix.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(OMP) $(BLAS)

sequential_prod_diagonal_block_matrix.exe : sequential_prod_diagonal_block_matrix.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(BLAS) $(OMP)

stl_vector_addition.exe : stl_vector_addition.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(TBB)

mpi_vector_addition_non_uniform_scattering.exe : mpi_vector_addition_non_uniform_scattering.cpp  
	$(MPICXX) $(CXXFLAGS) -o $@ $^ 

omp_vector_addition.exe : omp_vector_addition.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(OMP)

sequential_vector_addition.exe : sequential_vector_addition.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

tbb_vector_addition.exe : tbb_vector_addition.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(TBB)

mpi_naive_prod_diagonal_block_matrix.exe : mpi_naive_prod_diagonal_block_matrix.cpp
	$(MPICXX) $(CXXFLAGS) -o $@ $^ $(BLAS)

mpi_prod_diagonal_block_matrix.exe : mpi_prod_diagonal_block_matrix.cpp
	$(MPICXX) $(CXXFLAGS) -o $@ $^ $(BLAS)

sequential_syracuse.exe : sequential_syracuse.cpp 
	$(CXX) $(CXXFLAGS) -o $@ $^ 

omp_syracuse.exe : omp_syracuse.cpp 
	$(CXX) $(CXXFLAGS) -o $@ $^  $(OMP)

mpi_static_syracuse.exe : mpi_static_syracuse.cpp 
	$(MPICXX) $(CXXFLAGS) -o $@ $^  $(OMP)

mpi_slave_master_syracuse.exe : mpi_slave_master_syracuse.cpp
	$(MPICXX) $(CXXFLAGS) -o $@ $^  $(OMP)

integral_computation.exe : integral_computation.cpp 
	$(CXX) $(CXXFLAGS) -o $@ $^ 

omp_integral_computation.exe : omp_integral_computation.cpp 
	$(CXX) $(CXXFLAGS) -o $@ $^ $(OMP)

mpi_integral_computation.exe : mpi_integral_computation.cpp
	$(MPICXX) $(CXXFLAGS) -o $@ $^  $(OMP)

sequential_bhudda_set.exe : sequential_bhudda_set.cpp 
	$(CXX) $(CXXFLAGS) -o $@ $^ 

mpi_bhudda_set.exe : mpi_bhudda_set.cpp
	$(MPICXX) $(CXXFLAGS) -o $@ $^  $(OMP)

help:
	@echo "Available targets : "
	@echo "    all            : compile all executables"
	@echo "Add DEBUG=yes to compile in debug"
	@echo "Configuration :"
	@echo "    CXX      :    $(CXX)"
	@echo "    CXXFLAGS :    $(CXXFLAGS)"
