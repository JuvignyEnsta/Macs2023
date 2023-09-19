#include Make_linux.inc
include Make_msys2.inc
#include Make_osx.inc

CXXVERS = -std=c++17 -fPIC
CXXFLAGS17 = -std=c++17 -fPIC
ifdef DEBUG
CFLAGS      = -g -O0 -Wall -fbounds-check   -pedantic -fPIC # -fsanitize=thread
CXXFLAGS    = ${CFLAGS} -std=c++17 -D_GLIBCXX_DEBUG -DDEBUG $(INCPATH)
else
CFLAGS      = -O3 -march=native -fPIC
CXXFLAGS    = ${CFLAGS} -std=c++17 $(INCPATH)
endif

ALL= rankSort.exe oddEvenParallelBlock.exe shearSort.exe hyperquicksort.exe bitonicsort_distributed.exe

default:	help

all: $(ALL)

clean:
	@rm -fr *.o *.exe *~ *.tga

rankSort.exe : rankSort.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(OMP)

oddEvenParallelBlock.exe : oddEvenParallelBlock.cpp
	$(MPICXX) $(CXXFLAGS) -o $@ $^ $(OMP) $(LIBMPI)

shearSort.exe : shearSort.cpp
	$(MPICXX) $(CXXFLAGS) -o $@ $^ $(OMP) $(LIBMPI)

hyperquicksort.exe : hyperquicksort.cpp 
	$(MPICXX) $(CXXFLAGS) -o $@ $^ $(LIBMPI)

bitonicsort_distributed.exe : bitonicsort_distributed.cpp
	$(MPICXX) $(CXXFLAGS) -o $@ $^ $(LIBMPI)

help:
	@echo "Available targets : "
	@echo "    all            : compile all executables"
	@echo "Add DEBUG=yes to compile in debug"
	@echo "Configuration :"
	@echo "    CXX      :    $(CXX)"
	@echo "    CXXFLAGS :    $(CXXFLAGS)"
