#include Make_linux.inc
include Make_msys2.inc
#include Make_osx.inc

CXXVERS = -std=c++17 -fPIC
CXXFLAGS17 = -std=c++17 -fPIC
ifdef DEBUG
CFLAGS      = -g -O0 -Wall -fbounds-check -pedantic -fPIC
CXXFLAGS    = ${CFLAGS} -std=c++14 -D_GLIBCXX_DEBUG -fopenmp
CXXFLAGS17  = ${CFLAGS} -std=c++17 -D_GLIBCXX_DEBUG -fopenmp
else
CFLAGS      = -O3 -march=native -fopenmp -fPIC
CXXFLAGS    = ${CFLAGS} -std=c++14
CXXFLAGS17  = ${CFLAGS} -std=c++17
endif

ALL= find_nb_threads.exe parallel_section.exe static_parallel_for.exe dynamic_parallel_for.exe \
     reduction.exe critical.exe

default:	help

all: $(ALL)

clean:
	@rm -fr *.o *.exe *~ *.tga

find_nb_threads.exe : find_nb_threads.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBMPI)

parallel_section.exe : parallel_section.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBMPI)

static_parallel_for.exe: static_parallel_for.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBMPI)

dynamic_parallel_for.exe: dynamic_parallel_for.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBMPI)

reduction.exe: reduction.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBMPI)

critical.exe: critical.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBMPI)

help:
	@echo "Available targets : "
	@echo "    all            : compile all executables"
	@echo "Add DEBUG=yes to compile in debug"
	@echo "Configuration :"
	@echo "    CXX      :    $(CXX)"
	@echo "    CXXFLAGS :    $(CXXFLAGS)"
