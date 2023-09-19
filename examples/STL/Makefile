#include Make_linux.inc
include Make_msys2.inc
#include Make_osx.inc

ifdef DEBUG
CFLAGS      = -g -O0 -Wall -fbounds-check -pedantic -fPIC
CXXFLAGS    = ${CFLAGS} -std=c++17 -D_GLIBCXX_DEBUG -fopenmp
else
CFLAGS      = -O3 -march=native -fopenmp -fPIC
CXXFLAGS    = ${CFLAGS} -std=c++17
endif

ALL= parallel_sort.exe parallel_for_each.exe async_deferred_example.exe shared_future_example.exe

default:	help

all: $(ALL)

clean:
	@rm -fr *.o *.exe *~ *.tga

parallel_sort.exe : parallel_sort.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBTBB)

parallel_for_each.exe : parallel_for_each.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBTBB)

async_deferred_example.exe : async_deferred_example.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBTBB)

shared_future_example.exe : shared_future_example.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBTBB)

help:
	@echo "Available targets : "
	@echo "    all            : compile all executables"
	@echo "Add DEBUG=yes to compile in debug"
	@echo "Configuration :"
	@echo "    CXX      :    $(CXX)"
	@echo "    CXXFLAGS :    $(CXXFLAGS)"
