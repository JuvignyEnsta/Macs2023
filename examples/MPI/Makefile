#include Make_linux.inc
include Make_msys2.inc
#include Make_osx.inc

CXXFLAGS = -std=c++11 -fPIC
CXXFLAGS17 = -std=c++17 -fPIC
ifdef DEBUG
CFLAGS   = -g -O0 -Wall -fbounds-check -pedantic
CXXFLAGS += ${CFLAGS} -D_GLIBCXX_DEBUG
CXXFLAGS17 += ${CFLAGS} -D_GLIBCXX_DEBUG
else
CFLAGS = -O3 -march=native
CXXFLAGS += ${CFLAGS}
CXXFLAGS17 += ${CFLAGS} 
endif

ALL= ParallelHelloWorld.exe PointTwoPointExchangesBetweenTwoProcesses.exe \
	point2point_covering_communication.exe non_blocking_message.exe broadcasting.exe \
	scattering.exe scattering_non_uniform.exe \
	reduce.exe allreduce.exe scan.exe all2all.exe

default:	help

all: $(ALL)

clean:
	@rm -fr *.o *.exe *~ *.tga

ParallelHelloWorld.exe : ParallelHelloWorld.cpp
	$(MPICXX) $(CFLAGS) -o $@ $^ $(LIBMPI)

PointTwoPointExchangesBetweenTwoProcesses.exe : PointTwoPointExchangesBetweenTwoProcesses.cpp
	$(MPICXX) $(CXXFLAGS) -o $@ $^ $(LIBMPI)

point2point_covering_communication.exe : point2point_covering_communication.cpp
	$(MPICXX) $(CXXFLAGS) -o $@ $^ $(LIBMPI)

non_blocking_message.exe : non_blocking_messages.cpp
	$(MPICXX) $(CXXFLAGS) -o $@ $^ $(LIBMPI)

broadcasting.exe : broadcasting.cpp
	$(MPICXX) $(CXXFLAGS) -o $@ $^ $(LIBMPI)

scattering.exe : scattering.cpp
	$(MPICXX) $(CXXFLAGS) -o $@ $^ $(LIBMPI)

scattering_non_uniform.exe : scattering_non_uniform.cpp
	$(MPICXX) $(CXXFLAGS) -o $@ $^ $(LIBMPI)

reduce.exe : reduce.cpp
	$(MPICXX) $(CXXFLAGS) -o $@ $^ $(LIBMPI)

allreduce.exe : allreduce.cpp
	$(MPICXX) $(CXXFLAGS) -o $@ $^ $(LIBMPI)

scan.exe : scan.cpp
	$(MPICXX) $(CXXFLAGS) -o $@ $^ $(LIBMPI)

all2all.exe : all2all.cpp
	$(MPICXX) $(CXXFLAGS) -o $@ $^ $(LIBMPI)

help:
	@echo "Available targets : "
	@echo "    all            : compile all executables"
	@echo "Add DEBUG=yes to compile in debug"
	@echo "Configuration :"
	@echo "    CXX      :    $(CXX)"
	@echo "    CXXFLAGS :    $(CXXFLAGS)"
