
ifndef COMPILER
define compiler_help
Set COMPILER to change flags (defaulting to GNU).
Available compilers are:
  GNU CLANG INTEL CRAY

endef
$(info $(compiler_help))
COMPILER=GNU
endif

COMPILER_GNU = g++ 
COMPILER_CLANG = clang++
COMPILER_INTEL = icpc
COMPILER_CRAY = CC
CXX = $(COMPILER_$(COMPILER))
CC=gcc

FLAGS_ = -O3 -std=c++11 -D WITH_OPENCL -I ~/metaCL_workspace/MetaMorph/include -I ~/metaCL_workspace/MetaMorph/metamorph-backends/opencl-backend
FLAGS_GNU = -O3 -std=c++11 -D WITH_OPENCL -I ../MetaMorph/include -I ../MetaMorph/metamorph-backends/opencl-backend -I/opt/intel/inteloneapi/compiler/2021.1-beta03/linux/lib/oclfpga/host/include
FLAGS_CLANG = -O3 -std=c++11 -D WITH_OPENCL -I ~/metaCL_workspace/MetaMorph/include -I ~/metaCL_workspace/MetaMorph/metamorph-backends/opencl-backend 
FLAGS_INTEL = -O3 -std=c++11 -D WITH_OPENCL -I ~/metaCL_workspace/MetaMorph/include -I ~/metaCL_workspace/MetaMorph/metamorph-backends/opencl-backend
FLAGS_CRAY = -O3 -hstd=c++11 -D WITH_OPENCL -I ~/metaCL_workspace/MetaMorph/include -I ~/metaCL_workspace/MetaMorph/metamorph-backends/opencl-backend
CXXFLAGS=$(FLAGS_$(COMPILER))


CFLAGS = -std=c99  -D WITH_OPENCL  $(shell aocl compile-config ) -I ../MetaMorph/include -I ../MetaMorph/metamorph-backends/opencl-backend 
CPPFLAGS= -D WITH_OPENCL  $(shell aocl compile-config ) -I ../MetaMorph/include -I ../MetaMorph/metamorph-backends/opencl-backend


PLATFORM = $(shell uname -s)
ifeq ($(PLATFORM), Darwin)
  LIBS = -framework OpenCL -g
else
  LIBS = $(shell aocl link-config) 
endif



#ocl-stream: main.cpp OCLStream.cpp
#	$(CXX) $(CXXFLAGS) -DOCL $^ $(EXTRA_FLAGS) $(LIBS) -o $@

.SUFFIXES:  .o

OBJS = OCLStream.o main.o $(OCL_OBJS)

OCL_OBJS =  metacl_module.o

TARGET = metababel

$(TARGET) :	$(OBJS)
		g++  -o $@ $(OBJS) -L ../MetaMorph/lib -lmm_opencl_backend -lmetamorph  $(LIBS)

#
# CPP rule
#
%.o:	%.cpp 
	$(CXX) $(CXXFLAGS) -DOCL -c $< $(LIBS) 


#
# C rule
#
%.o:	%.c 
	$(CC) $(CFLAGS) -c $< $(LIBS) $(CFLAGS2)

.PHONY: clean
clean:
	rm -f *.o *.mod *.bc metababel 


run: 
	 LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../MetaMorph/lib  ./metababel --device $(dev)
#run:        
#         env LD_LIBRARY_PATH=/usr/lib/gcc/x86_64-linux-gnu/7:$LD_LIBRARY_PATH ../MetaMorph/lib ./metababel --device$(dev)
