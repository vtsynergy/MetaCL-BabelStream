
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

METAMORPH_PATH=../../MetaMorph

FLAGS_ = -O3 -std=c++11 -I $(METAMORPH_PATH)/include -I $(METAMORPH_PATH)/metamorph-backends/opencl-backend
FLAGS_GNU = -O3 -std=c++11 -I $(METAMORPH_PATH)/include -I $(METAMORPH_PATH)/metamorph-backends/opencl-backend 
FLAGS_CLANG = -O3 -std=c++11 -I $(METAMORPH_PATH)/include -I $(METAMORPH_PATH)/metamorph-backends/opencl-backend 
FLAGS_INTEL = -O3 -std=c++11 -I $(METAMORPH_PATH)/include -I $(METAMORPH_PATH)/metamorph-backends/opencl-backend
FLAGS_CRAY = -O3 -hstd=c++11 -I $(METAMORPH_PATH)/include -I $(METAMORPH_PATH)/metamorph-backends/opencl-backend
CXXFLAGS=$(FLAGS_$(COMPILER))


CFLAGS=-std=c99 
CPPFLAGS=
ifeq ($(PLATFORM), INTEL_FPGA)
 CFLAGS +=  $(shell aocl compile-config )
 CPPFLAGS +=  $(shell aocl compile-config )
endif
CFLAGS += -I $(METAMORPH_PATH)/include -I $(METAMORPH_PATH)/metamorph-backends/opencl-backend 
CPPFLAGS += -I $(METAMORPH_PATH)/include -I $(METAMORPH_PATH)/metamorph-backends/opencl-backend


PLATFORM = $(shell uname -s)
ifeq ($(PLATFORM), Darwin)
  LIBS = -framework OpenCL -g
else ifeq ($(PLATFORM), INTEL_FPGA)
  LIBS = $(shell aocl link-config)
else
  LIBS = -lOpenCL 
endif

#ocl-stream: main.cpp OCLStream.cpp
#	$(CXX) $(CXXFLAGS) -DOCL $^ $(EXTRA_FLAGS) $(LIBS) -o $@

.SUFFIXES:  .o

OBJS = OCLStream.o main.o $(OCL_OBJS)

OCL_OBJS =  metacl_module.o

TARGET = metababel

$(TARGET) : $(OBJS)
		g++  -o $@ $(OBJS) -L $(METAMORPH_PATH)/lib -lmetamorph_opencl -lmetamorph -lm   $(LIBS)

OCLStream.o: OCLStream.cpp metacl_module.h
	$(CXX) $(CXXFLAGS) -D FOO -D OCL -c $< $(LIBS) -o $@


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

metacl_module.h: metacl_module.c

METACL_PATH := $(METAMORPH_PATH)/metamorph-generators/opencl
metacl_module.c: $(wildcard *.cl)
	$(METACL_PATH)/metaCL $(wildcard *.cl) --unified-output-file="metacl_module" --cuda-grid-block=true -- -cl-std=CL1.2 --include opencl-c.h -I /usr/lib/llvm-6.0/lib/clang/6.0.1/include/ -D TYPE=double -D startScalar=0.4


.PHONY: clean
clean:
	rm -f *.o *.mod *.bc metababel metacl_module.* 


run: 
	 LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(METAMORPH_PATH)/lib  ./metababel --device $(dev)
#run:        
#         env LD_LIBRARY_PATH=/usr/lib/gcc/x86_64-linux-gnu/7:$LD_LIBRARY_PATH $(METAMORPH_PATH)/lib ./metababel --device$(dev)
