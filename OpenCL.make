
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

METAMORPH_PATH=../../MetaMorph
FLAGS_ = -O3 -std=c++11
FLAGS_GNU = -O3 -std=c++11
FLAGS_CLANG = -O3 -std=c++11
FLAGS_INTEL = -O3 -std=c++11
FLAGS_CRAY = -O3 -hstd=c++11
CXXFLAGS=$(FLAGS_$(COMPILER))
CXXFLAGS := $(CXXFLAGS) -I $(METAMORPH_PATH)/include -I $(METAMORPH_PATH)/metamorph-backends/opencl-backend

LIBS = -lmetamorph -lmetamorph_opencl
PLATFORM = $(shell uname -s)
ifeq ($(PLATFORM), Darwin)
  LIBS := $(LIBS) -framework OpenCL
else ifeq ($(PLATFORM), INTEL_FPGA)
  LIBS := $(LIBS) $(shell aocl link-config)
else
  LIBS := $(LIBS) -lOpenCL
endif

ocl-stream: main.cpp OCLStream.cpp metacl_module.c
	$(CXX) $(CXXFLAGS) -DOCL $^ $(EXTRA_FLAGS) $(LIBS) -o $@

metacl_module.h: metacl_module.c

METACL_PATH := $(METAMORPH_PATH)/metamorph-generators/opencl
metacl_module.c: $(wildcard *.cl)
	$(METACL_PATH)/metaCL $(wildcard *.cl) --unified-output-file="metacl_module" --cuda-grid-block=false -- -cl-std=CL1.2 --include opencl-c.h -I /usr/lib/llvm-6.0/lib/clang/6.0.1/include/ -D TYPE=double -D startScalar=0.4


.PHONY: clean
clean:
	rm -f *.o *.mod *.bc metababel metacl_module.* 


.PHONY:		 gen_aocx_hw

gen_aocx_hw:
	aoc -v -DTYPE=double -DstartScalar=0.4 babelstream.cl 
	

.PHONY:		 gen_aocx_emu

gen_aocx_emu:
	aoc -v -march=emulator -DTYPE=double -DstartScalar=0.4 babelstream.cl

.PHONY: clean
clean:
	rm -f ocl-stream 

.PHONY: run
run: 
	 LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(METAMORPH_PATH)/lib  ./ocl-stream --device $(dev)

