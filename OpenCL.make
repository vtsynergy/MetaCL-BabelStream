
ifndef FPGA
define fpga_help
Set FPGA to change flags (defaulting to NONE).
Available FPGAs are:
  NONE INTEL INTEL_EMU

endef
$(info $(fpga_help))
FPGA=NONE
endif

ifndef METACL 
METACL := $(shell which metaCL)
define metaCL_help
Set METACL to change binary location, attempting to use MetaCL at "$(METACL)"

endef
$(info $(metaCL_help))
endif

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

FLAGS_ = -O3 -std=c++11
FLAGS_GNU = -O3 -std=c++11
FLAGS_CLANG = -O3 -std=c++11
FLAGS_INTEL = -O3 -std=c++11
FLAGS_CRAY = -O3 -hstd=c++11
CXXFLAGS:=$(CXXFLAGS) $(FLAGS_$(COMPILER))

LIBS := $(LIBS)
DEPS=
AOC_OPTS := $(AOC_OPTS) -DTYPE=double -DstartScalar=0.4
ifeq ($(FPGA), INTEL_EMU)
  AOC_OPTS := $(AOC_OPTS) -march=emulator
  FPGA=INTEL
endif
ifeq ($(FPGA), INTEL)
  AOC_OPTS := $(AOC_OPTS) -v
  DEPS := babelstream.aocx
  CXXFLAGS := $(CXXFLAGS) $(shell aocl compile-config)
  LIBS := $(LIBS) $(shell aocl link-config)
endif

LIBS = -lmetamorph -lmetamorph_opencl
PLATFORM = $(shell uname -s)
ifeq ($(PLATFORM), Darwin)
  LIBS := $(LIBS) -framework OpenCL
else
  LIBS := $(LIBS) -lOpenCL
endif

SRC = main.cpp OCLStream.cpp
ifdef METACL
SRC := $(SRC) metacl_module.c
EXTRA_FLAGS := $(EXTRA_FLAGS) -DMETACL
endif
ocl-stream: $(SRC) $(DEPS)
	$(CXX) $(CXXFLAGS) -DOCL $(SRC) $(EXTRA_FLAGS) $(LIBS) -o $@

metacl_module.h: metacl_module.c

metacl_module.c: $(wildcard *.cl)
	$(METACL) $(wildcard *.cl) --unified-output-file="metacl_module" --cuda-grid-block=false -- -cl-std=CL1.2 --include opencl-c.h $(METACL_CFLAGS) -D TYPE=double -D startScalar=0.4

babelstream.aocx: babelstream.cl
	aoc $(AOC_OPTS) babelstream.cl

.PHONY: clean
clean:
	rm -f ocl-stream metacl_module.* 

