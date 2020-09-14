
ifndef COMPILER
define compiler_help
Set COMPILER to change flags (defaulting to GNU).
Available compilers are:
  GNU CLANG INTEL CRAY

endef
$(info $(compiler_help))
COMPILER=GNU
endif

COMPILER_GNU = g++ -g
COMPILER_CLANG = clang++
COMPILER_INTEL = icpc
COMPILER_CRAY = CC
CXX = $(COMPILER_$(COMPILER))

FLAGS_ = -O3 -std=c++11 
FLAGS_GNU = -O3 -std=c++11
FLAGS_CLANG = -O3 -std=c++11
FLAGS_INTEL = -O3 -std=c++11
FLAGS_CRAY = -O3 -hstd=c++11
CXXFLAGS=$(FLAGS_$(COMPILER)) $(shell aocl compile-config )

PLATFORM = $(shell uname -s)
ifeq ($(PLATFORM), Darwin)
  LIBS = -framework OpenCL
else
  LIBS =  $(shell aocl link-config) -lOpenCL
endif

ocl-stream: main.cpp OCLStream.cpp
	$(CXX) $(CXXFLAGS) -DOCL $^ $(EXTRA_FLAGS) $(LIBS) -o $@




.PHONY:		 gen_aocx_hw

gen_aocx_hw:
	aoc -v babelstream.cl 
	

.PHONY:		 gen_aocx_emu

gen_aocx_emu:
	aoc -v -march=emulator babelstream.cl




.PHONY: clean
clean:
	rm -f ocl-stream

