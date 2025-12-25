ifneq ($(shell which nvcc),)
    CXX := nvcc
else
    CXX := gcc
endif

# Common include paths
INCLUDES := -I. -I/opt/petsc/include

# Select extra cxx flags for NVCC
ifeq ($(findstring nvcc,$(CXX)),nvcc)
    ADDITIONAL_CXXFLAGS := -x cu --extended-lambda -arch=sm_80 --forward-unknown-to-host-compiler
endif

# Optimization & language standard (tweak as needed)
CXXFLAGS := -O3 -std=c++17 -fopenmp $(INCLUDES) $(ADDITIONAL_CXXFLAGS)

# PETSc/Kokkos lib directory and rpath target
PETSC_LIBDIR := /opt/petsc/lib
RPATH        := $(PETSC_LIBDIR)

# Select rpath flags based on compiler
ifeq ($(findstring nvcc,$(CXX)),nvcc)
    # nvcc needs -Xlinker for linker flags
    ADDITIONAL_LDFLAGS := -Xlinker -rpath -Xlinker $(RPATH)
else
    ADDITIONAL_LDFLAGS := -Wl,-rpath,$(RPATH)
endif

# Link search path and rpath
LDFLAGS := -L$(PETSC_LIBDIR) -lstdc++ -lgcc_s $(ADDITIONAL_LDFLAGS)
# Library to link
LDLIBS := -lkokkoscore

# Pick all cpp files and create an executable for each
SRCS := $(wildcard */*.cpp)
EXES := $(patsubst %.cpp, %_$(CXX), $(SRCS))

.PHONY: all clean
all: $(EXES)

# Build "prog_nvcc" or "prog_gcc" from "prog.cpp"
%_$(CXX): %.cpp
		$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)

# Clean up generated binaries and temporary files
clean:
		$(RM) $(EXES)
