CXX = g++
MPICXX = mpicxx

CXXFLAGS = -O2 -std=c++17 -Wall -Wextra

TARGET_1D = apps/serial_1d
TARGET_2D = apps/serial_2d
TARGET_MPI = apps/mpi_pcg

SRC_1D = \
	apps/serial_1d.cpp \
	problems/poisson1d.cpp \
	problems/poisson2d.cpp \
	preconditioners/block_jacobi.cpp \
	solvers/cg_solver.cpp \
	solvers/pcg_solver.cpp

SRC_2D = \
	apps/serial_2d.cpp \
	problems/poisson1d.cpp \
	problems/poisson2d.cpp \
	preconditioners/block_jacobi.cpp \
	solvers/cg_solver.cpp \
	solvers/pcg_solver.cpp

SRC_MPI = \
	apps/mpi_pcg.cpp \
	problems/poisson2d_mpi.cpp \
	parallel/partition.cpp \
	parallel/halo.cpp \
parallel/mpi_vector.cpp \
	preconditioners/mpi_bj.cpp \
	preconditioners/additive_schwarz.cpp \
	preconditioners/local_sgs.cpp \
	solvers/mpi_pcg_solver.cpp

all: $(TARGET_1D) $(TARGET_2D) $(TARGET_MPI)

$(TARGET_1D): $(SRC_1D)
	$(CXX) $(CXXFLAGS) $(SRC_1D) -o $(TARGET_1D)

$(TARGET_2D): $(SRC_2D)
	$(CXX) $(CXXFLAGS) $(SRC_2D) -o $(TARGET_2D)

$(TARGET_MPI): $(SRC_MPI)
	$(MPICXX) $(CXXFLAGS) $(SRC_MPI) -o $(TARGET_MPI)

clean:
	rm -f $(TARGET_1D)
	rm -f $(TARGET_2D)
	rm -f $(TARGET_MPI)
	rm -f results/phase1/*.csv
	rm -f results/phase2/*.csv
	rm -f results/phase3/*.csv
