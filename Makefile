CXX = g++
CXXFLAGS = -O2 -std=c++17 -Wall -Wextra

TARGET_1D = apps/serial_1d
TARGET_2D = apps/serial_2d

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
	solvers/cg_solver.cpp \
        preconditioners/block_jacobi.cpp \
        solvers/pcg_solver.cpp 

all: $(TARGET_1D) $(TARGET_2D)

$(TARGET_1D): $(SRC_1D)
	$(CXX) $(CXXFLAGS) $(SRC_1D) -o $(TARGET_1D)

$(TARGET_2D): $(SRC_2D)
	$(CXX) $(CXXFLAGS) $(SRC_2D) -o $(TARGET_2D)

clean:
	rm -f $(TARGET_1D)
	rm -f $(TARGET_2D)
	rm -f results/phase1/*.csv
	rm -f results/phase2/*.csv
