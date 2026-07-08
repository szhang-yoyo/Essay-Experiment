CXX = g++
CXXFLAGS = -O2 -std=c++17 -Wall -Wextra

TARGET = apps/serial_1d

SRC = \
	apps/serial_1d.cpp \
	problems/poisson1d.cpp \
	preconditioners/block_jacobi.cpp \
	solvers/cg_solver.cpp \
	solvers/pcg_solver.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)
	rm -f results/phase1/*.csv
