#!/bin/bash
#SBATCH --job-name=as0_bj_check
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=00:05:00
#SBATCH --output=results/implementation_validation/as0_vs_bj_%j.out

module purge
module load mpi/latest

export OMP_NUM_THREADS=1
export SGS_SWEEPS=4

cd ~/pcg_project

echo "========================================"
echo "Block Jacobi"
echo "========================================"

mpirun -np 4 ./apps/mpi_pcg 16 bj

echo
echo "========================================"
echo "Additive Schwarz, delta=0"
echo "========================================"

mpirun -np 4 ./apps/mpi_pcg 16 as 0
