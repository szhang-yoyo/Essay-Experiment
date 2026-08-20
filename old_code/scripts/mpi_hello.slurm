#!/bin/bash

#SBATCH --job-name=mpi_hello
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=00:05:00
#SBATCH --mem=1G
#SBATCH --output=results/phase3/mpi_hello_%j.out
#SBATCH --error=results/phase3/mpi_hello_%j.err

module purge
module load mpi/latest

cd "$SLURM_SUBMIT_DIR"

echo "Node: $(hostname)"
echo "MPI processes: $SLURM_NTASKS"

mpirun -np "$SLURM_NTASKS" ./apps/mpi_hello
