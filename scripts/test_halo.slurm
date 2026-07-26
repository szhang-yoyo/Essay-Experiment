#!/bin/bash

#SBATCH --job-name=test_halo
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=00:05:00
#SBATCH --mem=1G
#SBATCH --output=results/phase3/test_halo_%j.out
#SBATCH --error=results/phase3/test_halo_%j.err

module purge
module load mpi/latest

cd "$SLURM_SUBMIT_DIR"

mpirun -np "$SLURM_NTASKS" ./apps/test_halo_mpi
