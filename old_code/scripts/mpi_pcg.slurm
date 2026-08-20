#!/bin/bash

#SBATCH --job-name=mpi_pcg
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=results/phase3/mpi_pcg_%j.out
#SBATCH --error=results/phase3/mpi_pcg_%j.err


module purge
module load mpi/latest


cd "$SLURM_SUBMIT_DIR"

# Print
echo "Job ID: $SLURM_JOB_ID"
echo "MPI processes: $SLURM_NTASKS"
echo "Start time: $(date)"


mpirun -np "$SLURM_NTASKS" ./apps/mpi_pcg 1024

echo "End time: $(date)"
