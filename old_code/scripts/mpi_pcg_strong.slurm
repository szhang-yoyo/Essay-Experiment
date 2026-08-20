#!/bin/bash

#SBATCH --job-name=pcg_strong
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --exclusive
#SBATCH --output=results/phase3/pcg_strong_%j.out
#SBATCH --error=results/phase3/pcg_strong_%j.err


module purge
module load mpi/latest

cd "$SLURM_SUBMIT_DIR"

echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

# Run each process count sequentially
for p in 1 2 4 8
do
    for r in 1 2 3
    do
        echo "================================"
        echo "Processes: $p"
        echo "Run: $r"

        mpirun -np "$p" ./apps/mpi_pcg 1024
    done
done

echo "End time: $(date)"
