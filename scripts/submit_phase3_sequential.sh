#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")/.."

# Current strong-scaling p=4 test job.
prev_job=211478

submit_job()
{
    local script="$1"
    local p="$2"

    local nodes=$(( (p + 63) / 64 ))
    local tasks_per_node

    if (( p < 64 )); then
        tasks_per_node="${p}"
    else
        tasks_per_node=64
    fi

    local job_id

    job_id=$(sbatch --parsable \
        --dependency="afterok:${prev_job}" \
        --nodes="${nodes}" \
        --ntasks="${p}" \
        --ntasks-per-node="${tasks_per_node}" \
        "${script}")

    echo "Submitted ${script}, p=${p}, job=${job_id}, after=${prev_job}"
    prev_job="${job_id}"
}

echo "Current strong p=4 job: ${prev_job}"

# Remaining strong-scaling jobs.
for p in 1 2 8 16 32 64 128
do
    submit_job scripts/phase3_strong.slurm "${p}"
done

# Weak-scaling jobs start after all strong-scaling jobs finish.
for p in 1 2 4 8 16 32 64 128
do
    submit_job scripts/phase3_weak.slurm "${p}"
done

echo "Final job ID: ${prev_job}"
echo "All jobs have been submitted sequentially."
