#!/bin/bash

set -euo pipefail

mkdir -p \
    results/final10/manifests \
    results/final10/logs \
    results/final10/raw/pcg \
    results/final10/raw/cg

manifest="results/final10/manifests/final10_submission_manifest.csv"

echo "scaling,processes,nodes,job_id,dependency" > "${manifest}"

previous_job=""

submit_case()
{
    local scaling="$1"
    local p="$2"

    local nodes
    local dependency=""

    if [[ "${p}" -eq 128 ]]; then
        nodes=2
    else
        nodes=1
    fi

    local args=(
        --parsable
        --job-name="final10-${scaling}-p${p}"
        --nodes="${nodes}"
        --ntasks="${p}"
        --export="ALL,SCALING=${scaling}"
    )

    if [[ "${p}" -eq 128 ]]; then
        args+=(--ntasks-per-node=64)
    fi

    if [[ -n "${previous_job}" ]]; then
        dependency="afterok:${previous_job}"
        args+=(--dependency="${dependency}")
    fi

    local job_id
    job_id=$(sbatch "${args[@]}" scripts/phase5_final10.slurm)
    job_id="${job_id%%;*}"

    echo "${scaling},${p},${nodes},${job_id},${dependency}" \
        >> "${manifest}"

    echo "Submitted ${scaling} p=${p}, nodes=${nodes}, job=${job_id}"

    previous_job="${job_id}"
}

# Strong scaling
for p in 1 2 4 8 16 32 64 128
do
    submit_case strong "${p}"
done

# Weak scaling
for p in 1 2 4 8 16 32 64 128
do
    submit_case weak "${p}"
done

echo "All jobs submitted in one dependency chain."
echo "Manifest: ${manifest}"
