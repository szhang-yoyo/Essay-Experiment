#!/bin/bash
# SGS-sweep sensitivity experiment used in the dissertation.
#
# Purpose:
#   Examine how the strength of the local SGS solve affects convergence
#   and total time-to-solution for BJ-PCG and overlapping AS-PCG.
#
# Configuration:
#   N = 1024, p = 16, q = 1, 2, 4, 8.
#   Methods: BJ, AS1 (delta=1), and AS2 (delta=2).
#   Repetitions = 10 per tested configuration.
#
# Output:
#   Raw measurements are written to results/sgs_sensitivity/raw/.
#
# Dissertation use:
#   Tests sensitivity to the fixed q=4 baseline used by the main experiments.
#SBATCH --job-name=sgs-sensitivity
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --exclude=callan-n[13-24]
#SBATCH --time=06:00:00
#SBATCH --output=results/sgs_sensitivity/logs/%x_%j.out
#SBATCH --error=results/sgs_sensitivity/logs/%x_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

module purge
module load mpi/latest

export OMP_NUM_THREADS=1

n=1024
p="${SLURM_NTASKS}"
repeats=10

mkdir -p \
    results/sgs_sensitivity/raw \
    results/sgs_sensitivity/logs

raw="results/sgs_sensitivity/raw/sgs_sensitivity_${SLURM_JOB_ID}.csv"

echo "grid_size,processes,method,overlap,sgs_sweeps,repeat,iterations,relative_residual,total_time,spmv_time,halo_time,preconditioner_time,reduction_time,vector_time,job_id" \
    > "${raw}"

run_case()
{
    local q="$1"
    local method="$2"
    local overlap="$3"
    local run="$4"

    export SGS_SWEEPS="${q}"

    echo "q=${q}, method=${method}, overlap=${overlap}, repeat=${run}"

    local output

    if [[ "${method}" == "BJ" ]]; then
        output=$(mpirun -np "${p}" ./apps/mpi_pcg "${n}" bj)
    else
        output=$(mpirun -np "${p}" ./apps/mpi_pcg "${n}" as "${overlap}")
    fi

    local iter
    local rel_res
    local total_time
    local spmv_time
    local halo_time
    local prec_time
    local reduction_time
    local vector_time

    iter=$(printf "%s\n" "${output}" |
        awk '/^Iterations:/ {print $2; exit}')

    rel_res=$(printf "%s\n" "${output}" |
        awk '/^Relative residual:/ {print $3; exit}')

    total_time=$(printf "%s\n" "${output}" |
        awk '/^Total time:/ {print $3; exit}')

    spmv_time=$(printf "%s\n" "${output}" |
        awk '/^SpMV time:/ {print $3; exit}')

    halo_time=$(printf "%s\n" "${output}" |
        awk '/^Halo time:/ {print $3; exit}')

    prec_time=$(printf "%s\n" "${output}" |
        awk '/^Preconditioner time:/ {print $3; exit}')

    reduction_time=$(printf "%s\n" "${output}" |
        awk '/^Reduction time:/ {print $3; exit}')

    vector_time=$(printf "%s\n" "${output}" |
        awk '/^Vector time:/ {print $3; exit}')

    if [[ -z "${iter}" ||
          -z "${rel_res}" ||
          -z "${total_time}" ||
          -z "${spmv_time}" ||
          -z "${halo_time}" ||
          -z "${prec_time}" ||
          -z "${reduction_time}" ||
          -z "${vector_time}" ]]; then
        echo "Error: failed to parse output." >&2
        printf "%s\n" "${output}" >&2
        exit 1
    fi

    echo "${n},${p},${method},${overlap},${q},${run},${iter},${rel_res},${total_time},${spmv_time},${halo_time},${prec_time},${reduction_time},${vector_time},${SLURM_JOB_ID}" \
        >> "${raw}"
}

for q in 1 2 4 8
do
    for run in $(seq 1 "${repeats}")
    do
        run_case "${q}" "BJ" 0 "${run}"
    done

    for run in $(seq 1 "${repeats}")
    do
        run_case "${q}" "AS1" 1 "${run}"
    done

    for run in $(seq 1 "${repeats}")
    do
        run_case "${q}" "AS2" 2 "${run}"
    done
done

echo
echo "SGS sensitivity experiment complete."
echo "Raw results: ${raw}"
