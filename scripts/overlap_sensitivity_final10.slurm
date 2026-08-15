#!/bin/bash
#SBATCH --job-name=overlap-sens
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --exclude=callan-n[13-24]
#SBATCH --output=results/overlap_sensitivity/logs/%x_%j.out
#SBATCH --error=results/overlap_sensitivity/logs/%x_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

module purge
module load mpi/latest

export OMP_NUM_THREADS=1
export SGS_SWEEPS=4

n=512
p=8
owned_rows=64
repeats=10
sgs_sweeps=4

raw="results/overlap_sensitivity/raw/overlap_N${n}_p${p}_${SLURM_JOB_ID}.csv"

echo "grid_size,unknowns,processes,owned_rows,method,overlap,relative_overlap_pct,sgs_sweeps,repeat,iterations,relative_residual,total_time,spmv_time,halo_time,preconditioner_time,reduction_time,vector_time,job_id" \
    > "${raw}"

run_case()
{
    local method="$1"
    local delta="$2"
    local run="$3"

    local output
    local iter
    local rel_res
    local total_time
    local spmv_time
    local halo_time
    local prec_time
    local reduction_time
    local vector_time
    local alpha_pct

    alpha_pct=$(awk -v d="${delta}" -v h="${owned_rows}" \
        'BEGIN {printf "%.6f", 100.0*d/h}')

    echo "N=${n}, p=${p}, method=${method}, delta=${delta}, repeat=${run}"

    if [[ "${method}" == "BJ" ]]; then
        output=$(mpirun -np "${p}" ./apps/mpi_pcg "${n}" bj)
    else
        output=$(mpirun -np "${p}" ./apps/mpi_pcg "${n}" as "${delta}")
    fi

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

    echo "${n},$((n*n)),${p},${owned_rows},${method},${delta},${alpha_pct},${sgs_sweeps},${run},${iter},${rel_res},${total_time},${spmv_time},${halo_time},${prec_time},${reduction_time},${vector_time},${SLURM_JOB_ID}" \
        >> "${raw}"
}

for run in $(seq 1 "${repeats}")
do
    run_case BJ 0 "${run}"

    for delta in 1 2 6 12 19 32
    do
        run_case AS "${delta}" "${run}"
    done
done

echo "Completed overlap sensitivity experiment."
echo "Raw data: ${raw}"
