#!/bin/bash
#SBATCH --job-name=p5-critical
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --output=results/phase5/critical_rerun/logs/%x_%j.out
#SBATCH --error=results/phase5/critical_rerun/logs/%x_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

module purge
module load mpi/latest

export OMP_NUM_THREADS=1

p="${SLURM_NTASKS}"
n="${N:?Submit with --export=ALL,N=<grid_size>}"

repeats=5
sgs_sweeps=4

case "${n}:${p}" in
    1024:64|2048:2|2896:128)
        ;;
    *)
        echo "Error: unsupported critical configuration n=${n}, p=${p}" >&2
        exit 1
        ;;
esac

mkdir -p \
    results/phase5/critical_rerun/raw \
    results/phase5/critical_rerun/logs

if [[ ! -x ./apps/mpi_pcg ]]; then
    echo "Error: ./apps/mpi_pcg does not exist or is not executable." >&2
    exit 1
fi

raw="results/phase5/critical_rerun/raw/critical_n${n}_p${p}_${SLURM_JOB_ID}.csv"

echo "scaling,grid_size,unknowns,processes,local_unknowns,method,overlap,sgs_sweeps,repeat,iterations,relative_residual,total_time,spmv_time,halo_time,preconditioner_time,reduction_time,vector_time,job_id" \
    > "${raw}"

run_case()
{
    local method="$1"
    local overlap="$2"
    local run="$3"

    local unknowns=$((n * n))
    local local_unknowns=$((n * ((n + p - 1) / p)))

    local output

    echo "n=${n}, p=${p}, method=${method}, repeat=${run}"

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
        echo "Error: failed to parse program output." >&2
        printf "%s\n" "${output}" >&2
        exit 1
    fi

    echo "critical,${n},${unknowns},${p},${local_unknowns},${method},${overlap},${sgs_sweeps},${run},${iter},${rel_res},${total_time},${spmv_time},${halo_time},${prec_time},${reduction_time},${vector_time},${SLURM_JOB_ID}" \
        >> "${raw}"
}

for run in $(seq 1 "${repeats}")
do
    run_case "BJ"  0 "${run}"
    run_case "AS1" 1 "${run}"
    run_case "AS2" 2 "${run}"
done

echo "Critical profile experiment completed."
echo "Raw results: ${raw}"
