#!/bin/bash
#SBATCH --job-name=p5-time-weak
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --output=results/phase5/logs/%x_%j.out
#SBATCH --error=results/phase5/logs/%x_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

module purge
module load mpi/latest

export OMP_NUM_THREADS=1

p="${SLURM_NTASKS}"
repeats=5
sgs_sweeps=4

# Keep approximately 65,536 unknowns per MPI process.
case "${p}" in
    1)   n=256  ;;
    2)   n=362  ;;
    4)   n=512  ;;
    8)   n=724  ;;
    16)  n=1024 ;;
    32)  n=1448 ;;
    64)  n=2048 ;;
    128) n=2896 ;;
    *)
        echo "Error: unsupported process count ${p}." >&2
        echo "Use 1, 2, 4, 8, 16, 32, 64 or 128 processes." >&2
        exit 1
        ;;
esac

mkdir -p \
    results/phase5/raw \
    results/phase5/summary \
    results/phase5/logs

if [[ ! -x ./apps/mpi_pcg ]]; then
    echo "Error: ./apps/mpi_pcg does not exist or is not executable." >&2
    exit 1
fi

raw="results/phase5/raw/timing_weak_p${p}_${SLURM_JOB_ID}.csv"
summary="results/phase5/summary/timing_weak_p${p}_${SLURM_JOB_ID}.csv"

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
    local iter
    local rel_res
    local total_time
    local spmv_time
    local halo_time
    local prec_time
    local reduction_time
    local vector_time

    echo "n=${n}, p=${p}, method=${method}, repeat=${run}"

    if [[ "${method}" == "BJ" ]]; then
        output=$(mpirun -np "${p}" ./apps/mpi_pcg "${n}" bj)
    else
        output=$(mpirun -np "${p}" ./apps/mpi_pcg "${n}" as "${overlap}")
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

        echo "Error: failed to parse program output." >&2
        printf "%s\n" "${output}" >&2
        exit 1
    fi

    echo "weak,${n},${unknowns},${p},${local_unknowns},${method},${overlap},${sgs_sweeps},${run},${iter},${rel_res},${total_time},${spmv_time},${halo_time},${prec_time},${reduction_time},${vector_time},${SLURM_JOB_ID}" \
        >> "${raw}"
}

for run in $(seq 1 "${repeats}")
do
    run_case "BJ"  0 "${run}"
    run_case "AS1" 1 "${run}"
    run_case "AS2" 2 "${run}"
done

awk -F, '
BEGIN {
    OFS=","

    print "scaling,grid_size,unknowns,processes,local_unknowns,method,overlap,sgs_sweeps,repeats,mean_iterations,mean_relative_residual,mean_total_time,std_total_time,min_total_time,max_total_time,mean_spmv_time,mean_halo_time,mean_preconditioner_time,mean_reduction_time,mean_vector_time,job_id"
}

NR == 1 {
    next
}

{
    key = $1 SUBSEP $2 SUBSEP $3 SUBSEP $4 SUBSEP \
          $5 SUBSEP $6 SUBSEP $7 SUBSEP $8

    count[key]++

    sum_iter[key] += $10
    sum_res[key] += $11

    sum_total[key] += $12
    sumsq_total[key] += $12 * $12

    sum_spmv[key] += $13
    sum_halo[key] += $14
    sum_prec[key] += $15
    sum_reduction[key] += $16
    sum_vector[key] += $17

    if (!(key in min_total) || $12 < min_total[key]) {
        min_total[key] = $12
    }

    if (!(key in max_total) || $12 > max_total[key]) {
        max_total[key] = $12
    }

    job[key] = $18
}

END {
    for (key in count) {
        split(key, a, SUBSEP)

        mean_iter = sum_iter[key] / count[key]
        mean_res = sum_res[key] / count[key]

        mean_total = sum_total[key] / count[key]

        variance = sumsq_total[key] / count[key] \
                   - mean_total * mean_total

        if (variance < 0) {
            variance = 0
        }

        std_total = sqrt(variance)

        mean_spmv = sum_spmv[key] / count[key]
        mean_halo = sum_halo[key] / count[key]
        mean_prec = sum_prec[key] / count[key]
        mean_reduction = sum_reduction[key] / count[key]
        mean_vector = sum_vector[key] / count[key]

        print a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], \
              count[key], mean_iter, mean_res, mean_total, std_total, \
              min_total[key], max_total[key], mean_spmv, mean_halo, \
              mean_prec, mean_reduction, mean_vector, job[key]
    }
}
' "${raw}" > "${summary}"

{
    head -n 1 "${summary}"

    tail -n +2 "${summary}" |
        sort -t, -k4,4n -k6,6
} > "${summary}.tmp"

mv "${summary}.tmp" "${summary}"

echo "Weak-scaling timing experiment completed."
echo "Raw results: ${raw}"
echo "Summary: ${summary}"
