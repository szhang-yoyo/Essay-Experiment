#!/bin/bash
#SBATCH --job-name=p3-weak
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --output=results/phase3/logs/%x_%j.out
#SBATCH --error=results/phase3/logs/%x_%j.err

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
    results/phase3/raw \
    results/phase3/summary \
    results/phase3/logs

if [[ ! -x ./apps/mpi_pcg ]]; then
    echo "Error: ./apps/mpi_pcg does not exist or is not executable." >&2
    exit 1
fi

raw="results/phase3/raw/weak_p${p}_${SLURM_JOB_ID}.csv"
summary="results/phase3/summary/weak_p${p}_${SLURM_JOB_ID}.csv"

echo "scaling,grid_size,unknowns,processes,local_unknowns,method,overlap,sgs_sweeps,repeat,iterations,relative_residual,runtime,job_id" \
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
    local runtime

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

    runtime=$(printf "%s\n" "${output}" |
        awk '/^Runtime:/ {print $2; exit}')

    if [[ -z "${iter}" || -z "${rel_res}" || -z "${runtime}" ]]; then
        echo "Error: failed to parse program output." >&2
        printf "%s\n" "${output}" >&2
        exit 1
    fi

    echo "weak,${n},${unknowns},${p},${local_unknowns},${method},${overlap},${sgs_sweeps},${run},${iter},${rel_res},${runtime},${SLURM_JOB_ID}" \
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
    print "scaling,grid_size,unknowns,processes,local_unknowns,method,overlap,sgs_sweeps,repeats,mean_iterations,mean_relative_residual,mean_runtime,std_runtime,min_runtime,max_runtime,job_id"
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
    sum_time[key] += $12
    sumsq_time[key] += $12 * $12

    if (!(key in min_time) || $12 < min_time[key]) {
        min_time[key] = $12
    }

    if (!(key in max_time) || $12 > max_time[key]) {
        max_time[key] = $12
    }

    job[key] = $13
}

END {
    for (key in count) {
        split(key, a, SUBSEP)

        mean_iter = sum_iter[key] / count[key]
        mean_res = sum_res[key] / count[key]
        mean_time = sum_time[key] / count[key]

        variance = sumsq_time[key] / count[key] \
                   - mean_time * mean_time

        if (variance < 0) {
            variance = 0
        }

        std_time = sqrt(variance)

        print a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], \
              count[key], mean_iter, mean_res, mean_time, std_time, \
              min_time[key], max_time[key], job[key]
    }
}
' "${raw}" > "${summary}"

{
    head -n 1 "${summary}"
    tail -n +2 "${summary}" |
        sort -t, -k4,4n -k6,6
} > "${summary}.tmp"

mv "${summary}.tmp" "${summary}"

echo "Weak-scaling experiment completed."
echo "Raw results: ${raw}"
echo "Summary: ${summary}"
