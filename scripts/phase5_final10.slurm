#!/bin/bash
# Final strong- and weak-scaling experiment used in the dissertation.
#
# Purpose:
#   Compare unpreconditioned CG with BJ-PCG, AS1-PCG, and AS2-PCG.
#
# Configuration:
#   Strong scaling: N = 512, 1024, 2048 with supported p up to 128.
#   Weak scaling: approximately 65,536 unknowns per MPI process.
#   Repetitions = 10, SGS sweeps q = 4.
#
# Output:
#   Raw PCG and CG measurements are written to results/final10/raw/.
#
# Dissertation use:
#   This is the core Final10 scaling dataset. The fixed q=4 setting is an
#   experimental baseline, not a claim of runtime optimality.
#SBATCH --job-name=p5-final10
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=06:00:00
#SBATCH --exclude=callan-n[13-24]
#SBATCH --output=results/final10/logs/%x_%j.out
#SBATCH --error=results/final10/logs/%x_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

module purge
module load mpi/latest

export OMP_NUM_THREADS=1

: "${SCALING:?SCALING is not set}"

scaling="${SCALING}"
p="${SLURM_NTASKS}"

repeats=10
sgs_sweeps=4

mkdir -p \
    results/final10/raw/pcg \
    results/final10/raw/cg \
    results/final10/logs

if [[ ! -x ./apps/mpi_pcg ]]; then
    echo "Error: ./apps/mpi_pcg not found." >&2
    exit 1
fi

if [[ ! -x ./apps/mpi_cg ]]; then
    echo "Error: ./apps/mpi_cg not found." >&2
    exit 1
fi

if [[ "${scaling}" != "strong" && "${scaling}" != "weak" ]]; then
    echo "Error: invalid SCALING=${scaling}" >&2
    exit 1
fi

pcg_raw="results/final10/raw/pcg/pcg_${scaling}_p${p}_${SLURM_JOB_ID}.csv"
cg_raw="results/final10/raw/cg/cg_${scaling}_p${p}_${SLURM_JOB_ID}.csv"

echo "scaling,grid_size,unknowns,processes,local_unknowns,method,overlap,sgs_sweeps,repeat,iterations,relative_residual,total_time,spmv_time,halo_time,preconditioner_time,reduction_time,vector_time,job_id" \
    > "${pcg_raw}"

echo "grid_size,processes,method,repeat,iterations,relative_residual,total_time,job_id" \
    > "${cg_raw}"

run_cg()
{
    local n="$1"
    local run="$2"

    local output
    local iter
    local rel_res
    local total_time

    echo "CG: scaling=${scaling}, N=${n}, p=${p}, repeat=${run}"

    output=$(mpirun -np "${p}" ./apps/mpi_cg "${n}")

    iter=$(printf "%s\n" "${output}" |
        awk '/^Iterations:/ {print $2; exit}')

    rel_res=$(printf "%s\n" "${output}" |
        awk '/^Relative residual:/ {print $3; exit}')

    total_time=$(printf "%s\n" "${output}" |
        awk '/^Runtime:/ {print $2; exit}')

    if [[ -z "${iter}" ||
          -z "${rel_res}" ||
          -z "${total_time}" ]]; then
        echo "Error: failed to parse CG output." >&2
        printf "%s\n" "${output}" >&2
        exit 1
    fi

    echo "${n},${p},CG,${run},${iter},${rel_res},${total_time},${SLURM_JOB_ID}" \
        >> "${cg_raw}"
}

run_pcg()
{
    local n="$1"
    local method="$2"
    local overlap="$3"
    local run="$4"

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

    echo "PCG: scaling=${scaling}, N=${n}, p=${p}, method=${method}, repeat=${run}"

    case "${method}" in
        BJ)
            output=$(mpirun -np "${p}" ./apps/mpi_pcg "${n}" bj)
            ;;
        AS1)
            output=$(mpirun -np "${p}" ./apps/mpi_pcg "${n}" as 1)
            ;;
        AS2)
            output=$(mpirun -np "${p}" ./apps/mpi_pcg "${n}" as 2)
            ;;
        *)
            echo "Error: invalid method=${method}" >&2
            exit 1
            ;;
    esac

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
        echo "Error: failed to parse PCG output." >&2
        printf "%s\n" "${output}" >&2
        exit 1
    fi

    echo "${scaling},${n},${unknowns},${p},${local_unknowns},${method},${overlap},${sgs_sweeps},${run},${iter},${rel_res},${total_time},${spmv_time},${halo_time},${prec_time},${reduction_time},${vector_time},${SLURM_JOB_ID}" \
        >> "${pcg_raw}"
}

run_configuration()
{
    local n="$1"
    local include_cg="$2"

    for run in $(seq 1 "${repeats}")
    do
        if [[ "${include_cg}" -eq 1 ]]; then
            run_cg "${n}" "${run}"
        fi

        run_pcg "${n}" BJ  0 "${run}"
        run_pcg "${n}" AS1 1 "${run}"
        run_pcg "${n}" AS2 2 "${run}"
    done
}

if [[ "${scaling}" == "strong" ]]; then

    case "${p}" in
        1|2|4|8|16|32)
            grids="512 1024 2048"
            ;;
        64)
            grids="1024 2048"
            ;;
        128)
            grids="2048"
            ;;
        *)
            echo "Error: invalid strong-scaling p=${p}" >&2
            exit 1
            ;;
    esac

    for n in ${grids}
    do
        # All 21 strong configurations provide the canonical CG baseline.
        run_configuration "${n}" 1
    done

else

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
            echo "Error: invalid weak-scaling p=${p}" >&2
            exit 1
            ;;
    esac

    # These three (N,p) pairs already have an identical CG baseline
    # in the strong-scaling experiment.
    case "${n},${p}" in
        "512,4"|"1024,16"|"2048,64")
            include_cg=0
            ;;
        *)
            include_cg=1
            ;;
    esac

    run_configuration "${n}" "${include_cg}"

fi

echo "Final10 job completed."
echo "PCG raw: ${pcg_raw}"
echo "CG raw:  ${cg_raw}"
