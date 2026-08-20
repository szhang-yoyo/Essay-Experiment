# Distributed PCG for the 2D Poisson Problem

This repository contains the implementation, experiments, and analysis used for an MSc dissertation on distributed preconditioned conjugate gradient (PCG) methods for the two-dimensional Poisson problem.

The final implementation compares unpreconditioned CG with PCG using Block Jacobi (BJ) and one-level Classical Additive Schwarz (AS) preconditioners. The distributed-memory implementation uses row-wise MPI decomposition and symmetric Gauss--Seidel (SGS) as the local solver.

## Repository structure

- `apps/` -- executable drivers and validation programs.
- `core/`, `parallel/`, `problems/`, `preconditioners/`, `solvers/` -- final solver implementation.
- `scripts/` -- scripts used for the final dissertation experiments and analysis.
- `results/final10/` -- final strong- and weak-scaling dataset, with ten repetitions per executed configuration.
- `results/overlap_sensitivity/` -- overlap-sensitivity experiment used in the dissertation.
- `results/sgs_sensitivity/` -- SGS-sweep sensitivity experiment used in the dissertation.
- `results/implementation_validation/` -- implementation-validation outputs.
- `results/figures/` and `results/tables/` -- figures, tables, and derived data used during dissertation analysis.
- `old_code/` -- earlier or superseded scripts retained to show the progression of the project.
- `old_experiments/` -- preliminary and superseded experimental datasets retained for provenance; these are not the final dissertation dataset.

No historical code or data have been deleted as part of the final repository organisation.

## Primary dissertation experiment scripts

| Script | Purpose | Dissertation use |
| --- | --- | --- |
| `scripts/phase5_final10.slurm` | Runs the final CG/BJ/AS strong- and weak-scaling configurations with ten repetitions. | Core scaling dataset in `results/final10/`. |
| `scripts/submit_phase5_final10_serial.sh` | Submits the Final10 jobs serially to preserve controlled execution conditions. | Final scaling experiment submission. |
| `scripts/overlap_sensitivity_final10.slurm` | Varies Schwarz overlap at fixed `N=512`, `p=8`, with `q=4`. | Overlap-sensitivity analysis. |
| `scripts/sgs_sensitivity.slurm` | Varies the number of SGS sweeps at fixed `N=1024`, `p=16`. | SGS-sweep sensitivity analysis. |
| `scripts/validate_as0_vs_bj.slurm` | Checks that zero-overlap AS reproduces BJ behaviour. | Implementation validation. |
| `scripts/build_table3_overlap_extremes.py` | Builds selected overlap-comparison outputs from final data. | Dissertation table generation. |
| `scripts/build_table4_scaling_summary.py` | Builds strong/weak scaling summary outputs. | Dissertation table generation. |
| `scripts/plot_overlap_tradeoff_final10.py` | Produces the Final10 overlap trade-off figure/data. | Dissertation figure generation. |

## Final experimental baseline

Unless a sensitivity experiment explicitly varies a parameter, the final dissertation experiments use:

- relative residual tolerance: `1e-8`;
- initial guess: `x0 = 0`;
- right-hand side: `b = 1`;
- SGS sweeps: `q = 4`;
- AS overlap: `delta = 1` (AS1) or `delta = 2` (AS2);
- `OMP_NUM_THREADS=1`;
- Intel MPI via `mpi/latest` on the TCHPC Callan cluster.

The `q=4` setting is a fixed experimental baseline rather than a claim of runtime optimality; the dedicated SGS sensitivity experiment records the effect of varying `q`.
