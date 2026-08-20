# Final Dissertation Experiments

This branch contains the final implementation, rerun datasets, sensitivity experiments, validation outputs, figures, tables, and scripts used for the dissertation.

The repository history is separated by branch:

- `main` preserves the early serial development (Phases 1–2).
- `phase3-mpi` preserves the MPI development and preliminary experimental work from Phases 3–5.
- `final-dissertation` contains the final dissertation implementation and experiments.

## Final dissertation content

- `apps/`, `core/`, `parallel/`, `preconditioners/`, `problems/`, `solvers/`
  - Final distributed CG/PCG implementation, MPI decomposition, and preconditioners.
- `tests/`
  - Solver and implementation validation programs.
- `results/final10/`
  - Final ten-repeat strong- and weak-scaling dataset, summaries, scaling plots, and the block-size sensitivity summary.
- `results/overlap_sensitivity/`
  - Fixed-block-count overlap sensitivity experiment.
- `results/sgs_sensitivity/`
  - SGS-sweep sensitivity experiment.
- `results/implementation_validation/`
  - Final implementation checks, including the zero-overlap AS versus BJ comparison.
- `results/figures/` and `results/tables/`
  - Derived figures and tables used in the final dissertation analysis.
- `scripts/phase5_final10.slurm`
  - Final ten-repeat CG/BJ/AS strong- and weak-scaling experiment.
- `scripts/submit_phase5_final10_serial.sh`
  - Serial submission workflow for the Final10 jobs.
- `scripts/overlap_sensitivity_final10.slurm`
  - Final overlap-sensitivity experiment.
- `scripts/sgs_sensitivity.slurm`
  - Final SGS-sweep sensitivity experiment.
- `scripts/validate_as0_vs_bj.slurm`
  - AS delta=0 versus BJ implementation validation.

## Final experimental baseline

Unless a sensitivity experiment explicitly varies a parameter, the final experiments use:

- relative residual tolerance: `1e-8`;
- initial guess: `x0 = 0`;
- right-hand side: `b = 1`;
- SGS sweeps: `q = 4`;
- AS overlap widths: `delta = 1` (AS1) and `delta = 2` (AS2);
- ten repetitions per executed final solver configuration;
- `OMP_NUM_THREADS=1`;
- Intel MPI via `mpi/latest` on the TCHPC Callan cluster.

The fixed `q=4` value is the main-experiment baseline rather than a claim of runtime optimality; the dedicated SGS sensitivity experiment varies `q` explicitly.

Historical Phase 3–5 scripts and datasets are retained on the `phase3-mpi` branch rather than duplicated here.
