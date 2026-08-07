from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PHASE5 = ROOT / "results" / "phase5"
OUTPUT = ROOT / "results" / "final"

OUTPUT.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 1. Read all original Phase 5 timing runs
# ------------------------------------------------------------

timing_files = sorted(
    (PHASE5 / "raw").glob("timing_*.csv")
)

if not timing_files:
    raise RuntimeError("No Phase 5 timing raw files found.")

original_parts = []

for path in timing_files:
    df = pd.read_csv(path)
    df["source_file"] = str(path.relative_to(ROOT))
    original_parts.append(df)

raw = pd.concat(original_parts, ignore_index=True)


# ------------------------------------------------------------
# 2. Define the critical reruns and their correct studies
# ------------------------------------------------------------

critical_specs = [
    {
        "path": PHASE5
        / "critical_rerun"
        / "raw"
        / "critical_n1024_p64_213717.csv",
        "scaling": "strong",
        "grid_size": 1024,
        "processes": 64,
    },
    {
        "path": PHASE5
        / "critical_rerun"
        / "raw"
        / "critical_n2048_p2_213718.csv",
        "scaling": "strong",
        "grid_size": 2048,
        "processes": 2,
    },
    {
        "path": PHASE5
        / "critical_rerun"
        / "raw"
        / "critical_n2896_p128_213719.csv",
        "scaling": "weak",
        "grid_size": 2896,
        "processes": 128,
    },
]


# ------------------------------------------------------------
# 3. Remove old versions and insert critical reruns
# ------------------------------------------------------------

critical_parts = []

for spec in critical_specs:
    path = spec["path"]

    if not path.exists():
        raise FileNotFoundError(path)

    # Remove the superseded original measurements.
    old_mask = (
        (raw["scaling"] == spec["scaling"])
        & (raw["grid_size"] == spec["grid_size"])
        & (raw["processes"] == spec["processes"])
    )

    removed = int(old_mask.sum())

    print(
        f"Replacing {spec['scaling']} "
        f"N={spec['grid_size']}, "
        f"p={spec['processes']}: "
        f"removed {removed} old rows"
    )

    raw = raw.loc[~old_mask].copy()

    rerun = pd.read_csv(path)

    # Critical files use "critical" as the scaling label.
    rerun["scaling"] = spec["scaling"]
    rerun["source_file"] = str(path.relative_to(ROOT))

    critical_parts.append(rerun)

raw = pd.concat(
    [raw, *critical_parts],
    ignore_index=True,
)


# ------------------------------------------------------------
# 4. Validate the merged raw data
# ------------------------------------------------------------

required_columns = [
    "scaling",
    "grid_size",
    "unknowns",
    "processes",
    "local_unknowns",
    "method",
    "overlap",
    "sgs_sweeps",
    "repeat",
    "iterations",
    "relative_residual",
    "total_time",
    "spmv_time",
    "halo_time",
    "preconditioner_time",
    "reduction_time",
    "vector_time",
    "job_id",
]

missing = [
    column
    for column in required_columns
    if column not in raw.columns
]

if missing:
    raise RuntimeError(
        f"Missing required columns: {missing}"
    )

group_columns = [
    "scaling",
    "grid_size",
    "processes",
    "method",
    "overlap",
]

duplicate_runs = raw.duplicated(
    subset=group_columns + ["repeat"],
    keep=False,
)

if duplicate_runs.any():
    print("\nDuplicate configuration/repeat rows:")
    print(
        raw.loc[
            duplicate_runs,
            group_columns
            + ["repeat", "job_id", "source_file"],
        ]
        .sort_values(group_columns + ["repeat"])
        .to_string(index=False)
    )
    raise RuntimeError("Duplicate runs remain after replacement.")

repeat_counts = (
    raw.groupby(group_columns)
    .agg(
        row_count=("repeat", "size"),
        unique_repeats=("repeat", "nunique"),
    )
    .reset_index()
)

bad_repeats = repeat_counts[
    (repeat_counts["row_count"] != 5)
    | (repeat_counts["unique_repeats"] != 5)
]

if not bad_repeats.empty:
    print("\nConfigurations without exactly five runs:")
    print(bad_repeats.to_string(index=False))
    raise RuntimeError(
        "Some configurations do not contain five unique repeats."
    )


# ------------------------------------------------------------
# 5. Add per-run derived metrics
# ------------------------------------------------------------

raw["time_per_iteration"] = (
    raw["total_time"] / raw["iterations"]
)

raw["spmv_time_per_iteration"] = (
    raw["spmv_time"] / raw["iterations"]
)

raw["halo_time_per_iteration"] = (
    raw["halo_time"] / raw["iterations"]
)

raw["preconditioner_time_per_iteration"] = (
    raw["preconditioner_time"]
    / raw["iterations"]
)

raw["reduction_time_per_iteration"] = (
    raw["reduction_time"]
    / raw["iterations"]
)

raw["vector_time_per_iteration"] = (
    raw["vector_time"] / raw["iterations"]
)


# ------------------------------------------------------------
# 6. Produce the final summary
# ------------------------------------------------------------

summary_group_columns = [
    "scaling",
    "grid_size",
    "unknowns",
    "processes",
    "local_unknowns",
    "method",
    "overlap",
    "sgs_sweeps",
]

summary = (
    raw.groupby(
        summary_group_columns,
        dropna=False,
    )
    .agg(
        repeats=("repeat", "nunique"),
        mean_iterations=("iterations", "mean"),
        std_iterations=("iterations", "std"),
        min_iterations=("iterations", "min"),
        max_iterations=("iterations", "max"),
        mean_relative_residual=(
            "relative_residual",
            "mean",
        ),
        std_relative_residual=(
            "relative_residual",
            "std",
        ),
        max_relative_residual=(
            "relative_residual",
            "max",
        ),
        mean_total_time=("total_time", "mean"),
        std_total_time=("total_time", "std"),
        min_total_time=("total_time", "min"),
        max_total_time=("total_time", "max"),
        mean_time_per_iteration=(
            "time_per_iteration",
            "mean",
        ),
        std_time_per_iteration=(
            "time_per_iteration",
            "std",
        ),
        mean_spmv_time=("spmv_time", "mean"),
        std_spmv_time=("spmv_time", "std"),
        mean_halo_time=("halo_time", "mean"),
        std_halo_time=("halo_time", "std"),
        mean_preconditioner_time=(
            "preconditioner_time",
            "mean",
        ),
        std_preconditioner_time=(
            "preconditioner_time",
            "std",
        ),
        mean_reduction_time=(
            "reduction_time",
            "mean",
        ),
        std_reduction_time=(
            "reduction_time",
            "std",
        ),
        mean_vector_time=("vector_time", "mean"),
        std_vector_time=("vector_time", "std"),
        mean_preconditioner_time_per_iteration=(
            "preconditioner_time_per_iteration",
            "mean",
        ),
        mean_reduction_time_per_iteration=(
            "reduction_time_per_iteration",
            "mean",
        ),
    )
    .reset_index()
)

method_order = pd.CategoricalDtype(
    categories=["BJ", "AS1", "AS2"],
    ordered=True,
)

summary["method"] = summary["method"].astype(
    method_order
)

summary = summary.sort_values(
    [
        "scaling",
        "grid_size",
        "processes",
        "method",
    ]
).reset_index(drop=True)

summary["method"] = summary["method"].astype(str)


# ------------------------------------------------------------
# 7. Produce BJ-relative overlap comparison data
# ------------------------------------------------------------

bj = summary[
    summary["method"] == "BJ"
].copy()

as_data = summary[
    summary["method"].isin(["AS1", "AS2"])
].copy()

comparison_keys = [
    "scaling",
    "grid_size",
    "processes",
]

tradeoff = as_data.merge(
    bj,
    on=comparison_keys,
    suffixes=("_as", "_bj"),
    validate="many_to_one",
)

# Positive values mean an improvement/reduction.
tradeoff["iteration_reduction_pct"] = (
    100.0
    * (
        tradeoff["mean_iterations_bj"]
        - tradeoff["mean_iterations_as"]
    )
    / tradeoff["mean_iterations_bj"]
)

# Positive values mean AS is slower than BJ.
tradeoff["runtime_change_pct"] = (
    100.0
    * (
        tradeoff["mean_total_time_as"]
        - tradeoff["mean_total_time_bj"]
    )
    / tradeoff["mean_total_time_bj"]
)

tradeoff["time_per_iteration_change_pct"] = (
    100.0
    * (
        tradeoff["mean_time_per_iteration_as"]
        / tradeoff["mean_time_per_iteration_bj"]
        - 1.0
    )
)

tradeoff[
    "preconditioner_time_per_iteration_change_pct"
] = (
    100.0
    * (
        tradeoff[
            "mean_preconditioner_time_per_iteration_as"
        ]
        / tradeoff[
            "mean_preconditioner_time_per_iteration_bj"
        ]
        - 1.0
    )
)

tradeoff_final = tradeoff[
    [
        "scaling",
        "grid_size",
        "processes",
        "local_unknowns_as",
        "method_as",
        "iteration_reduction_pct",
        "runtime_change_pct",
        "time_per_iteration_change_pct",
        "preconditioner_time_per_iteration_change_pct",
    ]
].rename(
    columns={
        "local_unknowns_as": "local_unknowns",
        "method_as": "method",
    }
)

tradeoff_final = tradeoff_final.sort_values(
    [
        "scaling",
        "grid_size",
        "processes",
        "method",
    ]
).reset_index(drop=True)


# ------------------------------------------------------------
# 8. Save outputs
# ------------------------------------------------------------

raw_output = OUTPUT / "phase5_raw_clean.csv"
summary_output = OUTPUT / "phase5_summary_clean.csv"
tradeoff_output = OUTPUT / "phase5_tradeoff_clean.csv"

raw.to_csv(raw_output, index=False)
summary.to_csv(summary_output, index=False)
tradeoff_final.to_csv(
    tradeoff_output,
    index=False,
)

print("\nCreated:")
print(raw_output.relative_to(ROOT))
print(summary_output.relative_to(ROOT))
print(tradeoff_output.relative_to(ROOT))

print("\nFinal counts:")
print(f"Raw runs: {len(raw)}")
print(f"Summary rows: {len(summary)}")
print(
    "Physical configurations:",
    summary[
        [
            "scaling",
            "grid_size",
            "processes",
        ]
    ]
    .drop_duplicates()
    .shape[0],
)

print("\nRows by scaling and method:")
print(
    summary.groupby(
        ["scaling", "method"]
    )
    .size()
    .unstack(fill_value=0)
)

print("\nMaximum final relative residual:")
print(
    f"{summary['max_relative_residual'].max():.6e}"
)

print("\nTrade-off ranges:")
print(
    tradeoff_final[
        [
            "iteration_reduction_pct",
            "runtime_change_pct",
            "time_per_iteration_change_pct",
        ]
    ]
    .agg(["min", "max"])
    .to_string()
)
