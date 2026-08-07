from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results" / "final" / "phase5_raw_clean.csv"
RERUN_DIR = (
    ROOT
    / "results"
    / "phase5"
    / "stability_rerun"
    / "raw"
)
OUTPUT_DIR = ROOT / "results" / "final"

BASE_COLUMNS = [
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

CONFIG_KEYS = [
    "scaling",
    "grid_size",
    "processes",
]

METHOD_KEYS = CONFIG_KEYS + [
    "method",
    "overlap",
]


# ------------------------------------------------------------
# 1. Read the existing cleaned data
#    This already contains the three critical reruns.
# ------------------------------------------------------------

old = pd.read_csv(INPUT)

missing = [
    column
    for column in BASE_COLUMNS
    if column not in old.columns
]

if missing:
    raise RuntimeError(
        f"Missing columns in {INPUT}: {missing}"
    )

if "source_file" not in old.columns:
    old["source_file"] = str(INPUT.relative_to(ROOT))

old = old[BASE_COLUMNS + ["source_file"]].copy()


# ------------------------------------------------------------
# 2. Read the 15 stability rerun files
# ------------------------------------------------------------

rerun_files = sorted(RERUN_DIR.glob("*.csv"))

if len(rerun_files) != 15:
    raise RuntimeError(
        f"Expected 15 rerun files, found {len(rerun_files)}"
    )

rerun_parts = []
rerun_configs = []

for path in rerun_files:
    df = pd.read_csv(path)

    missing = [
        column
        for column in BASE_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"{path} is missing columns: {missing}"
        )

    if len(df) != 15:
        raise RuntimeError(
            f"{path} contains {len(df)} rows; expected 15"
        )

    configurations = (
        df[CONFIG_KEYS]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    if len(configurations) != 1:
        raise RuntimeError(
            f"{path} contains more than one configuration"
        )

    scaling = configurations.loc[0, "scaling"]
    grid_size = int(
        configurations.loc[0, "grid_size"]
    )
    processes = int(
        configurations.loc[0, "processes"]
    )

    rerun_configs.append(
        (scaling, grid_size, processes)
    )

    df["source_file"] = str(path.relative_to(ROOT))

    rerun_parts.append(
        df[BASE_COLUMNS + ["source_file"]]
    )

if len(set(rerun_configs)) != 15:
    raise RuntimeError(
        "The rerun files do not contain 15 unique configurations"
    )


# ------------------------------------------------------------
# 3. Remove the 15 superseded configurations
# ------------------------------------------------------------

for scaling, grid_size, processes in rerun_configs:
    mask = (
        (old["scaling"] == scaling)
        & (old["grid_size"] == grid_size)
        & (old["processes"] == processes)
    )

    removed = int(mask.sum())

    print(
        f"Replacing {scaling} "
        f"N={grid_size}, p={processes}: "
        f"removed {removed} old rows"
    )

    if removed != 15:
        raise RuntimeError(
            f"Expected to remove 15 rows, removed {removed}"
        )

    old = old.loc[~mask].copy()


# ------------------------------------------------------------
# 4. Combine retained data and new reruns
# ------------------------------------------------------------

raw = pd.concat(
    [old, *rerun_parts],
    ignore_index=True,
)

duplicate_mask = raw.duplicated(
    subset=METHOD_KEYS + ["repeat"],
    keep=False,
)

if duplicate_mask.any():
    print(
        raw.loc[
            duplicate_mask,
            METHOD_KEYS
            + ["repeat", "job_id", "source_file"],
        ]
        .sort_values(METHOD_KEYS + ["repeat"])
        .to_string(index=False)
    )
    raise RuntimeError(
        "Duplicate configuration/repeat rows detected"
    )

run_counts = (
    raw.groupby(METHOD_KEYS)
    .agg(
        rows=("repeat", "size"),
        repeats=("repeat", "nunique"),
    )
    .reset_index()
)

bad_counts = run_counts[
    (run_counts["rows"] != 5)
    | (run_counts["repeats"] != 5)
]

if not bad_counts.empty:
    print(bad_counts.to_string(index=False))
    raise RuntimeError(
        "Some method configurations do not have five runs"
    )


# ------------------------------------------------------------
# 5. Derived per-run metrics
# ------------------------------------------------------------

raw["time_per_iteration"] = (
    raw["total_time"]
    / raw["iterations"]
)

raw["preconditioner_time_per_iteration"] = (
    raw["preconditioner_time"]
    / raw["iterations"]
)

raw["reduction_time_per_iteration"] = (
    raw["reduction_time"]
    / raw["iterations"]
)


# ------------------------------------------------------------
# 6. Final summary
#
# Use population standard deviation, ddof=0, matching
# the original Phase 5 summary scripts.
# ------------------------------------------------------------

SUMMARY_KEYS = [
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
        SUMMARY_KEYS,
        dropna=False,
    )
    .agg(
        repeats=("repeat", "nunique"),
        mean_iterations=("iterations", "mean"),
        std_iterations=(
            "iterations",
            lambda values: values.std(ddof=0),
        ),
        min_iterations=("iterations", "min"),
        max_iterations=("iterations", "max"),
        mean_relative_residual=(
            "relative_residual",
            "mean",
        ),
        std_relative_residual=(
            "relative_residual",
            lambda values: values.std(ddof=0),
        ),
        max_relative_residual=(
            "relative_residual",
            "max",
        ),
        mean_total_time=("total_time", "mean"),
        std_total_time=(
            "total_time",
            lambda values: values.std(ddof=0),
        ),
        min_total_time=("total_time", "min"),
        max_total_time=("total_time", "max"),
        mean_time_per_iteration=(
            "time_per_iteration",
            "mean",
        ),
        std_time_per_iteration=(
            "time_per_iteration",
            lambda values: values.std(ddof=0),
        ),
        mean_spmv_time=("spmv_time", "mean"),
        std_spmv_time=(
            "spmv_time",
            lambda values: values.std(ddof=0),
        ),
        mean_halo_time=("halo_time", "mean"),
        std_halo_time=(
            "halo_time",
            lambda values: values.std(ddof=0),
        ),
        mean_preconditioner_time=(
            "preconditioner_time",
            "mean",
        ),
        std_preconditioner_time=(
            "preconditioner_time",
            lambda values: values.std(ddof=0),
        ),
        mean_reduction_time=(
            "reduction_time",
            "mean",
        ),
        std_reduction_time=(
            "reduction_time",
            lambda values: values.std(ddof=0),
        ),
        mean_vector_time=("vector_time", "mean"),
        std_vector_time=(
            "vector_time",
            lambda values: values.std(ddof=0),
        ),
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

method_order = {
    "BJ": 0,
    "AS1": 1,
    "AS2": 2,
}

summary["_method_order"] = (
    summary["method"]
    .map(method_order)
)

summary = (
    summary.sort_values(
        [
            "scaling",
            "grid_size",
            "processes",
            "_method_order",
        ]
    )
    .drop(columns="_method_order")
    .reset_index(drop=True)
)

summary["cv_total_time"] = (
    summary["std_total_time"]
    / summary["mean_total_time"]
)


# ------------------------------------------------------------
# 7. BJ-relative overlap trade-off
#
# Positive iteration_reduction_pct:
# AS uses fewer iterations than BJ.
#
# Positive runtime_change_pct:
# AS is slower than BJ.
# ------------------------------------------------------------

bj = summary[
    summary["method"] == "BJ"
][
    CONFIG_KEYS
    + [
        "mean_iterations",
        "mean_total_time",
        "mean_time_per_iteration",
        "mean_preconditioner_time_per_iteration",
    ]
].rename(
    columns={
        "mean_iterations": "bj_iterations",
        "mean_total_time": "bj_total_time",
        "mean_time_per_iteration":
            "bj_time_per_iteration",
        "mean_preconditioner_time_per_iteration":
            "bj_preconditioner_time_per_iteration",
    }
)

as_data = summary[
    summary["method"].isin(["AS1", "AS2"])
][
    CONFIG_KEYS
    + [
        "local_unknowns",
        "method",
        "mean_iterations",
        "mean_total_time",
        "mean_time_per_iteration",
        "mean_preconditioner_time_per_iteration",
    ]
].copy()

tradeoff = as_data.merge(
    bj,
    on=CONFIG_KEYS,
    how="left",
    validate="many_to_one",
)

tradeoff["iteration_reduction_pct"] = (
    100.0
    * (
        tradeoff["bj_iterations"]
        - tradeoff["mean_iterations"]
    )
    / tradeoff["bj_iterations"]
)

tradeoff["runtime_change_pct"] = (
    100.0
    * (
        tradeoff["mean_total_time"]
        - tradeoff["bj_total_time"]
    )
    / tradeoff["bj_total_time"]
)

tradeoff["time_per_iteration_change_pct"] = (
    100.0
    * (
        tradeoff["mean_time_per_iteration"]
        / tradeoff["bj_time_per_iteration"]
        - 1.0
    )
)

tradeoff[
    "preconditioner_time_per_iteration_change_pct"
] = (
    100.0
    * (
        tradeoff[
            "mean_preconditioner_time_per_iteration"
        ]
        / tradeoff[
            "bj_preconditioner_time_per_iteration"
        ]
        - 1.0
    )
)

tradeoff = tradeoff[
    CONFIG_KEYS
    + [
        "local_unknowns",
        "method",
        "iteration_reduction_pct",
        "runtime_change_pct",
        "time_per_iteration_change_pct",
        "preconditioner_time_per_iteration_change_pct",
    ]
].sort_values(
    CONFIG_KEYS + ["method"]
).reset_index(drop=True)


# ------------------------------------------------------------
# 8. Save final outputs
# ------------------------------------------------------------

raw_path = (
    OUTPUT_DIR
    / "phase5_raw_final.csv"
)

summary_path = (
    OUTPUT_DIR
    / "phase5_summary_final.csv"
)

tradeoff_path = (
    OUTPUT_DIR
    / "phase5_tradeoff_final.csv"
)

quality_path = (
    OUTPUT_DIR
    / "phase5_timing_quality_final.csv"
)

raw.to_csv(raw_path, index=False)
summary.to_csv(summary_path, index=False)
tradeoff.to_csv(tradeoff_path, index=False)

quality = summary[
    [
        "scaling",
        "grid_size",
        "processes",
        "method",
        "repeats",
        "mean_total_time",
        "std_total_time",
        "cv_total_time",
        "min_total_time",
        "max_total_time",
    ]
].copy()

quality["status"] = quality[
    "cv_total_time"
].apply(
    lambda value: (
        "PASS"
        if value <= 0.20
        else "REVIEW"
    )
)

quality.to_csv(quality_path, index=False)


# ------------------------------------------------------------
# 9. Report
# ------------------------------------------------------------

physical_configurations = (
    summary[CONFIG_KEYS]
    .drop_duplicates()
    .shape[0]
)

print("\nCreated:")
print(raw_path.relative_to(ROOT))
print(summary_path.relative_to(ROOT))
print(tradeoff_path.relative_to(ROOT))
print(quality_path.relative_to(ROOT))

print("\nFinal counts:")
print(f"Raw runs: {len(raw)}")
print(f"Summary rows: {len(summary)}")
print(
    "Physical configurations:",
    physical_configurations,
)

print("\nMaximum relative residual:")
print(
    f"{summary['max_relative_residual'].max():.6e}"
)

print("\nTiming-quality status:")
print(
    quality["status"]
    .value_counts()
    .to_string()
)

review = quality[
    quality["status"] == "REVIEW"
].sort_values(
    "cv_total_time",
    ascending=False,
)

print("\nConfigurations with CV above 20%:")

if review.empty:
    print("None")
else:
    print(review.to_string(index=False))

print("\nTrade-off ranges:")
print(
    tradeoff[
        [
            "iteration_reduction_pct",
            "runtime_change_pct",
            "time_per_iteration_change_pct",
        ]
    ]
    .agg(["min", "max"])
    .to_string()
)
