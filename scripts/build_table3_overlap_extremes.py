from pathlib import Path

import numpy as np
import pandas as pd


RAW_PATH = Path(
    "results/crossover/pcg_canonical_raw_final_validated.csv"
)
QUALITY_PATH = Path(
    "results/crossover/pcg_quality_final_validated.csv"
)

OUT_DIR = Path("results/tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_OUTPUT = OUT_DIR / "table3_overlap_all_comparisons.csv"
SELECTED_OUTPUT = OUT_DIR / "table3_overlap_selected.csv"
LATEX_OUTPUT = OUT_DIR / "table3_overlap_extremes.tex"


def rename_aliases(df):
    aliases = {
        "n": "grid_size",
        "p": "processes",
        "runtime": "total_time",
        "mean_runtime": "mean_total_time",
        "residual": "relative_residual",
        "status": "quality_check",
        "repeats": "runs",
    }

    return df.rename(
        columns={
            old: new
            for old, new in aliases.items()
            if old in df.columns and new not in df.columns
        }
    )


def normalise_method(value):
    value = str(value).strip().upper().replace(" ", "")

    if value.startswith("BJ"):
        return "BJ"

    if value in {"AS1", "AS2"}:
        return value

    raise ValueError(f"Unknown method: {value}")


def signed(value):
    return f"{value:+.2f}"


if not RAW_PATH.exists():
    raise FileNotFoundError(RAW_PATH)

if not QUALITY_PATH.exists():
    raise FileNotFoundError(QUALITY_PATH)


# ---------------------------------------------------------------------
# 1. Read and normalise final raw data
# ---------------------------------------------------------------------

raw = rename_aliases(pd.read_csv(RAW_PATH))
raw["method"] = raw["method"].map(normalise_method)

required_raw = {
    "scaling",
    "grid_size",
    "processes",
    "method",
    "iterations",
    "relative_residual",
    "total_time",
}

missing_raw = required_raw - set(raw.columns)

if missing_raw:
    raise ValueError(
        f"Raw dataset missing columns: {sorted(missing_raw)}"
    )

raw = raw[raw["method"].isin(["BJ", "AS1", "AS2"])].copy()

keys = ["scaling", "grid_size", "processes", "method"]

if raw.duplicated(keys + ["job_id"], keep=False).all():
    print(
        "Note: repeated job IDs are expected when one job contains "
        "multiple experimental repeats."
    )


# ---------------------------------------------------------------------
# 2. Independently audit the raw data
# ---------------------------------------------------------------------

summary = (
    raw.groupby(keys, as_index=False)
    .agg(
        runs=("total_time", "size"),
        mean_iterations=("iterations", "mean"),
        iteration_values=("iterations", "nunique"),
        mean_relative_residual=("relative_residual", "mean"),
        max_relative_residual=("relative_residual", "max"),
        mean_total_time=("total_time", "mean"),
        std_total_time=("total_time", "std"),
    )
)

summary["std_total_time"] = summary["std_total_time"].fillna(0.0)

summary["cv_pct"] = np.where(
    summary["mean_total_time"] > 0.0,
    100.0
    * summary["std_total_time"]
    / summary["mean_total_time"],
    np.nan,
)

summary["mean_time_per_iteration_ms"] = (
    1000.0
    * summary["mean_total_time"]
    / summary["mean_iterations"]
)

audit_failures = []

if len(raw) != 437:
    audit_failures.append(
        f"Expected 437 final raw rows, observed {len(raw)}."
    )

if len(summary) != 87:
    audit_failures.append(
        f"Expected 87 method configurations, observed {len(summary)}."
    )

if (summary["runs"] < 5).any():
    bad = summary[summary["runs"] < 5][keys + ["runs"]]
    audit_failures.append(
        "Configurations with fewer than five runs:\n"
        + bad.to_string(index=False)
    )

if (summary["iteration_values"] != 1).any():
    bad = summary[
        summary["iteration_values"] != 1
    ][keys + ["iteration_values"]]

    audit_failures.append(
        "Configurations with inconsistent iteration counts:\n"
        + bad.to_string(index=False)
    )

if (summary["max_relative_residual"] > 1.0e-8).any():
    bad = summary[
        summary["max_relative_residual"] > 1.0e-8
    ][keys + ["max_relative_residual"]]

    audit_failures.append(
        "Configurations failing the residual tolerance:\n"
        + bad.to_string(index=False)
    )

if (summary["mean_total_time"] <= 0.0).any():
    audit_failures.append(
        "One or more configurations have non-positive runtime."
    )


# Every physical scaling configuration must contain BJ, AS1 and AS2.
method_coverage = (
    summary.groupby(
        ["scaling", "grid_size", "processes"]
    )["method"]
    .agg(lambda values: set(values))
)

expected_methods = {"BJ", "AS1", "AS2"}

bad_coverage = method_coverage[
    method_coverage.map(lambda values: values != expected_methods)
]

if not bad_coverage.empty:
    audit_failures.append(
        "Incomplete BJ/AS1/AS2 method coverage:\n"
        + bad_coverage.to_string()
    )

if len(method_coverage) != 29:
    audit_failures.append(
        "Expected 29 scaling configurations, observed "
        f"{len(method_coverage)}."
    )


# ---------------------------------------------------------------------
# 3. Verify the official quality file
# ---------------------------------------------------------------------

quality = rename_aliases(pd.read_csv(QUALITY_PATH))
quality["method"] = quality["method"].map(normalise_method)

required_quality = {
    "scaling",
    "grid_size",
    "processes",
    "method",
}

missing_quality = required_quality - set(quality.columns)

if missing_quality:
    raise ValueError(
        "Quality dataset missing columns: "
        f"{sorted(missing_quality)}"
    )

status_column = None

for candidate in [
    "quality_check",
    "status",
    "quality_status",
]:
    if candidate in quality.columns:
        status_column = candidate
        break

if status_column is None:
    raise ValueError(
        "Could not find the quality-status column."
    )

quality[status_column] = (
    quality[status_column].astype(str).str.strip().str.upper()
)

accepted_statuses = {
    "PASS",
    "ACCEPTED",
    "ACCEPTED_AFTER_FOLLOW_UP",
    "STABLE",
}

quality["quality_pass"] = quality[status_column].isin(
    accepted_statuses
)

if quality.duplicated(keys).any():
    audit_failures.append(
        "Duplicate configuration keys found in quality dataset."
    )

summary = summary.merge(
    quality[keys + ["quality_pass", status_column]],
    on=keys,
    how="left",
    validate="one_to_one",
)

if summary["quality_pass"].isna().any():
    bad = summary[
        summary["quality_pass"].isna()
    ][keys]

    audit_failures.append(
        "Configurations missing from quality dataset:\n"
        + bad.to_string(index=False)
    )

if (~summary["quality_pass"].fillna(False)).any():
    bad = summary[
        ~summary["quality_pass"].fillna(False)
    ][keys + [status_column]]

    audit_failures.append(
        "Configurations that did not pass quality checks:\n"
        + bad.to_string(index=False)
    )


if audit_failures:
    print("\nTABLE 3 DATA AUDIT: FAIL\n")

    for failure in audit_failures:
        print(failure)
        print()

    raise SystemExit(1)

print("TABLE 3 DATA AUDIT: PASS")
print(f"Raw rows: {len(raw)}")
print(f"Method configurations: {len(summary)}")
print(f"Scaling configurations: {len(method_coverage)}")
print("All configurations have at least five valid runs.")
print("All configurations passed numerical and quality checks.")


# ---------------------------------------------------------------------
# 4. Match each AS configuration with its BJ baseline
# ---------------------------------------------------------------------

baseline_keys = [
    "scaling",
    "grid_size",
    "processes",
]

bj = summary[summary["method"] == "BJ"].copy()

bj = bj.rename(
    columns={
        "mean_iterations": "bj_iterations",
        "mean_total_time": "bj_total_time",
        "mean_time_per_iteration_ms": "bj_time_per_iteration_ms",
        "cv_pct": "bj_cv_pct",
    }
)

bj = bj[
    baseline_keys
    + [
        "bj_iterations",
        "bj_total_time",
        "bj_time_per_iteration_ms",
        "bj_cv_pct",
    ]
]

overlap = summary[
    summary["method"].isin(["AS1", "AS2"])
].copy()

trade = overlap.merge(
    bj,
    on=baseline_keys,
    how="inner",
    validate="many_to_one",
)

trade["iteration_reduction_pct"] = (
    100.0
    * (
        trade["bj_iterations"]
        - trade["mean_iterations"]
    )
    / trade["bj_iterations"]
)

trade["time_per_iteration_change_pct"] = (
    100.0
    * (
        trade["mean_time_per_iteration_ms"]
        - trade["bj_time_per_iteration_ms"]
    )
    / trade["bj_time_per_iteration_ms"]
)

trade["total_runtime_change_pct"] = (
    100.0
    * (
        trade["mean_total_time"]
        - trade["bj_total_time"]
    )
    / trade["bj_total_time"]
)


# Exact consistency identity:
#
# T_AS / T_BJ =
# (k_AS / k_BJ) *
# ((T_AS / k_AS) / (T_BJ / k_BJ))

runtime_ratio = (
    1.0 + trade["total_runtime_change_pct"] / 100.0
)

factorised_ratio = (
    1.0 - trade["iteration_reduction_pct"] / 100.0
) * (
    1.0
    + trade["time_per_iteration_change_pct"] / 100.0
)

trade["identity_error"] = (
    runtime_ratio - factorised_ratio
).abs()

max_identity_error = trade["identity_error"].max()

if max_identity_error > 1.0e-10:
    raise ValueError(
        "Runtime decomposition identity failed: "
        f"{max_identity_error:.3e}"
    )


trade = trade[
    [
        "scaling",
        "method",
        "grid_size",
        "processes",
        "runs",
        "mean_iterations",
        "bj_iterations",
        "mean_total_time",
        "bj_total_time",
        "mean_time_per_iteration_ms",
        "bj_time_per_iteration_ms",
        "iteration_reduction_pct",
        "time_per_iteration_change_pct",
        "total_runtime_change_pct",
        "cv_pct",
        "bj_cv_pct",
        status_column,
    ]
].sort_values(
    [
        "scaling",
        "grid_size",
        "processes",
        "method",
    ]
)

trade.to_csv(ALL_OUTPUT, index=False)


# ---------------------------------------------------------------------
# 5. Select the extreme and explanatory cases
# ---------------------------------------------------------------------

selected_records = {}


def add_selection(criterion, row):
    key = (
        row["scaling"],
        int(row["grid_size"]),
        int(row["processes"]),
        row["method"],
    )

    if key not in selected_records:
        selected_records[key] = {
            "criteria": [],
            "row": row.copy(),
        }

    selected_records[key]["criteria"].append(criterion)


# Greatest numerical benefit.
add_selection(
    "Largest iteration reduction",
    trade.loc[trade["iteration_reduction_pct"].idxmax()],
)

# Configuration nearest zero runtime change.
add_selection(
    "Closest to runtime break-even",
    trade.loc[
        trade["total_runtime_change_pct"].abs().idxmin()
    ],
)

# Worst runtime cost caused by overlap.
add_selection(
    "Largest runtime increase",
    trade.loc[trade["total_runtime_change_pct"].idxmax()],
)

# Include this criterion only when an actual validated speedup exists.
runtime_reductions = trade[
    trade["total_runtime_change_pct"] < 0.0
]

if not runtime_reductions.empty:
    add_selection(
        "Largest runtime reduction",
        runtime_reductions.loc[
            runtime_reductions[
                "total_runtime_change_pct"
            ].idxmin()
        ],
    )

# Best runtime result for each overlap width.
for method in ["AS1", "AS2"]:
    method_rows = trade[trade["method"] == method]

    add_selection(
        f"Best {method} runtime result",
        method_rows.loc[
            method_rows[
                "total_runtime_change_pct"
            ].idxmin()
        ],
    )


selected_rows = []

for data in selected_records.values():
    row = data["row"].copy()
    row["criterion"] = "; ".join(data["criteria"])
    selected_rows.append(row)

selected = pd.DataFrame(selected_rows)

selected = selected[
    [
        "criterion",
        "scaling",
        "method",
        "grid_size",
        "processes",
        "iteration_reduction_pct",
        "time_per_iteration_change_pct",
        "total_runtime_change_pct",
        "runs",
        "cv_pct",
        "bj_cv_pct",
    ]
]

selected.to_csv(SELECTED_OUTPUT, index=False)


# ---------------------------------------------------------------------
# 6. Generate Overleaf-ready LaTeX
# ---------------------------------------------------------------------

latex_lines = [
    r"\begin{table}[htbp]",
    r"    \centering",
    r"    \caption{Selected extreme effects of Classical Additive Schwarz overlap relative to Block Jacobi. Positive iteration-reduction values indicate fewer Krylov iterations, whereas positive time changes indicate an increased cost relative to Block Jacobi.}",
    r"    \label{tab:overlap-extreme-effects}",
    r"    \small",
    r"    \setlength{\tabcolsep}{4pt}",
    r"    \begin{tabular}{p{4.2cm}llrrrrr}",
    r"        \toprule",
    r"        Criterion & Scaling & Method & $N$ & $p$ & Iteration reduction (\%) & Time/iteration change (\%) & Runtime change (\%) \\",
    r"        \midrule",
]

for row in selected.itertuples(index=False):
    criterion = str(row.criterion).replace("_", r"\_")

    latex_lines.append(
        "        "
        f"{criterion} & "
        f"{row.scaling} & "
        f"{row.method} & "
        f"{int(row.grid_size)} & "
        f"{int(row.processes)} & "
        f"{signed(row.iteration_reduction_pct)} & "
        f"{signed(row.time_per_iteration_change_pct)} & "
        f"{signed(row.total_runtime_change_pct)} "
        r"\\"
    )

latex_lines.extend(
    [
        r"        \bottomrule",
        r"    \end{tabular}",
        r"\end{table}",
        "",
    ]
)

LATEX_OUTPUT.write_text(
    "\n".join(latex_lines),
    encoding="utf-8",
)


# ---------------------------------------------------------------------
# 7. Report
# ---------------------------------------------------------------------

print()
print("ALL AS–BJ COMPARISONS")
print(f"Rows: {len(trade)}")
print(f"Expected rows: 58")
print(f"Maximum identity error: {max_identity_error:.3e}")

print()
print("VALIDATED RUNTIME REDUCTIONS")
print(f"Count: {len(runtime_reductions)}")

if runtime_reductions.empty:
    print(
        "No validated AS configuration was faster than BJ. "
        "Do not include a 'Largest runtime reduction' row."
    )
else:
    print(
        runtime_reductions[
            [
                "scaling",
                "grid_size",
                "processes",
                "method",
                "iteration_reduction_pct",
                "time_per_iteration_change_pct",
                "total_runtime_change_pct",
            ]
        ]
        .sort_values("total_runtime_change_pct")
        .to_string(index=False)
    )

print()
print("SELECTED TABLE 3 ROWS")
print(selected.to_string(index=False))

print()
print(f"Created: {ALL_OUTPUT}")
print(f"Created: {SELECTED_OUTPUT}")
print(f"Created: {LATEX_OUTPUT}")
