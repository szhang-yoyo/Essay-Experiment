from pathlib import Path
import sys

import numpy as np
import pandas as pd


CG_RAW_PATH = Path("results/crossover/cg_canonical_raw.csv")
PCG_RAW_PATH = Path(
    "results/crossover/pcg_canonical_raw_final_validated.csv"
)
OUTPUT_PATH = Path(
    "results/crossover/ratio_point_estimates.csv"
)

EXPECTED_SCALING_CONFIGS = 29
EXPECTED_METHOD_CONFIGS = 87
EXPECTED_METHODS = ["BJ", "AS1", "AS2"]

IDENTITY_RTOL = 1.0e-12
IDENTITY_ATOL = 1.0e-12


def require_column(df, candidates, description):
    for column in candidates:
        if column in df.columns:
            return column

    raise KeyError(
        f"Cannot find {description}. "
        f"Tried {candidates}. Available columns: {list(df.columns)}"
    )


def optional_column(df, candidates):
    for column in candidates:
        if column in df.columns:
            return column
    return None


def normalise_method(value, overlap=None):
    text = str(value).strip().upper()
    compact = (
        text.replace("-", "")
        .replace("_", "")
        .replace(" ", "")
    )

    if compact == "CG":
        return "CG"

    if "BLOCKJACOBI" in compact or compact == "BJ":
        return "BJ"

    if compact == "AS1":
        return "AS1"

    if compact == "AS2":
        return "AS2"

    if (
        "ADDITIVESCHWARZ" in compact
        or "CLASSICALAS" in compact
        or compact in {"AS", "CAS"}
    ):
        if overlap is not None and not pd.isna(overlap):
            overlap_value = int(float(overlap))
            if overlap_value == 1:
                return "AS1"
            if overlap_value == 2:
                return "AS2"

    return text


def prepare_numeric_columns(
    df,
    n_col,
    p_col,
    time_col,
    iteration_col,
):
    result = df.copy()

    result["_N"] = pd.to_numeric(
        result[n_col], errors="coerce"
    )
    result["_p"] = pd.to_numeric(
        result[p_col], errors="coerce"
    )
    result["_time"] = pd.to_numeric(
        result[time_col], errors="coerce"
    )
    result["_iterations"] = pd.to_numeric(
        result[iteration_col], errors="coerce"
    )

    valid = (
        result["_N"].notna()
        & result["_p"].notna()
        & result["_time"].notna()
        & result["_iterations"].notna()
        & np.isfinite(result["_time"])
        & np.isfinite(result["_iterations"])
        & (result["_N"] > 0)
        & (result["_p"] > 0)
        & (result["_time"] > 0)
        & (result["_iterations"] > 0)
    )

    if not valid.all():
        bad_rows = result.loc[
            ~valid,
            [n_col, p_col, time_col, iteration_col],
        ]
        raise ValueError(
            "Invalid numeric ratio inputs detected:\n"
            + bad_rows.to_string(index=False)
        )

    result["_N"] = result["_N"].astype(int)
    result["_p"] = result["_p"].astype(int)

    return result


for input_path in [CG_RAW_PATH, PCG_RAW_PATH]:
    if not input_path.exists():
        print(f"ERROR: missing input file: {input_path}")
        sys.exit(1)

cg = pd.read_csv(CG_RAW_PATH)
pcg = pd.read_csv(PCG_RAW_PATH)

cg_n_col = require_column(
    cg, ["grid_size", "N"], "CG grid-size column"
)
cg_p_col = require_column(
    cg,
    ["processes", "p", "np", "nprocs"],
    "CG process-count column",
)
cg_time_col = require_column(
    cg,
    ["total_time", "solve_time", "runtime"],
    "CG runtime column",
)
cg_iteration_col = require_column(
    cg,
    ["iterations", "iteration_count", "iters"],
    "CG iteration-count column",
)

pcg_scaling_col = require_column(
    pcg,
    ["scaling", "scaling_type"],
    "PCG scaling column",
)
pcg_n_col = require_column(
    pcg, ["grid_size", "N"], "PCG grid-size column"
)
pcg_p_col = require_column(
    pcg,
    ["processes", "p", "np", "nprocs"],
    "PCG process-count column",
)
pcg_method_col = require_column(
    pcg, ["method"], "PCG method column"
)
pcg_time_col = require_column(
    pcg,
    ["total_time", "solve_time", "runtime"],
    "PCG runtime column",
)
pcg_iteration_col = require_column(
    pcg,
    ["iterations", "iteration_count", "iters"],
    "PCG iteration-count column",
)
pcg_overlap_col = optional_column(
    pcg,
    ["overlap", "delta", "overlap_width"],
)

cg = prepare_numeric_columns(
    cg,
    cg_n_col,
    cg_p_col,
    cg_time_col,
    cg_iteration_col,
)

pcg = prepare_numeric_columns(
    pcg,
    pcg_n_col,
    pcg_p_col,
    pcg_time_col,
    pcg_iteration_col,
)

pcg["_scaling"] = (
    pcg[pcg_scaling_col]
    .astype(str)
    .str.strip()
    .str.lower()
)

if pcg_overlap_col is None:
    pcg["_method"] = pcg[pcg_method_col].map(
        normalise_method
    )
else:
    pcg["_method"] = [
        normalise_method(method, overlap)
        for method, overlap in zip(
            pcg[pcg_method_col],
            pcg[pcg_overlap_col],
        )
    ]

unexpected_methods = sorted(
    set(pcg["_method"]) - set(EXPECTED_METHODS)
)

if unexpected_methods:
    raise ValueError(
        f"Unexpected PCG methods: {unexpected_methods}"
    )

cg_summary = (
    cg.groupby(["_N", "_p"], as_index=False)
    .agg(
        n_CG=("_time", "size"),
        T_CG_mean_s=("_time", "mean"),
        k_CG_mean=("_iterations", "mean"),
    )
)

pcg_summary = (
    pcg.groupby(
        ["_scaling", "_N", "_p", "_method"],
        as_index=False,
    )
    .agg(
        n_M=("_time", "size"),
        T_M_mean_s=("_time", "mean"),
        k_M_mean=("_iterations", "mean"),
    )
)

ratios = pcg_summary.merge(
    cg_summary,
    on=["_N", "_p"],
    how="left",
    validate="many_to_one",
)

required_columns = [
    "n_CG",
    "T_CG_mean_s",
    "k_CG_mean",
]

if ratios[required_columns].isna().any().any():
    missing = ratios[
        ratios[required_columns].isna().any(axis=1)
    ]

    raise ValueError(
        "PCG configurations without matching CG input:\n"
        + missing.to_string(index=False)
    )

ratios["C_CG_s_per_iter"] = (
    ratios["T_CG_mean_s"]
    / ratios["k_CG_mean"]
)

ratios["C_M_s_per_iter"] = (
    ratios["T_M_mean_s"]
    / ratios["k_M_mean"]
)

# Required definitions:
#
# RT = T_M / T_CG
# Rk = k_M / k_CG
# RC = (T_M / k_M) / (T_CG / k_CG)

ratios["R_T"] = (
    ratios["T_M_mean_s"]
    / ratios["T_CG_mean_s"]
)

ratios["R_k"] = (
    ratios["k_M_mean"]
    / ratios["k_CG_mean"]
)

ratios["R_C"] = (
    ratios["C_M_s_per_iter"]
    / ratios["C_CG_s_per_iter"]
)

ratios["R_k_times_R_C"] = (
    ratios["R_k"] * ratios["R_C"]
)

ratios["identity_abs_error"] = np.abs(
    ratios["R_T"] - ratios["R_k_times_R_C"]
)

ratios["identity_pass"] = np.isclose(
    ratios["R_T"],
    ratios["R_k_times_R_C"],
    rtol=IDENTITY_RTOL,
    atol=IDENTITY_ATOL,
)

# Screening flag only. This is not a statistical conclusion.
ratios["screening_RT_le_1_20"] = (
    ratios["R_T"] <= 1.20
)

ratios = ratios.rename(
    columns={
        "_scaling": "scaling",
        "_N": "grid_size",
        "_p": "processes",
        "_method": "method",
    }
)

method_order = pd.CategoricalDtype(
    categories=EXPECTED_METHODS,
    ordered=True,
)

ratios["method"] = ratios["method"].astype(method_order)

ratios = ratios.sort_values(
    ["scaling", "grid_size", "processes", "method"]
).reset_index(drop=True)

ratios["method"] = ratios["method"].astype(str)

output_columns = [
    "scaling",
    "grid_size",
    "processes",
    "method",
    "n_CG",
    "n_M",
    "T_CG_mean_s",
    "T_M_mean_s",
    "k_CG_mean",
    "k_M_mean",
    "C_CG_s_per_iter",
    "C_M_s_per_iter",
    "R_T",
    "R_k",
    "R_C",
    "R_k_times_R_C",
    "identity_abs_error",
    "identity_pass",
    "screening_RT_le_1_20",
]

ratios = ratios[output_columns]

method_config_count = len(ratios)

scaling_config_count = (
    ratios[
        ["scaling", "grid_size", "processes"]
    ]
    .drop_duplicates()
    .shape[0]
)

duplicate_keys = ratios.duplicated(
    ["scaling", "grid_size", "processes", "method"],
    keep=False,
)

group_counts = (
    ratios.groupby(["scaling", "method"])
    .size()
    .rename("configurations")
    .reset_index()
)

maximum_identity_error = (
    ratios["identity_abs_error"].max()
)

all_identity_pass = bool(
    ratios["identity_pass"].all()
)

checks = {
    "29 scaling configurations": (
        scaling_config_count
        == EXPECTED_SCALING_CONFIGS
    ),
    "87 method configurations": (
        method_config_count
        == EXPECTED_METHOD_CONFIGS
    ),
    "No duplicate method keys": (
        not duplicate_keys.any()
    ),
    "All ratio values finite": (
        np.isfinite(
            ratios[["R_T", "R_k", "R_C"]]
        ).all().all()
    ),
    "All ratio values positive": (
        (
            ratios[["R_T", "R_k", "R_C"]]
            > 0
        ).all().all()
    ),
    "RT = Rk * RC for all configurations": (
        all_identity_pass
    ),
}

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

ratios.to_csv(
    OUTPUT_PATH,
    index=False,
    float_format="%.15g",
)

print("=" * 78)
print("RATIO POINT-ESTIMATE AUDIT")
print("=" * 78)
print(
    f"Scaling configurations:       "
    f"{scaling_config_count}"
)
print(
    f"Method configurations:        "
    f"{method_config_count}"
)
print(
    f"Maximum identity abs. error:  "
    f"{maximum_identity_error:.3e}"
)
print()

print("CONFIGURATION COUNTS")
print(group_counts.to_string(index=False))
print()

print("CHECKS")
for name, passed in checks.items():
    print(
        f"{'PASS' if passed else 'FAIL':4s}  "
        f"{name}"
    )

print()
print("RUN-COUNT EXCEPTIONS")
run_count_exceptions = ratios[
    (ratios["n_CG"] != 5)
    | (ratios["n_M"] != 5)
][
    [
        "scaling",
        "grid_size",
        "processes",
        "method",
        "n_CG",
        "n_M",
    ]
]

if run_count_exceptions.empty:
    print("None")
else:
    print(
        run_count_exceptions.to_string(index=False)
    )

print()
print("SCREENING SUMMARY — RT <= 1.20 ONLY")
screening_summary = (
    ratios.groupby(["scaling", "method"])
    ["screening_RT_le_1_20"]
    .agg(["sum", "count"])
    .reset_index()
    .rename(
        columns={
            "sum": "screened_in",
            "count": "total",
        }
    )
)

print(screening_summary.to_string(index=False))

print()
print(f"Output: {OUTPUT_PATH}")

overall_pass = all(checks.values())

print(
    f"OVERALL RESULT: "
    f"{'PASS' if overall_pass else 'FAIL'}"
)

sys.exit(0 if overall_pass else 1)
