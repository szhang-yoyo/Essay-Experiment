from pathlib import Path
import sys

import numpy as np
import pandas as pd


CG_RAW_PATH = Path("results/crossover/cg_canonical_raw.csv")
PCG_RAW_PATH = Path(
    "results/crossover/pcg_canonical_raw_final_validated.csv"
)
OUTPUT_PATH = Path("results/crossover/ratio_input_audit.csv")

EXPECTED_SCALING_CONFIGS = 29
EXPECTED_CG_CONFIGS = 26
EXPECTED_PCG_METHOD_CONFIGS = 87
EXPECTED_METHODS = ["BJ", "AS1", "AS2"]
MIN_VALID_RUNS = 5


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

    if "BJ" in compact or "BLOCKJACOBI" in compact:
        return "BJ"

    if "AS1" in compact:
        return "AS1"

    if "AS2" in compact:
        return "AS2"

    if "ADDITIVESCHWARZ" in compact or compact in {"AS", "CAS"}:
        if overlap is not None and not pd.isna(overlap):
            overlap_value = int(float(overlap))
            if overlap_value == 1:
                return "AS1"
            if overlap_value == 2:
                return "AS2"

    if compact == "CG":
        return "CG"

    return text


def prepare_numeric_columns(df, n_col, p_col, time_col, iter_col):
    result = df.copy()

    result["_N"] = pd.to_numeric(result[n_col], errors="coerce")
    result["_p"] = pd.to_numeric(result[p_col], errors="coerce")
    result["_time"] = pd.to_numeric(result[time_col], errors="coerce")
    result["_iterations"] = pd.to_numeric(
        result[iter_col], errors="coerce"
    )

    result["_numeric_valid"] = (
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

    result["_N"] = result["_N"].astype("Int64")
    result["_p"] = result["_p"].astype("Int64")

    return result


for path in [CG_RAW_PATH, PCG_RAW_PATH]:
    if not path.exists():
        print(f"ERROR: missing input file: {path}")
        sys.exit(1)

cg = pd.read_csv(CG_RAW_PATH)
pcg = pd.read_csv(PCG_RAW_PATH)

cg_n_col = require_column(
    cg, ["grid_size", "N"], "CG grid-size column"
)
cg_p_col = require_column(
    cg, ["processes", "p", "np", "nprocs"], "CG process-count column"
)
cg_time_col = require_column(
    cg, ["total_time", "solve_time", "runtime"], "CG runtime column"
)
cg_iter_col = require_column(
    cg, ["iterations", "iteration_count", "iters"],
    "CG iteration-count column",
)

pcg_scaling_col = require_column(
    pcg, ["scaling", "scaling_type"], "PCG scaling column"
)
pcg_n_col = require_column(
    pcg, ["grid_size", "N"], "PCG grid-size column"
)
pcg_p_col = require_column(
    pcg, ["processes", "p", "np", "nprocs"],
    "PCG process-count column",
)
pcg_method_col = require_column(
    pcg, ["method"], "PCG method column"
)
pcg_time_col = require_column(
    pcg, ["total_time", "solve_time", "runtime"],
    "PCG runtime column",
)
pcg_iter_col = require_column(
    pcg, ["iterations", "iteration_count", "iters"],
    "PCG iteration-count column",
)
pcg_overlap_col = optional_column(
    pcg, ["overlap", "delta", "overlap_width"]
)

cg = prepare_numeric_columns(
    cg, cg_n_col, cg_p_col, cg_time_col, cg_iter_col
)
pcg = prepare_numeric_columns(
    pcg, pcg_n_col, pcg_p_col, pcg_time_col, pcg_iter_col
)

pcg["_scaling"] = (
    pcg[pcg_scaling_col].astype(str).str.strip().str.lower()
)

if pcg_overlap_col is None:
    pcg["_method"] = pcg[pcg_method_col].map(normalise_method)
else:
    pcg["_method"] = [
        normalise_method(method, overlap)
        for method, overlap in zip(
            pcg[pcg_method_col], pcg[pcg_overlap_col]
        )
    ]

cg_method_col = optional_column(cg, ["method"])
if cg_method_col is not None:
    cg["_method"] = cg[cg_method_col].map(normalise_method)
    unexpected_cg_methods = sorted(
        set(cg["_method"].dropna()) - {"CG"}
    )
else:
    unexpected_cg_methods = []

missing_key_rows_cg = int(
    (cg["_N"].isna() | cg["_p"].isna()).sum()
)
missing_key_rows_pcg = int(
    (
        pcg["_scaling"].eq("")
        | pcg["_N"].isna()
        | pcg["_p"].isna()
        | pcg["_method"].isna()
    ).sum()
)

invalid_numeric_cg = int((~cg["_numeric_valid"]).sum())
invalid_numeric_pcg = int((~pcg["_numeric_valid"]).sum())

unexpected_pcg_methods = sorted(
    set(pcg["_method"].dropna()) - set(EXPECTED_METHODS)
)

pcg_configs = (
    pcg[["_scaling", "_N", "_p"]]
    .drop_duplicates()
    .sort_values(["_scaling", "_N", "_p"])
    .reset_index(drop=True)
)

cg_counts = (
    cg[cg["_numeric_valid"]]
    .groupby(["_N", "_p"])
    .size()
    .rename("n_CG")
    .reset_index()
)

pcg_expected = pcg[
    pcg["_numeric_valid"]
    & pcg["_method"].isin(EXPECTED_METHODS)
]

pcg_counts = (
    pcg_expected
    .groupby(["_scaling", "_N", "_p", "_method"])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=EXPECTED_METHODS, fill_value=0)
    .rename(
        columns={
            "BJ": "n_BJ",
            "AS1": "n_AS1",
            "AS2": "n_AS2",
        }
    )
    .reset_index()
)

audit = pcg_configs.merge(
    cg_counts,
    on=["_N", "_p"],
    how="left",
    validate="many_to_one",
)

audit = audit.merge(
    pcg_counts,
    on=["_scaling", "_N", "_p"],
    how="left",
    validate="one_to_one",
)

for column in ["n_CG", "n_BJ", "n_AS1", "n_AS2"]:
    audit[column] = audit[column].fillna(0).astype(int)

reuse_counts = (
    pcg_configs
    .groupby(["_N", "_p"])["_scaling"]
    .nunique()
    .rename("scaling_uses")
    .reset_index()
)

audit = audit.merge(
    reuse_counts,
    on=["_N", "_p"],
    how="left",
    validate="many_to_one",
)

audit["cg_reused_across_scaling"] = audit["scaling_uses"] > 1

audit["complete"] = (
    (audit["n_CG"] >= MIN_VALID_RUNS)
    & (audit["n_BJ"] >= MIN_VALID_RUNS)
    & (audit["n_AS1"] >= MIN_VALID_RUNS)
    & (audit["n_AS2"] >= MIN_VALID_RUNS)
)

audit = audit.rename(
    columns={
        "_scaling": "scaling",
        "_N": "grid_size",
        "_p": "processes",
    }
)

audit = audit[
    [
        "scaling",
        "grid_size",
        "processes",
        "n_CG",
        "n_BJ",
        "n_AS1",
        "n_AS2",
        "scaling_uses",
        "cg_reused_across_scaling",
        "complete",
    ]
].sort_values(["scaling", "grid_size", "processes"])

cg_config_count = int(
    cg[["_N", "_p"]].drop_duplicates().shape[0]
)

pcg_scaling_config_count = int(pcg_configs.shape[0])

pcg_method_config_count = int(
    pcg[
        pcg["_method"].isin(EXPECTED_METHODS)
    ][["_scaling", "_N", "_p", "_method"]]
    .drop_duplicates()
    .shape[0]
)

reused_pairs = (
    audit[audit["cg_reused_across_scaling"]]
    [["grid_size", "processes", "scaling_uses"]]
    .drop_duplicates()
    .sort_values(["grid_size", "processes"])
)

checks = {
    "CG unique (N,p) configurations": (
        cg_config_count == EXPECTED_CG_CONFIGS
    ),
    "PCG scaling configurations": (
        pcg_scaling_config_count == EXPECTED_SCALING_CONFIGS
    ),
    "PCG method configurations": (
        pcg_method_config_count == EXPECTED_PCG_METHOD_CONFIGS
    ),
    "No missing CG keys": missing_key_rows_cg == 0,
    "No missing PCG keys": missing_key_rows_pcg == 0,
    "All CG numeric inputs valid": invalid_numeric_cg == 0,
    "All PCG numeric inputs valid": invalid_numeric_pcg == 0,
    "Only CG appears in CG input": not unexpected_cg_methods,
    "Only BJ/AS1/AS2 appear in PCG input": not unexpected_pcg_methods,
    "All 29 configurations have >=5 runs per method": (
        bool(audit["complete"].all())
    ),
}

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
audit.to_csv(OUTPUT_PATH, index=False)

print("=" * 78)
print("RATIO INPUT AUDIT")
print("=" * 78)
print(f"CG raw rows:                 {len(cg)}")
print(f"CG unique (N,p):             {cg_config_count}")
print(f"PCG raw rows:                {len(pcg)}")
print(f"PCG scaling configurations:  {pcg_scaling_config_count}")
print(f"PCG method configurations:   {pcg_method_config_count}")
print()

print("CHECKS")
for name, passed in checks.items():
    print(f"{'PASS' if passed else 'FAIL':4s}  {name}")

print()
print("CONFIGURATION RUN COUNTS")
print(audit.to_string(index=False))

print()
print("CG INPUTS REUSED ACROSS STRONG/WEAK CONFIGURATIONS")
if reused_pairs.empty:
    print("None")
else:
    print(reused_pairs.to_string(index=False))

failed = audit[~audit["complete"]]

print()
if failed.empty:
    print("Incomplete configurations: 0")
else:
    print(f"Incomplete configurations: {len(failed)}")
    print(failed.to_string(index=False))

overall_pass = all(checks.values())

print()
print(f"Audit file: {OUTPUT_PATH}")
print(f"OVERALL RESULT: {'PASS' if overall_pass else 'FAIL'}")

sys.exit(0 if overall_pass else 1)
