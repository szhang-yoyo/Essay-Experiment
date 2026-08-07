from pathlib import Path
import sys

import numpy as np
import pandas as pd


CG_RAW_PATH = Path("results/crossover/cg_canonical_raw.csv")
PCG_RAW_PATH = Path(
    "results/crossover/pcg_canonical_raw_final_validated.csv"
)
POINT_PATH = Path(
    "results/crossover/ratio_point_estimates.csv"
)
OUTPUT_PATH = Path(
    "results/crossover/ratio_bootstrap_ci.csv"
)

BOOTSTRAP_SEED = 20260802
BOOTSTRAP_SAMPLES = 10000
CI_LEVEL = 0.95

EXPECTED_METHODS = ["BJ", "AS1", "AS2"]
EXPECTED_CONFIGURATIONS = 87


def require_column(df, candidates, description):
    for column in candidates:
        if column in df.columns:
            return column
    raise KeyError(
        f"Cannot find {description}. "
        f"Tried {candidates}. "
        f"Available columns: {list(df.columns)}"
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

    if compact == "BJ" or "BLOCKJACOBI" in compact:
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
            overlap = int(float(overlap))
            if overlap == 1:
                return "AS1"
            if overlap == 2:
                return "AS2"

    return text


def prepare_numeric(df, n_col, p_col, time_col, iter_col):
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
        result[iter_col], errors="coerce"
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
        raise ValueError(
            "Invalid numeric input rows:\n"
            + result.loc[
                ~valid,
                [n_col, p_col, time_col, iter_col],
            ].to_string(index=False)
        )

    result["_N"] = result["_N"].astype(int)
    result["_p"] = result["_p"].astype(int)

    return result


def percentile_interval(values):
    alpha = 1.0 - CI_LEVEL
    lower = 100.0 * alpha / 2.0
    upper = 100.0 * (1.0 - alpha / 2.0)

    return np.percentile(values, [lower, upper])


def runtime_conclusion(ci_lower, ci_upper):
    if ci_upper < 1.0:
        return "PCG_faster"
    if ci_lower > 1.0:
        return "CG_faster"
    return "inconclusive"


for path in [CG_RAW_PATH, PCG_RAW_PATH, POINT_PATH]:
    if not path.exists():
        print(f"ERROR: missing input file: {path}")
        sys.exit(1)

cg = pd.read_csv(CG_RAW_PATH)
pcg = pd.read_csv(PCG_RAW_PATH)
points = pd.read_csv(POINT_PATH)

cg_n_col = require_column(
    cg, ["grid_size", "N"], "CG grid-size column"
)
cg_p_col = require_column(
    cg, ["processes", "p", "np"], "CG process column"
)
cg_time_col = require_column(
    cg, ["total_time", "runtime"], "CG runtime column"
)
cg_iter_col = require_column(
    cg, ["iterations", "iters"], "CG iteration column"
)

pcg_scaling_col = require_column(
    pcg, ["scaling", "scaling_type"], "PCG scaling column"
)
pcg_n_col = require_column(
    pcg, ["grid_size", "N"], "PCG grid-size column"
)
pcg_p_col = require_column(
    pcg, ["processes", "p", "np"], "PCG process column"
)
pcg_method_col = require_column(
    pcg, ["method"], "PCG method column"
)
pcg_time_col = require_column(
    pcg, ["total_time", "runtime"], "PCG runtime column"
)
pcg_iter_col = require_column(
    pcg, ["iterations", "iters"], "PCG iteration column"
)
pcg_overlap_col = optional_column(
    pcg, ["overlap", "delta", "overlap_width"]
)

cg = prepare_numeric(
    cg,
    cg_n_col,
    cg_p_col,
    cg_time_col,
    cg_iter_col,
)

pcg = prepare_numeric(
    pcg,
    pcg_n_col,
    pcg_p_col,
    pcg_time_col,
    pcg_iter_col,
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

keys = (
    pcg[
        ["_scaling", "_N", "_p", "_method"]
    ]
    .drop_duplicates()
    .sort_values(
        ["_scaling", "_N", "_p", "_method"]
    )
    .reset_index(drop=True)
)

if len(keys) != EXPECTED_CONFIGURATIONS:
    raise ValueError(
        f"Expected {EXPECTED_CONFIGURATIONS} configurations, "
        f"found {len(keys)}"
    )

seed_sequence = np.random.SeedSequence(BOOTSTRAP_SEED)
child_seeds = seed_sequence.spawn(len(keys))

records = []
maximum_bootstrap_identity_error = 0.0

for row_index, key in keys.iterrows():
    scaling = key["_scaling"]
    grid_size = int(key["_N"])
    processes = int(key["_p"])
    method = key["_method"]

    cg_group = cg[
        (cg["_N"] == grid_size)
        & (cg["_p"] == processes)
    ]

    method_group = pcg[
        (pcg["_scaling"] == scaling)
        & (pcg["_N"] == grid_size)
        & (pcg["_p"] == processes)
        & (pcg["_method"] == method)
    ]

    if cg_group.empty or method_group.empty:
        raise ValueError(
            "Missing raw input for "
            f"{scaling}, N={grid_size}, p={processes}, "
            f"method={method}"
        )

    cg_time = cg_group["_time"].to_numpy(dtype=float)
    cg_iter = cg_group["_iterations"].to_numpy(dtype=float)

    method_time = method_group["_time"].to_numpy(dtype=float)
    method_iter = method_group[
        "_iterations"
    ].to_numpy(dtype=float)

    rng = np.random.default_rng(child_seeds[row_index])

    cg_indices = rng.integers(
        0,
        len(cg_time),
        size=(BOOTSTRAP_SAMPLES, len(cg_time)),
    )

    method_indices = rng.integers(
        0,
        len(method_time),
        size=(BOOTSTRAP_SAMPLES, len(method_time)),
    )

    # Time and iteration count use the same sampled row indices,
    # preserving their within-run pairing.
    cg_time_mean = cg_time[cg_indices].mean(axis=1)
    cg_iter_mean = cg_iter[cg_indices].mean(axis=1)

    method_time_mean = method_time[
        method_indices
    ].mean(axis=1)

    method_iter_mean = method_iter[
        method_indices
    ].mean(axis=1)

    rt_boot = method_time_mean / cg_time_mean
    rk_boot = method_iter_mean / cg_iter_mean

    rc_boot = (
        (method_time_mean / method_iter_mean)
        / (cg_time_mean / cg_iter_mean)
    )

    identity_error = np.max(
        np.abs(rt_boot - rk_boot * rc_boot)
    )

    maximum_bootstrap_identity_error = max(
        maximum_bootstrap_identity_error,
        float(identity_error),
    )

    rt_lower, rt_upper = percentile_interval(rt_boot)
    rk_lower, rk_upper = percentile_interval(rk_boot)
    rc_lower, rc_upper = percentile_interval(rc_boot)

    point_row = points[
        (points["scaling"] == scaling)
        & (points["grid_size"] == grid_size)
        & (points["processes"] == processes)
        & (points["method"] == method)
    ]

    if len(point_row) != 1:
        raise ValueError(
            "Point-estimate key mismatch for "
            f"{scaling}, N={grid_size}, p={processes}, "
            f"method={method}"
        )

    point_row = point_row.iloc[0]

    records.append(
        {
            "scaling": scaling,
            "grid_size": grid_size,
            "processes": processes,
            "method": method,
            "n_CG": len(cg_time),
            "n_M": len(method_time),
            "R_T": float(point_row["R_T"]),
            "R_T_ci_lower": float(rt_lower),
            "R_T_ci_upper": float(rt_upper),
            "R_k": float(point_row["R_k"]),
            "R_k_ci_lower": float(rk_lower),
            "R_k_ci_upper": float(rk_upper),
            "R_C": float(point_row["R_C"]),
            "R_C_ci_lower": float(rc_lower),
            "R_C_ci_upper": float(rc_upper),
            "runtime_conclusion": runtime_conclusion(
                rt_lower, rt_upper
            ),
            "screening_RT_le_1_20": bool(
                point_row["screening_RT_le_1_20"]
            ),
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "ci_level": CI_LEVEL,
        }
    )

result = pd.DataFrame(records)

method_order = pd.CategoricalDtype(
    EXPECTED_METHODS,
    ordered=True,
)

result["method"] = result["method"].astype(method_order)

result = result.sort_values(
    ["scaling", "grid_size", "processes", "method"]
).reset_index(drop=True)

result["method"] = result["method"].astype(str)

finite_columns = [
    "R_T",
    "R_T_ci_lower",
    "R_T_ci_upper",
    "R_k",
    "R_k_ci_lower",
    "R_k_ci_upper",
    "R_C",
    "R_C_ci_lower",
    "R_C_ci_upper",
]

checks = {
    "87 method configurations": (
        len(result) == EXPECTED_CONFIGURATIONS
    ),
    "All bootstrap outputs finite": (
        np.isfinite(result[finite_columns]).all().all()
    ),
    "All RT intervals ordered": (
        (
            result["R_T_ci_lower"]
            <= result["R_T_ci_upper"]
        ).all()
    ),
    "All Rk intervals ordered": (
        (
            result["R_k_ci_lower"]
            <= result["R_k_ci_upper"]
        ).all()
    ),
    "All RC intervals ordered": (
        (
            result["R_C_ci_lower"]
            <= result["R_C_ci_upper"]
        ).all()
    ),
    "All runtime conclusions valid": (
        result["runtime_conclusion"].isin(
            ["PCG_faster", "CG_faster", "inconclusive"]
        ).all()
    ),
    "Bootstrap identity RT = Rk * RC": (
        maximum_bootstrap_identity_error <= 1.0e-12
    ),
}

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

result.to_csv(
    OUTPUT_PATH,
    index=False,
    float_format="%.15g",
)

classification_summary = (
    result.groupby(
        ["scaling", "method", "runtime_conclusion"]
    )
    .size()
    .rename("configurations")
    .reset_index()
)

non_cg_faster = result[
    result["runtime_conclusion"] != "CG_faster"
][
    [
        "scaling",
        "grid_size",
        "processes",
        "method",
        "R_T",
        "R_T_ci_lower",
        "R_T_ci_upper",
        "runtime_conclusion",
    ]
]

print("=" * 78)
print("RATIO BOOTSTRAP AUDIT")
print("=" * 78)
print(f"Bootstrap seed:             {BOOTSTRAP_SEED}")
print(f"Bootstrap samples:          {BOOTSTRAP_SAMPLES}")
print(f"Confidence level:           {CI_LEVEL:.0%}")
print(f"Method configurations:      {len(result)}")
print(
    "Maximum identity error:    "
    f"{maximum_bootstrap_identity_error:.3e}"
)
print()

print("CHECKS")
for name, passed in checks.items():
    print(f"{'PASS' if passed else 'FAIL':4s}  {name}")

print()
print("RUNTIME CONCLUSION SUMMARY")
print(classification_summary.to_string(index=False))

print()
print("CONFIGURATIONS NOT CLASSIFIED AS CG_FASTER")
if non_cg_faster.empty:
    print("None")
else:
    print(non_cg_faster.to_string(index=False))

print()
print("SCREENING STATUS")
print(
    "Configurations with RT <= 1.20: "
    f"{int(result['screening_RT_le_1_20'].sum())}"
)
print(
    "Note: screening_RT_le_1_20 is not a "
    "statistical conclusion."
)

print()
print(f"Output: {OUTPUT_PATH}")

overall_pass = all(checks.values())

print(
    "OVERALL RESULT: "
    f"{'PASS' if overall_pass else 'FAIL'}"
)

sys.exit(0 if overall_pass else 1)
