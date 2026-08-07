from pathlib import Path
import sys

import numpy as np
import pandas as pd


INPUT_PATH = Path(
    "results/crossover/ratio_bootstrap_ci.csv"
)
FULL_OUTPUT_PATH = Path(
    "results/crossover/crossover_boundary_audit.csv"
)
NEAREST_OUTPUT_PATH = Path(
    "results/crossover/crossover_nearest_cases.csv"
)

EXPECTED_CONFIGURATIONS = 87
EXPECTED_METHODS = ["BJ", "AS1", "AS2"]
NEAREST_COUNT = 12


if not INPUT_PATH.exists():
    print(f"ERROR: missing input file: {INPUT_PATH}")
    sys.exit(1)

df = pd.read_csv(INPUT_PATH)

required_columns = [
    "scaling",
    "grid_size",
    "processes",
    "method",
    "R_T",
    "R_T_ci_lower",
    "R_T_ci_upper",
    "R_k",
    "R_k_ci_lower",
    "R_k_ci_upper",
    "R_C",
    "R_C_ci_lower",
    "R_C_ci_upper",
    "runtime_conclusion",
    "screening_RT_le_1_20",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise KeyError(
        f"Missing required columns: {missing_columns}"
    )

numeric_columns = [
    "grid_size",
    "processes",
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

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column], errors="coerce"
    )

if df[numeric_columns].isna().any().any():
    bad_rows = df[
        df[numeric_columns].isna().any(axis=1)
    ]
    raise ValueError(
        "Non-numeric or missing ratio values detected:\n"
        + bad_rows.to_string(index=False)
    )

df["scaling"] = (
    df["scaling"].astype(str).str.strip().str.lower()
)
df["method"] = (
    df["method"].astype(str).str.strip().str.upper()
)

unexpected_methods = sorted(
    set(df["method"]) - set(EXPECTED_METHODS)
)

if unexpected_methods:
    raise ValueError(
        f"Unexpected methods: {unexpected_methods}"
    )

# Percentage interpretation of the required ratios.
df["runtime_penalty_pct"] = (
    (df["R_T"] - 1.0) * 100.0
)

df["iteration_change_pct"] = (
    (df["R_k"] - 1.0) * 100.0
)

df["iteration_reduction_pct"] = (
    (1.0 - df["R_k"]) * 100.0
)

df["per_iteration_cost_change_pct"] = (
    (df["R_C"] - 1.0) * 100.0
)

df["ci_lower_runtime_penalty_pct"] = (
    (df["R_T_ci_lower"] - 1.0) * 100.0
)

# A configuration is closer to crossover when RT is closer to one.
df["distance_from_crossover"] = np.abs(
    df["R_T"] - 1.0
)

# Interpret the point-estimate decomposition.
df["decomposition_pattern"] = np.select(
    [
        (df["R_k"] < 1.0) & (df["R_C"] > 1.0),
        (df["R_k"] >= 1.0) & (df["R_C"] > 1.0),
        (df["R_k"] < 1.0) & (df["R_C"] <= 1.0),
        (df["R_k"] >= 1.0) & (df["R_C"] <= 1.0),
    ],
    [
        "iteration_gain_but_per_iteration_penalty",
        "no_iteration_gain_and_per_iteration_penalty",
        "iteration_gain_and_no_per_iteration_penalty",
        "no_iteration_gain_but_no_per_iteration_penalty",
    ],
    default="unclassified",
)

method_order = pd.CategoricalDtype(
    EXPECTED_METHODS,
    ordered=True,
)

df["method"] = df["method"].astype(method_order)

df = df.sort_values(
    [
        "distance_from_crossover",
        "scaling",
        "grid_size",
        "processes",
        "method",
    ]
).reset_index(drop=True)

df["crossover_rank"] = np.arange(1, len(df) + 1)

nearest = df.head(NEAREST_COUNT).copy()

nearest_by_group = (
    df.sort_values("R_T")
    .groupby(
        ["scaling", "method"],
        observed=True,
        as_index=False,
    )
    .first()
)

pattern_summary = (
    df.groupby(
        ["scaling", "method", "decomposition_pattern"],
        observed=True,
    )
    .size()
    .rename("configurations")
    .reset_index()
)

checks = {
    "87 method configurations": (
        len(df) == EXPECTED_CONFIGURATIONS
    ),
    "No duplicate configuration keys": (
        not df.duplicated(
            [
                "scaling",
                "grid_size",
                "processes",
                "method",
            ]
        ).any()
    ),
    "All classifications are CG_faster": (
        df["runtime_conclusion"]
        .eq("CG_faster")
        .all()
    ),
    "All RT confidence intervals are above 1": (
        (df["R_T_ci_lower"] > 1.0).all()
    ),
    "All RT point estimates exceed 1.20": (
        (df["R_T"] > 1.20).all()
    ),
    "No screening candidates": (
        not df["screening_RT_le_1_20"]
        .astype(bool)
        .any()
    ),
    "Required identity holds": (
        np.allclose(
            df["R_T"],
            df["R_k"] * df["R_C"],
            rtol=1.0e-12,
            atol=1.0e-12,
        )
    ),
}

output_columns = [
    "crossover_rank",
    "scaling",
    "grid_size",
    "processes",
    "method",
    "R_T",
    "R_T_ci_lower",
    "R_T_ci_upper",
    "runtime_penalty_pct",
    "ci_lower_runtime_penalty_pct",
    "R_k",
    "R_k_ci_lower",
    "R_k_ci_upper",
    "iteration_change_pct",
    "iteration_reduction_pct",
    "R_C",
    "R_C_ci_lower",
    "R_C_ci_upper",
    "per_iteration_cost_change_pct",
    "decomposition_pattern",
    "runtime_conclusion",
    "screening_RT_le_1_20",
]

df["method"] = df["method"].astype(str)
nearest["method"] = nearest["method"].astype(str)
nearest_by_group["method"] = (
    nearest_by_group["method"].astype(str)
)
pattern_summary["method"] = (
    pattern_summary["method"].astype(str)
)

FULL_OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

df[output_columns].to_csv(
    FULL_OUTPUT_PATH,
    index=False,
    float_format="%.15g",
)

nearest[output_columns].to_csv(
    NEAREST_OUTPUT_PATH,
    index=False,
    float_format="%.15g",
)

print("=" * 88)
print("CROSSOVER BOUNDARY AUDIT")
print("=" * 88)
print(f"Method configurations: {len(df)}")
print()

print("CHECKS")
for name, passed in checks.items():
    print(f"{'PASS' if passed else 'FAIL':4s}  {name}")

print()
print("NEAREST CONFIGURATION FOR EACH SCALING–METHOD GROUP")
print(
    nearest_by_group[
        [
            "scaling",
            "method",
            "grid_size",
            "processes",
            "R_T",
            "R_T_ci_lower",
            "R_T_ci_upper",
            "R_k",
            "R_C",
            "runtime_penalty_pct",
            "iteration_reduction_pct",
            "per_iteration_cost_change_pct",
        ]
    ].to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}",
    )
)

print()
print(f"{NEAREST_COUNT} CONFIGURATIONS CLOSEST TO RT = 1")
print(
    nearest[
        [
            "crossover_rank",
            "scaling",
            "grid_size",
            "processes",
            "method",
            "R_T",
            "R_T_ci_lower",
            "R_T_ci_upper",
            "R_k",
            "R_C",
            "runtime_penalty_pct",
            "iteration_reduction_pct",
            "per_iteration_cost_change_pct",
            "decomposition_pattern",
        ]
    ].to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}",
    )
)

print()
print("DECOMPOSITION PATTERN SUMMARY")
print(pattern_summary.to_string(index=False))

print()
print(
    "Minimum point-estimate RT: "
    f"{df['R_T'].min():.6f}"
)
print(
    "Minimum RT confidence lower bound: "
    f"{df['R_T_ci_lower'].min():.6f}"
)
print(
    "Maximum iteration reduction: "
    f"{df['iteration_reduction_pct'].max():.2f}%"
)
print(
    "Minimum per-iteration cost increase: "
    f"{df['per_iteration_cost_change_pct'].min():.2f}%"
)

print()
print(f"Full output:    {FULL_OUTPUT_PATH}")
print(f"Nearest cases:  {NEAREST_OUTPUT_PATH}")

overall_pass = all(checks.values())

print(
    "OVERALL RESULT: "
    f"{'PASS' if overall_pass else 'FAIL'}"
)

sys.exit(0 if overall_pass else 1)
