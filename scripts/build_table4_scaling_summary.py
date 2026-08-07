from pathlib import Path

import numpy as np
import pandas as pd


PCG_RAW = Path(
    "results/crossover/pcg_canonical_raw_final_validated.csv"
)
PCG_QUALITY = Path(
    "results/crossover/pcg_quality_final_validated.csv"
)
CG_RAW = Path(
    "results/crossover/cg_canonical_raw.csv"
)
CG_QUALITY = Path(
    "results/crossover/cg_quality.csv"
)

OUT_DIR = Path("results/tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_OUTPUT = OUT_DIR / "table4_scaling_all.csv"
STRONG_OUTPUT = OUT_DIR / "table4_strong_summary.csv"
WEAK_OUTPUT = OUT_DIR / "table4_weak_summary.csv"
LATEX_OUTPUT = OUT_DIR / "table4_scaling_summary.tex"

METHOD_ORDER = ["CG", "BJ", "AS1", "AS2"]


def rename_aliases(df):
    aliases = {
        "n": "grid_size",
        "p": "processes",
        "runtime": "total_time",
        "residual": "relative_residual",
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

    if value == "CG":
        return "CG"

    if value.startswith("BJ"):
        return "BJ"

    if value in {"AS1", "AS2"}:
        return value

    raise ValueError(f"Unknown method: {value}")


def format_time_pair(t1, tp):
    return f"{t1:.3f}--{tp:.3f}"


def format_iteration_pair(k1, kp):
    return f"{int(round(k1))}--{int(round(kp))}"


for path in [PCG_RAW, PCG_QUALITY, CG_RAW, CG_QUALITY]:
    if not path.exists():
        raise FileNotFoundError(path)


# ------------------------------------------------------------------
# 1. Load final validated PCG data
# ------------------------------------------------------------------

pcg = rename_aliases(pd.read_csv(PCG_RAW))
pcg["method"] = pcg["method"].map(normalise_method)

required_pcg = {
    "scaling",
    "grid_size",
    "processes",
    "method",
    "iterations",
    "relative_residual",
    "total_time",
}

missing_pcg = required_pcg - set(pcg.columns)

if missing_pcg:
    raise ValueError(
        f"PCG raw data missing columns: {sorted(missing_pcg)}"
    )

pcg = pcg[
    pcg["method"].isin(["BJ", "AS1", "AS2"])
].copy()


# The PCG data defines the 29 tested scaling configurations.
scale_map = (
    pcg[
        ["scaling", "grid_size", "processes"]
    ]
    .drop_duplicates()
    .sort_values(
        ["scaling", "grid_size", "processes"]
    )
)

if len(scale_map) != 29:
    raise ValueError(
        f"Expected 29 scaling configurations, found {len(scale_map)}"
    )


# ------------------------------------------------------------------
# 2. Load CG data and attach strong/weak labels
# ------------------------------------------------------------------

cg = rename_aliases(pd.read_csv(CG_RAW))

if "method" not in cg.columns:
    cg["method"] = "CG"
else:
    cg["method"] = cg["method"].map(normalise_method)

required_cg = {
    "grid_size",
    "processes",
    "method",
    "iterations",
    "relative_residual",
    "total_time",
}

missing_cg = required_cg - set(cg.columns)

if missing_cg:
    raise ValueError(
        f"CG raw data missing columns: {sorted(missing_cg)}"
    )

cg = cg[cg["method"] == "CG"].copy()

# Remove any pre-existing scaling label and reconstruct it using the
# final PCG configuration map. Physical configurations that belong to
# both strong and weak scaling are intentionally used in both groups.
cg = cg.drop(columns=["scaling"], errors="ignore")

cg = scale_map.merge(
    cg,
    on=["grid_size", "processes"],
    how="left",
    validate="many_to_many",
)

if cg["total_time"].isna().any():
    missing = (
        cg[cg["total_time"].isna()][
            ["scaling", "grid_size", "processes"]
        ]
        .drop_duplicates()
    )

    raise ValueError(
        "Missing CG data for scaling configurations:\n"
        + missing.to_string(index=False)
    )


# ------------------------------------------------------------------
# 3. Summarise raw runs
# ------------------------------------------------------------------

keys = [
    "scaling",
    "grid_size",
    "processes",
    "method",
]


def summarise(df):
    return (
        df.groupby(keys, as_index=False)
        .agg(
            runs=("total_time", "size"),
            mean_iterations=("iterations", "mean"),
            iteration_values=("iterations", "nunique"),
            max_relative_residual=(
                "relative_residual",
                "max",
            ),
            mean_total_time=("total_time", "mean"),
            std_total_time=("total_time", "std"),
        )
    )


pcg_summary = summarise(pcg)
cg_summary = summarise(cg)

summary = pd.concat(
    [cg_summary, pcg_summary],
    ignore_index=True,
)

summary["std_total_time"] = (
    summary["std_total_time"].fillna(0.0)
)

summary["cv_pct"] = (
    100.0
    * summary["std_total_time"]
    / summary["mean_total_time"]
)

summary["method"] = pd.Categorical(
    summary["method"],
    categories=METHOD_ORDER,
    ordered=True,
)

summary = summary.sort_values(
    [
        "scaling",
        "grid_size",
        "processes",
        "method",
    ]
).reset_index(drop=True)


# ------------------------------------------------------------------
# 4. Audit
# ------------------------------------------------------------------

failures = []

if len(pcg) != 437:
    failures.append(
        f"Expected 437 PCG raw rows, observed {len(pcg)}."
    )

if len(pcg_summary) != 87:
    failures.append(
        "Expected 87 PCG method configurations, observed "
        f"{len(pcg_summary)}."
    )

if len(cg_summary) != 29:
    failures.append(
        "Expected 29 expanded CG scaling configurations, observed "
        f"{len(cg_summary)}."
    )

if len(summary) != 116:
    failures.append(
        "Expected 116 total method configurations, observed "
        f"{len(summary)}."
    )

if (summary["runs"] < 5).any():
    bad = summary[
        summary["runs"] < 5
    ][keys + ["runs"]]

    failures.append(
        "Configurations with fewer than five runs:\n"
        + bad.to_string(index=False)
    )

if (summary["iteration_values"] != 1).any():
    bad = summary[
        summary["iteration_values"] != 1
    ][keys + ["iteration_values"]]

    failures.append(
        "Configurations with inconsistent iteration counts:\n"
        + bad.to_string(index=False)
    )

if (summary["max_relative_residual"] > 1.0e-8).any():
    bad = summary[
        summary["max_relative_residual"] > 1.0e-8
    ][keys + ["max_relative_residual"]]

    failures.append(
        "Residual-tolerance failures:\n"
        + bad.to_string(index=False)
    )

if (summary["mean_total_time"] <= 0.0).any():
    failures.append(
        "One or more configurations have non-positive runtime."
    )

coverage = (
    summary.assign(method_name=summary["method"].astype(str))
    .groupby(
        ["scaling", "grid_size", "processes"],
        observed=True,
    )["method_name"]
    .agg(lambda values: set(values))
)

expected_methods = set(METHOD_ORDER)

bad_coverage = coverage[
    coverage.map(
        lambda methods: methods != expected_methods
    )
]

if not bad_coverage.empty:
    failures.append(
        "Incomplete method coverage:\n"
        + bad_coverage.to_string()
    )


# Check official quality files.
pcg_quality = pd.read_csv(PCG_QUALITY)
cg_quality = pd.read_csv(CG_QUALITY)

if (
    "quality_check" not in pcg_quality.columns
    or not pcg_quality["quality_check"]
    .astype(str)
    .str.upper()
    .eq("PASS")
    .all()
):
    failures.append(
        "One or more PCG configurations failed the official "
        "quality audit."
    )

if (
    "quality_check" not in cg_quality.columns
    or not cg_quality["quality_check"]
    .astype(str)
    .str.upper()
    .eq("PASS")
    .all()
):
    failures.append(
        "One or more CG configurations failed the official "
        "quality audit."
    )

if failures:
    print("TABLE 4 DATA AUDIT: FAIL")
    print()

    for failure in failures:
        print(failure)
        print()

    raise SystemExit(1)

print("TABLE 4 DATA AUDIT: PASS")
print(f"PCG raw rows: {len(pcg)}")
print(f"PCG method configurations: {len(pcg_summary)}")
print(f"CG scaling configurations: {len(cg_summary)}")
print(f"Combined method configurations: {len(summary)}")
print("All configurations passed numerical and quality checks.")

summary.to_csv(ALL_OUTPUT, index=False)


# ------------------------------------------------------------------
# 5. Strong-scaling endpoint summary
# ------------------------------------------------------------------

strong = summary[
    summary["scaling"] == "strong"
].copy()

strong_rows = []

for (grid_size, method), part in strong.groupby(
    ["grid_size", "method"],
    observed=True,
):
    part = part.sort_values("processes")

    baseline = part.iloc[0]
    endpoint = part.iloc[-1]

    if int(baseline["processes"]) != 1:
        raise ValueError(
            f"Missing p=1 baseline for N={grid_size}, "
            f"method={method}."
        )

    p_max = int(endpoint["processes"])
    speedup = (
        baseline["mean_total_time"]
        / endpoint["mean_total_time"]
    )
    efficiency_pct = 100.0 * speedup / p_max

    strong_rows.append(
        {
            "grid_size": int(grid_size),
            "method": str(method),
            "p_max": p_max,
            "k_1": int(round(
                baseline["mean_iterations"]
            )),
            "k_pmax": int(round(
                endpoint["mean_iterations"]
            )),
            "T_1": baseline["mean_total_time"],
            "T_pmax": endpoint["mean_total_time"],
            "speedup": speedup,
            "efficiency_pct": efficiency_pct,
        }
    )

strong_table = pd.DataFrame(strong_rows)

strong_table["method_order"] = (
    strong_table["method"].map(
        {method: i for i, method in enumerate(METHOD_ORDER)}
    )
)

strong_table = (
    strong_table.sort_values(
        ["grid_size", "method_order"]
    )
    .drop(columns="method_order")
    .reset_index(drop=True)
)

if len(strong_table) != 12:
    raise ValueError(
        f"Expected 12 strong-scaling rows, found "
        f"{len(strong_table)}."
    )

strong_table.to_csv(STRONG_OUTPUT, index=False)


# ------------------------------------------------------------------
# 6. Weak-scaling endpoint summary
# ------------------------------------------------------------------

weak = summary[
    summary["scaling"] == "weak"
].copy()

weak_rows = []

for method, part in weak.groupby(
    "method",
    observed=True,
):
    part = part.sort_values("processes")

    baseline = part.iloc[0]
    endpoint = part.iloc[-1]

    if int(baseline["processes"]) != 1:
        raise ValueError(
            f"Missing weak p=1 baseline for {method}."
        )

    if int(endpoint["processes"]) != 128:
        raise ValueError(
            f"Missing weak p=128 endpoint for {method}."
        )

    weak_efficiency_pct = (
        100.0
        * baseline["mean_total_time"]
        / endpoint["mean_total_time"]
    )

    weak_rows.append(
        {
            "method": str(method),
            "N_1": int(baseline["grid_size"]),
            "N_128": int(endpoint["grid_size"]),
            "k_1": int(round(
                baseline["mean_iterations"]
            )),
            "k_128": int(round(
                endpoint["mean_iterations"]
            )),
            "T_1": baseline["mean_total_time"],
            "T_128": endpoint["mean_total_time"],
            "weak_efficiency_pct": weak_efficiency_pct,
        }
    )

weak_table = pd.DataFrame(weak_rows)

weak_table["method_order"] = (
    weak_table["method"].map(
        {method: i for i, method in enumerate(METHOD_ORDER)}
    )
)

weak_table = (
    weak_table.sort_values("method_order")
    .drop(columns="method_order")
    .reset_index(drop=True)
)

if len(weak_table) != 4:
    raise ValueError(
        f"Expected four weak-scaling rows, found "
        f"{len(weak_table)}."
    )

weak_table.to_csv(WEAK_OUTPUT, index=False)


# ------------------------------------------------------------------
# 7. Generate width-safe LaTeX
# ------------------------------------------------------------------

latex = [
    r"\begin{table}[htbp]",
    r"    \centering",
    r"    \caption{Endpoint summary of strong- and weak-scaling performance.}",
    r"    \label{tab:scaling-summary}",
    r"    \small",
    "",
    r"    \textbf{(a) Strong scaling}",
    r"    \vspace{1mm}",
    "",
    r"    \setlength{\tabcolsep}{4pt}",
    r"    \begin{tabular}{llrccrr}",
    r"        \toprule",
    r"        $N$ & Method & $p_{\max}$"
    r" & $k_1\!\rightarrow\!k_{p_{\max}}$"
    r" & $T_1\!\rightarrow\!T_{p_{\max}}$ (s)"
    r" & $S_{p_{\max}}$"
    r" & $E_{p_{\max}}$ (\%) \\",
    r"        \midrule",
]

previous_n = None

for row in strong_table.itertuples(index=False):
    if previous_n is not None and row.grid_size != previous_n:
        latex.append(r"        \addlinespace")

    latex.append(
        "        "
        f"{row.grid_size} & "
        f"{row.method} & "
        f"{row.p_max} & "
        f"{format_iteration_pair(row.k_1, row.k_pmax)} & "
        f"{format_time_pair(row.T_1, row.T_pmax)} & "
        f"{row.speedup:.2f} & "
        f"{row.efficiency_pct:.1f} "
        r"\\"
    )

    previous_n = row.grid_size

latex.extend(
    [
        r"        \bottomrule",
        r"    \end{tabular}",
        "",
        r"    \vspace{3mm}",
        r"    \textbf{(b) Weak scaling}",
        r"    \vspace{1mm}",
        "",
        r"    \begin{tabular}{lcccc}",
        r"        \toprule",
        r"        Method"
        r" & $N_1\!\rightarrow\!N_{128}$"
        r" & $k_1\!\rightarrow\!k_{128}$"
        r" & $T_1\!\rightarrow\!T_{128}$ (s)"
        r" & $E^{\mathrm{weak}}_{128}$ (\%) \\",
        r"        \midrule",
    ]
)

for row in weak_table.itertuples(index=False):
    latex.append(
        "        "
        f"{row.method} & "
        f"{row.N_1}--{row.N_128} & "
        f"{format_iteration_pair(row.k_1, row.k_128)} & "
        f"{format_time_pair(row.T_1, row.T_128)} & "
        f"{row.weak_efficiency_pct:.1f} "
        r"\\"
    )

latex.extend(
    [
        r"        \bottomrule",
        r"    \end{tabular}",
        "",
        r"    \vspace{2mm}",
        r"    \begin{minipage}{0.97\linewidth}",
        r"        \footnotesize",
        r"        For strong scaling, "
        r"$S_p=T_1/T_p$ and "
        r"$E_p=100S_p/p$. "
        r"For weak scaling, "
        r"$E^{\mathrm{weak}}_p=100T_1/T_p$. "
        r"The symbols $k$ and $T$ denote the mean iteration "
        r"count and mean total runtime, respectively.",
        r"    \end{minipage}",
        r"\end{table}",
        "",
    ]
)

LATEX_OUTPUT.write_text(
    "\n".join(latex),
    encoding="utf-8",
)

print()
print("STRONG-SCALING SUMMARY")
print(strong_table.to_string(index=False))

print()
print("WEAK-SCALING SUMMARY")
print(weak_table.to_string(index=False))

print()
print(f"Created: {ALL_OUTPUT}")
print(f"Created: {STRONG_OUTPUT}")
print(f"Created: {WEAK_OUTPUT}")
print(f"Created: {LATEX_OUTPUT}")
