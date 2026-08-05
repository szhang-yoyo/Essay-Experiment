from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
import pandas as pd



# File locations
CG_SUMMARY_FILE = Path(
    "results/crossover/cg_canonical_summary.csv"
)

PCG_QUALITY_FILE = Path(
    "results/crossover/pcg_quality_final_validated.csv"
)

OUTPUT_DIR = Path("results/figures")

FIGURE3_PDF = OUTPUT_DIR / "figure3_strong_scaling.pdf"
FIGURE3_PNG = OUTPUT_DIR / "figure3_strong_scaling.png"

FIGURE4_PDF = OUTPUT_DIR / "figure4_weak_scaling.pdf"
FIGURE4_PNG = OUTPUT_DIR / "figure4_weak_scaling.png"

FIGURE3_DATA = OUTPUT_DIR / "figure3_strong_scaling_data.csv"
FIGURE4_DATA = OUTPUT_DIR / "figure4_weak_scaling_data.csv"



# Plot configuration
METHOD_ORDER = ["CG", "BJ", "AS1", "AS2"]
STRONG_GRID_SIZES = [512, 1024, 2048]

METHOD_STYLE = {
    "CG": {
        "color": "black",
        "marker": "o",
        "label": "CG",
    },
    "BJ": {
        "color": "#E78AC3",
        "marker": "s",
        "label": "BJ",
    },
    "AS1": {
        "color": "#F2C14E",
        "marker": "^",
        "label": "AS1",
    },
    "AS2": {
        "color": "#6EC5E9",
        "marker": "D",
        "label": "AS2",
    },
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})



# Validation helpers
def require_columns(
    dataframe: pd.DataFrame,
    required_columns: list[str],
    dataset_name: str,
) -> None:
    """Raise an error when required columns are missing."""

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing columns: {missing_columns}"
        )


def require_unique(
    dataframe: pd.DataFrame,
    key_columns: list[str],
    dataset_name: str,
) -> None:
    """Raise an error when duplicate configuration keys are present."""

    duplicate_rows = dataframe[
        dataframe.duplicated(key_columns, keep=False)
    ]

    if not duplicate_rows.empty:
        raise ValueError(
            f"{dataset_name} contains duplicate keys:\n"
            f"{duplicate_rows[key_columns].to_string(index=False)}"
        )


def configure_log2_process_axis(
    axis: plt.Axes,
    process_counts: list[int],
) -> None:
    """Configure a base-2 logarithmic MPI-process axis."""

    axis.set_xscale("log", base=2)
    axis.set_xticks(process_counts)
    axis.xaxis.set_major_formatter(ScalarFormatter())
    axis.minorticks_off()



# Load and validate canonical summary data
if not CG_SUMMARY_FILE.exists():
    raise FileNotFoundError(CG_SUMMARY_FILE)

if not PCG_QUALITY_FILE.exists():
    raise FileNotFoundError(PCG_QUALITY_FILE)

cg = pd.read_csv(CG_SUMMARY_FILE)
pcg = pd.read_csv(PCG_QUALITY_FILE)

require_columns(
    cg,
    [
        "grid_size",
        "processes",
        "method",
        "mean_total_time",
        "std_total_time",
        "quality_check",
    ],
    "CG summary",
)

require_columns(
    pcg,
    [
        "scaling",
        "grid_size",
        "processes",
        "method",
        "mean_total_time",
        "std_total_time",
        "quality_check",
    ],
    "PCG quality summary",
)

# Retain only configurations that passed all validation checks.
cg = cg[
    cg["quality_check"].astype(str).str.upper().eq("PASS")
].copy()

pcg = pcg[
    pcg["quality_check"].astype(str).str.upper().eq("PASS")
].copy()

cg = cg[
    [
        "grid_size",
        "processes",
        "method",
        "mean_total_time",
        "std_total_time",
    ]
].copy()

pcg = pcg[
    [
        "scaling",
        "grid_size",
        "processes",
        "method",
        "mean_total_time",
        "std_total_time",
    ]
].copy()

require_unique(
    cg,
    ["grid_size", "processes", "method"],
    "CG summary",
)

require_unique(
    pcg,
    ["scaling", "grid_size", "processes", "method"],
    "PCG quality summary",
)



# Classify CG configurations using the validated PCG scaling keys
strong_keys = (
    pcg.loc[
        pcg["scaling"].eq("strong"),
        ["grid_size", "processes"],
    ]
    .drop_duplicates()
    .sort_values(["grid_size", "processes"])
)

weak_keys = (
    pcg.loc[
        pcg["scaling"].eq("weak"),
        ["grid_size", "processes"],
    ]
    .drop_duplicates()
    .sort_values("processes")
)

cg_strong = strong_keys.merge(
    cg,
    on=["grid_size", "processes"],
    how="inner",
    validate="one_to_one",
)

cg_strong.insert(0, "scaling", "strong")

cg_weak = weak_keys.merge(
    cg,
    on=["grid_size", "processes"],
    how="inner",
    validate="one_to_one",
)

cg_weak.insert(0, "scaling", "weak")

strong_data = pd.concat(
    [
        cg_strong,
        pcg.loc[pcg["scaling"].eq("strong")],
    ],
    ignore_index=True,
)

weak_data = pd.concat(
    [
        cg_weak,
        pcg.loc[pcg["scaling"].eq("weak")],
    ],
    ignore_index=True,
)

require_unique(
    strong_data,
    ["grid_size", "processes", "method"],
    "Combined strong-scaling data",
)

require_unique(
    weak_data,
    ["grid_size", "processes", "method"],
    "Combined weak-scaling data",
)


# Figure 3 data: strong-scaling speedup
#
# S_M(N,p) = T_M(N,1) / T_M(N,p)
strong_data = strong_data[
    strong_data["grid_size"].isin(STRONG_GRID_SIZES)
].copy()

strong_baselines = (
    strong_data.loc[
        strong_data["processes"].eq(1),
        ["grid_size", "method", "mean_total_time"],
    ]
    .rename(columns={"mean_total_time": "baseline_total_time"})
)

require_unique(
    strong_baselines,
    ["grid_size", "method"],
    "Strong-scaling baselines",
)

expected_baselines = {
    (grid_size, method)
    for grid_size in STRONG_GRID_SIZES
    for method in METHOD_ORDER
}

observed_baselines = set(
    strong_baselines[
        ["grid_size", "method"]
    ].itertuples(index=False, name=None)
)

missing_baselines = expected_baselines - observed_baselines

if missing_baselines:
    raise ValueError(
        "Missing p=1 strong-scaling baselines: "
        f"{sorted(missing_baselines)}"
    )

strong_plot_data = strong_data.merge(
    strong_baselines,
    on=["grid_size", "method"],
    how="left",
    validate="many_to_one",
)

strong_plot_data["speedup"] = (
    strong_plot_data["baseline_total_time"]
    / strong_plot_data["mean_total_time"]
)

strong_plot_data["parallel_efficiency"] = (
    strong_plot_data["speedup"]
    / strong_plot_data["processes"]
)

strong_plot_data = strong_plot_data.sort_values(
    ["grid_size", "method", "processes"]
)

strong_plot_data[
    [
        "grid_size",
        "processes",
        "method",
        "mean_total_time",
        "std_total_time",
        "baseline_total_time",
        "speedup",
        "parallel_efficiency",
    ]
].to_csv(FIGURE3_DATA, index=False)



# Draw Figure 3
fig, axes = plt.subplots(
    nrows=1,
    ncols=3,
    figsize=(12.6, 4.2),
)

subplot_labels = ["a", "b", "c"]

for axis, grid_size, subplot_label in zip(
    axes,
    STRONG_GRID_SIZES,
    subplot_labels,
):
    panel = strong_plot_data[
        strong_plot_data["grid_size"].eq(grid_size)
    ].copy()

    process_counts = sorted(
        panel["processes"].drop_duplicates().astype(int).tolist()
    )

    # Ideal strong scaling.
    axis.plot(
        process_counts,
        process_counts,
        color="0.55",
        linestyle="--",
        linewidth=1.4,
        zorder=1,
    )

    for method in METHOD_ORDER:
        method_data = panel[
            panel["method"].eq(method)
        ].sort_values("processes")

        if method_data.empty:
            continue

        style = METHOD_STYLE[method]

        axis.plot(
            method_data["processes"],
            method_data["speedup"],
            color=style["color"],
            marker=style["marker"],
            linestyle="-",
            linewidth=1.8,
            markersize=5.5,
            markeredgecolor="black",
            markeredgewidth=0.4,
            zorder=2,
        )

    configure_log2_process_axis(axis, process_counts)

    axis.set_title(
        rf"({subplot_label}) $N={grid_size}$"
    )

    axis.set_xlabel(r"MPI processes, $p$")
    axis.set_ylabel(r"Speedup, $S_M(N,p)$")

    maximum_observed_speedup = panel["speedup"].max()
    maximum_ideal_speedup = max(process_counts)

    axis.set_ylim(
        bottom=0,
        top=max(
            maximum_observed_speedup,
            maximum_ideal_speedup,
        ) * 1.08,
    )    

    axis.grid(
        True,
        which="major",
        linestyle=":",
        linewidth=0.7,
        alpha=0.8,
    )

legend_handles = [
    Line2D(
        [0],
        [0],
        color=METHOD_STYLE[method]["color"],
        marker=METHOD_STYLE[method]["marker"],
        linestyle="-",
        linewidth=1.8,
        markersize=5.5,
        markeredgecolor="black",
        markeredgewidth=0.4,
        label=METHOD_STYLE[method]["label"],
    )
    for method in METHOD_ORDER
]

legend_handles.append(
    Line2D(
        [0],
        [0],
        color="0.55",
        linestyle="--",
        linewidth=1.4,
        label="Ideal",
    )
)

fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=5,
    frameon=False,
    bbox_to_anchor=(0.5, -0.01),
)

fig.tight_layout(rect=(0, 0.12, 1, 1))

fig.savefig(
    FIGURE3_PDF,
    bbox_inches="tight",
)

fig.savefig(
    FIGURE3_PNG,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# Figure 4 data: normalised weak-scaling runtime
# W_M(p) = T_M(N(p),p) / T_M(256,1)
weak_baselines = (
    weak_data.loc[
        weak_data["grid_size"].eq(256)
        & weak_data["processes"].eq(1),
        ["method", "mean_total_time"],
    ]
    .rename(columns={"mean_total_time": "baseline_total_time"})
)

require_unique(
    weak_baselines,
    ["method"],
    "Weak-scaling baselines",
)

observed_weak_baselines = set(
    weak_baselines["method"].tolist()
)

missing_weak_baselines = (
    set(METHOD_ORDER) - observed_weak_baselines
)

if missing_weak_baselines:
    raise ValueError(
        "Missing weak-scaling N=256, p=1 baselines: "
        f"{sorted(missing_weak_baselines)}"
    )

weak_plot_data = weak_data.merge(
    weak_baselines,
    on="method",
    how="left",
    validate="many_to_one",
)

weak_plot_data["normalised_runtime"] = (
    weak_plot_data["mean_total_time"]
    / weak_plot_data["baseline_total_time"]
)

weak_plot_data = weak_plot_data.sort_values(
    ["method", "processes"]
)

weak_plot_data[
    [
        "grid_size",
        "processes",
        "method",
        "mean_total_time",
        "std_total_time",
        "baseline_total_time",
        "normalised_runtime",
    ]
].to_csv(FIGURE4_DATA, index=False)



# Draw Figure 4
fig, axis = plt.subplots(
    figsize=(7.2, 4.8),
)

weak_process_counts = sorted(
    weak_plot_data["processes"]
    .drop_duplicates()
    .astype(int)
    .tolist()
)

axis.axhline(
    y=1.0,
    color="0.55",
    linestyle="--",
    linewidth=1.4,
    zorder=1,
)

for method in METHOD_ORDER:
    method_data = weak_plot_data[
        weak_plot_data["method"].eq(method)
    ].sort_values("processes")

    if method_data.empty:
        continue

    style = METHOD_STYLE[method]

    axis.plot(
        method_data["processes"],
        method_data["normalised_runtime"],
        color=style["color"],
        marker=style["marker"],
        linestyle="-",
        linewidth=1.8,
        markersize=5.5,
        markeredgecolor="black",
        markeredgewidth=0.4,
        label=style["label"],
        zorder=2,
    )

configure_log2_process_axis(
    axis,
    weak_process_counts,
)

axis.set_xlabel(r"MPI processes, $p$")
axis.set_ylabel(r"Normalised runtime, $W_M(p)$")

axis.set_ylim(bottom=0)

axis.grid(
    True,
    which="major",
    linestyle=":",
    linewidth=0.7,
    alpha=0.8,
)

weak_legend_handles = [
    Line2D(
        [0],
        [0],
        color=METHOD_STYLE[method]["color"],
        marker=METHOD_STYLE[method]["marker"],
        linestyle="-",
        linewidth=1.8,
        markersize=5.5,
        markeredgecolor="black",
        markeredgewidth=0.4,
        label=METHOD_STYLE[method]["label"],
    )
    for method in METHOD_ORDER
]

weak_legend_handles.append(
    Line2D(
        [0],
        [0],
        color="0.55",
        linestyle="--",
        linewidth=1.4,
        label="Ideal",
    )
)

axis.legend(
    handles=weak_legend_handles,
    loc="best",
    frameon=False,
    ncol=2,
)

fig.tight_layout()

fig.savefig(
    FIGURE4_PDF,
    bbox_inches="tight",
)

fig.savefig(
    FIGURE4_PNG,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# Terminal audit output
print("=" * 80)
print("FIGURE 3: STRONG-SCALING CONFIGURATIONS")
print("=" * 80)

for grid_size in STRONG_GRID_SIZES:
    configuration = strong_plot_data[
        strong_plot_data["grid_size"].eq(grid_size)
    ]

    process_counts = sorted(
        configuration["processes"]
        .drop_duplicates()
        .astype(int)
        .tolist()
    )

    print(f"N={grid_size}: p={process_counts}")

print()
print("=" * 80)
print("FIGURE 4: WEAK-SCALING CONFIGURATIONS")
print("=" * 80)

weak_configuration_table = (
    weak_plot_data[
        ["processes", "grid_size"]
    ]
    .drop_duplicates()
    .sort_values("processes")
)

print(
    weak_configuration_table.to_string(index=False)
)

print()
print("=" * 80)
print("OUTPUT FILES")
print("=" * 80)
print(FIGURE3_PDF)
print(FIGURE3_PNG)
print(FIGURE3_DATA)
print(FIGURE4_PDF)
print(FIGURE4_PNG)
print(FIGURE4_DATA)
