from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


INPUT_DIR = Path("results/final10/raw/pcg")

OUTPUT_DIR = Path("results/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PNG_FILE = OUTPUT_DIR / "figure2_overlap_tradeoff_final10.png"
PDF_FILE = OUTPUT_DIR / "figure2_overlap_tradeoff_final10.pdf"
POINTS_FILE = OUTPUT_DIR / "figure2_overlap_tradeoff_final10_points.csv"



# 1. Read and validate the final canonical PCG data
input_files = sorted(INPUT_DIR.glob("*.csv"))

if not input_files:
    raise FileNotFoundError(
        f"No final10 PCG CSV files found in {INPUT_DIR}"
    )

df = pd.concat(
    [pd.read_csv(f) for f in input_files],
    ignore_index=True,
)

required_columns = {
    "scaling",
    "grid_size",
    "processes",
    "method",
    "iterations",
    "total_time",
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {sorted(missing_columns)}"
    )

df["scaling"] = df["scaling"].str.lower()
df["method"] = df["method"].str.upper()

df = df[
    df["method"].isin(["BJ", "AS1", "AS2"])
].copy()



# 2. Average repeated runs for each method configuration
summary = (
    df.groupby(
        ["scaling", "grid_size", "processes", "method"],
        as_index=False,
    )
    .agg(
        mean_iterations=("iterations", "mean"),
        mean_total_time=("total_time", "mean"),
        runs=("total_time", "size"),
    )
)



# 3. Extract BJ baseline values

bj = (
    summary[summary["method"] == "BJ"]
    [
        [
            "scaling",
            "grid_size",
            "processes",
            "mean_iterations",
            "mean_total_time",
        ]
    ]
    .rename(
        columns={
            "mean_iterations": "bj_iterations",
            "mean_total_time": "bj_total_time",
        }
    )
)


# 4. Match AS1 and AS2 against BJ for the same physical configuration

as_data = summary[
    summary["method"].isin(["AS1", "AS2"])
].copy()

plot_data = as_data.merge(
    bj,
    on=["scaling", "grid_size", "processes"],
    how="inner",
    validate="many_to_one",
)

plot_data["iteration_reduction_pct"] = 100.0 * (
    1.0
    - plot_data["mean_iterations"]
    / plot_data["bj_iterations"]
)

plot_data["runtime_change_pct"] = 100.0 * (
    plot_data["mean_total_time"]
    / plot_data["bj_total_time"]
    - 1.0
)


#6. Classify quadrants for checking

def classify_quadrant(row):
    delta_k = row["iteration_reduction_pct"]
    delta_t = row["runtime_change_pct"]

    if delta_k >= 0.0 and delta_t < 0.0:
        return "right-lower: net benefit"

    if delta_k >= 0.0 and delta_t >= 0.0:
        return "right-upper: fewer iterations but slower"

    if delta_k < 0.0 and delta_t < 0.0:
        return "left-lower: more iterations but faster"

    return "left-upper: both worse"


plot_data["quadrant"] = plot_data.apply(
    classify_quadrant,
    axis=1,
)

plot_data = plot_data.sort_values(
    ["scaling", "method", "grid_size", "processes"]
)

plot_data.to_csv(POINTS_FILE, index=False)


# 6. Draw the scatter plot
#
# Marker shape:
#   circle   = AS1
#   triangle = AS2
#
# Marker fill:
#   hollow = strong scaling
#   filled = weak scaling
fig, ax = plt.subplots(figsize=(7.2, 5.4))

plot_styles = {
    ("AS1", "strong"): {
        "marker": "o",
        "facecolors": "none",
        "edgecolors": "black",
        "label": "AS1, strong scaling",
    },
    ("AS2", "strong"): {
        "marker": "^",
        "facecolors": "none",
        "edgecolors": "black",
        "label": "AS2, strong scaling",
    },
    ("AS1", "weak"): {
        "marker": "o",
        "facecolors": "black",
        "edgecolors": "black",
        "label": "AS1, weak scaling",
    },
    ("AS2", "weak"): {
        "marker": "^",
        "facecolors": "black",
        "edgecolors": "black",
        "label": "AS2, weak scaling",
    },
}

for (method, scaling), style in plot_styles.items():
    subset = plot_data[
        (plot_data["method"] == method)
        & (plot_data["scaling"] == scaling)
    ]

    ax.scatter(
        subset["iteration_reduction_pct"],
        subset["runtime_change_pct"],
        marker=style["marker"],
        facecolors=style["facecolors"],
        edgecolors=style["edgecolors"],
        s=55,
        linewidths=1.0,
        label=style["label"],
        zorder=3,
    )

ax.axvline(
    x=0.0,
    linestyle="--",
    linewidth=0.9,
    color="0.4",
    zorder=1,
)

ax.axhline(
    y=0.0,
    linestyle="--",
    linewidth=0.9,
    color="0.4",
    zorder=1,
)

# Axis labels
ax.set_xlabel(
    r"Iteration reduction relative to BJ, $\Delta k$ (%)"
)

ax.set_ylabel(
    r"Total-runtime change relative to BJ, $\Delta T$ (%)"
)


# Ensure that both zero reference lines remain visible
x_values = plot_data["iteration_reduction_pct"]
y_values = plot_data["runtime_change_pct"]

x_range = max(x_values.max() - x_values.min(), 1.0)
y_range = max(y_values.max() - y_values.min(), 1.0)

x_padding = 0.07 * x_range
y_padding = 0.07 * y_range

ax.set_xlim(
    min(0.0, x_values.min()) - x_padding,
    max(0.0, x_values.max()) + x_padding,
)

ax.set_ylim(
    min(0.0, y_values.min()) - y_padding,
    max(0.0, y_values.max()) + y_padding,
)


ax.legend(
    frameon=False,
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=2,
)

fig.tight_layout()

fig.savefig(
    PNG_FILE,
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    PDF_FILE,
    bbox_inches="tight",
)

plt.close(fig)


# 7. Print audit information

print("=" * 72)
print("OVERLAP TRADE-OFF FIGURE")
print("=" * 72)

print(f"Physical configurations: {len(bj)}")
print(f"Scatter points: {len(plot_data)}")

print("\nPoints by scaling and method:")
print(
    plot_data.groupby(
        ["scaling", "method"]
    ).size()
)

print("\nQuadrant counts:")
print(
    plot_data.groupby(
        ["method", "quadrant"]
    ).size()
)

print("\nRanges:")
print(
    plot_data[
        [
            "iteration_reduction_pct",
            "runtime_change_pct",
        ]
    ].agg(["min", "max"])
)

print(f"\nPoint data: {POINTS_FILE}")
print(f"PNG figure: {PNG_FILE}")
print(f"PDF figure: {PDF_FILE}")
