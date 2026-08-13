import pandas as pd
import matplotlib.pyplot as plt

# Load final10 summary
df = pd.read_csv("results/final10/summary/final10_summary.csv")

# Keep only weak-scaling configurations
weak = df[df["scaling"] == "weak"].copy()

# Keep colours consistent with the strong-scaling figure
colors = {
    "CG": "#1f77b4",
    "BJ": "#ff7f0e",
    "AS1": "#2ca02c",
    "AS2": "#d62728",
}

markers = {
    "CG": "o",
    "BJ": "s",
    "AS1": "^",
    "AS2": "D",
}

labels = {
    "CG": "CG",
    "BJ": "BJ-PCG",
    "AS1": "AS1-PCG",
    "AS2": "AS2-PCG",
}

fig, ax = plt.subplots(figsize=(6.2, 4.2))

# CG values are repeated in the BJ/AS1/AS2 rows,
# so retain one value for each process count.
cg = (
    weak[["processes", "cg_mean_runtime"]]
    .drop_duplicates(subset=["processes"])
    .sort_values("processes")
)

ax.plot(
    cg["processes"],
    cg["cg_mean_runtime"],
    marker=markers["CG"],
    color=colors["CG"],
    linewidth=1.6,
    markersize=5,
    label=labels["CG"],
)

# PCG methods
for method in ["BJ", "AS1", "AS2"]:

    m = (
        weak[weak["method"] == method]
        .sort_values("processes")
    )

    ax.plot(
        m["processes"],
        m["mean_runtime"],
        marker=markers[method],
        color=colors[method],
        linewidth=1.6,
        markersize=5,
        label=labels[method],
    )

# MPI process counts are powers of two
ax.set_xscale("log", base=2)

p_values = sorted(weak["processes"].unique())

ax.set_xticks(p_values)
ax.set_xticklabels([str(int(p)) for p in p_values])

ax.set_xlabel("MPI processes")
ax.set_ylabel("Mean runtime (s)")

ax.grid(True, alpha=0.25)

ax.legend(
    ncol=2,
    frameon=False,
)

plt.tight_layout()

plt.savefig(
    "results/final10/figures/weak_scaling_runtime.pdf",
    bbox_inches="tight",
)

plt.savefig(
    "results/final10/figures/weak_scaling_runtime.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print("Created:")
print("results/final10/figures/weak_scaling_runtime.pdf")
print("results/final10/figures/weak_scaling_runtime.png")
