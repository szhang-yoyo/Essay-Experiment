import pandas as pd
import matplotlib.pyplot as plt

# Load final10 summary
df = pd.read_csv("results/final10/summary/final10_summary.csv")
strong = df[df["scaling"] == "strong"].copy()

grid_sizes = [512, 1024, 2048]

labels = {
    "BJ": "BJ-PCG",
    "AS1": "AS1-PCG",
    "AS2": "AS2-PCG",
}

markers = {
    "CG": "o",
    "BJ": "s",
    "AS1": "^",
    "AS2": "D",
}

fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))

for ax, N in zip(axes, grid_sizes):

    sub = strong[strong["grid_size"] == N].copy()

    # CG values are repeated in BJ/AS1/AS2 rows,
    # so keep one CG value for each (N, p).
    cg = (
        sub[["processes", "cg_mean_runtime"]]
        .drop_duplicates(subset=["processes"])
        .sort_values("processes")
    )

    ax.plot(
        cg["processes"],
        cg["cg_mean_runtime"],
        marker=markers["CG"],
        linewidth=1.6,
        markersize=4.5,
        label="CG",
    )

    # PCG methods
    for method in ["BJ", "AS1", "AS2"]:
        m = (
            sub[sub["method"] == method]
            .sort_values("processes")
        )

        ax.plot(
            m["processes"],
            m["mean_runtime"],
            marker=markers[method],
            linewidth=1.6,
            markersize=4.5,
            label=labels[method],
        )

    # MPI process counts are powers of two
    ax.set_xscale("log", base=2)

    p_values = sorted(sub["processes"].unique())
    ax.set_xticks(p_values)
    ax.set_xticklabels([str(int(p)) for p in p_values])

    ax.set_title(rf"$N={N}$")
    ax.grid(True, alpha=0.25)

fig.supxlabel("MPI processes")
fig.supylabel("Mean runtime (s)")

# One shared legend
handles, legend_labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    legend_labels,
    loc="upper center",
    ncol=4,
    frameon=False,
)

plt.tight_layout(rect=[0, 0, 1, 0.88])

plt.savefig(
    "results/final10/figures/strong_scaling_runtime.pdf",
    bbox_inches="tight",
)

plt.savefig(
    "results/final10/figures/strong_scaling_runtime.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print("Created:")
print("results/final10/figures/strong_scaling_runtime.pdf")
print("results/final10/figures/strong_scaling_runtime.png")
