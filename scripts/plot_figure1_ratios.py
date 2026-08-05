from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


POINT_FILE = Path("results/crossover/ratio_point_estimates.csv")
CI_FILE = Path("results/crossover/ratio_bootstrap_ci.csv")
OUTPUT_FILE = Path("results/crossover/plots/figure_iteration_runtime_ratios.pdf")


# Representative configurations used in the corresponding results table.
CONFIGURATIONS = [
    {
        "scaling": "strong",
        "grid_size": 1024,
        "processes": 1,
        "label": "Strong\n" + r"$N=1024,\ p=1$",
    },
    {
        "scaling": "strong",
        "grid_size": 1024,
        "processes": 64,
        "label": "Strong\n" + r"$N=1024,\ p=64$",
    },
    {
        "scaling": "strong",
        "grid_size": 2048,
        "processes": 32,
        "label": "Strong\n" + r"$N=2048,\ p=32$",
    },
    {
        "scaling": "weak",
        "grid_size": 2896,
        "processes": 128,
        "label": "Weak\n" + r"$N=2896,\ p=128$",
    },
]




METHODS = ["BJ", "AS1", "AS2"]
METHOD_LABELS = {
    "BJ": "BJ",
    "AS1": r"AS ($\delta=1$)",
    "AS2": r"AS ($\delta=2$)",
}

COLORS = {
    "BJ": "#9ECAE1",    # light blue
    "AS1": "#FFD966",   # light yellow
    "AS2": "#F4A6C1",   # light pink
}

HATCHES = {
    "BJ": "",
    "AS1": "//",
    "AS2": "xx",
}



def select_configurations(data: pd.DataFrame) -> pd.DataFrame:
    """Extract the selected configurations in a fixed plotting order."""
    selected_rows = []

    for config_index, config in enumerate(CONFIGURATIONS):
        for method_index, method in enumerate(METHODS):
            mask = (
                (data["scaling"] == config["scaling"])
                & (data["grid_size"] == config["grid_size"])
                & (data["processes"] == config["processes"])
                & (data["method"] == method)
            )

            rows = data.loc[mask].copy()

            if len(rows) != 1:
                raise ValueError(
                    "Expected exactly one row for "
                    f"{config['scaling']}, N={config['grid_size']}, "
                    f"p={config['processes']}, method={method}; "
                    f"found {len(rows)}."
                )

            rows["config_order"] = config_index
            rows["method_order"] = method_index
            rows["config_label"] = config["label"]
            selected_rows.append(rows)

    selected = pd.concat(selected_rows, ignore_index=True)

    return selected.sort_values(
        ["config_order", "method_order"]
    ).reset_index(drop=True)


def main() -> None:
    point_data = pd.read_csv(POINT_FILE)
    ci_data = pd.read_csv(CI_FILE)

    point_selected = select_configurations(point_data)
    ci_selected = select_configurations(ci_data)

    # Confirm that the point estimates agree between the two files.
    merged_check = point_selected.merge(
        ci_selected,
        on=[
            "scaling",
            "grid_size",
            "processes",
            "method",
        ],
        suffixes=("_point", "_ci"),
    )

    if not np.allclose(
        merged_check["R_T_point"],
        merged_check["R_T_ci"],
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError("R_T point estimates differ between the two input files.")

    n_configs = len(CONFIGURATIONS)
    x = np.arange(n_configs)
    bar_width = 0.24

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(11.0, 4.4),
        constrained_layout=True,
    )

    ax_iterations = axes[0]
    ax_runtime = axes[1]

    for method_index, method in enumerate(METHODS):
        offset = (method_index - 1) * bar_width

        point_method = point_selected[
            point_selected["method"] == method
        ].sort_values("config_order")

        ci_method = ci_selected[
            ci_selected["method"] == method
        ].sort_values("config_order")

        # Panel (a): iteration-count ratio.
        ax_iterations.bar(
            x + offset,
            point_method["R_k"],
            width=bar_width,
            label=METHOD_LABELS[method],
            color=COLORS[method],
            edgecolor="black", 
            linewidth=0.7, 
            hatch=HATCHES[method],
        )

        # Asymmetric bootstrap confidence intervals for R_T.
        lower_error = (
            ci_method["R_T"] - ci_method["R_T_ci_lower"]
        ).to_numpy()

        upper_error = (
            ci_method["R_T_ci_upper"] - ci_method["R_T"]
        ).to_numpy()

        runtime_error = np.vstack([lower_error, upper_error])

        # Panel (b): runtime ratio.
        ax_runtime.bar(
            x + offset,
            ci_method["R_T"],
            width=bar_width,
            yerr=runtime_error,
            capsize=3,
            label=METHOD_LABELS[method],
            edgecolor="black",
            linewidth=0.7,
            hatch=HATCHES[method],
            error_kw={"elinewidth": 0.9, "capthick": 0.9, "ecolor": "black",},
        )

    config_labels = [config["label"] for config in CONFIGURATIONS]

    ax_iterations.axhline(
        y=1.0,
        linestyle="--",
        linewidth=1.0,
    )
    ax_runtime.axhline(
        y=1.0,
        linestyle="--",
        linewidth=1.0,
    )

    ax_iterations.set_xticks(x)
    ax_iterations.set_xticklabels(
        config_labels,
        rotation=0,
        ha="center",
    )
    ax_runtime.set_xticks(x)
    ax_runtime.set_xticklabels(
        config_labels,
        rotation=0,
        ha="center",
    )

    ax_iterations.set_ylabel(
        r"Iteration-count ratio "
        r"$R_k=\overline{k}_M/\overline{k}_{\mathrm{CG}}$"
    )

    ax_runtime.set_ylabel(
        r"Runtime ratio "
        r"$R_T=\overline{T}_M/\overline{T}_{\mathrm{CG}}$"
    )

    ax_iterations.set_title("(a) Iteration-count ratio")
    ax_runtime.set_title("(b) Total-runtime ratio")

    ax_iterations.set_ylim(0, 1.05)
    ax_runtime.set_ylim(bottom=0)

    ax_iterations.grid(
        axis="y",
        linestyle=":",
        linewidth=0.6,
    )
    ax_runtime.grid(
        axis="y",
        linestyle=":",
        linewidth=0.6,
    )

    # A single shared legend avoids duplication.
    handles, labels = ax_runtime.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.08),
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        OUTPUT_FILE,
        bbox_inches="tight",
    )

    print(f"Saved figure: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
