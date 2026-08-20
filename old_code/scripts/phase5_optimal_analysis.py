from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SRC = Path("results/phase4/phase3_summary.csv")
OUT = Path("results/phase5")
PLOT = Path("plots/phase5")

METHODS = ["BJ", "AS1", "AS2"]
PRIORITY = ["BJ", "AS1", "AS2"]
TIE_LIMIT = 0.03

OUT.mkdir(parents=True, exist_ok=True)
PLOT.mkdir(parents=True, exist_ok=True)


def normalise_method(row):
    name = str(row["method"]).strip().upper().replace(" ", "")

    if name.startswith("BJ"):
        return "BJ"

    if name in {"AS1", "AS2"}:
        return name

    overlap = int(row["overlap"])
    return f"AS{overlap}"


df = pd.read_csv(SRC)

if "processes" in df.columns and "p" not in df.columns:
    df = df.rename(columns={"processes": "p"})

required = {
    "n",
    "p",
    "method",
    "overlap",
    "local_unknowns",
    "avg_iterations",
    "avg_runtime",
}

missing = required - set(df.columns)

if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

df["method"] = df.apply(normalise_method, axis=1)
df = df[df["method"].isin(METHODS)].copy()

keys = ["n", "p", "local_unknowns"]

# Average duplicate summary rows if any exist.
base = (
    df.groupby(keys + ["method"], as_index=False)
    .agg(
        avg_runtime=("avg_runtime", "mean"),
        avg_iterations=("avg_iterations", "mean"),
    )
)

runtime = base.pivot_table(
    index=keys,
    columns="method",
    values="avg_runtime",
    aggfunc="mean",
)

iterations = base.pivot_table(
    index=keys,
    columns="method",
    values="avg_iterations",
    aggfunc="mean",
)

# Record incomplete configurations instead of silently using them.
incomplete = runtime[runtime.reindex(columns=METHODS).isna().any(axis=1)]
incomplete.reset_index().to_csv(
    OUT / "optimal_incomplete.csv",
    index=False,
)

runtime = runtime.dropna(subset=METHODS).reset_index()
iterations = iterations.dropna(subset=METHODS).reset_index()

combined = runtime.merge(
    iterations,
    on=keys,
    suffixes=("_time", "_iterations"),
)

optimal_rows = []
audit_rows = []

for _, row in combined.iterrows():
    times = {
        method: float(row[f"{method}_time"])
        for method in METHODS
    }

    minimum = min(times.values())
    raw_fastest = min(times, key=times.get)

    # A method is approximately tied if it is within 3% of the minimum.
    near_tie = [
        method
        for method in PRIORITY
        if times[method] <= minimum * (1.0 + TIE_LIMIT)
    ]

    best_method = near_tie[0]
    best_time = times[best_method]

    optimal_rows.append(
        {
            "n": int(row["n"]),
            "p": int(row["p"]),
            "local_unknowns": int(row["local_unknowns"]),
            "bj_time": times["BJ"],
            "as1_time": times["AS1"],
            "as2_time": times["AS2"],
            "best_method": best_method,
            "best_time": best_time,
        }
    )

    audit_rows.append(
        {
            "n": int(row["n"]),
            "p": int(row["p"]),
            "local_unknowns": int(row["local_unknowns"]),
            "raw_fastest": raw_fastest,
            "raw_minimum_time": minimum,
            "near_tie_methods": ",".join(near_tie),
            "selected_method": best_method,
            "selected_penalty_pct": 100.0 * (best_time / minimum - 1.0),
        }
    )

optimal = pd.DataFrame(optimal_rows).sort_values(["p", "n"])
audit = pd.DataFrame(audit_rows).sort_values(["p", "n"])

optimal.to_csv(OUT / "optimal_map.csv", index=False)
audit.to_csv(OUT / "optimal_audit.csv", index=False)


# Plot 1: total runtime comparison

plot_data = optimal.reset_index(drop=True)
x = np.arange(len(plot_data))
width = 0.25

plt.figure(figsize=(max(14, len(plot_data) * 0.55), 7))

for offset, method, column in [
    (-width, "BJ", "bj_time"),
    (0.0, "AS1", "as1_time"),
    (width, "AS2", "as2_time"),
]:
    plt.bar(
        x + offset,
        plot_data[column],
        width=width,
        label=method,
    )

labels = [
    f"n={int(row.n)}\np={int(row.p)}"
    for row in plot_data.itertuples()
]

plt.xticks(x, labels, rotation=60, ha="right")
plt.yscale("log")
plt.ylabel("Mean total runtime (s)")
plt.xlabel("Configuration")
plt.title("Total Runtime Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(PLOT / "runtime_comparison.png", dpi=300)
plt.close()


# Plot 2: optimal configuration regions

plt.figure(figsize=(9, 7))

for method in METHODS:
    part = optimal[optimal["best_method"] == method]

    plt.scatter(
        part["local_unknowns"],
        part["p"],
        s=85,
        label=method,
    )

plt.xscale("log")
plt.yscale("log", base=2)

process_values = sorted(optimal["p"].unique())
plt.yticks(process_values, process_values)

plt.xlabel("Local unknowns per process")
plt.ylabel("MPI processes")
plt.title("Time-Optimal Preconditioner Regions")
plt.legend()
plt.grid(True, which="both", alpha=0.25)
plt.tight_layout()
plt.savefig(PLOT / "optimal_regions.png", dpi=300)
plt.close()

# Plot 3: iteration reduction versus runtime change


trade_rows = []

for method in ["AS1", "AS2"]:
    for _, row in combined.iterrows():
        bj_iterations = float(row["BJ_iterations"])
        method_iterations = float(row[f"{method}_iterations"])

        bj_time = float(row["BJ_time"])
        method_time = float(row[f"{method}_time"])

        trade_rows.append(
            {
                "n": int(row["n"]),
                "p": int(row["p"]),
                "local_unknowns": int(row["local_unknowns"]),
                "method": method,
                "iteration_reduction_pct": (
                    100.0
                    * (bj_iterations - method_iterations)
                    / bj_iterations
                ),
                "runtime_change_pct": (
                    100.0 * (method_time - bj_time) / bj_time
                ),
            }
        )

trade = pd.DataFrame(trade_rows)
trade.to_csv(
    OUT / "iteration_runtime_tradeoff.csv",
    index=False,
)

plt.figure(figsize=(9, 7))

for method in ["AS1", "AS2"]:
    part = trade[trade["method"] == method]

    plt.scatter(
        part["iteration_reduction_pct"],
        part["runtime_change_pct"],
        s=75,
        label=method,
    )

plt.axhline(0.0, linewidth=1)
plt.axvline(0.0, linewidth=1)

plt.xlabel("Iteration reduction about BJ (%)")
plt.ylabel("Runtime change about  BJ (%)")
plt.title("Iteration Reduction versus Runtime Change")
plt.legend()
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(
    PLOT / "iteration_runtime_comparison.png",
    dpi=300,
)
plt.close()


print(f"Complete configurations: {len(optimal)}")
print(f"Incomplete configurations: {len(incomplete)}")
print()
print("Best-method counts:")
print(optimal["best_method"].value_counts())
print()
print(f"Created: {OUT / 'optimal_map.csv'}")
print(f"Created: {OUT / 'optimal_audit.csv'}")
print(f"Created plots in: {PLOT}")
