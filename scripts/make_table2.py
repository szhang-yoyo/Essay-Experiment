
from pathlib import Path

import pandas as pd


ROOT = Path("results/phase5")
OUTPUT_CSV = ROOT / "table2_representative_results.csv"
OUTPUT_TEX = ROOT / "table2_representative_results.tex"


def read_summary(
    path: Path,
    case: str,
    grid_size: int | None = None,
) -> pd.DataFrame:
    """Read an existing Phase 5 summary file."""
    if not path.exists():
        raise FileNotFoundError(f"Missing summary file: {path}")

    frame = pd.read_csv(path)

    if grid_size is not None:
        frame = frame.loc[frame["grid_size"] == grid_size].copy()

    required = {
        "grid_size",
        "processes",
        "method",
        "mean_iterations",
        "mean_relative_residual",
        "mean_total_time",
        "std_total_time",
    }

    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    frame["case"] = case

    return frame[
        [
            "case",
            "grid_size",
            "processes",
            "method",
            "mean_iterations",
            "mean_relative_residual",
            "mean_total_time",
            "std_total_time",
        ]
    ]


def summarise_critical_rerun(path: Path, case: str) -> pd.DataFrame:
    """Aggregate the five raw runs of the N=1024, p=64 rerun."""
    if not path.exists():
        raise FileNotFoundError(f"Missing critical-rerun file: {path}")

    raw = pd.read_csv(path)

    required = {
        "grid_size",
        "processes",
        "method",
        "iterations",
        "relative_residual",
        "total_time",
    }

    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    summary = (
        raw.groupby(
            ["grid_size", "processes", "method"],
            as_index=False,
        )
        .agg(
            mean_iterations=("iterations", "mean"),
            mean_relative_residual=("relative_residual", "mean"),
            mean_total_time=("total_time", "mean"),
            std_total_time=("total_time", "std"),
        )
    )

    summary["case"] = case

    return summary[
        [
            "case",
            "grid_size",
            "processes",
            "method",
            "mean_iterations",
            "mean_relative_residual",
            "mean_total_time",
            "std_total_time",
        ]
    ]


def read_optional_cg(
    path: Path,
    case: str,
) -> pd.DataFrame | None:
    """Read one same-environment CG reference summary."""
    if not path.exists():
        print(f"WARNING: CG reference file was not found: {path}")
        return None

    frame = pd.read_csv(path)
    frame["case"] = case

    return frame[
        [
            "case",
            "grid_size",
            "processes",
            "method",
            "mean_iterations",
            "mean_relative_residual",
            "mean_total_time",
            "std_total_time",
        ]
    ]


def scientific_latex(value: float) -> str:
    """Convert a floating-point number to LaTeX scientific notation."""
    coefficient, exponent = f"{value:.3e}".split("e")
    exponent_value = int(exponent)
    return rf"\({coefficient}\times10^{{{exponent_value}}}\)"


def method_latex(method: str) -> str:
    labels = {
        "CG": "CG",
        "BJ": "BJ-PCG",
        "AS1": r"AS-PCG (\(\delta=1\))",
        "AS2": r"AS-PCG (\(\delta=2\))",
    }
    return labels.get(method, method)


def create_latex_table(frame: pd.DataFrame, output_path: Path) -> None:
    case_labels = {
        "Single-process reference":
            r"Single-process reference",
        "Medium strong-scaling case":
            r"Medium strong-scaling configuration",
        "Representative high-process strong-scaling case":
            r"Large-problem strong-scaling configuration",
        "Largest weak-scaling case":
            r"Largest weak-scaling configuration",
    }

    lines = [
        r"\begin{table}[htbp]",
        r"    \centering",
        (
            r"    \caption{Representative convergence and runtime results "
            r"for CG, BJ-PCG, and Classical AS-PCG.}"
        ),
        r"    \label{tab:representative-results}",
        r"    \small",
        r"    \setlength{\tabcolsep}{3.5pt}",
        r"    \begin{tabular}{rrlrrrr}",
        r"        \toprule",
        (
            r"        \(N\) & \(p\) & Method & \(N_{\mathrm{it}}\) & "
            r"\(r_{\mathrm{rel}}\) & \(T_{\mathrm{total}}\) (s) & "
            r"\(T_{\mathrm{iter}}\) (ms) \\"
        ),
        r"        \midrule",
    ]

    case_order = [
        "Single-process reference",
        "Representative high-process strong-scaling case",
        "Largest weak-scaling case",
    ]

    for case_index, case in enumerate(case_order):
        group = frame.loc[frame["case"] == case]

        if group.empty:
            continue

        if case_index > 0:
            lines.append(r"        \addlinespace")

        label = case_labels[case]
        lines.append(
            rf"        \multicolumn{{7}}{{l}}{{\textit{{{label}}}}} \\"
        )

        for _, row in group.iterrows():
            total_time = (
                rf"\({row['mean_total_time']:.3f}"
                rf"\pm{row['std_total_time']:.3f}\)"
            )

            line = (
                rf"        {int(row['grid_size'])} & "
                rf"{int(row['processes'])} & "
                rf"{method_latex(str(row['method']))} & "
                rf"{int(round(row['mean_iterations']))} & "
                rf"{scientific_latex(row['mean_relative_residual'])} & "
                rf"{total_time} & "
                rf"{row['time_per_iteration_ms']:.3f} \\"
            )
            lines.append(line)

    lines.extend(
        [
            r"        \bottomrule",
            r"    \end{tabular}",
            r"    \par\smallskip",
            r"    \begin{minipage}{0.96\textwidth}",
            r"        \footnotesize",
            r"        Reported runtimes are means \(\pm\) standard deviations over five runs. "
            r"The per-iteration time is calculated as "
            r"\(T_{\mathrm{iter}}=1000T_{\mathrm{total}}/N_{\mathrm{it}}\).",
            r"    \end{minipage}",
            r"\end{table}",
        ]
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    frames: list[pd.DataFrame] = []

    # 1. Single-process configuration: retain only N=1024.
    frames.append(
        read_summary(
            ROOT / "summary" / "timing_strong_p1_212466.csv",
            case="Single-process reference",
            grid_size=1024,
        )
    )

    # 3. Large-problem strong-scaling configuration.
    frames.append(
        read_summary(
            ROOT / "summary" / "timing_strong_p32_212471.csv",
            case="Representative high-process strong-scaling case",
            grid_size=2048,
        )
    )

    # 4. Largest weak-scaling configuration.
    frames.append(
        read_summary(
            ROOT / "summary" / "timing_weak_p128_212464.csv",
            case="Largest weak-scaling case",
        )
    )

    # CG references generated using the same environment and timing scope.
    cg_cases = [
        (
            ROOT / "summary" / "cg_reference_n1024_p1.csv",
            "Single-process reference",
        ),
        (
            ROOT / "summary" / "cg_reference_n2048_p32.csv",
            "Representative high-process strong-scaling case",
        ),
        (
            ROOT / "summary" / "cg_reference_n2896_p128.csv",
            "Largest weak-scaling case",
        ),
    ]

    for cg_path, case in cg_cases:
        cg_frame = read_optional_cg(cg_path, case)
        if cg_frame is not None:
            frames.append(cg_frame)

    table = pd.concat(frames, ignore_index=True)

    case_rank = {
        "Single-process reference": 0,
        "Representative high-process strong-scaling case": 1,
        "Largest weak-scaling case": 2,
    }

    method_rank = {
        "CG": 0,
        "BJ": 1,
        "AS1": 2,
        "AS2": 3,
    }

    table["case_rank"] = table["case"].map(case_rank)
    table["method_rank"] = table["method"].map(method_rank)

    if table["case_rank"].isna().any():
        raise ValueError("An unknown representative case was encountered.")

    if table["method_rank"].isna().any():
        unknown = sorted(
            table.loc[table["method_rank"].isna(), "method"].unique()
        )
        raise ValueError(f"Unknown methods encountered: {unknown}")

    table["time_per_iteration_ms"] = (
        1000.0
        * table["mean_total_time"]
        / table["mean_iterations"]
    )

    table = (
        table.sort_values(["case_rank", "method_rank"])
        .drop(columns=["case_rank", "method_rank"])
        .reset_index(drop=True)
    )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    table.to_csv(OUTPUT_CSV, index=False)
    create_latex_table(table, OUTPUT_TEX)

    print(f"Created: {OUTPUT_CSV}")
    print(f"Created: {OUTPUT_TEX}")
    print()
    print(
        table[
            [
                "case",
                "grid_size",
                "processes",
                "method",
                "mean_iterations",
                "mean_relative_residual",
                "mean_total_time",
                "std_total_time",
                "time_per_iteration_ms",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
