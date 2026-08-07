from pathlib import Path
import shutil
import numpy as np
import pandas as pd


CANONICAL = Path(
    "results/crossover/pcg_canonical_raw_final_validated.csv"
)
QUALITY = Path(
    "results/crossover/pcg_quality_final_validated.csv"
)
NEW_RAW = Path(
    "results/phase5/raw/timing_weak_p64_216783.csv"
)

RAW_BACKUP = CANONICAL.with_name(
    "pcg_canonical_raw_before_216783.csv"
)
QUALITY_BACKUP = QUALITY.with_name(
    "pcg_quality_before_216783.csv"
)

keys = ["scaling", "grid_size", "processes", "method"]

canonical = pd.read_csv(CANONICAL)
quality = pd.read_csv(QUALITY)
new = pd.read_csv(NEW_RAW)

case_mask_new = (
    (new["scaling"] == "weak")
    & (new["grid_size"] == 2048)
    & (new["processes"] == 64)
)

new = new[case_mask_new].copy()

# Validate the replacement dataset.
assert len(new) == 15, f"Expected 15 rows, found {len(new)}"
assert set(new["method"]) == {"BJ", "AS1", "AS2"}
assert new.groupby("method").size().eq(5).all()
assert set(new["job_id"]) == {216783}
assert (new["sgs_sweeps"] == 4).all()
assert (new["relative_residual"] <= 1.0e-8).all()
assert new.groupby("method")["iterations"].nunique().eq(1).all()

timing_columns = [
    "total_time",
    "spmv_time",
    "halo_time",
    "preconditioner_time",
    "reduction_time",
    "vector_time",
]

assert (new[timing_columns] >= 0.0).all().all()
assert (new["halo_time"] <= new["spmv_time"]).all()

for column in [
    "spmv_time",
    "halo_time",
    "preconditioner_time",
    "reduction_time",
    "vector_time",
]:
    assert (new[column] <= new["total_time"]).all()

# Add canonical provenance fields.
new["source_type"] = "same_job_rerun"
new["source_file"] = NEW_RAW.name

missing_columns = set(canonical.columns) - set(new.columns)
assert not missing_columns, (
    f"Replacement data missing columns: {sorted(missing_columns)}"
)

new = new[canonical.columns]

case_mask_old = (
    (canonical["scaling"] == "weak")
    & (canonical["grid_size"] == 2048)
    & (canonical["processes"] == 64)
)

old_rows = canonical[case_mask_old].copy()
assert len(old_rows) == 15, (
    f"Expected to replace 15 canonical rows, found {len(old_rows)}"
)

# Preserve the current validated files.
if not RAW_BACKUP.exists():
    shutil.copy2(CANONICAL, RAW_BACKUP)

if not QUALITY_BACKUP.exists():
    shutil.copy2(QUALITY, QUALITY_BACKUP)

# Replace all BJ, AS1 and AS2 rows for this configuration.
updated = pd.concat(
    [canonical[~case_mask_old], new],
    ignore_index=True,
)

updated = updated.sort_values(
    [
        "scaling",
        "grid_size",
        "processes",
        "method",
        "repeat",
    ]
).reset_index(drop=True)

assert len(updated) == 437
assert len(
    updated.groupby(
        ["scaling", "grid_size", "processes", "method"]
    )
) == 87

updated.to_csv(CANONICAL, index=False)


# Rebuild only the three affected quality rows.
quality_rows = []

for method, part in new.groupby("method", sort=True):
    mean_time = part["total_time"].mean()
    std_time = part["total_time"].std(ddof=0)
    cv_pct = 100.0 * std_time / mean_time

    repeat_check = "PASS" if len(part) >= 5 else "FAIL"
    residual_check = (
        "PASS"
        if part["relative_residual"].max() <= 1.0e-8
        else "FAIL"
    )
    iteration_limit_check = (
        "PASS" if part["iterations"].max() < 10000 else "FAIL"
    )
    iteration_consistency_check = (
        "PASS"
        if part["iterations"].nunique() == 1
        else "FAIL"
    )
    timing_nonnegative_check = (
        "PASS"
        if (part[timing_columns] >= 0.0).all().all()
        else "FAIL"
    )
    halo_check = (
        "PASS"
        if (part["halo_time"] <= part["spmv_time"]).all()
        else "FAIL"
    )

    phase_check = "PASS"

    for column in [
        "spmv_time",
        "halo_time",
        "preconditioner_time",
        "reduction_time",
        "vector_time",
    ]:
        if not (part[column] <= part["total_time"]).all():
            phase_check = "FAIL"

    checks = [
        repeat_check,
        residual_check,
        iteration_limit_check,
        iteration_consistency_check,
        timing_nonnegative_check,
        halo_check,
        phase_check,
    ]

    quality_rows.append(
        {
            "scaling": "weak",
            "grid_size": 2048,
            "processes": 64,
            "method": method,
            "runs": len(part),
            "iterations": int(part["iterations"].iloc[0]),
            "max_relative_residual": (
                part["relative_residual"].max()
            ),
            "mean_total_time": mean_time,
            "std_total_time": std_time,
            "min_total_time": part["total_time"].min(),
            "max_total_time": part["total_time"].max(),
            "cv_pct": cv_pct,
            "repeat_check": repeat_check,
            "residual_check": residual_check,
            "iteration_limit_check": iteration_limit_check,
            "iteration_consistency_check": (
                iteration_consistency_check
            ),
            "timing_nonnegative_check": (
                timing_nonnegative_check
            ),
            "halo_check": halo_check,
            "phase_check": phase_check,
            "quality_check": (
                "PASS" if all(x == "PASS" for x in checks)
                else "FAIL"
            ),
            "cv_status": (
                "accepted" if cv_pct <= 20.0 else "review"
            ),
            "job_ids": "216783",
        }
    )

new_quality = pd.DataFrame(quality_rows)
new_quality = new_quality[quality.columns]

quality_case_mask = (
    (quality["scaling"] == "weak")
    & (quality["grid_size"] == 2048)
    & (quality["processes"] == 64)
)

assert quality_case_mask.sum() == 3

updated_quality = pd.concat(
    [quality[~quality_case_mask], new_quality],
    ignore_index=True,
)

updated_quality = updated_quality.sort_values(
    ["scaling", "grid_size", "processes", "method"]
).reset_index(drop=True)

assert len(updated_quality) == 87
assert (updated_quality["quality_check"] == "PASS").all()

updated_quality.to_csv(QUALITY, index=False)

print("Replacement completed.")
print(f"Canonical rows: {len(updated)}")
print(f"Quality rows: {len(updated_quality)}")
print()
print(
    new_quality[
        [
            "method",
            "runs",
            "iterations",
            "mean_total_time",
            "std_total_time",
            "cv_pct",
            "quality_check",
            "job_ids",
        ]
    ].to_string(index=False)
)
print()
print(f"Raw backup: {RAW_BACKUP}")
print(f"Quality backup: {QUALITY_BACKUP}")
