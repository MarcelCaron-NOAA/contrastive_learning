#!/usr/bin/env python3
"""Structural/data preflight and visual alignment check for forecast-hour archive."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd


PRECIP_LEVELS = [0, 0.1, 0.25, 0.5, 1, 2, 4, 6, 8, 10, 15, 25, 50]
PRECIP_COLORS = [
    "#7CFC00", "#00cc00", "#008f00", "#2468a2", "#3595e8", "#29bdd2",
    "#7a60c5", "#8d28d1", "#990099", "#cc0000", "#ff8500", "#ffcc00",
]
DIFF_LEVELS = [-25, -10, -5, -2, -1, -0.25, 0.25, 1, 2, 5, 10, 25]
DIFF_COLORS = [
    "#08306b", "#08519c", "#2171b5", "#6baed6", "#9ecae1",
    "#e7e3df", "#fcbf9f", "#ef8a62", "#d6604d", "#b2182b", "#67001f",
]


def full_finite(arr: np.ndarray, chunk: int = 64) -> bool:
    for s in range(0, len(arr), chunk):
        if not np.isfinite(np.asarray(arr[s:s+chunk])).all():
            return False
    return True


def main() -> None:
    data = Path(os.environ["DATA_DIR"])
    p = argparse.ArgumentParser()
    p.add_argument(
        "--archive-dir",
        default=str(data / "derived/evaluation/stageiv_forecast_hour_v1"),
    )
    p.add_argument("--forecast-hours", type=int, nargs="+", default=[6,12,18,24,30,36,42,48])
    p.add_argument("--alignment-count", type=int, default=8)
    p.add_argument("--alignment-fhr", type=int, default=24)
    args = p.parse_args()

    arc = Path(args.archive_dir)
    needed = [arc/"cases.csv", arc/"valid_times.csv", arc/"hrrr_qpf6_mm.npy", arc/"stageiv_budget_qpf6_mm.npy"]
    missing = [str(x) for x in needed if not x.is_file()]
    if missing:
        raise SystemExit("Archive preflight missing files:\n" + "\n".join(missing))

    cases = pd.read_csv(arc / "cases.csv")
    valids = pd.read_csv(arc / "valid_times.csv")
    fcst = np.load(arc / "hrrr_qpf6_mm.npy", mmap_mode="r")
    obs = np.load(arc / "stageiv_budget_qpf6_mm.npy", mmap_mode="r")

    expected = sorted(set(int(x) for x in args.forecast_hours))
    if len(fcst) != len(cases):
        raise SystemExit(f"HRRR array length {len(fcst)} != cases rows {len(cases)}")
    if len(obs) != len(valids):
        raise SystemExit(f"Stage-IV array length {len(obs)} != valid_times rows {len(valids)}")
    if fcst.shape[1:] != obs.shape[1:]:
        raise SystemExit(f"Patch-shape mismatch: HRRR {fcst.shape[1:]} vs Stage IV {obs.shape[1:]}")

    cases["forecast_hour"] = pd.to_numeric(cases["forecast_hour"], errors="raise").astype(int)
    cases["init_hour"] = pd.to_numeric(cases["init_hour"], errors="raise").astype(int)
    cases["stageiv_index"] = pd.to_numeric(cases["stageiv_index"], errors="raise").astype(int)

    got_fhrs = sorted(cases["forecast_hour"].unique().tolist())
    if got_fhrs != expected:
        raise SystemExit(f"Forecast-hour set mismatch: got {got_fhrs}, expected {expected}")

    counts = cases["forecast_hour"].value_counts().sort_index()
    if counts.nunique() != 1:
        raise SystemExit(f"Unbalanced case counts by lead:\n{counts}")

    group_sets = cases.groupby("valid_id")["forecast_hour"].apply(lambda x: tuple(sorted(x.tolist())))
    bad = group_sets[group_sets != tuple(expected)]
    if len(bad):
        raise SystemExit(f"{len(bad)} valid times lack the full forecast-hour set")

    # Shared crop and shared Stage-IV field must be identical within each valid time.
    for col in ["y0", "x0", "patch_size", "stageiv_index"]:
        nunique = cases.groupby("valid_id")[col].nunique()
        if (nunique != 1).any():
            raise SystemExit(f"Column {col!r} is not constant within valid time")

    if cases["stageiv_index"].min() < 0 or cases["stageiv_index"].max() >= len(obs):
        raise SystemExit("cases.csv contains out-of-range stageiv_index")

    if not full_finite(fcst):
        raise SystemExit("HRRR archive contains non-finite values")
    if not full_finite(obs):
        raise SystemExit("Stage-IV archive contains non-finite values")
    if float(np.min(fcst)) < -1e-5 or float(np.min(obs)) < -1e-5:
        raise SystemExit("Archive contains materially negative precipitation")

    # Pick the wettest valid times so the alignment figures are informative.
    wet_order = np.argsort(np.max(np.asarray(obs), axis=(1,2)))[::-1]
    selected_valid_idx = wet_order[: min(args.alignment_count, len(wet_order))]
    align_dir = arc / "alignment_check_forecast_hour"
    align_dir.mkdir(parents=True, exist_ok=True)

    pnorm = BoundaryNorm(PRECIP_LEVELS, len(PRECIP_COLORS))
    pcmap = ListedColormap(PRECIP_COLORS)
    dnorm = BoundaryNorm(DIFF_LEVELS, len(DIFF_COLORS))
    dcmap = ListedColormap(DIFF_COLORS)

    for n, st_idx in enumerate(selected_valid_idx, 1):
        sub = cases[(cases["stageiv_index"] == int(st_idx)) & (cases["forecast_hour"] == int(args.alignment_fhr))]
        if sub.empty:
            sub = cases[cases["stageiv_index"] == int(st_idx)].iloc[[0]]
        row = sub.iloc[0]
        ci = int(row["case_index"])
        f = np.asarray(fcst[ci])
        o = np.asarray(obs[int(st_idx)])
        d = f - o

        fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), constrained_layout=True)
        im0 = axes[0].imshow(f, origin="lower", cmap=pcmap, norm=pnorm)
        axes[1].imshow(o, origin="lower", cmap=pcmap, norm=pnorm)
        im2 = axes[2].imshow(d, origin="lower", cmap=dcmap, norm=dnorm)
        for ax, title in zip(axes, ["HRRR forecast", "Stage IV", "Difference: HRRR - Stage IV"]):
            ax.set_title(title, fontweight="bold")
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(
            f"Alignment check {n} | valid {row['valid_end']} | F{int(row['forecast_hour']):02d} | "
            f"patch y0={int(row['y0'])} x0={int(row['x0'])}",
            fontweight="bold",
        )
        fig.colorbar(im0, ax=axes[:2], orientation="horizontal", fraction=0.06, pad=0.03, label="mm / 6 h")
        fig.colorbar(im2, ax=axes[2], orientation="horizontal", fraction=0.06, pad=0.03, label="mm / 6 h")
        fig.savefig(align_dir / f"alignment_{n:02d}_f{int(row['forecast_hour']):02d}.png", dpi=180)
        plt.close(fig)

    report = {
        "n_cases": int(len(cases)),
        "n_valid_times": int(len(valids)),
        "forecast_hours": expected,
        "cases_per_forecast_hour": {str(int(k)): int(v) for k, v in counts.items()},
        "init_hours": sorted(cases["init_hour"].unique().astype(int).tolist()),
        "hrrr_shape": list(fcst.shape),
        "stageiv_shape": list(obs.shape),
        "hrrr_min_mm": float(np.min(fcst)),
        "hrrr_max_mm": float(np.max(fcst)),
        "stageiv_min_mm": float(np.min(obs)),
        "stageiv_max_mm": float(np.max(obs)),
        "alignment_dir": str(align_dir),
        "status": "PASS",
    }
    (arc / "preflight_report.json").write_text(json.dumps(report, indent=2) + "\n")

    print("=" * 88)
    print("STAGE-IV FORECAST-HOUR ARCHIVE PREFLIGHT: PASS")
    print("=" * 88)
    print(f"Valid times              : {len(valids):,}")
    print(f"Cases                    : {len(cases):,}")
    print(f"Forecast hours           : {expected}")
    print(f"Cases per forecast hour  : {counts.iloc[0]:,} each")
    print(f"Initialization hours     : {sorted(cases['init_hour'].unique().tolist())}")
    print(f"HRRR array               : {fcst.shape}")
    print(f"Stage-IV unique array    : {obs.shape}")
    print(f"Alignment figures        : {align_dir}")
    print("\nIMPORTANT: inspect the alignment figures before submitting the GPU job.")


if __name__ == "__main__":
    main()
