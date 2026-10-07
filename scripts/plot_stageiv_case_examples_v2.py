#!/usr/bin/env python3
"""Select and plot informative HRRR–Stage-IV V1/V2 cases.

Uses the existing Stage-IV reference archive and the per-case metrics written by
scripts/evaluate_stageiv_v1_v2.py. No additional model inference is required.

Buckets
-------
A: high-quality successes (V2 and conventional metrics all high)
B: largest V2 - V1 score increases
C: largest V2 - V1 score decreases
D: largest V2-vs-conventional disagreements (half each direction)

Each selected case is plotted as HRRR / Stage IV / HRRR-Stage IV, in inches.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import uniform_filter

MM_TO_IN = 1.0 / 25.4
THRESH_01IN_MM = 2.54


def defaults():
    data = Path(os.environ["DATA_DIR"])
    exp = data / "derived/experiments/hard_ranking_finetune_v1"
    return {
        "archive": data / "derived/evaluation/stageiv_reference_v1",
        "metrics": exp / "evaluation/stageiv_v1_vs_v2/per_case_metrics.csv.gz",
        "out": exp / "evaluation/stageiv_case_examples_v2",
    }


def fss_score(fcst_mm: np.ndarray, obs_mm: np.ndarray, threshold_mm: float, width: int) -> float:
    f = (np.asarray(fcst_mm) >= threshold_mm).astype(np.float32)
    o = (np.asarray(obs_mm) >= threshold_mm).astype(np.float32)
    ff = uniform_filter(f, size=width, mode="nearest")
    oo = uniform_filter(o, size=width, mode="nearest")
    num = float(np.mean((ff - oo) ** 2))
    den = float(np.mean(ff ** 2 + oo ** 2))
    if den <= 0:
        return np.nan
    return 1.0 - num / den


def ets_score(fcst_mm: np.ndarray, obs_mm: np.ndarray, threshold_mm: float) -> float:
    f = np.asarray(fcst_mm) >= threshold_mm
    o = np.asarray(obs_mm) >= threshold_mm
    h = float(np.logical_and(f, o).sum())
    m = float(np.logical_and(~f, o).sum())
    fa = float(np.logical_and(f, ~o).sum())
    n = float(f.size)
    hr = ((h + m) * (h + fa)) / n if n else np.nan
    den = h + m + fa - hr
    if not np.isfinite(den) or den <= 0:
        return np.nan
    return (h - hr) / den


def rank01(s: pd.Series, higher_better: bool = True) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    # percentile rank is robust to very different metric scales
    r = x.rank(method="average", pct=True)
    return r if higher_better else 1.0 - r


def fmt_time(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    try:
        t = pd.to_datetime(v)
        return t.strftime("%Y-%m-%d %HZ")
    except Exception:
        return str(v)


def first_present(row: pd.Series, names):
    for n in names:
        if n in row.index and pd.notna(row[n]):
            return row[n]
    return None


def nice_upper(v: float) -> float:
    """Round a positive plotting max upward to a readable value."""
    if not np.isfinite(v) or v <= 0:
        return 0.1
    targets = np.array([0.05, 0.10, 0.20, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0])
    bigger = targets[targets >= v]
    return float(bigger[0] if len(bigger) else np.ceil(v))


def select_unique(sorted_df: pd.DataFrame, n: int, used: set[int]) -> pd.DataFrame:
    idx = []
    for i in sorted_df.index:
        ii = int(sorted_df.loc[i, "case_index"])
        if ii in used:
            continue
        idx.append(i)
        used.add(ii)
        if len(idx) >= n:
            break
    return sorted_df.loc[idx].copy()


def build_case_metadata(metrics: pd.DataFrame, fcst, obs, min_wet_fraction: float, min_max_mm: float) -> pd.DataFrame:
    df = metrics.copy().reset_index(drop=True)
    df["case_index"] = np.arange(len(df), dtype=int)

    # Compute verification quantities on the exact fields used for the F/O learned score.
    fss5, fss10, ets01 = [], [], []
    fw, ow, fmax, omax = [], [], [], []
    for i in range(len(df)):
        f = np.asarray(fcst[i], dtype=np.float32)
        o = np.asarray(obs[i], dtype=np.float32)
        fss5.append(fss_score(f, o, THRESH_01IN_MM, 5))
        fss10.append(fss_score(f, o, THRESH_01IN_MM, 10))
        ets01.append(ets_score(f, o, THRESH_01IN_MM))
        fw.append(float(np.mean(f >= THRESH_01IN_MM)))
        ow.append(float(np.mean(o >= THRESH_01IN_MM)))
        fmax.append(float(np.max(f)))
        omax.append(float(np.max(o)))

    df["fss5_0p1in"] = fss5
    df["fss10_0p1in"] = fss10
    df["ets_0p1in"] = ets01
    df["fcst_wet_fraction_0p1in"] = fw
    df["obs_wet_fraction_0p1in"] = ow
    df["fcst_max_mm_recalc"] = fmax
    df["obs_max_mm_recalc"] = omax
    df["meaningfully_wet"] = (
        (df["fcst_wet_fraction_0p1in"] >= min_wet_fraction)
        & (df["obs_wet_fraction_0p1in"] >= min_wet_fraction)
        & (df["fcst_max_mm_recalc"] >= min_max_mm)
        & (df["obs_max_mm_recalc"] >= min_max_mm)
    )
    return df


def choose_buckets(df: pd.DataFrame, n_per_bucket: int) -> pd.DataFrame:
    wet = df[df["meaningfully_wet"]].copy()
    if len(wet) < 4 * n_per_bucket:
        raise RuntimeError(
            f"Only {len(wet)} meaningfully-wet cases remain; need at least {4*n_per_bucket}. "
            "Relax --min-wet-fraction-0p1in or --min-max-mm."
        )

    # Conventional composite used only for case selection, not as a target metric.
    wet["traditional_rank"] = (
        rank01(wet["fss5_0p1in"]) +
        rank01(wet["fss10_0p1in"]) +
        rank01(wet["ets_0p1in"]) +
        rank01(wet["corr"])
    ) / 4.0
    wet["v2_rank"] = rank01(wet["learned_v2"])
    wet["success_score"] = 0.5 * wet["v2_rank"] + 0.5 * wet["traditional_rank"]
    wet["delta"] = wet["learned_v2"] - wet["learned_v1"]
    wet["disagreement"] = wet["v2_rank"] - wet["traditional_rank"]

    used: set[int] = set()
    out = []

    a = select_unique(wet.sort_values("success_score", ascending=False), n_per_bucket, used)
    a["bucket"] = "A"; a["bucket_desc"] = "high-quality success"
    a["bucket_rank"] = np.arange(1, len(a)+1)
    out.append(a)

    b = select_unique(wet.sort_values("delta", ascending=False), n_per_bucket, used)
    b["bucket"] = "B"; b["bucket_desc"] = "largest V2 improvement"
    b["bucket_rank"] = np.arange(1, len(b)+1)
    out.append(b)

    c = select_unique(wet.sort_values("delta", ascending=True), n_per_bucket, used)
    c["bucket"] = "C"; c["bucket_desc"] = "largest V2 degradation"
    c["bucket_rank"] = np.arange(1, len(c)+1)
    out.append(c)

    # D: split evenly between V2-high/traditional-low and V2-low/traditional-high.
    n_hi = n_per_bucket // 2
    n_lo = n_per_bucket - n_hi
    d_hi = select_unique(wet.sort_values("disagreement", ascending=False), n_hi, used)
    d_hi["disagreement_direction"] = "V2 high / traditional low"
    d_lo = select_unique(wet.sort_values("disagreement", ascending=True), n_lo, used)
    d_lo["disagreement_direction"] = "V2 low / traditional high"
    d = pd.concat([d_hi, d_lo], ignore_index=True)
    d["bucket"] = "D"; d["bucket_desc"] = "V2 vs traditional disagreement"
    d["bucket_rank"] = np.arange(1, len(d)+1)
    out.append(d)

    return pd.concat(out, ignore_index=True)


def plot_case(row: pd.Series, fcst_mm: np.ndarray, obs_mm: np.ndarray, out_png: Path):
    f_in = np.asarray(fcst_mm, dtype=np.float32) * MM_TO_IN
    o_in = np.asarray(obs_mm, dtype=np.float32) * MM_TO_IN
    d_in = f_in - o_in

    # Shared forecast/obs scale; robust upper bound but never smaller than 0.25 in.
    pmax = np.nanpercentile(np.concatenate([f_in.ravel(), o_in.ravel()]), 99.5)
    pmax = nice_upper(max(float(pmax), 0.25))
    dmax = np.nanpercentile(np.abs(d_in), 99.5)
    dmax = nice_upper(max(float(dmax), 0.10))

    pnorm = Normalize(vmin=0.0, vmax=pmax)
    dnorm = TwoSlopeNorm(vmin=-dmax, vcenter=0.0, vmax=dmax)

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.55), constrained_layout=False)
    ax1, ax2, ax3 = axes
    im1 = ax1.imshow(f_in, origin="lower", cmap="turbo", norm=pnorm, interpolation="nearest")
    ax2.imshow(o_in, origin="lower", cmap="turbo", norm=pnorm, interpolation="nearest")
    im3 = ax3.imshow(d_in, origin="lower", cmap="RdBu_r", norm=dnorm, interpolation="nearest")

    for ax, title in zip(axes, ["HRRR", "Stage IV", "HRRR − Stage IV"]):
        ax.set_title(title, fontsize=14, fontweight="bold", pad=5)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_linewidth(1.0)

    # Two slender bars, both exactly the height of panel 3, entirely to its right.
    divider = make_axes_locatable(ax3)
    cax_p = divider.append_axes("right", size="3.4%", pad=0.08)
    cax_d = divider.append_axes("right", size="3.4%", pad=0.42)
    cbp = fig.colorbar(im1, cax=cax_p)
    cbd = fig.colorbar(im3, cax=cax_d)
    cbp.set_label("QPF (in)", fontsize=10, fontweight="bold")
    cbd.set_label("ΔQPF (in)", fontsize=10, fontweight="bold")
    cbp.ax.tick_params(labelsize=9); cbd.ax.tick_params(labelsize=9)
    for lab in cbp.ax.get_yticklabels() + cbd.ax.get_yticklabels(): lab.set_fontweight("bold")

    init = first_present(row, ["init_time", "anchor_init_time", "init", "cycle_time"])
    valid = first_present(row, ["valid_time", "anchor_valid_time", "valid"])
    lead = first_present(row, ["lead_hours", "fhr", "forecast_hour", "anchor_lead"])
    parts = [f"Bucket {row['bucket']}{int(row['bucket_rank'])}: {row['bucket_desc']}"]
    detail = []
    if init is not None: detail.append(f"init {fmt_time(init)}")
    if valid is not None: detail.append(f"valid {fmt_time(valid)}")
    if lead is not None:
        try: detail.append(f"F{int(lead):02d}")
        except Exception: detail.append(f"lead {lead}")
    if "disagreement_direction" in row.index and pd.notna(row.get("disagreement_direction", np.nan)):
        detail.append(str(row["disagreement_direction"]))
    if detail: parts.append(" | ".join(detail))

    stats = (
        f"V1={row['learned_v1']:.3f}   V2={row['learned_v2']:.3f}   "
        f"Δ={row['delta']:+.3f}   FSS-5={row['fss5_0p1in']:.3f}   "
        f"FSS-10={row['fss10_0p1in']:.3f}   ETS(0.10\")={row['ets_0p1in']:.3f}   "
        f"r={row['corr']:.3f}"
    )

    fig.suptitle("\n".join(parts), fontsize=15.5, fontweight="bold", y=0.985)
    fig.text(0.5, 0.905, stats, ha="center", va="center", fontsize=11.5, fontweight="bold")
    fig.subplots_adjust(left=0.015, right=0.925, bottom=0.025, top=0.84, wspace=0.055)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=190, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main():
    d = defaults()
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive-dir", default=str(d["archive"]))
    ap.add_argument("--metrics-csv", default=str(d["metrics"]))
    ap.add_argument("--output-dir", default=str(d["out"]))
    ap.add_argument("--n-per-bucket", type=int, default=6)
    ap.add_argument("--min-wet-fraction-0p1in", type=float, default=0.001,
                    help="Minimum fraction of pixels >=0.10 in in BOTH HRRR and Stage IV (default ~66 pixels in 256x256).")
    ap.add_argument("--min-max-mm", type=float, default=2.54,
                    help="Minimum max QPF in BOTH fields; default 2.54 mm = 0.10 in.")
    args = ap.parse_args()

    arc = Path(args.archive_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cases_path = arc / "cases.csv"
    fcst_path = arc / "hrrr_qpf6_mm.npy"
    obs_path = arc / "stageiv_budget_qpf6_mm.npy"
    metrics_path = Path(args.metrics_csv)
    for p in [cases_path, fcst_path, obs_path, metrics_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required input does not exist: {p}")

    cases = pd.read_csv(cases_path)
    metrics = pd.read_csv(metrics_path, compression="infer")
    fcst = np.load(fcst_path, mmap_mode="r")
    obs = np.load(obs_path, mmap_mode="r")
    n = len(cases)
    if not (len(metrics) == len(fcst) == len(obs) == n):
        raise RuntimeError(
            f"Length mismatch: cases={len(cases)}, metrics={len(metrics)}, HRRR={len(fcst)}, StageIV={len(obs)}"
        )

    # The V1/V2 diagnostic preserved every cases.csv column and row order. If somehow
    # the current metrics table lacks metadata, merge the archive metadata by row index.
    for c in cases.columns:
        if c not in metrics.columns:
            metrics[c] = cases[c].values

    enriched = build_case_metadata(metrics, fcst, obs, args.min_wet_fraction_0p1in, args.min_max_mm)
    enriched.to_csv(out / "all_cases_with_example_metrics.csv.gz", index=False, compression="gzip")
    selected = choose_buckets(enriched, args.n_per_bucket)
    selected.to_csv(out / "selected_cases.csv", index=False)

    for _, row in selected.iterrows():
        i = int(row["case_index"])
        name = f"{row['bucket']}{int(row['bucket_rank']):02d}_case{i:04d}.png"
        plot_case(row, fcst[i], obs[i], out / f"bucket_{row['bucket']}" / name)

    summary = {
        "archive_dir": str(arc),
        "metrics_csv": str(metrics_path),
        "n_archive_cases": n,
        "n_meaningfully_wet": int(enriched["meaningfully_wet"].sum()),
        "n_per_bucket": args.n_per_bucket,
        "threshold_for_fss_ets_mm": THRESH_01IN_MM,
        "threshold_for_fss_ets_in": 0.10,
        "fss_widths": [5, 10],
        "selection_buckets": {
            "A": "high V2 plus high conventional-metric composite",
            "B": "largest V2 minus V1 increases",
            "C": "largest V2 minus V1 decreases",
            "D": "largest rank disagreement between V2 and conventional-metric composite, split both directions",
        },
    }
    (out / "run_metadata.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("=" * 88)
    print("STAGE-IV CASE EXAMPLE DIAGNOSTIC COMPLETE")
    print("=" * 88)
    print(f"Archive       : {arc}")
    print(f"Metrics       : {metrics_path}")
    print(f"Wet candidates: {summary['n_meaningfully_wet']} / {n}")
    print(f"Selected      : {len(selected)} ({args.n_per_bucket} per bucket)")
    print(f"Manifest      : {out/'selected_cases.csv'}")
    print(f"Figures       : {out}/bucket_A ... bucket_D")


if __name__ == "__main__":
    main()

