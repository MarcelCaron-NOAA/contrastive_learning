#!/usr/bin/env python3
"""Select and plot Stage-IV galleries for V3."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

MM_TO_IN = 1.0 / 25.4

PRECIP_BOUNDS = np.array([0.0, 0.10, 0.25, 0.50, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0], dtype=float)
PRECIP_COLORS = [
    "#7CFC00", "#00CC00", "#008E00", "#1f5fa2", "#3399ff",
    "#22cfe0", "#8a6bd1", "#8f2adf", "#a00096", "#c00000", "#ff8c00",
]
DIFF_BOUNDS = np.array([-2.0, -1.0, -0.5, -0.10, 0.10, 0.50, 1.0, 2.0], dtype=float)
DIFF_COLORS = ["#08306b", "#2171b5", "#9ecae1", "#e8e3de", "#f2c4a6", "#e0705c", "#99002f"]


def defaults():
    data = Path(os.environ["DATA_DIR"])
    exp = data / "derived/experiments/hard_ranking_v3_full_v1"
    return {
        "archive": data / "derived/evaluation/stageiv_reference_v1",
        "metrics": exp / "evaluation/stageiv_v1_v2_v3/per_case_metrics.csv.gz",
        "out": exp / "evaluation/stageiv_case_examples_v3",
    }


def rank01(s: pd.Series, higher_better: bool = True) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
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


def prepare_metrics(metrics: pd.DataFrame, min_wet_fraction: float, min_max_in: float) -> pd.DataFrame:
    df = metrics.copy().reset_index(drop=True)
    df["case_index"] = np.arange(len(df), dtype=int)
    df["meaningfully_wet"] = (
        (df["fcst_wet_fraction_0p1in"] >= min_wet_fraction)
        & (df["obs_wet_fraction_0p1in"] >= min_wet_fraction)
        & (df["fcst_max_in"] >= min_max_in)
        & (df["obs_max_in"] >= min_max_in)
    )
    df["fss_rank"] = 0.5 * (rank01(df["fss5_0p1in"]) + rank01(df["fss10_0p1in"]))
    df["v2_rank"] = rank01(df["learned_v2"])
    df["v3_rank"] = rank01(df["learned_v3"])
    df["agreement_v3_fss"] = 1.0 - np.abs(df["v3_rank"] - df["fss_rank"])
    df["departure_v3_v2"] = np.abs(df["learned_v3"] - df["learned_v2"])
    df["departure_v3_fss"] = np.abs(df["v3_rank"] - df["fss_rank"])
    df["agreement_score"] = 0.8 * df["agreement_v3_fss"] + 0.2 * (0.5 * df["v3_rank"] + 0.5 * df["fss_rank"])
    return df


def choose_buckets(df: pd.DataFrame, n_per_bucket: int) -> pd.DataFrame:
    wet = df[df["meaningfully_wet"]].copy()
    if len(wet) < 4 * n_per_bucket:
        raise RuntimeError(f"Only {len(wet)} meaningfully wet cases remain; need at least {4 * n_per_bucket}.")

    used: set[int] = set()
    out = []

    a = select_unique(wet.sort_values("learned_v3", ascending=False), n_per_bucket, used)
    a["bucket"] = "A"
    a["bucket_desc"] = "top V3 learned similarity"
    a["bucket_rank"] = np.arange(1, len(a) + 1)
    out.append(a)

    b = select_unique(wet.sort_values("agreement_score", ascending=False), n_per_bucket, used)
    b["bucket"] = "B"
    b["bucket_desc"] = "best V3 agreement with FSS"
    b["bucket_rank"] = np.arange(1, len(b) + 1)
    out.append(b)

    c = select_unique(wet.sort_values("departure_v3_v2", ascending=False), n_per_bucket, used)
    c["bucket"] = "C"
    c["bucket_desc"] = "largest V3 departures from V2"
    c["bucket_rank"] = np.arange(1, len(c) + 1)
    out.append(c)

    d = select_unique(wet.sort_values("departure_v3_fss", ascending=False), n_per_bucket, used)
    d["bucket"] = "D"
    d["bucket_desc"] = "largest V3 departures from FSS"
    d["bucket_rank"] = np.arange(1, len(d) + 1)
    out.append(d)

    return pd.concat(out, ignore_index=True)


def precip_setup():
    cmap = ListedColormap(PRECIP_COLORS)
    norm = BoundaryNorm(PRECIP_BOUNDS, cmap.N, clip=True)
    return cmap, norm


def diff_setup():
    cmap = ListedColormap(DIFF_COLORS)
    norm = BoundaryNorm(DIFF_BOUNDS, cmap.N, clip=True)
    return cmap, norm


def add_colorbar(ax, mappable, ticks, label, height="6%", ypad=-0.14, extend="max"):
    cax = inset_axes(ax, width="88%", height=height, loc="lower center",
                     bbox_to_anchor=(0.0, ypad, 1.0, 1.0), bbox_transform=ax.transAxes, borderpad=0)
    cb = plt.colorbar(mappable, cax=cax, orientation="horizontal", extend=extend)
    cb.set_ticks(ticks)
    cb.ax.tick_params(labelsize=8, pad=1)
    for lab in cb.ax.get_xticklabels():
        lab.set_fontweight("bold")
    cb.set_label(label, fontsize=9, fontweight="bold", labelpad=2)
    return cb


def plot_case(row: pd.Series, fcst_mm: np.ndarray, obs_mm: np.ndarray, out_png: Path):
    f_in = np.asarray(fcst_mm, dtype=np.float32) * MM_TO_IN
    o_in = np.asarray(obs_mm, dtype=np.float32) * MM_TO_IN
    d_in = f_in - o_in
    p_cmap, p_norm = precip_setup()
    d_cmap, d_norm = diff_setup()

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.35), constrained_layout=False)
    im1 = axes[0].imshow(f_in, origin="lower", cmap=p_cmap, norm=p_norm, interpolation="nearest")
    axes[1].imshow(o_in, origin="lower", cmap=p_cmap, norm=p_norm, interpolation="nearest")
    im3 = axes[2].imshow(d_in, origin="lower", cmap=d_cmap, norm=d_norm, interpolation="nearest")

    for ax, title in zip(axes, ["HRRR forecast", "Stage IV", "Diff: HRRR − Stage IV"]):
        ax.set_title(title, fontsize=14, fontweight="bold", pad=7)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(1.0)

    add_colorbar(axes[0], im1, PRECIP_BOUNDS[1:-1], "inches", extend="max")
    add_colorbar(axes[2], im3, DIFF_BOUNDS, "inches", extend="both")

    init = first_present(row, ["init_time", "anchor_init_time", "init", "cycle_time"])
    valid = first_present(row, ["valid_end", "valid_time", "anchor_valid_time", "valid"])
    lead = first_present(row, ["forecast_hour", "lead_hours", "fhr", "anchor_lead"])
    y0 = first_present(row, ["y0"])
    x0 = first_present(row, ["x0"])

    line1 = f"Bucket {row['bucket']}{int(row['bucket_rank'])}: {row['bucket_desc']}"
    detail = []
    if init is not None:
        detail.append(f"init {fmt_time(init)}")
    if valid is not None:
        detail.append(f"valid {fmt_time(valid)}")
    if lead is not None:
        try:
            detail.append(f"F{int(lead):02d}")
        except Exception:
            detail.append(f"lead {lead}")
    if y0 is not None and x0 is not None:
        detail.append(f"patch y0={int(y0)} x0={int(x0)}")
    line2 = " | ".join(detail)
    line3 = f"V1={row['learned_v1']:.3f}   V2={row['learned_v2']:.3f}   V3={row['learned_v3']:.3f}"
    line4 = f"FSS-5={row['fss5_0p1in']:.3f}   FSS-10={row['fss10_0p1in']:.3f}   ETS(0.10\")={row['ets_0p1in']:.3f}"

    fig.suptitle(f"{line1}\n{line2}\n{line3}\n{line4}", fontsize=13.0, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.79, bottom=0.08, wspace=0.03)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=190, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main():
    d = defaults()
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive-dir", default=str(d["archive"]))
    ap.add_argument("--metrics-csv", default=str(d["metrics"]))
    ap.add_argument("--output-dir", default=str(d["out"]))
    ap.add_argument("--n-per-bucket", type=int, default=5)
    ap.add_argument("--min-wet-fraction-0p1in", type=float, default=0.001)
    ap.add_argument("--min-max-in", type=float, default=0.10)
    args = ap.parse_args()

    arc = Path(args.archive_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cases = pd.read_csv(arc / "cases.csv")
    metrics = pd.read_csv(args.metrics_csv, compression="infer")
    fcst = np.load(arc / "hrrr_qpf6_mm.npy", mmap_mode="r")
    obs = np.load(arc / "stageiv_budget_qpf6_mm.npy", mmap_mode="r")

    if not (len(cases) == len(metrics) == len(fcst) == len(obs)):
        raise RuntimeError("Length mismatch between archive and metrics")

    for c in cases.columns:
        if c not in metrics.columns:
            metrics[c] = cases[c].values

    enriched = prepare_metrics(metrics, args.min_wet_fraction_0p1in, args.min_max_in)
    enriched.to_csv(out / "all_cases_with_bucket_metrics.csv.gz", index=False, compression="gzip")
    selected = choose_buckets(enriched, args.n_per_bucket)
    selected.to_csv(out / "selected_cases.csv", index=False)

    for _, row in selected.iterrows():
        i = int(row["case_index"])
        name = f"{row['bucket']}{int(row['bucket_rank']):02d}_case{i:04d}.png"
        plot_case(row, fcst[i], obs[i], out / f"bucket_{row['bucket']}" / name)

    summary = {
        "archive_dir": str(arc),
        "metrics_csv": str(args.metrics_csv),
        "n_archive_cases": len(cases),
        "n_meaningfully_wet": int(enriched["meaningfully_wet"].sum()),
        "n_per_bucket": args.n_per_bucket,
        "selection_buckets": {
            "A": "highest V3 learned similarity",
            "B": "best V3 agreement with FSS composite (rank mean of FSS-5 and FSS-10)",
            "C": "largest absolute V3-V2 learned-similarity difference",
            "D": "largest absolute rank disagreement between V3 and FSS composite",
        },
    }
    (out / "run_metadata.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("=" * 88)
    print("STAGE-IV V3 GALLERIES COMPLETE")
    print("=" * 88)
    print(f"Wet candidates: {summary['n_meaningfully_wet']} / {len(cases)}")
    print(f"Selected     : {len(selected)} ({args.n_per_bucket} per bucket)")
    print(f"Manifest     : {out / 'selected_cases.csv'}")
    print(f"Figures      : {out}/bucket_A ... bucket_D")


if __name__ == "__main__":
    main()
