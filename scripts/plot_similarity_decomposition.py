#!/usr/bin/env python3
"""
Plot similarity decomposition by anchor precipitation intensity.

Uses the pair-level and anchor-summary CSVs written by
evaluate_validation_similarity_pairs.py.

Primary outputs:
  - grouped boxplots of positive vs negative learned similarity
  - mean AP / mean AN / hardest-AN similarity by intensity
  - the same decomposition for backbone h for completeness

The projection-space z plots are the primary scientific diagnostic because z
is the space directly optimized by InfoNCE.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ORDER = ["dry", "0.1-1", "1-5", "5-10", "10-25", "25-50", ">=50"]


def default_exp():
    return (
        Path(os.environ["DATA_DIR"])
        / "derived/experiments/first_real_train_validation_v1"
    )


def newest(exp, pattern):
    found = sorted(exp.glob(pattern))
    if not found:
        raise SystemExit(f"No files match {pattern} in {exp}")
    return found[-1]


def ci95(x):
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna().to_numpy(float)
    if len(x) == 0:
        return np.nan, np.nan
    mean = float(np.mean(x))
    if len(x) < 2:
        return mean, np.nan
    half = 1.96 * float(np.std(x, ddof=1) / np.sqrt(len(x)))
    return mean, half


def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def grouped_boxplot(pair_df, space, outpath):
    bins = [b for b in ORDER if b in set(pair_df["intensity_bin"])]
    positions = []
    data = []
    labels = []

    pos = 1.0
    for b in bins:
        g = pair_df[pair_df["intensity_bin"] == b]
        pvals = g.loc[
            g["pair_type"] == "positive", f"{space}_similarity"
        ].dropna().to_numpy()
        nvals = g.loc[
            g["pair_type"] == "negative", f"{space}_similarity"
        ].dropna().to_numpy()

        if len(pvals):
            positions.append(pos - 0.18)
            data.append(pvals)
            labels.append(("P", b, len(pvals)))
        if len(nvals):
            positions.append(pos + 0.18)
            data.append(nvals)
            labels.append(("N", b, len(nvals)))
        pos += 1.0

    plt.figure(figsize=(11, 6))
    bp = plt.boxplot(
        data,
        positions=positions,
        widths=0.28,
        patch_artist=False,
        showfliers=False,
        whis=(5, 95),
    )

    centers = np.arange(1, len(bins) + 1, dtype=float)
    xlabels = []
    for b in bins:
        g = pair_df[pair_df["intensity_bin"] == b]
        np_ = int((g["pair_type"] == "positive").sum())
        nn_ = int((g["pair_type"] == "negative").sum())
        xlabels.append(f"{b}\nP={np_}, N={nn_}")

    plt.xticks(centers, xlabels)
    plt.ylabel("Cosine similarity")
    plt.xlabel("Anchor wet-pixel P90 intensity bin (mm / 6 h)")
    plt.title(
        f"{space}: positive and negative learned-similarity distributions"
    )
    plt.grid(axis="y", alpha=0.25)

    # Minimal legend without relying on manually assigned colors.
    plt.plot([], [], linestyle="-", label="Each bin: left=P, right=N")
    plt.legend()
    savefig(outpath)


def component_plot(anchor_df, space, outpath):
    bins = [b for b in ORDER if b in set(anchor_df["intensity_bin"])]
    x = np.arange(len(bins))

    specs = [
        (f"{space}_cosine_ap", "A-P"),
        (f"{space}_cosine_an_mean", "mean A-N"),
        (f"{space}_cosine_an_max", "hardest A-N"),
    ]

    plt.figure(figsize=(10, 6))
    for col, label in specs:
        means = []
        errs = []
        for b in bins:
            g = anchor_df.loc[
                anchor_df["intensity_bin"] == b, col
            ]
            m, e = ci95(g)
            means.append(m)
            errs.append(e)

        plt.errorbar(
            x,
            means,
            yerr=errs,
            marker="o",
            capsize=3,
            label=label,
        )

    labels = [
        f"{b}\n(n={(anchor_df['intensity_bin'] == b).sum()})"
        for b in bins
    ]
    plt.xticks(x, labels)
    plt.xlabel("Anchor wet-pixel P90 intensity bin (mm / 6 h)")
    plt.ylabel("Cosine similarity")
    plt.title(f"{space}: similarity components by precipitation intensity")
    plt.grid(alpha=0.25)
    plt.legend()
    savefig(outpath)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-dir", default=str(default_exp()))
    p.add_argument("--pair-csv", default=None)
    p.add_argument("--anchor-csv", default=None)
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()

    exp = Path(args.experiment_dir)
    pair_path = (
        Path(args.pair_csv)
        if args.pair_csv
        else newest(exp, "validation_similarity_pairs_step*.csv")
    )
    anchor_path = (
        Path(args.anchor_csv)
        if args.anchor_csv
        else newest(exp, "validation_similarity_anchor_summary_step*.csv")
    )

    out = (
        Path(args.output_dir)
        if args.output_dir
        else exp / "diagnostic_plots" / "similarity_decomposition"
    )
    out.mkdir(parents=True, exist_ok=True)

    pair_df = pd.read_csv(pair_path)
    anchor_df = pd.read_csv(anchor_path)

    for space in ("z", "h"):
        grouped_boxplot(
            pair_df,
            space,
            out / f"{space}_positive_negative_boxplot_by_intensity.png",
        )
        component_plot(
            anchor_df,
            space,
            out / f"{space}_similarity_components_by_intensity.png",
        )

    # Numerical summary for quick inspection.
    rows = []
    for b in ORDER:
        g = anchor_df[anchor_df["intensity_bin"] == b]
        if g.empty:
            continue
        rows.append({
            "intensity_bin": b,
            "n": len(g),
            "z_ap_mean": g["z_cosine_ap"].mean(),
            "z_an_mean": g["z_cosine_an_mean"].mean(),
            "z_an_max_mean": g["z_cosine_an_max"].mean(),
            "z_margin_mean": g["z_margin"].mean(),
            "z_top1_pct": 100.0 * g["z_top1"].mean(),
            "h_ap_mean": g["h_cosine_ap"].mean(),
            "h_an_mean": g["h_cosine_an_mean"].mean(),
            "h_an_max_mean": g["h_cosine_an_max"].mean(),
        })

    summary = pd.DataFrame(rows)
    summary_path = out / "similarity_by_intensity_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"Pair input   : {pair_path}")
    print(f"Anchor input : {anchor_path}")
    print(f"Output       : {out}")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

