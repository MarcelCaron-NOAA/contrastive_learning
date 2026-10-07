#!/usr/bin/env python3
"""
Plot fixed-validation performance stratified by meteorological/sample regime.

Requires the CSV written by evaluate_fixed_validation_diagnostics.py.

Produces separate plots (no subplots) for:
  - forecast lead
  - season
  - valid UTC hour
  - wet/dry status
  - precipitation-intensity bin

For each grouping, plots:
  - mean loss
  - mean A-vs-hardest-N margin
  - mean top-1 positive retrieval

Both projection z and normalized backbone h are shown.
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


SEASON_ORDER = ["DJF", "MAM", "JJA", "SON"]
INTENSITY_ORDER = [
    "dry",
    "0.1-1",
    "1-5",
    "5-10",
    "10-25",
    "25-50",
    ">=50",
]


def default_exp():
    return (
        Path(os.environ["DATA_DIR"])
        / "derived/experiments/first_real_train_validation_v1"
    )


def ci95(series):
    x = pd.to_numeric(series, errors="coerce").dropna().to_numpy(float)
    n = len(x)
    if n == 0:
        return np.nan, np.nan, 0
    mean = float(np.mean(x))
    if n < 2:
        return mean, np.nan, n
    sem = float(np.std(x, ddof=1) / np.sqrt(n))
    return mean, 1.96 * sem, n


def summarize(df, group_col, order=None):
    rows = []
    metrics = [
        "z_loss", "h_loss",
        "z_margin_a", "h_margin_a",
        "z_top1_a_pct", "h_top1_a_pct",
    ]

    for group, g in df.groupby(group_col, observed=True):
        row = {group_col: group, "n": len(g)}
        for metric in metrics:
            if metric not in g.columns:
                continue
            mean, half, n_valid = ci95(g[metric])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci95"] = half
            row[f"{metric}_n"] = n_valid
        rows.append(row)

    out = pd.DataFrame(rows)

    if order is not None:
        rank = {str(v): i for i, v in enumerate(order)}
        out["_rank"] = out[group_col].astype(str).map(rank)
        out = out.sort_values("_rank").drop(columns="_rank")
    else:
        try:
            out["_numeric"] = pd.to_numeric(out[group_col])
            out = out.sort_values("_numeric").drop(columns="_numeric")
        except Exception:
            out = out.sort_values(group_col)

    return out.reset_index(drop=True)


def labels_with_n(summary, group_col):
    return [
        f"{v}\n(n={int(n)})"
        for v, n in zip(summary[group_col], summary["n"])
    ]


def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_metric(summary, group_col, metric, ylabel, title, path):
    x = np.arange(len(summary))
    labels = labels_with_n(summary, group_col)

    plt.figure(figsize=(max(8, len(summary) * 0.55), 5.5))

    for space, display in (("z", "projection z"), ("h", "backbone h")):
        mean_col = f"{space}_{metric}_mean"
        ci_col = f"{space}_{metric}_ci95"
        if mean_col not in summary.columns:
            continue
        plt.errorbar(
            x,
            summary[mean_col],
            yerr=summary[ci_col],
            marker="o",
            capsize=3,
            label=display,
        )

    if metric == "margin_a":
        plt.axhline(0.0, linewidth=1.0, linestyle="--")
    if metric == "top1_a_pct":
        plt.axhline(
            100.0 / 9.0,
            linewidth=1.0,
            linestyle="--",
            label="random-choice chance",
        )
        plt.ylim(0, 100)

    plt.xticks(x, labels, rotation=45 if len(summary) > 8 else 0)
    plt.xlabel(group_col.replace("_", " "))
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend()
    savefig(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-dir", default=str(default_exp()))
    p.add_argument("--input-csv", default=None)
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()

    exp = Path(args.experiment_dir)

    if args.input_csv:
        infile = Path(args.input_csv)
    else:
        candidates = sorted(exp.glob("validation_per_example_step*.csv"))
        if not candidates:
            raise SystemExit(
                "No validation_per_example_step*.csv found. "
                "Run evaluate_fixed_validation_diagnostics.py first."
            )
        infile = candidates[-1]

    out = (
        Path(args.output_dir)
        if args.output_dir
        else exp / "diagnostic_plots" / "validation_stratified"
    )
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(infile)

    specs = [
        (
            "anchor_forecast_hour",
            None,
            "forecast_lead",
            "Forecast lead (h)",
        ),
        ("season", SEASON_ORDER, "season", "Season"),
        (
            "anchor_valid_hour_utc",
            None,
            "valid_hour",
            "Valid UTC hour",
        ),
        ("wet_dry", ["dry", "wet"], "wet_dry", "Wet/dry status"),
        (
            "intensity_bin",
            INTENSITY_ORDER,
            "intensity",
            "Anchor wet-pixel P90 intensity bin (mm / 6 h)",
        ),
    ]

    for group_col, order, stem, display in specs:
        if group_col not in df.columns:
            print(f"Skipping missing grouping column: {group_col}")
            continue

        summary = summarize(df, group_col, order)
        summary.to_csv(out / f"{stem}_summary.csv", index=False)

        plot_metric(
            summary,
            group_col,
            "loss",
            "Symmetric explicit-negative InfoNCE loss",
            f"Validation loss by {display}",
            out / f"{stem}_loss.png",
        )
        plot_metric(
            summary,
            group_col,
            "margin_a",
            "cos(A,P) - max cos(A,N)",
            f"Validation margin by {display}",
            out / f"{stem}_margin.png",
        )
        plot_metric(
            summary,
            group_col,
            "top1_a_pct",
            "Positive top-1 retrieval (%)",
            f"Validation retrieval by {display}",
            out / f"{stem}_top1.png",
        )

    print(f"Input: {infile}")
    print(f"Wrote stratified diagnostics to: {out}")


if __name__ == "__main__":
    main()

