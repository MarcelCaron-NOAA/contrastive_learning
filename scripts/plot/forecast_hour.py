#!/usr/bin/env python3
"""Plot the canonical Stage-IV forecast-hour/die-off diagnostic."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def defaults() -> tuple[Path, Path]:
    data = Path(os.environ["DATA_DIR"])
    base = data / "derived/evaluation/stageiv_forecast_hour_v1"
    return base / "verification_scores_by_forecast_hour.csv", base / "plots"


def main() -> None:
    default_csv, default_out = defaults()
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-csv", default=str(default_csv))
    ap.add_argument("--output-dir", default=str(default_out))
    args = ap.parse_args()

    df = pd.read_csv(args.scores_csv).sort_values("forecast_hour")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    required = [
        "forecast_hour", "n_cases", "ets_0p1in", "pod_0p1in", "success_ratio_0p1in",
        "fss_0p1in_w5", "fss_0p1in_w10", "syn3", "syn8", "nfv",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required score columns: {missing}")

    mpl.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 17,
        "axes.labelsize": 14,
        "axes.labelweight": "bold",
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.linewidth": 1.0,
    })

    fig, ax = plt.subplots(figsize=(14.5, 8.5), constrained_layout=True)
    ax.set_facecolor("#f5f5f5")
    ax.grid(True, alpha=0.28, linewidth=0.7)
    x = df["forecast_hour"].to_numpy(dtype=int)

    styles = [
        ("ets_0p1in", "ETS-0.1", "darkgreen", "-", 2.2),
        ("pod_0p1in", "POD-0.1", "green", "-", 2.0),
        ("success_ratio_0p1in", "SR-0.1", "lightgreen", "-", 2.0),
        ("fss_0p1in_w5", "FSS-5", "lightskyblue", "-", 2.2),
        ("fss_0p1in_w10", "FSS-10", "navy", "-", 2.4),
        ("syn3", "SYN-3", "firebrick", "-", 2.6),
        ("syn8", "SYN-8", "firebrick", "--", 2.6),
        ("nfv", "NFV", "black", "-", 2.7),
    ]

    plotted = []
    for col, label, color, ls, lw in styles:
        y = df[col].to_numpy(dtype=float)
        plotted.append(y)
        ax.plot(x, y, color=color, linestyle=ls, linewidth=lw, marker="o", markersize=5.5, label=label)

    finite = np.concatenate([a[np.isfinite(a)] for a in plotted if np.isfinite(a).any()])
    ax.set_ylim(min(-0.05, float(finite.min()) - 0.04), 1.04)
    ax.set_xlim(x.min() - 1, x.max() + 1)
    ax.set_xticks(x)
    ax.set_xlabel("Forecast Lead Hour")
    ax.set_ylabel("Verification Score")
    ax.set_title("HRRR 6-h Precipitation Verification by Forecast Hour", fontweight="bold")

    for xi, n in zip(x, df["n_cases"].to_numpy(dtype=int)):
        ax.text(xi, 1.012, str(n), transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                color="dimgray", fontsize=9.5, fontweight="bold", clip_on=False)
    ax.text(x.min() - 0.8, 1.020, "#SAMPLES", transform=ax.get_xaxis_transform(),
            ha="right", va="bottom", color="dimgray", fontsize=9.5, fontweight="bold", clip_on=False)

    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=4,
              frameon=True, fontsize=10.5, handlelength=3.0, columnspacing=1.3)

    png = out / "hrrr_stageiv_verification_scores_by_forecast_hour.png"
    pdf = out / "hrrr_stageiv_verification_scores_by_forecast_hour.pdf"
    fig.savefig(png, dpi=250, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
