#!/usr/bin/env python3
"""Make a Figure-9-style three-panel controlled-transformation plot."""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ref_curve(family, x):
    x = np.asarray(x, float)
    if family == "displacement":
        d = np.abs(x) / 10.0
    else:
        d = np.where(
            x <= 1.0,
            np.abs(x-1.0) / 0.5,
            np.abs(x-1.0) / 1.0,
        )
    return np.maximum(0.0, 1.0 - d*d)


def plot_metric(ax, g, col, label, color):
    x = g["bias"].to_numpy()
    y = g[f"{col}_mean"].to_numpy()
    lo = g[f"{col}_ci95_lo"].to_numpy()
    hi = g[f"{col}_ci95_hi"].to_numpy()
    ax.plot(x, y, label=label, color=color, lw=1.5)
    ax.fill_between(x, lo, hi, color=color, alpha=0.16, linewidth=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("summary_csv")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    df = pd.read_csv(args.summary_csv)
    outfile = (
        Path(args.output)
        if args.output
        else Path(args.summary_csv).with_name("wang_style_comparison.png")
    )

    # Deliberately close to Wang Fig. 9.
    metrics = [
        ("ts", "TS", "forestgreen"),
        ("pod", "POD", "limegreen"),
        ("normalized_far", "normalized FAR", "seagreen"),
        ("fss10", "FSS-10", "blue"),
        ("fss5", "FSS-5", "lightskyblue"),
        ("learned_similarity_z", "Learned similarity", "red"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(7.5, 10.0))
    titles = {
        "displacement": "Displacement biases experiments",
        "intensity": "Intensity biases experiments",
        "area_size": "Area size biases experiments",
    }

    for ax, family in zip(axes, ["displacement","intensity","area_size"]):
        g = df[df.family == family].sort_values("bias")
        for col, label, color in metrics:
            plot_metric(ax, g, col, label, color)

        x = g["bias"].to_numpy()
        ax.plot(x, ref_curve(family, x), "k:", lw=2, label="reference parabola")
        ax.set_title(titles[family])
        ax.set_ylabel("verification scores")
        ax.set_xlabel("biases")
        ax.set_ylim(-0.02, 1.05)
        ax.grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1.23, 0.5))
    fig.tight_layout(rect=[0,0,0.83,1])
    fig.savefig(outfile, dpi=220, bbox_inches="tight")
    print(outfile)


if __name__ == "__main__":
    main()

