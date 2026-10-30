#!/usr/bin/env python3
"""Plot the 8-family morphology suite in Wang-style form."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FAMILY_ORDER = [
    "displacement",
    "intensity",
    "area_size",
    "smoothing_sharpening",
    "object_dispersion_cells",
    "principal_axis_log2_scale",
    "rotation_deg",
    "tile_fragmentation_cells",
]

TITLES = {
    "displacement": "Displacement biases experiments",
    "intensity": "Intensity biases experiments",
    "area_size": "Area size biases experiments",
    "smoothing_sharpening": "Sharpness vs. smoothing experiments",
    "object_dispersion_cells": "Object concentration vs. dispersion experiments",
    "principal_axis_log2_scale": "Principal-axis compression vs. elongation experiments",
    "rotation_deg": "Rotation biases experiments",
    "tile_fragmentation_cells": "Tile-fragmentation experiments",
}

XLABELS = {
    "displacement": "biases",
    "intensity": "biases",
    "area_size": "biases",
    "smoothing_sharpening": "sigma (− sharpen, + smooth)",
    "object_dispersion_cells": "grid cells (− concentrate, + disperse)",
    "principal_axis_log2_scale": "log2 principal-axis scale",
    "rotation_deg": "rotation degrees",
    "tile_fragmentation_cells": "tile jitter grid cells",
}

METRICS = [
    ("ts", "TS", "forestgreen"),
    ("pod", "POD", "limegreen"),
    ("normalized_far", "normalized FAR", "seagreen"),
    ("fss10", "FSS-10", "blue"),
    ("fss5", "FSS-5", "lightskyblue"),
    ("normalized_sal", "normalized SAL", "orange"),
    ("learned_similarity_v3", "V3 learned similarity", "red"),
]

LEARNED = [
    ("learned_similarity_v2", "V2 learned similarity", "darkred", "--"),
    ("learned_similarity_v3", "V3 learned similarity", "red", "-"),
]


def ref_curve(family, x):
    x = np.asarray(x, float)
    if family == "displacement":
        d = np.abs(x) / max(1e-12, np.max(np.abs(x)))
    elif family in ("intensity", "area_size"):
        d = np.where(x <= 1.0, np.abs(x - 1.0) / 0.5, np.abs(x - 1.0) / 1.0)
    else:
        d = np.abs(x) / max(1e-12, np.max(np.abs(x)))
    return np.maximum(0.0, 1.0 - d * d)


def plot_ci(ax, g, stem, label, color, ls="-", fill=True, lw=1.6):
    x = g.bias.to_numpy()
    y = g[f"{stem}_mean"].to_numpy()
    ax.plot(x, y, label=label, color=color, lw=lw, linestyle=ls)
    lo = f"{stem}_ci95_lo"
    hi = f"{stem}_ci95_hi"
    if fill and lo in g.columns and hi in g.columns:
        ax.fill_between(x, g[lo], g[hi], color=color, alpha=.16, linewidth=0)


def decorate(ax, fam):
    ax.axvline(0 if fam not in ("intensity", "area_size") else 1.0, color="0.5", lw=.8, alpha=.55)
    ax.set_title(TITLES[fam], fontsize=11.5)
    ax.set_ylabel("verification scores")
    ax.set_xlabel(XLABELS[fam])
    ax.set_ylim(-.02, 1.05)
    ax.grid(alpha=.25)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summary_csv")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()
    df = pd.read_csv(args.summary_csv)
    out = Path(args.output_dir) if args.output_dir else Path(args.summary_csv).parent
    out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 2, figsize=(13.2, 16.8))
    axes = axes.ravel()
    for ax, fam in zip(axes, FAMILY_ORDER):
        g = df[df.family == fam].sort_values("bias")
        for stem, label, color in METRICS:
            plot_ci(ax, g, stem, label, color)
        ax.plot(g.bias.to_numpy(), ref_curve(fam, g.bias.to_numpy()), "k:", lw=2.0, label="reference parabola")
        decorate(ax, fam)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="center right", bbox_to_anchor=(1.18, .5), frameon=False)
    fig.tight_layout(rect=[0, 0, .84, 1])
    fig.savefig(out / "morphology_wang_style_v3.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(4, 2, figsize=(13.2, 16.8))
    axes = axes.ravel()
    for ax, fam in zip(axes, FAMILY_ORDER):
        g = df[df.family == fam].sort_values("bias")
        for stem, label, color, ls in LEARNED:
            x = g.bias.to_numpy()
            y = g[f"{stem}_mean"].to_numpy()
            ax.plot(x, y, label=label, color=color, lw=1.8, linestyle=ls)
        ax.plot(g.bias.to_numpy(), ref_curve(fam, g.bias.to_numpy()), "k:", lw=1.9, label="reference parabola")
        ax.axvline(0 if fam not in ("intensity", "area_size") else 1.0, color="0.5", lw=.8, alpha=.55)
        ax.set_title(TITLES[fam], fontsize=11.5)
        ax.set_ylabel("learned cosine similarity")
        ax.set_xlabel(XLABELS[fam])
        ax.set_ylim(-.02, 1.05)
        ax.grid(alpha=.25)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="center right", bbox_to_anchor=(1.14, .5), frameon=False)
    fig.tight_layout(rect=[0, 0, .86, 1])
    fig.savefig(out / "morphology_learned_v2_vs_v3.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(out / "morphology_wang_style_v3.png")
    print(out / "morphology_learned_v2_vs_v3.png")


if __name__ == "__main__":
    main()
