#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

FAMILIES = [
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

XLABEL = {
    "displacement": "horizontal displacement (grid cells)",
    "intensity": "multiplicative intensity factor",
    "area_size": "area scale factor",
    "smoothing_sharpening": "sigma (− sharpen, + smooth)",
    "object_dispersion_cells": "grid cells (− concentrate, + disperse)",
    "principal_axis_log2_scale": "log2 principal-axis scale",
    "rotation_deg": "rotation (degrees)",
    "tile_fragmentation_cells": "tile jitter (grid cells)",
}

COLORS = {
    "ets": "darkgreen",
    "pod": "green",
    "success_ratio": "lightgreen",
    "fss10": "navy",
    "fss5": "lightskyblue",
    "synthetic": "crimson",
}


def add_metric(ax, g, stem, label, color, *, ci=True, ls="-", lw=1.8):
    x = g.bias.to_numpy(dtype=float)
    y = g[f"{stem}_mean"].to_numpy(dtype=float)
    ax.plot(x, y, color=color, label=label, linestyle=ls, linewidth=lw)
    lo = f"{stem}_ci95_lo"
    hi = f"{stem}_ci95_hi"
    if ci and lo in g.columns and hi in g.columns:
        ax.fill_between(
            x,
            g[lo].to_numpy(dtype=float),
            g[hi].to_numpy(dtype=float),
            color=color,
            alpha=0.13,
            linewidth=0,
        )


def decorate(ax, family):
    identity = 1.0 if family in ("intensity", "area_size") else 0.0
    ax.axvline(identity, color="0.45", lw=0.8, alpha=0.6)
    ax.set_title(TITLES[family], fontsize=11)
    ax.set_xlabel(XLABEL[family])
    ax.set_ylabel("verification score")
    ax.set_ylim(-0.03, 1.04)
    ax.grid(alpha=0.22)


def main():
    data = Path(os.environ["DATA_DIR"])
    default_dir = data / "derived/experiments/synthetic_perturb_syn3_v1/evaluation/morphology_vs_v3"
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluation-dir", default=str(default_dir))
    ap.add_argument("--synthetic-label", default="SYN-3")
    ap.add_argument("--output-file", default=None)
    args = ap.parse_args()

    ev = Path(args.evaluation_dir)
    df = pd.read_csv(ev / "morphology_summary.csv")
    out = Path(args.output_file) if args.output_file else ev / "morphology_traditional_plus_syn3.png"

    fig, axes = plt.subplots(4, 2, figsize=(12.0, 16.2))
    axes = axes.ravel()
    metrics = [
        ("ets", "ETS", COLORS["ets"]),
        ("pod", "POD", COLORS["pod"]),
        ("success_ratio", "Success Ratio", COLORS["success_ratio"]),
        ("fss5", "FSS-5", COLORS["fss5"]),
        ("fss10", "FSS-10", COLORS["fss10"]),
        ("learned_synthetic", args.synthetic_label, COLORS["synthetic"]),
    ]
    for ax, family in zip(axes, FAMILIES):
        g = df[df.family == family].sort_values("bias")
        if len(g) == 0:
            raise RuntimeError(f"Missing morphology family: {family}")
        for stem, label, color in metrics:
            add_metric(ax, g, stem, label, color, ci=True)
        ax.plot(
            g.bias.to_numpy(dtype=float),
            g.synthetic_target.to_numpy(dtype=float),
            color="black",
            linestyle=":",
            linewidth=1.8,
            label="synthetic training target",
        )
        decorate(ax, family)

    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="center right", bbox_to_anchor=(1.14, 0.5), frameon=False)
    fig.suptitle("Controlled precipitation perturbation response", fontsize=15, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 0.86, 0.985])
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
