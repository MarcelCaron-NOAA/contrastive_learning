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


def add_line(ax, g, stem, label, color, lw=2.0, ls="-"):
    x = g.bias.to_numpy(dtype=float)
    y = g[f"{stem}_mean"].to_numpy(dtype=float)
    ax.plot(x, y, color=color, linewidth=lw, linestyle=ls, label=label)


def decorate(ax, family):
    identity = 1.0 if family in ("intensity", "area_size") else 0.0
    ax.axvline(identity, color="0.45", lw=0.8, alpha=0.6)
    ax.set_title(TITLES[family], fontsize=11)
    ax.set_xlabel(XLABEL[family])
    ax.set_ylabel("learned cosine similarity")
    ax.set_ylim(-0.03, 1.04)
    ax.grid(alpha=0.22)


def main():
    data = Path(os.environ["DATA_DIR"])
    default_syn3 = data / "derived/experiments/synthetic_perturb_syn3_v1/evaluation/morphology_vs_v3"
    default_syn8 = data / "derived/experiments/synthetic_perturb_all8_v1/evaluation/morphology_vs_v3"

    ap = argparse.ArgumentParser()
    ap.add_argument("--syn3-evaluation-dir", default=str(default_syn3))
    ap.add_argument("--syn8-evaluation-dir", default=str(default_syn8))
    ap.add_argument("--output-file", default=None)
    ap.add_argument("--syn3-label", default="SYN-3")
    ap.add_argument("--syn8-label", default="SYN-8")
    args = ap.parse_args()

    syn3_dir = Path(args.syn3_evaluation_dir)
    syn8_dir = Path(args.syn8_evaluation_dir)
    df3 = pd.read_csv(syn3_dir / "morphology_summary.csv")
    df8 = pd.read_csv(syn8_dir / "morphology_summary.csv")

    out = Path(args.output_file) if args.output_file else syn3_dir / "morphology_syn3_vs_syn8.png"

    fig, axes = plt.subplots(4, 2, figsize=(12.0, 16.2))
    axes = axes.ravel()
    for ax, family in zip(axes, FAMILIES):
        g3 = df3[df3.family == family].sort_values("bias")
        g8 = df8[df8.family == family].sort_values("bias")
        if len(g3) == 0 or len(g8) == 0:
            raise RuntimeError(f"Missing family {family} in one of the morphology summaries")
        add_line(ax, g3, "learned_synthetic", args.syn3_label, "crimson")
        add_line(ax, g8, "learned_synthetic", args.syn8_label, "black")
        ax.plot(
            g3.bias.to_numpy(dtype=float),
            g3.synthetic_target.to_numpy(dtype=float),
            color="0.35",
            linestyle=":",
            linewidth=1.5,
            label="synthetic training target",
        )
        decorate(ax, family)

    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="center right", bbox_to_anchor=(1.14, 0.5), frameon=False)
    fig.suptitle("Learned similarity response to controlled precipitation perturbations", fontsize=15, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 0.86, 0.985])
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
