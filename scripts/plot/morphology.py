#!/usr/bin/env python3
"""Make the four compact 8-panel morphology figures.

Traditional ETS/POD/SR/FSS values in the revised summary all use a 0.1-inch
(2.54-mm internal) precipitation threshold. The three model-vs-traditional
figures show per-case 95% confidence intervals as translucent bands; the
learned-model comparison figure intentionally has no confidence intervals.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FAMILIES = [
    "displacement",
    "intensity",
    "area_size",
    "smoothing",
    "object_dispersion_cells",
    "principal_axis_log2_scale",
    "rotation_deg",
    "tile_fragmentation_cells",
]

TITLES = {
    "displacement": "Displacement",
    "intensity": "Intensity",
    "area_size": "Area size",
    "smoothing": "Smoothing",
    "object_dispersion_cells": "Object concentration / dispersion",
    "principal_axis_log2_scale": "Principal-axis compression / elongation",
    "rotation_deg": "Rotation",
    "tile_fragmentation_cells": "Tile fragmentation",
}

XLABELS = {
    "displacement": "horizontal displacement (grid cells)",
    "intensity": "multiplicative intensity factor",
    "area_size": "area scale factor",
    "smoothing": "Gaussian smoothing sigma",
    "object_dispersion_cells": "grid cells (− concentrate, + disperse)",
    "principal_axis_log2_scale": "log2 principal-axis scale",
    "rotation_deg": "rotation (degrees)",
    "tile_fragmentation_cells": "tile jitter (grid cells)",
}

COLORS = {
    "ets": "darkgreen",
    "pod": "green",
    "success_ratio": "lightgreen",
    "fss5": "lightskyblue",
    "fss10": "navy",
    "learned": "black",
    "target": "0.45",
    "syn3": "red",
    "syn8": "red",
    "nfv": "black",
}


def defaults():
    data = Path(os.environ["DATA_DIR"])
    base = data / "derived/evaluation/final_syn_nfv_morphology_v1"
    return {"summary": base / "morphology_summary_inches.csv", "out": base / "plots"}


def add_curve(ax, g, stem, label, color, linestyle="-", linewidth=1.8, show_ci=False):
    x = g["bias"].to_numpy(dtype=float)
    y = g[f"{stem}_mean"].to_numpy(dtype=float)
    ax.plot(x, y, color=color, linestyle=linestyle, linewidth=linewidth, label=label)
    if show_ci:
        lo = f"{stem}_ci95_lo"
        hi = f"{stem}_ci95_hi"
        if lo in g.columns and hi in g.columns:
            ax.fill_between(
                x,
                g[lo].to_numpy(dtype=float),
                g[hi].to_numpy(dtype=float),
                color=color, alpha=0.10, linewidth=0,
            )


def decorate(ax, family):
    identity = 1.0 if family in ("intensity", "area_size") else 0.0
    ax.axvline(identity, color="0.65", linewidth=0.8, alpha=0.65)
    ax.set_title(TITLES[family], fontsize=10.6, pad=4)
    ax.set_xlabel(XLABELS[family], fontsize=9.2, labelpad=2)
    ax.set_ylabel("verification score", fontsize=9.2, labelpad=2)
    ax.set_ylim(-0.055, 1.035)
    ax.tick_params(labelsize=8.5, pad=2)
    ax.grid(alpha=0.20, linewidth=0.7)


def compact_layout(fig, axes, title, legend_handles, legend_labels):
    fig.suptitle(title, fontsize=14.5, fontweight="bold", y=0.988)
    # Keep panels close together and leave only a narrow strip for the legend.
    fig.subplots_adjust(
        left=0.065, right=0.845, bottom=0.055, top=0.947,
        wspace=0.20, hspace=0.28,
    )
    fig.legend(
        legend_handles, legend_labels,
        loc="center left", bbox_to_anchor=(0.855, 0.50),
        frameon=False, fontsize=9.5,
        borderaxespad=0.0, handlelength=2.8, labelspacing=0.55,
    )


def plot_model_vs_traditional(df, model_key, model_label, output):
    fig, axes = plt.subplots(4, 2, figsize=(11.9, 13.7))
    axes = axes.ravel()

    for ax, family in zip(axes, FAMILIES):
        g = df[df.family == family].sort_values("bias")
        if g.empty:
            raise RuntimeError(f"Missing morphology family: {family}")
        add_curve(ax, g, "ets", "ETS-0.1", COLORS["ets"], show_ci=True)
        add_curve(ax, g, "pod", "POD-0.1", COLORS["pod"], show_ci=True)
        add_curve(ax, g, "success_ratio", "SR-0.1", COLORS["success_ratio"], show_ci=True)
        add_curve(ax, g, "fss5", "FSS-5", COLORS["fss5"], show_ci=True)
        add_curve(ax, g, "fss10", "FSS-10", COLORS["fss10"], show_ci=True)
        add_curve(ax, g, model_key, model_label, "black", linewidth=2.1, show_ci=True)
        ax.plot(
            g["bias"].to_numpy(dtype=float),
            g["wang_target_similarity"].to_numpy(dtype=float),
            color=COLORS["target"], linestyle="--", linewidth=1.7,
            label="Wang target",
        )
        decorate(ax, family)

    handles, labels = axes[0].get_legend_handles_labels()
    compact_layout(
        fig, axes,
        f"{model_label} response to controlled precipitation perturbations",
        handles, labels,
    )
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_models(df, output):
    fig, axes = plt.subplots(4, 2, figsize=(11.9, 13.7))
    axes = axes.ravel()

    for ax, family in zip(axes, FAMILIES):
        g = df[df.family == family].sort_values("bias")
        if g.empty:
            raise RuntimeError(f"Missing morphology family: {family}")
        add_curve(ax, g, "nfv", "NFV", "black", linewidth=2.1, show_ci=False)
        add_curve(ax, g, "syn3", "SYN-3", "red", linewidth=2.0, show_ci=False)
        add_curve(ax, g, "syn8", "SYN-8", "red", linestyle="--", linewidth=2.0, show_ci=False)
        decorate(ax, family)
        ax.set_ylabel("learned cosine similarity", fontsize=9.2, labelpad=2)

    handles, labels = axes[0].get_legend_handles_labels()
    compact_layout(
        fig, axes,
        "Learned similarity response to controlled precipitation perturbations",
        handles, labels,
    )
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    d = defaults()
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-csv", default=str(d["summary"]))
    ap.add_argument("--output-dir", default=str(d["out"]))
    args = ap.parse_args()

    df = pd.read_csv(args.summary_csv)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    plot_model_vs_traditional(
        df, "syn3", "SYN-3", out / "morphology_syn3_vs_traditional.png"
    )
    plot_model_vs_traditional(
        df, "syn8", "SYN-8", out / "morphology_syn8_vs_traditional.png"
    )
    plot_model_vs_traditional(
        df, "nfv", "NFV", out / "morphology_nfv_vs_traditional.png"
    )
    plot_models(df, out / "morphology_nfv_vs_syn3_syn8.png")

    print("MORPHOLOGY PLOTS COMPLETE")
    for p in sorted(out.glob("*.png")):
        print(p)


if __name__ == "__main__":
    main()
