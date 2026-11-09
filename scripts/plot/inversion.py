#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd

from src.eval.feature_inversion import MM_PER_IN, PRECIP_LEVELS_IN, PRECIP_COLORS_IN, PRECIP_OVER_IN


def tag_for(target: float) -> str:
    return ("p" if target >= 0 else "m") + f"{abs(target):.2f}".replace(".", "p")


def load_target(model_dir: Path, target: float):
    td = model_dir / f"target_{tag_for(target)}"
    field = np.load(td / "optimized_precip_mm.npy") / MM_PER_IN
    meta = json.loads((td / "summary.json").read_text())
    return field, meta


def make_cmap():
    cmap = ListedColormap(PRECIP_COLORS_IN)
    cmap.set_under("white")
    cmap.set_over(PRECIP_OVER_IN)
    return cmap, BoundaryNorm(PRECIP_LEVELS_IN, cmap.N)

def plot_model(outroot: Path, model: str, targets: list[float], original_in: np.ndarray):
    cmap, norm = make_cmap()
    fields = [original_in]
    metas = [{
        "target_similarity": 1.0,
        "achieved_similarity": 1.0,
        "reached_target_tolerance": True,
    }]

    for t in targets:
        f, m = load_target(outroot / model, t)
        fields.append(f)
        metas.append(m)

    # Original + 8 target similarities = 9 panels.
    n = len(fields)
    ncol = 3 if n == 9 else min(5, n)
    nrow = int(np.ceil(n / ncol))

    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=(3.45 * ncol, 3.65 * nrow),
        squeeze=False,
    )
    axes_flat = axes.ravel()
    mesh = None

    # Reserve dedicated space below the panels for the colorbar.
    fig.subplots_adjust(
        left=0.035,
        right=0.995,
        top=0.915,
        bottom=0.145,
        wspace=0.06,
        hspace=0.16,
    )

    for j, ax in enumerate(axes_flat):
        if j >= n:
            ax.axis("off")
            continue

        f, m = fields[j], metas[j]

        mesh = ax.pcolormesh(
            f,
            shading="auto",
            cmap=cmap,
            norm=norm,
        )

        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

        # Single-line, left/right-aligned panel headings.
        if j == 0:
            left_label = "Original"
            right_label = "S = 1.000"
        else:
            hit = bool(m.get("reached_target_tolerance", False))

            left_label = (
                f"Target {m['target_similarity']:+.2f}"
            )

            right_label = (
                f"Achieved {m['achieved_similarity']:+.3f}"
            )

            # Mark the achieved value when the optimizer could not
            # reach the requested target within tolerance.
            if not hit:
                right_label += "*"

        ax.text(
            0.0,
            1.015,
            left_label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

        ax.text(
            1.0,
            1.015,
            right_label,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Dedicated colorbar axes: completely below the panel grid.
    cax = fig.add_axes([0.22, 0.055, 0.56, 0.024])

    cbar = fig.colorbar(
        mesh,
        cax=cax,
        orientation="horizontal",
        extend="max",
    )
    cbar.set_label("6-h precipitation (in)", fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    # Define the asterisk without cluttering the individual panels.
    fig.text(
        0.995,
        0.105,
        "* target not reached within tolerance",
        ha="right",
        va="bottom",
        fontsize=8,
    )

    display_name = {
        "nfv": "NFV",
        "syn3": "SYN-3",
        "syn8": "SYN-8",
    }.get(model.lower(), model.upper())

    fig.suptitle(
        f"{display_name} learned-similarity inversion",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )

    fig.savefig(
        outroot / f"continuation_gallery_{model}.png",
        dpi=210,
        bbox_inches="tight",
    )

    plt.close(fig)

def plot_target_vs_achieved(outroot: Path):
    csv_path = outroot / "continuation_summary.csv"
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    lo = min(-1.0, float(df["target_similarity"].min()))
    hi = 1.0
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2, label="target = achieved")
    for model, g in df.groupby("model", sort=False):
        g = g.sort_values("target_similarity", ascending=False)
        ax.plot(g["target_similarity"], g["achieved_similarity"], marker="o", label=model.upper())
    ax.set_xlabel("Target cosine similarity")
    ax.set_ylabel("Achieved cosine similarity")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.grid(alpha=0.25)
    ax.legend()
    ax.set_title("Reachability of constrained feature-space targets")
    fig.tight_layout()
    fig.savefig(outroot / "continuation_target_vs_achieved.png", dpi=210)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--models", nargs="+", default=["nfv", "syn3", "syn8"])
    ap.add_argument(
        "--targets", nargs="+",
        default=["0.75", "0.50", "0.25", "0.00", "-0.25", "-0.50", "-0.75", "-1.00"],
    )
    args = ap.parse_args()
    outroot = Path(args.input_dir)
    targets = [float(v) for v in args.targets]
    original_in = np.load(outroot / "original_stageiv_precip_mm.npy") / MM_PER_IN
    for model in args.models:
        plot_model(outroot, model, targets, original_in)
    plot_target_vs_achieved(outroot)
    print(f"Wrote continuation galleries to {outroot}")


if __name__ == "__main__":
    main()
