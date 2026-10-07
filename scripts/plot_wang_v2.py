#!/usr/bin/env python3
"""Wang-style plots for V2 plus a dedicated V1-vs-V2 learned-score comparison."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ref_curve(family, x):
    x = np.asarray(x, float)
    if family == "displacement":
        d = np.abs(x)/10.0
    else:
        d = np.where(x <= 1.0, np.abs(x-1.0)/0.5, np.abs(x-1.0)/1.0)
    return np.maximum(0.0, 1.0-d*d)


def plot_ci(ax, g, stem, label, color, ls="-", fill=True, lw=1.5):
    x = g.bias.to_numpy()
    y = g[f"{stem}_mean"].to_numpy()
    ax.plot(x, y, label=label, color=color, lw=lw, linestyle=ls)
    if fill and f"{stem}_ci95_lo" in g:
        ax.fill_between(x, g[f"{stem}_ci95_lo"], g[f"{stem}_ci95_hi"], color=color, alpha=.16, linewidth=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summary_csv")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()
    df = pd.read_csv(args.summary_csv)
    out = Path(args.output_dir) if args.output_dir else Path(args.summary_csv).parent
    out.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("ts", "TS", "forestgreen"),
        ("pod", "POD", "limegreen"),
        ("normalized_far", "normalized FAR", "seagreen"),
        ("fss10", "FSS-10", "blue"),
        ("fss5", "FSS-5", "lightskyblue"),
        ("learned_similarity_z_v2", "Learned similarity", "red"),
    ]
    titles = {"displacement":"Displacement biases experiments", "intensity":"Intensity biases experiments", "area_size":"Area size biases experiments"}
    fig, axes = plt.subplots(3,1,figsize=(7.5,10.0))
    for ax,family in zip(axes,["displacement","intensity","area_size"]):
        g = df[df.family==family].sort_values("bias")
        for stem,label,color in metrics:
            plot_ci(ax,g,stem,label,color)
        x = g.bias.to_numpy()
        ax.plot(x, ref_curve(family,x), "k:", lw=2, label="reference parabola")
        ax.set_title(titles[family]); ax.set_ylabel("verification scores"); ax.set_xlabel("biases")
        ax.set_ylim(-.02,1.05); ax.grid(alpha=.25)
    h,l = axes[0].get_legend_handles_labels()
    fig.legend(h,l,loc="center right",bbox_to_anchor=(1.23,.5))
    fig.tight_layout(rect=[0,0,.83,1])
    fig.savefig(out/"wang_style_v2.png",dpi=220,bbox_inches="tight"); plt.close(fig)

    fig,axes = plt.subplots(3,1,figsize=(7.5,10.0))
    for ax,family in zip(axes,["displacement","intensity","area_size"]):
        g = df[df.family==family].sort_values("bias")
        plot_ci(ax,g,"learned_similarity_z_v1","Baseline V1","darkred",ls="--",fill=False,lw=1.7)
        plot_ci(ax,g,"learned_similarity_z_v2","Hard-ranking V2","red",ls="-",fill=True,lw=1.7)
        ax.set_title(titles[family]); ax.set_ylabel("learned cosine similarity"); ax.set_xlabel("biases")
        ax.set_ylim(-.02,1.05); ax.grid(alpha=.25)
    h,l=axes[0].get_legend_handles_labels(); fig.legend(h,l,loc="center right",bbox_to_anchor=(1.20,.5))
    fig.tight_layout(rect=[0,0,.84,1]); fig.savefig(out/"wang_learned_v1_vs_v2.png",dpi=220,bbox_inches="tight"); plt.close(fig)
    print(out/"wang_style_v2.png")
    print(out/"wang_learned_v1_vs_v2.png")

if __name__ == "__main__": main()

