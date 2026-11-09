#!/usr/bin/env python3
"""Shared Wang-like plotting conventions for controlled-transform diagnostics."""
from __future__ import annotations

METRICS = [
    ("ts", "TS", "forestgreen"),
    ("pod", "POD", "limegreen"),
    ("normalized_far", "normalized FAR", "seagreen"),
    ("fss10", "FSS-10", "blue"),
    ("fss5", "FSS-5", "lightskyblue"),
    ("learned_similarity_z", "Learned similarity", "red"),
]


def plot_metric(ax, g, col, label, color, *, suffix="", linestyle="-", alpha_fill=0.16, lw=1.5):
    mean_col = f"{col}{suffix}_mean"
    lo_col = f"{col}{suffix}_ci95_lo"
    hi_col = f"{col}{suffix}_ci95_hi"
    if mean_col not in g.columns:
        return
    x = g["bias"].to_numpy()
    y = g[mean_col].to_numpy()
    ax.plot(x, y, label=label, color=color, lw=lw, linestyle=linestyle)
    if lo_col in g.columns and hi_col in g.columns and alpha_fill > 0:
        lo = g[lo_col].to_numpy()
        hi = g[hi_col].to_numpy()
        ax.fill_between(x, lo, hi, color=color, alpha=alpha_fill, linewidth=0)

