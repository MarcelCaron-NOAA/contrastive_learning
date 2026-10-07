#!/usr/bin/env python3
"""
Compare the GroupNorm pilot against the existing BatchNorm full-epoch baseline
over their common optimization steps.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def summarize(name, df):
    v = df["z_loss"].to_numpy(float)
    jumps = np.abs(np.diff(v)) if len(v) > 1 else np.array([np.nan])
    return {
        "model": name,
        "n_validations": len(df),
        "best_z_loss": float(np.nanmin(v)),
        "final_z_loss": float(v[-1]),
        "z_loss_std": float(np.nanstd(v)),
        "median_abs_loss_jump": float(np.nanmedian(jumps)),
        "max_abs_loss_jump": float(np.nanmax(jumps)),
        "final_margin": float(df.iloc[-1]["z_margin_a"]),
        "final_top1_pct": float(df.iloc[-1]["z_top1_a_pct"]),
    }


def save(path):
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def main():
    data = Path(os.environ["DATA_DIR"])
    bn_dir = data / "derived/experiments/full_epoch_train_validation_v1"
    gn_dir = data / "derived/experiments/groupnorm_pilot_v1"

    bn = pd.read_csv(bn_dir / "validation_history.csv")
    gn = pd.read_csv(gn_dir / "validation_history.csv")
    gn_train = pd.read_csv(gn_dir / "training_history.csv")

    max_step = int(gn["step"].max())
    bn = bn[bn["step"] <= max_step].copy()

    out = gn_dir / "diagnostic_plots" / "bn_vs_gn"
    out.mkdir(parents=True, exist_ok=True)

    # Loss.
    plt.figure(figsize=(11, 6))
    plt.plot(bn["step"], bn["z_loss"], marker="o", label="BatchNorm baseline")
    plt.plot(gn["step"], gn["z_loss"], marker="o", label="GroupNorm pilot")
    plt.xlabel("Optimization step")
    plt.ylabel("Fixed-validation z loss")
    plt.title("BatchNorm vs GroupNorm: fixed-validation loss")
    plt.grid(alpha=0.25)
    plt.legend()
    save(out / "bn_vs_gn_validation_loss.png")

    # AP.
    plt.figure(figsize=(11, 5.5))
    plt.plot(bn["step"], bn["z_cosine_ap"], marker="o", label="BatchNorm A-P")
    plt.plot(gn["step"], gn["z_cosine_ap"], marker="o", label="GroupNorm A-P")
    plt.xlabel("Optimization step")
    plt.ylabel("cos(A,P)")
    plt.title("BatchNorm vs GroupNorm: positive similarity")
    plt.grid(alpha=0.25)
    plt.legend()
    save(out / "bn_vs_gn_ap.png")

    # AN.
    plt.figure(figsize=(11, 5.5))
    plt.plot(bn["step"], bn["z_cosine_an"], marker="o", label="BatchNorm mean A-N")
    plt.plot(gn["step"], gn["z_cosine_an"], marker="o", label="GroupNorm mean A-N")
    plt.xlabel("Optimization step")
    plt.ylabel("mean cos(A,N)")
    plt.title("BatchNorm vs GroupNorm: negative similarity")
    plt.grid(alpha=0.25)
    plt.legend()
    save(out / "bn_vs_gn_an.png")

    # Margin.
    plt.figure(figsize=(11, 5.5))
    plt.plot(bn["step"], bn["z_margin_a"], marker="o", label="BatchNorm")
    plt.plot(gn["step"], gn["z_margin_a"], marker="o", label="GroupNorm")
    plt.axhline(0.0, linestyle="--", alpha=0.5)
    plt.xlabel("Optimization step")
    plt.ylabel("cos(A,P) - max cos(A,N)")
    plt.title("BatchNorm vs GroupNorm: fixed-validation margin")
    plt.grid(alpha=0.25)
    plt.legend()
    save(out / "bn_vs_gn_margin.png")

    # Top1.
    plt.figure(figsize=(11, 5.5))
    plt.plot(bn["step"], bn["z_top1_a_pct"], marker="o", label="BatchNorm")
    plt.plot(gn["step"], gn["z_top1_a_pct"], marker="o", label="GroupNorm")
    plt.xlabel("Optimization step")
    plt.ylabel("Positive top-1 retrieval (%)")
    plt.title("BatchNorm vs GroupNorm: fixed-validation retrieval")
    plt.grid(alpha=0.25)
    plt.legend()
    save(out / "bn_vs_gn_top1.png")

    # GN train + validation loss.
    plt.figure(figsize=(11, 6))
    roll = gn_train["loss"].rolling(250, min_periods=25).mean()
    plt.plot(
        gn_train["step"],
        roll,
        label="GroupNorm train loss (250-step rolling mean)",
    )
    plt.plot(
        gn["step"],
        gn["z_loss"],
        marker="o",
        label="GroupNorm fixed validation",
    )
    plt.xlabel("Optimization step")
    plt.ylabel("InfoNCE loss")
    plt.title("GroupNorm pilot: train and fixed-validation loss")
    plt.grid(alpha=0.25)
    plt.legend()
    save(out / "groupnorm_train_validation_loss.png")

    summary = pd.DataFrame([
        summarize("BatchNorm", bn),
        summarize("GroupNorm", gn),
    ])
    summary_path = out / "bn_vs_gn_stability_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("=" * 84)
    print("BATCHNORM vs GROUPNORM PILOT SUMMARY")
    print("=" * 84)
    print(summary.to_string(index=False))
    print()
    print(f"Plots/summary: {out}")


if __name__ == "__main__":
    main()

