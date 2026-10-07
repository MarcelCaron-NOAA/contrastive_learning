#!/usr/bin/env python3
"""
Plot the one-epoch training/validation trajectory.

Reads:
  training_history.csv
  validation_history.csv

Writes separate figures (no subplots) under:
  diagnostic_plots/full_epoch_history/

The validation set is fixed across checkpoints, so checkpoint-to-checkpoint
oscillation is directly comparable.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def default_exp():
    return (
        Path(os.environ["DATA_DIR"])
        / "derived/experiments/full_epoch_train_validation_v1"
    )


def save(path):
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-dir", default=str(default_exp()))
    p.add_argument("--rolling", type=int, default=250)
    args = p.parse_args()

    exp = Path(args.experiment_dir)
    train = pd.read_csv(exp / "training_history.csv")
    val = pd.read_csv(exp / "validation_history.csv")

    out = exp / "diagnostic_plots" / "full_epoch_history"
    out.mkdir(parents=True, exist_ok=True)

    best_i = val["z_loss"].idxmin()
    best_step = int(val.loc[best_i, "step"])
    best_loss = float(val.loc[best_i, "z_loss"])

    # 1) Train/validation loss.
    plt.figure(figsize=(11, 6))
    rolling = train["loss"].rolling(
        args.rolling, min_periods=max(10, args.rolling // 10)
    ).mean()
    plt.plot(train["step"], rolling, label=f"train loss ({args.rolling}-step rolling mean)")
    plt.plot(val["step"], val["z_loss"], marker="o", label="fixed validation z loss")
    plt.axvline(best_step, linestyle="--", alpha=0.6, label=f"best val step {best_step}")
    plt.xlabel("Optimization step")
    plt.ylabel("Symmetric explicit-negative InfoNCE loss")
    plt.title("Full-epoch training and fixed-validation loss")
    plt.grid(alpha=0.25)
    plt.legend()
    save(out / "loss_vs_step.png")

    # 2) Learning rate.
    plt.figure(figsize=(11, 5.5))
    plt.plot(train["step"], train["learning_rate"])
    plt.axvline(best_step, linestyle="--", alpha=0.6)
    plt.xlabel("Optimization step")
    plt.ylabel("Learning rate")
    plt.title("Warmup + cosine learning-rate schedule")
    plt.grid(alpha=0.25)
    save(out / "learning_rate_vs_step.png")

    # 3) Validation geometry.
    plt.figure(figsize=(11, 6))
    plt.plot(val["step"], val["z_cosine_ap"], marker="o", label="z cos(A,P)")
    plt.plot(val["step"], val["z_cosine_an"], marker="o", label="z mean cos(A,N)")
    plt.plot(val["step"], val["z_margin_a"], marker="o", label="z A margin")
    plt.axvline(best_step, linestyle="--", alpha=0.6)
    plt.xlabel("Optimization step")
    plt.ylabel("Cosine / margin")
    plt.title("Fixed-validation projection-space geometry")
    plt.grid(alpha=0.25)
    plt.legend()
    save(out / "validation_geometry_vs_step.png")

    # 4) Validation top-1.
    plt.figure(figsize=(11, 5.5))
    plt.plot(val["step"], val["z_top1_a_pct"], marker="o", label="projection z")
    plt.plot(val["step"], val["h_top1_a_pct"], marker="o", label="backbone h")
    plt.axvline(best_step, linestyle="--", alpha=0.6)
    plt.xlabel("Optimization step")
    plt.ylabel("Positive top-1 retrieval (%)")
    plt.title("Fixed-validation retrieval")
    plt.grid(alpha=0.25)
    plt.legend()
    save(out / "validation_top1_vs_step.png")

    # 5) Late-run validation loss, useful for diagnosing whether low LR stabilized eval.
    late = val[val["step"] >= 25000]
    plt.figure(figsize=(11, 5.5))
    plt.plot(late["step"], late["z_loss"], marker="o", label="z loss")
    plt.axvline(best_step, linestyle="--", alpha=0.6)
    plt.xlabel("Optimization step")
    plt.ylabel("Validation z loss")
    plt.title("Late-epoch fixed-validation loss")
    plt.grid(alpha=0.25)
    plt.legend()
    save(out / "late_validation_loss.png")

    # 6) Input wait and GPU time.
    if {"data_wait_seconds", "gpu_compute_seconds"}.issubset(train.columns):
        plt.figure(figsize=(11, 5.5))
        w = train["data_wait_seconds"].rolling(args.rolling, min_periods=10).mean()
        g = train["gpu_compute_seconds"].rolling(args.rolling, min_periods=10).mean()
        plt.plot(train["step"], w, label="DataLoader wait")
        plt.plot(train["step"], g, label="GPU compute")
        plt.xlabel("Optimization step")
        plt.ylabel("Seconds per step")
        plt.title("Input wait versus GPU compute")
        plt.grid(alpha=0.25)
        plt.legend()
        save(out / "io_vs_compute.png")

    print("=" * 80)
    print("FULL-EPOCH HISTORY SUMMARY")
    print("=" * 80)
    print(f"Best fixed-validation z loss : {best_loss:.4f} at step {best_step}")
    print(f"Best-step z margin           : {val.loc[best_i, 'z_margin_a']:+.4f}")
    print(f"Best-step z top1             : {val.loc[best_i, 'z_top1_a_pct']:.1f}%")
    print(f"Final z loss                 : {val.iloc[-1]['z_loss']:.4f}")
    print(f"Final z margin               : {val.iloc[-1]['z_margin_a']:+.4f}")
    print(f"Final z top1                 : {val.iloc[-1]['z_top1_a_pct']:.1f}%")
    print(f"Plots                        : {out}")


if __name__ == "__main__":
    main()

