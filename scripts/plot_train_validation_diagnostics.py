#!/usr/bin/env python3
"""
Plot first-run training/validation diagnostics.

Outputs separate figures for:
  1. contrastive loss vs step
  2. positive-vs-hardest-negative margin vs step
  3. positive top-1 retrieval vs step
  4. DataLoader wait time vs GPU compute time (engineering diagnostic)

No model/data reads are required; this script only reads the saved CSV histories.
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


def defaults():
    data = Path(os.environ["DATA_DIR"])
    exp = data / "derived/experiments/first_real_train_validation_v1"
    return exp


def savefig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-dir", default=str(defaults()))
    p.add_argument("--rolling-window", type=int, default=100)
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()

    exp = Path(args.experiment_dir)
    out = (
        Path(args.output_dir)
        if args.output_dir
        else exp / "diagnostic_plots"
    )
    out.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(exp / "training_history.csv")
    val = pd.read_csv(exp / "validation_history.csv")

    if args.rolling_window < 1:
        raise SystemExit("--rolling-window must be >= 1")

    # Centered rolling means are visually easier to compare against sparse
    # validation checkpoints; min_periods keeps endpoints visible.
    w = args.rolling_window

    # ------------------------------------------------------------------
    # 1) Loss
    # ------------------------------------------------------------------
    train_loss = train["loss"].rolling(
        w, center=True, min_periods=max(1, w // 5)
    ).mean()

    plt.figure(figsize=(9, 5.5))
    plt.plot(train["step"], train_loss, label=f"train z loss ({w}-step rolling mean)")
    plt.plot(val["step"], val["z_loss"], marker="o", label="fixed validation z loss")
    if "h_loss" in val.columns:
        plt.plot(val["step"], val["h_loss"], marker="o", label="fixed validation h loss")
    plt.xlabel("Optimization step")
    plt.ylabel("Symmetric explicit-negative InfoNCE loss")
    plt.title("Training and fixed-validation contrastive loss")
    plt.grid(alpha=0.25)
    plt.legend()
    savefig(out / "01_loss_vs_step.png")

    # ------------------------------------------------------------------
    # 2) Margin
    # ------------------------------------------------------------------
    train_margin = train["margin_a"].rolling(
        w, center=True, min_periods=max(1, w // 5)
    ).mean()

    plt.figure(figsize=(9, 5.5))
    plt.plot(
        train["step"],
        train_margin,
        label=f"train A margin ({w}-step rolling mean)",
    )
    plt.plot(
        val["step"],
        val["z_margin_a"],
        marker="o",
        label="fixed validation z A margin",
    )
    if "h_margin_a" in val.columns:
        plt.plot(
            val["step"],
            val["h_margin_a"],
            marker="o",
            label="fixed validation h A margin",
        )
    plt.axhline(0.0, linewidth=1.0, linestyle="--")
    plt.xlabel("Optimization step")
    plt.ylabel("cos(A,P) - max cos(A,N)")
    plt.title("Positive-vs-hardest-negative margin")
    plt.grid(alpha=0.25)
    plt.legend()
    savefig(out / "02_margin_vs_step.png")

    # ------------------------------------------------------------------
    # 3) Retrieval
    # ------------------------------------------------------------------
    train_top1 = train["top1_a_pct"].rolling(
        w, center=True, min_periods=max(1, w // 5)
    ).mean()

    plt.figure(figsize=(9, 5.5))
    plt.plot(
        train["step"],
        train_top1,
        label=f"train A top-1 ({w}-step rolling mean)",
    )
    plt.plot(
        val["step"],
        val["z_top1_a_pct"],
        marker="o",
        label="fixed validation z top-1",
    )
    if "h_top1_a_pct" in val.columns:
        plt.plot(
            val["step"],
            val["h_top1_a_pct"],
            marker="o",
            label="fixed validation h top-1",
        )

    # K=8 in the current experiment -> 1 positive among 9 candidates.
    chance = 100.0 / 9.0
    plt.axhline(
        chance,
        linewidth=1.0,
        linestyle="--",
        label=f"random-choice chance ({chance:.1f}%)",
    )
    plt.xlabel("Optimization step")
    plt.ylabel("Positive top-1 retrieval (%)")
    plt.ylim(0, 100)
    plt.title("Positive retrieval on train and fixed validation")
    plt.grid(alpha=0.25)
    plt.legend()
    savefig(out / "03_top1_retrieval_vs_step.png")

    # ------------------------------------------------------------------
    # 4) I/O vs GPU compute
    # ------------------------------------------------------------------
    if {"data_wait_seconds", "compute_seconds"}.issubset(train.columns):
        wait = train["data_wait_seconds"].rolling(
            w, center=True, min_periods=max(1, w // 5)
        ).mean()
        compute = train["compute_seconds"].rolling(
            w, center=True, min_periods=max(1, w // 5)
        ).mean()

        plt.figure(figsize=(9, 5.5))
        plt.plot(train["step"], wait, label="DataLoader wait")
        plt.plot(train["step"], compute, label="GPU compute")
        plt.xlabel("Optimization step")
        plt.ylabel("Seconds per step")
        plt.title(f"Input wait vs GPU compute ({w}-step rolling mean)")
        plt.grid(alpha=0.25)
        plt.legend()
        savefig(out / "04_io_wait_vs_gpu_compute.png")

    # Small numerical checkpoint table for easy inspection.
    keep = ["step"]
    for c in (
        "z_loss",
        "z_margin_a",
        "z_top1_a_pct",
        "h_loss",
        "h_margin_a",
        "h_top1_a_pct",
    ):
        if c in val.columns:
            keep.append(c)
    val[keep].to_csv(out / "validation_checkpoint_summary.csv", index=False)

    print(f"Wrote diagnostics to: {out}")
    for path in sorted(out.glob("*.png")):
        print(path)
    print(out / "validation_checkpoint_summary.csv")


if __name__ == "__main__":
    main()

