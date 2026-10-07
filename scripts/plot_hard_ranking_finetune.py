#!/usr/bin/env python3
"""Plot compact diagnostics from hard_ranking_finetune_v1 histories."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def defaults():
    data = Path(os.environ["DATA_DIR"])
    exp = data / "derived/experiments/hard_ranking_finetune_v1"
    return exp, exp / "plots"


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(path)


def main():
    exp0, out0 = defaults()
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-dir", default=str(exp0))
    p.add_argument("--output-dir", default=str(out0))
    a = p.parse_args()

    exp = Path(a.experiment_dir)
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tr = pd.read_csv(exp / "training_history.csv")
    va = pd.read_csv(exp / "validation_history.csv")

    # 1) Training objective components (rolling mean).
    roll = min(200, max(1, len(tr) // 20))
    fig, ax = plt.subplots(figsize=(10, 6))
    for col, label in (
        ("global_contribution", "0.25 × normalized global"),
        ("tl_contribution", "1.0 × normalized TL ranking"),
        ("evolution_contribution", "1.0 × normalized evolution ranking"),
        ("total_loss", "total"),
    ):
        ax.plot(tr["step"], tr[col].rolling(roll, min_periods=1).mean(), label=label)
    ax.set_xlabel("fine-tuning step")
    ax.set_ylabel("loss contribution")
    ax.set_title(f"Hard-ranking fine-tune: training objective ({roll}-step rolling mean)")
    ax.legend()
    ax.grid(alpha=0.25)
    save(fig, out / "01_training_objective_components.png")

    # 2) Primary fixed-validation losses.
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(va["step"], va["z_global_norm_loss"], marker="o", label="global retention")
    ax.plot(va["step"], va["z_tl_norm_loss"], marker="o", label="TL ranking")
    ax.plot(va["step"], va["z_evolution_norm_loss"], marker="o", label="evolution ranking")
    ax.plot(va["step"], va["z_total_objective"], marker="o", label="weighted total")
    ax.set_xlabel("fine-tuning step")
    ax.set_ylabel("normalized loss")
    ax.set_title("Hard-ranking fine-tune: fixed-validation losses")
    ax.legend()
    ax.grid(alpha=0.25)
    save(fig, out / "02_validation_losses.png")

    # 3) Ranking success + old-task retention.
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(va["step"], va["z_global_top1_a_pct"], marker="o", label="global P top-1 vs N")
    ax.plot(va["step"], va["z_tl_full_order_pct"], marker="o", label="P > TL3 > TL6")
    ax.plot(va["step"], va["z_evolution_full_order_pct"], marker="o", label="E1 > E2 > E3 > E6")
    ax.set_xlabel("fine-tuning step")
    ax.set_ylabel("success rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Hard-ranking fine-tune: validation ranking success")
    ax.legend()
    ax.grid(alpha=0.25)
    save(fig, out / "03_validation_ranking_success.png")

    # 4) Raw TL cosine geometry.
    fig, ax = plt.subplots(figsize=(10, 6))
    for col, label in (("z_sim_p", "P / 1-h TL"), ("z_sim_tl3", "TL3"), ("z_sim_tl6", "TL6")):
        ax.plot(va["step"], va[col], marker="o", label=label)
    ax.set_xlabel("fine-tuning step")
    ax.set_ylabel("mean z cosine similarity to A")
    ax.set_title("Fixed validation: same-valid-time lag geometry")
    ax.legend()
    ax.grid(alpha=0.25)
    save(fig, out / "04_validation_timelag_geometry.png")

    # 5) Raw evolution cosine geometry.
    fig, ax = plt.subplots(figsize=(10, 6))
    for col, label in (
        ("z_sim_e1", "E1"), ("z_sim_e2", "E2"),
        ("z_sim_e3", "E3"), ("z_sim_e6", "E6"),
    ):
        ax.plot(va["step"], va[col], marker="o", label=label)
    ax.set_xlabel("fine-tuning step")
    ax.set_ylabel("mean z cosine similarity to A")
    ax.set_title("Fixed validation: same-initialization evolution geometry")
    ax.legend()
    ax.grid(alpha=0.25)
    save(fig, out / "05_validation_evolution_geometry.png")

    # 6) Backbone h secondary diagnostics.
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(va["step"], va["h_global_top1_a_pct"], marker="o", label="h global top-1")
    ax.plot(va["step"], va["h_tl_full_order_pct"], marker="o", label="h TL full order")
    ax.plot(va["step"], va["h_evolution_full_order_pct"], marker="o", label="h evolution full order")
    ax.set_xlabel("fine-tuning step")
    ax.set_ylabel("success rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Backbone h: secondary fixed-validation ranking diagnostics")
    ax.legend()
    ax.grid(alpha=0.25)
    save(fig, out / "06_validation_backbone_h.png")


if __name__ == "__main__":
    main()

