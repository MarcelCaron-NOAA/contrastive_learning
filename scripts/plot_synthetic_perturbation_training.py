#!/usr/bin/env python3
"""Quick-look plots for synthetic-perturbation training/validation history."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def save(path):
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment-dir", required=True)
    args = ap.parse_args()
    d = Path(args.experiment_dir)
    out = d / "plots"
    out.mkdir(parents=True, exist_ok=True)
    tr = pd.read_csv(d / "training_history.csv")
    va = pd.read_csv(d / "validation_history.csv")

    plt.figure(figsize=(9,5))
    plt.plot(tr.step, tr.total_loss, alpha=.45, label="train total")
    plt.plot(va.step, va.total_objective, "o-", label="validation total")
    plt.xlabel("step"); plt.ylabel("objective"); plt.title("Synthetic perturbation objective")
    plt.legend(); save(out / "objective_history.png")

    plt.figure(figsize=(9,5))
    plt.plot(va.step, va.global_norm, "o-", label="global InfoNCE / log(K+1)")
    plt.plot(va.step, va.penalty, "o-", label="penalty target MAE")
    plt.xlabel("step"); plt.ylabel("loss"); plt.title("Validation objective components")
    plt.legend(); save(out / "validation_loss_components.png")

    fams = []
    for c in va.columns:
        if c.endswith("_mae") and c not in ("target_mae",):
            fams.append(c[:-4])
    if fams:
        plt.figure(figsize=(10,6))
        for fam in fams:
            plt.plot(va.step, va[fam + "_mae"], marker="o", label=fam)
        plt.xlabel("step"); plt.ylabel("|similarity - target|")
        plt.title("Validation penalty error by transformation")
        plt.legend(fontsize=8, ncol=2)
        save(out / "validation_penalty_mae_by_transform.png")

        plt.figure(figsize=(10,6))
        for fam in fams:
            plt.plot(va.step, va[fam + "_sim"], marker="o", label=fam)
        plt.xlabel("step"); plt.ylabel("mean cosine similarity")
        plt.title("Validation similarity by transformation")
        plt.legend(fontsize=8, ncol=2)
        save(out / "validation_similarity_by_transform.png")

    print(f"Wrote plots to {out}")


if __name__ == "__main__":
    main()

