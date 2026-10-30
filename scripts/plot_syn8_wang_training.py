#!/usr/bin/env python3
"""Quick-look plots for revised SYN-8 Wang-adapted training."""
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
    plt.plot(tr.step, tr.total_loss, alpha=.35, label="train total")
    plt.plot(va.step, va.total_objective, "o-", label="validation total")
    plt.xlabel("step"); plt.ylabel("weighted objective")
    plt.title("Revised SYN-8 objective")
    plt.legend(); save(out / "objective_history.png")

    plt.figure(figsize=(9,5))
    plt.plot(va.step, va.supcon_raw, "o-", label="SupCon raw")
    plt.plot(va.step, va.penalty_mae, "o-", label="penalty MAE (raw)")
    plt.xlabel("step"); plt.ylabel("raw loss")
    plt.title("Validation loss components (unweighted)")
    plt.legend(); save(out / "validation_loss_components_raw.png")

    plt.figure(figsize=(9,5))
    plt.plot(va.step, va.supcon_contribution, "o-", label="SupCon contribution")
    plt.plot(va.step, va.penalty_contribution, "o-", label="penalty contribution (x1000)")
    plt.xlabel("step"); plt.ylabel("weighted contribution")
    plt.title("Validation objective contributions")
    plt.legend(); save(out / "validation_loss_components_weighted.png")

    fams = (
        "displacement",
        "intensity",
        "area_size",
        "smoothing",
        "object_dispersion_cells",
        "principal_axis_log2_scale",
        "rotation_deg",
        "tile_fragmentation_cells",
    )
    plt.figure(figsize=(9,5))
    for f in fams:
        plt.plot(va.step, va[f+"_mae"], "o-", label=f)
    plt.xlabel("step"); plt.ylabel("|cosine distance - target distance|")
    plt.title("Validation Wang-target error by transformation")
    plt.legend(); save(out / "validation_penalty_mae_by_transform.png")

    plt.figure(figsize=(9,5))
    for f in fams:
        plt.plot(va.step, va[f+"_sim"], "o-", label=f)
    plt.xlabel("step"); plt.ylabel("mean cosine similarity")
    plt.title("Validation similarity by transformation")
    plt.legend(); save(out / "validation_similarity_by_transform.png")

    print(f"Wrote plots to {out}")


if __name__ == "__main__":
    main()

