#!/usr/bin/env python3
"""
One-time visual sanity check before scientific Stage-IV analysis.

Plots HRRR, Stage-IV BUDGET, and Stage-IV BILIN for a small deterministic
selection of archive cases. This is intended to catch gross orientation,
translation, or unit problems before computing model scores.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm


LEVELS = [0,.1,.25,.5,1,2,5,10,15,25]


def main():
    data = Path(os.environ["DATA_DIR"])
    p = argparse.ArgumentParser()
    p.add_argument(
        "--archive-dir",
        default=str(data/"derived/evaluation/stageiv_reference_v1"),
    )
    p.add_argument("--count", type=int, default=8)
    p.add_argument("--seed", type=int, default=20261004)
    args = p.parse_args()

    archive = Path(args.archive_dir)
    cases = pd.read_csv(archive/"cases.csv")
    f = np.load(archive/"hrrr_qpf6_mm.npy", mmap_mode="r")
    b = np.load(archive/"stageiv_budget_qpf6_mm.npy", mmap_mode="r")
    l = np.load(archive/"stageiv_bilin_qpf6_mm.npy", mmap_mode="r")

    out = archive/"alignment_check"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(cases), size=min(args.count,len(cases)), replace=False)

    norm = BoundaryNorm(LEVELS, 256, clip=True)
    for n, i in enumerate(idx,1):
        row = cases.iloc[i]
        fig, axes = plt.subplots(1,3,figsize=(13,4.2))
        im = axes[0].imshow(f[i], origin="lower", norm=norm)
        axes[1].imshow(b[i], origin="lower", norm=norm)
        axes[2].imshow(l[i], origin="lower", norm=norm)
        for ax,title in zip(
            axes, ["HRRR QPF6","Stage IV BUDGET","Stage IV BILIN"]
        ):
            ax.set_title(title)
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(
            f"Alignment check {n} | valid={row.valid_end} | "
            f"F{int(row.forecast_hour):02d}"
        )
        fig.colorbar(im, ax=axes, fraction=.024, pad=.02, label="mm / 6 h")
        fig.savefig(out/f"alignment_{n:02d}.png",dpi=170,bbox_inches="tight")
        plt.close(fig)

    print(out)
    print(
        "Inspect for: same geographic orientation, plausible spatial overlap, "
        "same units/order of magnitude, and BUDGET/BILIN differences that look "
        "like interpolation rather than translation/rotation."
    )


if __name__ == "__main__":
    main()

