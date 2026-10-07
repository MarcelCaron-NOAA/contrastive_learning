#!/usr/bin/env python3
"""
Create compact meteorological galleries for Stage-IV validation:
  * highest learned similarity
  * lowest learned similarity
  * largest learned-vs-FSS rank disagreements
  * largest learned-vs-correlation rank disagreements
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm


LEVELS = [0, .1, .25, .5, 1, 2, 5, 10, 15, 25]


def make_gallery(df, indices, fcst, obs, outdir, prefix, fss_col):
    outdir.mkdir(parents=True, exist_ok=True)
    for rank, idx in enumerate(indices, start=1):
        r = df.loc[idx]
        f = np.asarray(fcst[int(r.case_index)])
        o = np.asarray(obs[int(r.case_index)])

        fig, axes = plt.subplots(1,3,figsize=(13.5,4.5))
        norm = BoundaryNorm(LEVELS, 256, clip=True)
        im = axes[0].imshow(f, origin="lower", norm=norm)
        axes[1].imshow(o, origin="lower", norm=norm)
        diff = f-o
        lim = max(1.0, float(np.nanpercentile(np.abs(diff), 99)))
        axes[2].imshow(diff, origin="lower", vmin=-lim, vmax=lim)

        axes[0].set_title("HRRR forecast")
        axes[1].set_title("Stage IV (BUDGET)")
        axes[2].set_title("HRRR − Stage IV")
        for ax in axes:
            ax.set_xticks([]); ax.set_yticks([])

        valid = str(r.get("valid_end",""))
        fhr = int(r.get("forecast_hour",-1))
        fig.suptitle(
            f"{prefix} #{rank} | valid {valid} | F{fhr:02d} | "
            f"S={r.learned_budget:.3f} | corr={r.budget_corr:.3f} | "
            f"{fss_col}={r[fss_col]:.3f} | RMSE={r.budget_rmse:.2f} mm"
        )
        fig.colorbar(im, ax=axes[:2], fraction=.025, pad=.02, label="QPF6 (mm)")
        fig.tight_layout(rect=[0,0,1,.92])
        fig.savefig(
            outdir/f"{prefix.lower().replace(' ','_')}_{rank:02d}.png",
            dpi=170, bbox_inches="tight"
        )
        plt.close(fig)


def main():
    data = Path(os.environ["DATA_DIR"])
    p = argparse.ArgumentParser()
    p.add_argument(
        "--archive-dir",
        default=str(data/"derived/evaluation/stageiv_reference_v1"),
    )
    p.add_argument("--count", type=int, default=6)
    p.add_argument(
        "--fss-column",
        default="budget_fss_t1_w15",
    )
    args = p.parse_args()

    archive = Path(args.archive_dir)
    result = archive/"results"
    df = pd.read_csv(result/"per_case_metrics.csv.gz")
    fcst = np.load(archive/"hrrr_qpf6_mm.npy", mmap_mode="r")
    obs = np.load(archive/"stageiv_budget_qpf6_mm.npy", mmap_mode="r")
    out = result/"galleries"

    n = args.count
    make_gallery(
        df,
        df["learned_budget"].nlargest(n).index,
        fcst, obs, out/"highest", "Highest similarity", args.fss_column,
    )
    make_gallery(
        df,
        df["learned_budget"].nsmallest(n).index,
        fcst, obs, out/"lowest", "Lowest similarity", args.fss_column,
    )

    x = df["learned_budget"].rank(pct=True) - df[args.fss_column].rank(pct=True)
    make_gallery(
        df,
        x.abs().nlargest(n).index,
        fcst, obs, out/"fss_disagreement",
        "Learned–FSS disagreement", args.fss_column,
    )

    x = df["learned_budget"].rank(pct=True) - df["budget_corr"].rank(pct=True)
    make_gallery(
        df,
        x.abs().nlargest(n).index,
        fcst, obs, out/"corr_disagreement",
        "Learned–corr disagreement", args.fss_column,
    )

    print(out)


if __name__ == "__main__":
    main()

