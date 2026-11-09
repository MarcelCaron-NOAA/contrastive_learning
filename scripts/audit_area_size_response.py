#!/usr/bin/env python3
"""Print the area-size response table to diagnose local non-monotonicity."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import pandas as pd


def main():
    data = Path(os.environ["DATA_DIR"])
    default = data / "derived/evaluation/final_syn_nfv_morphology_v1/morphology_summary_inches.csv"
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-csv", default=str(default))
    args = ap.parse_args()
    df = pd.read_csv(args.summary_csv)
    g = df[df.family == "area_size"].sort_values("bias").copy()
    cols = [
        "bias", "normalized_severity", "wang_target_similarity",
        "syn3_mean", "syn8_mean", "nfv_mean",
        "ets_mean", "fss5_mean", "fss10_mean",
    ]
    for c in ("pixel_mae_mm_mean", "pixel_rmse_mm_mean", "pixel_corr_mean", "total_precip_ratio_mean"):
        if c in g.columns:
            cols.append(c)
    print(g[cols].to_string(index=False))
    print("\nExpected revised training mapping on the contraction side:")
    for rank in range(1, 11):
        factor = 1.0 - 0.05 * rank
        target = 1.0 - 0.009 * rank * rank
        print(f"  factor={factor:4.2f}  rank={rank:2d}  target_similarity={target:.3f}")


if __name__ == "__main__":
    main()
