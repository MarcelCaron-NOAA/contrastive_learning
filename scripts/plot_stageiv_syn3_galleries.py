#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, TwoSlopeNorm
import numpy as np
import pandas as pd

PRECIP_LEVELS_MM = [0, 0.1, 0.5, 1, 2.5, 5, 10, 20, 35, 50, 75, 100]


def take_unique(order: pd.DataFrame, n: int, used: set[int]) -> pd.DataFrame:
    rows = []
    for _, row in order.iterrows():
        ci = int(row["case_index"])
        if ci in used:
            continue
        rows.append(row)
        used.add(ci)
        if len(rows) == n:
            break
    return pd.DataFrame(rows)


def first_present(row, names, default=""):
    for name in names:
        if name in row.index and pd.notna(row[name]):
            return row[name]
    return default


def plot_case(row, fcst, obs, path, bucket_desc, primary_label, compare_label, natural_label, threshold_mm, dpi):
    i = int(row["case_index"])
    f = np.asarray(fcst[i], dtype=float)
    o = np.asarray(obs[i], dtype=float)
    d = f - o

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8))
    norm = BoundaryNorm(PRECIP_LEVELS_MM, ncolors=256, clip=True)
    im0 = axes[0].imshow(f, origin="lower", norm=norm, cmap="viridis")
    axes[1].imshow(o, origin="lower", norm=norm, cmap="viridis")
    dlim = max(1.0, float(np.nanpercentile(np.abs(d), 99)))
    im2 = axes[2].imshow(
        d,
        origin="lower",
        norm=TwoSlopeNorm(vcenter=0.0, vmin=-dlim, vmax=dlim),
        cmap="RdBu_r",
    )

    axes[0].set_title("HRRR forecast")
    axes[1].set_title("Stage IV (BUDGET)")
    axes[2].set_title("HRRR − Stage IV")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    valid = first_present(row, ["valid_end", "valid_time", "valid"], "")
    fhr = first_present(row, ["forecast_hour", "lead_hours", "fhr"], None)
    fhr_txt = f"F{int(fhr):02d}" if fhr is not None and str(fhr) != "" else ""
    rank = int(row["bucket_rank"])
    bucket = str(row["bucket"])
    inches = threshold_mm / 25.4

    pieces = [f"{primary_label}={row.learned_synthetic:.3f}"]
    if "learned_compare" in row.index and pd.notna(row["learned_compare"]):
        pieces.append(f"{compare_label}={row.learned_compare:.3f}")
    if "learned_v3" in row.index and pd.notna(row["learned_v3"]):
        pieces.append(f"{natural_label}={row.learned_v3:.3f}")
    pieces.extend([
        f"FSS-5={row.fss5:.3f}",
        f"FSS-10={row.fss10:.3f}",
        f"ETS({inches:.2f}\")={row.ets:.3f}",
    ])

    fig.suptitle(
        f"{bucket}{rank}: {bucket_desc} | valid {valid} {fhr_txt}\n" + " | ".join(pieces),
        fontsize=13,
        fontweight="bold",
    )
    cb0 = fig.colorbar(im0, ax=axes[:2], fraction=0.024, pad=0.02)
    cb0.set_label("6-h precipitation (mm)")
    cb2 = fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.03)
    cb2.set_label("difference (mm)")
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    data = Path(os.environ["DATA_DIR"])
    default_syn3 = data / "derived/experiments/synthetic_perturb_syn3_v1/evaluation/stageiv_vs_v3"
    default_syn8 = data / "derived/experiments/synthetic_perturb_all8_v1/evaluation/stageiv_vs_v3"
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--primary-evaluation-dir",
        default=str(default_syn3),
        help="Stage-IV evaluation dir for the model being ranked (SYN-3).",
    )
    ap.add_argument(
        "--comparison-evaluation-dir",
        default=str(default_syn8),
        help="Stage-IV evaluation dir for the comparison model used in bucket C (SYN-8).",
    )
    ap.add_argument("--archive-dir", default=str(data / "derived/evaluation/stageiv_reference_v1"))
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--count", type=int, default=5)
    ap.add_argument("--min-wet-pixels", type=int, default=16)
    ap.add_argument("--primary-label", default="SYN-3")
    ap.add_argument("--comparison-label", default="SYN-8")
    ap.add_argument("--natural-label", default="NFV (V3)")
    ap.add_argument("--dpi", type=int, default=180)
    args = ap.parse_args()

    pev = Path(args.primary_evaluation_dir)
    cev = Path(args.comparison_evaluation_dir)
    arc = Path(args.archive_dir)
    out = Path(args.output_dir) if args.output_dir else pev / "galleries_syn3_vs_syn8"
    out.mkdir(parents=True, exist_ok=True)

    base = pd.read_csv(pev / "per_case_metrics.csv.gz").copy()
    cmp_df = pd.read_csv(cev / "per_case_metrics.csv.gz")[["case_index", "learned_synthetic"]].copy()
    cmp_df = cmp_df.rename(columns={"learned_synthetic": "learned_compare"})
    df = base.merge(cmp_df, on="case_index", how="left", validate="one_to_one")
    if df["learned_compare"].isna().any():
        missing = int(df["learned_compare"].isna().sum())
        raise RuntimeError(f"Comparison learned scores missing for {missing} cases after merge")

    fcst = np.load(arc / "hrrr_qpf6_mm.npy", mmap_mode="r")
    obs = np.load(arc / "stageiv_budget_qpf6_mm.npy", mmap_mode="r")
    meta = json.loads((pev / "run_metadata.json").read_text())
    threshold_mm = float(meta["metric_threshold_mm"])

    wet = df[(df.fcst_wet_pixels >= args.min_wet_pixels) & (df.obs_wet_pixels >= args.min_wet_pixels)].copy()
    if len(wet) < 4 * args.count:
        raise RuntimeError(f"Only {len(wet)} non-dry cases; need at least {4*args.count} for unique A-D galleries")

    wet["primary_rank"] = wet.learned_synthetic.rank(method="average", pct=True)
    wet["fss_rank"] = wet.fss_mean.rank(method="average", pct=True)
    wet["rank_gap_fss"] = (wet.primary_rank - wet.fss_rank).abs()
    wet["joint_rank_fss"] = 0.5 * (wet.primary_rank + wet.fss_rank)
    wet["abs_delta_vs_compare"] = (wet.learned_synthetic - wet.learned_compare).abs()

    used: set[int] = set()
    selected = []

    a = take_unique(wet.sort_values("learned_synthetic", ascending=False), args.count, used)
    a["bucket"] = "A"
    a["bucket_rank"] = np.arange(1, len(a) + 1)
    a["bucket_desc"] = "highest similarities"
    selected.append(a)

    b_order = wet.sort_values(["rank_gap_fss", "joint_rank_fss"], ascending=[True, False])
    b = take_unique(b_order, args.count, used)
    b["bucket"] = "B"
    b["bucket_rank"] = np.arange(1, len(b) + 1)
    b["bucket_desc"] = f"best {args.primary_label} vs FSS agreement"
    selected.append(b)

    c = take_unique(wet.sort_values("abs_delta_vs_compare", ascending=False), args.count, used)
    c["bucket"] = "C"
    c["bucket_rank"] = np.arange(1, len(c) + 1)
    c["bucket_desc"] = f"least {args.primary_label} vs {args.comparison_label} agreement"
    selected.append(c)

    d = take_unique(wet.sort_values("rank_gap_fss", ascending=False), args.count, used)
    d["bucket"] = "D"
    d["bucket_rank"] = np.arange(1, len(d) + 1)
    d["bucket_desc"] = f"least {args.primary_label} vs FSS agreement"
    selected.append(d)

    picks = pd.concat(selected, ignore_index=True)
    picks.to_csv(out / "selected_gallery_cases.csv", index=False)

    for _, row in picks.iterrows():
        bucket = str(row.bucket)
        rank = int(row.bucket_rank)
        bdir = out / f"bucket_{bucket}"
        bdir.mkdir(parents=True, exist_ok=True)
        plot_case(
            row,
            fcst,
            obs,
            bdir / f"{bucket}{rank:02d}_case{int(row.case_index):04d}.png",
            str(row.bucket_desc),
            args.primary_label,
            args.comparison_label,
            args.natural_label,
            threshold_mm,
            args.dpi,
        )

    print("=" * 88)
    print("STAGE-IV SYNTHETIC GALLERIES COMPLETE")
    print("=" * 88)
    print(f"Non-dry candidates : {len(wet)}")
    print(f"Selected cases     : {len(picks)}")
    print(f"Output             : {out}")
    cols = ["bucket", "bucket_rank", "case_index", "learned_synthetic", "learned_compare", "learned_v3", "fss5", "fss10", "ets"]
    print(
        picks[cols]
        .rename(columns={
            "learned_synthetic": args.primary_label,
            "learned_compare": args.comparison_label,
            "learned_v3": args.natural_label,
        })
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
