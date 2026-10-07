#!/usr/bin/env python3
"""
Create a small qualitative gallery from the fixed validation set.

Selections (default 3 each):
  - best positive: highest z(A,P)
  - worst positive: lowest z(A,P)
  - hard negative: highest z(A,N_hard)

Each case image contains A, P, and the hardest valid N using the identical
geographic crop. Fields are displayed as log1p(QPF6 mm) with a common scale
within the case so spatial structure can be compared directly.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import HRRRArchive


def default_exp():
    return (
        Path(os.environ["DATA_DIR"])
        / "derived/experiments/first_real_train_validation_v1"
    )


def default_manifest():
    return (
        Path(os.environ["DATA_DIR"])
        / "forecasts/hrrr/qpf6_native/metadata/hrrr_qpf6_manifest.csv.gz"
    )


def newest(exp, pattern):
    found = sorted(exp.glob(pattern))
    if not found:
        raise SystemExit(f"No files match {pattern} in {exp}")
    return found[-1]


def load_recipe_map(path):
    out = {}
    with Path(path).open() as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            out[str(d["anchor_id"])] = d
    return out


def choose_cases(df, n):
    chosen = {}
    used = set()

    best = df.sort_values("z_cosine_ap", ascending=False)
    rows = []
    for _, r in best.iterrows():
        if r["anchor_id"] in used:
            continue
        rows.append(r)
        used.add(r["anchor_id"])
        if len(rows) == n:
            break
    chosen["best_positive"] = rows

    worst = df.sort_values("z_cosine_ap", ascending=True)
    rows = []
    for _, r in worst.iterrows():
        if r["anchor_id"] in used:
            continue
        rows.append(r)
        used.add(r["anchor_id"])
        if len(rows) == n:
            break
    chosen["worst_positive"] = rows

    hard = df.sort_values("z_cosine_an_max", ascending=False)
    rows = []
    for _, r in hard.iterrows():
        if r["anchor_id"] in used:
            continue
        rows.append(r)
        used.add(r["anchor_id"])
        if len(rows) == n:
            break
    chosen["hard_negative"] = rows

    return chosen


def make_case_plot(
    archive,
    recipe,
    summary_row,
    category,
    rank,
    outpath,
):
    y0 = int(recipe["y0"])
    x0 = int(recipe["x0"])
    size = int(recipe["patch_size"])

    aid = str(summary_row["anchor_id"])
    pid = str(summary_row["positive_id"])
    nid = str(summary_row["z_hardest_negative_id"])

    a = archive.read_patch(aid, y0, x0, size)
    p = archive.read_patch(pid, y0, x0, size)
    n = archive.read_patch(nid, y0, x0, size)

    fields = [a, p, n]
    disp = [np.log1p(np.maximum(x, 0.0)) for x in fields]

    vmax = max(
        float(np.quantile(x[np.isfinite(x)], 0.995))
        if np.isfinite(x).any()
        else 1.0
        for x in disp
    )
    vmax = max(vmax, 1.0e-6)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    titles = [
        (
            "Anchor A\n"
            f"F{int(summary_row['anchor_forecast_hour']):02d} | "
            f"P90wet={summary_row['anchor_wet_pixel_p90_mm']:.2f} mm"
        ),
        (
            "Positive P\n"
            f"z(A,P)={summary_row['z_cosine_ap']:.3f}"
        ),
        (
            "Hardest valid N\n"
            f"z(A,N)={summary_row['z_cosine_an_max']:.3f} | "
            f"margin={summary_row['z_margin']:+.3f}"
        ),
    ]

    last_im = None
    for ax, arr, title in zip(axes, disp, titles):
        last_im = ax.imshow(
            arr,
            origin="lower",
            vmin=0.0,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    cbar = fig.colorbar(
        last_im,
        ax=axes.ravel().tolist(),
        shrink=0.88,
        pad=0.02,
    )
    cbar.set_label("log1p(QPF6 mm)")

    fig.suptitle(
        f"{category.replace('_', ' ').title()} #{rank} | "
        f"valid {summary_row['anchor_valid_time']} | "
        f"{summary_row['season']} | "
        f"{summary_row['intensity_bin']}"
    )
    fig.subplots_adjust(
        left=0.02,
        right=0.91,
        bottom=0.05,
        top=0.83,
        wspace=0.06,
    )
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-dir", default=str(default_exp()))
    p.add_argument("--manifest", default=str(default_manifest()))
    p.add_argument("--anchor-csv", default=None)
    p.add_argument("--recipes", default=None)
    p.add_argument("--cases-per-category", type=int, default=3)
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()

    exp = Path(args.experiment_dir)
    anchor_path = (
        Path(args.anchor_csv)
        if args.anchor_csv
        else newest(exp, "validation_similarity_anchor_summary_step*.csv")
    )
    recipe_path = (
        Path(args.recipes)
        if args.recipes
        else exp / "fixed_validation_recipes.jsonl"
    )
    out = (
        Path(args.output_dir)
        if args.output_dir
        else exp / "diagnostic_plots" / "case_gallery"
    )
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(anchor_path)
    df = df.loc[
        np.isfinite(df["z_cosine_ap"])
        & np.isfinite(df["z_cosine_an_max"])
    ].copy()

    recipes = load_recipe_map(recipe_path)
    choices = choose_cases(df, args.cases_per_category)

    archive = HRRRArchive(args.manifest)
    selection_rows = []

    try:
        for category, rows in choices.items():
            for rank, row in enumerate(rows, start=1):
                aid = str(row["anchor_id"])
                if aid not in recipes:
                    raise RuntimeError(
                        f"Anchor {aid} missing from fixed recipe file"
                    )

                outpath = out / f"{category}_{rank:02d}.png"
                make_case_plot(
                    archive,
                    recipes[aid],
                    row,
                    category,
                    rank,
                    outpath,
                )

                selection_rows.append({
                    "category": category,
                    "rank": rank,
                    "anchor_id": aid,
                    "positive_id": row["positive_id"],
                    "hardest_negative_id": row[
                        "z_hardest_negative_id"
                    ],
                    "valid_time": row["anchor_valid_time"],
                    "forecast_hour": row[
                        "anchor_forecast_hour"
                    ],
                    "season": row["season"],
                    "intensity_bin": row["intensity_bin"],
                    "z_ap": row["z_cosine_ap"],
                    "z_an_hard": row["z_cosine_an_max"],
                    "z_margin": row["z_margin"],
                    "file": str(outpath),
                })
    finally:
        archive.close()

    selections = pd.DataFrame(selection_rows)
    selection_path = out / "gallery_selections.csv"
    selections.to_csv(selection_path, index=False)

    print(f"Input summary : {anchor_path}")
    print(f"Output gallery: {out}")
    print()
    print(
        selections[
            [
                "category",
                "rank",
                "valid_time",
                "intensity_bin",
                "z_ap",
                "z_an_hard",
                "z_margin",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()

