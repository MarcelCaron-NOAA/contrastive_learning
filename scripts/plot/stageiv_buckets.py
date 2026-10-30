#!/usr/bin/env python3
"""Create the 11 final Stage-IV gallery buckets using inch-based verification.

All precipitation fields are stored internally in mm, but display and threshold
labels are in inches.  Raw and difference fields are rendered with pcolormesh;
contourf is deliberately not used for difference plots.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd

MM_PER_IN = 25.4

PRECIP_LEVELS_IN = [
    0.01, 0.10, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50,
    1.75, 2.00, 2.50, 3.00, 4.00, 5.00, 7.00, 10.00,
    15.00, 20.00,
]
PRECIP_COLORS_IN = [
    "#7fff00", "#00cd00", "#008b00", "#104e8b",
    "#1e90ff", "#00b2ee", "#00eeee", "#8968cd",
    "#912cee", "#8b008b", "#8b0000", "#cd0000",
    "#ee4000", "#ff7f00", "#cd8500", "#ffd700",
    "#ffff00", "#ffff02",
]
PRECIP_OVER_IN = "#ffaeb9"


def precip_cmap_norm():
    cmap = ListedColormap(PRECIP_COLORS_IN, name="weather_precip_in")
    cmap.set_under("white")
    cmap.set_over(PRECIP_OVER_IN)
    norm = BoundaryNorm(PRECIP_LEVELS_IN, cmap.N, clip=False)
    return cmap, norm


def defaults():
    data = Path(os.environ["DATA_DIR"])
    base = data / "derived/evaluation/final_syn_nfv_stageiv_v1"
    return {
        "archive": data / "derived/evaluation/stageiv_reference_v1",
        "metrics": base / "stageiv_per_case_metrics_inches.csv.gz",
        "out": base / "galleries_inches",
    }


def first_present(row, names, default=""):
    for name in names:
        if name in row.index and pd.notna(row[name]):
            return row[name]
    return default


def rank_gap(df: pd.DataFrame, a: str, b: str) -> pd.Series:
    ar = df[a].rank(method="average", pct=True)
    br = df[b].rank(method="average", pct=True)
    return (ar - br).abs()


def choose_rank_alignment(df, model, metric, count, best):
    g = df.dropna(subset=[model, metric]).copy()
    g["criterion_value"] = rank_gap(g, model, metric)
    g["joint_percentile"] = 0.5 * (
        g[model].rank(method="average", pct=True) +
        g[metric].rank(method="average", pct=True)
    )
    if best:
        g = g.sort_values(["criterion_value", "joint_percentile"], ascending=[True, False])
    else:
        g = g.sort_values(["criterion_value", "joint_percentile"], ascending=[False, False])
    return g.head(count).copy()


def choose_model_disagreement(df, a, b, count):
    """Largest disagreement in relative ranking, not raw score calibration.

    Each model is independently percentile-ranked across the same non-dry
    candidate population, then cases are ordered by the absolute percentile
    gap.  This makes the learned-vs-learned buckets consistent with the
    learned-vs-traditional alignment buckets.
    """
    g = df.dropna(subset=[a, b]).copy()
    g["criterion_value"] = rank_gap(g, a, b)
    g["joint_percentile"] = 0.5 * (
        g[a].rank(method="average", pct=True) +
        g[b].rank(method="average", pct=True)
    )
    return g.sort_values(["criterion_value", "joint_percentile"], ascending=[False, False]).head(count).copy()


def fmt(v, digits=3):
    try:
        if not np.isfinite(float(v)):
            return "NA"
        return f"{float(v):.{digits}f}"
    except Exception:
        return "NA"


def format_valid_time(row):
    valid = first_present(row, ["valid_end", "valid_time", "valid"], "")
    fhr = first_present(row, ["forecast_hour", "lead_hours", "fhr"], None)
    fhr_txt = f"F{int(fhr):02d}" if fhr is not None and str(fhr) != "" else ""
    try:
        ts = pd.to_datetime(valid, utc=True, errors="raise")
        date_txt = ts.strftime("%HZ %b %d, %Y")
    except Exception:
        date_txt = str(valid)
    return f"valid {date_txt} ({fhr_txt})" if fhr_txt else f"valid {date_txt}"


def nice_diff_limit(d_in):
    finite = np.abs(d_in[np.isfinite(d_in)])
    if finite.size == 0:
        return 0.25
    q = max(0.10, float(np.nanpercentile(finite, 99.0)))
    candidates = np.asarray([0.10, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0])
    above = candidates[candidates >= q]
    return float(above[0] if len(above) else np.ceil(q / 5.0) * 5.0)


def plot_case(row, fcst, obs, path: Path, bucket_title: str, rank: int, dpi: int):
    j = int(row["archive_row"])
    f_in = np.asarray(fcst[j], dtype=np.float64) / MM_PER_IN
    o_in = np.asarray(obs[j], dtype=np.float64) / MM_PER_IN
    d_in = f_in - o_in

    # Compact 3-panel layout.  The axes are placed manually with square
    # physical dimensions so equal-aspect images do not shrink inside wide
    # GridSpec cells and create large apparent gaps.  Horizontal inter-panel
    # spacing and vertical panel-to-colorbar spacing are intentionally matched
    # in physical units.
    fig = plt.figure(figsize=(10.6, 5.25))
    panel_bottom = 0.245
    panel_top = 0.735
    panel_height_frac = panel_top - panel_bottom
    panel_height_in = panel_height_frac * fig.get_figheight()
    panel_width_frac = panel_height_in / fig.get_figwidth()

    gap_in = 0.13
    gap_x = gap_in / fig.get_figwidth()
    total_width = 3.0 * panel_width_frac + 2.0 * gap_x
    left = 0.5 * (1.0 - total_width)

    axes = []
    for i in range(3):
        x0 = left + i * (panel_width_frac + gap_x)
        axes.append(fig.add_axes([x0, panel_bottom, panel_width_frac, panel_height_frac]))

    pcmap, pnorm = precip_cmap_norm()
    im0 = axes[0].pcolormesh(f_in, cmap=pcmap, norm=pnorm, shading="nearest", rasterized=True)
    axes[1].pcolormesh(o_in, cmap=pcmap, norm=pnorm, shading="nearest", rasterized=True)

    dlim = nice_diff_limit(d_in)
    dcmap = plt.get_cmap("RdBu_r").copy()
    dnorm = TwoSlopeNorm(vmin=-dlim, vcenter=0.0, vmax=dlim)
    im2 = axes[2].pcolormesh(d_in, cmap=dcmap, norm=dnorm, shading="nearest", rasterized=True)

    axes[0].set_title("HRRR forecast", fontsize=12.5, pad=6)
    axes[1].set_title("Stage IV (BUDGET)", fontsize=12.5, pad=6)
    axes[2].set_title("HRRR − Stage IV", fontsize=12.5, pad=6)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

    fig.text(
        0.5, 0.970,
        f"{rank}. {bucket_title} | {format_valid_time(row)}",
        ha="center", va="top", fontsize=14.8, fontweight="bold",
    )
    fig.text(
        0.5, 0.910,
        f"SYN-3={fmt(row.syn3)}  |  SYN-8={fmt(row.syn8)}  |  NFV={fmt(row.nfv)}",
        ha="center", va="top", fontsize=12.2, fontweight="bold",
    )
    fig.text(
        0.5, 0.858,
        "  |  ".join([
            f"FSS-5={fmt(row.fss5)}",
            f"FSS-10={fmt(row.fss10)}",
            f"ETS-0.1={fmt(row.ets_0p1)}",
            f"ETS-0.5={fmt(row.ets_0p5)}",
            f"FBIAS-0.1={fmt(row.fbias_0p1)}",
            f"FBIAS-0.5={fmt(row.fbias_0p5)}",
        ]),
        ha="center", va="top", fontsize=10.9,
    )

    # The vertical gap from panels to colorbars is the same physical distance
    # as the horizontal panel-to-panel gap.  Colorbars retain the exact spans
    # requested: panels 1+2 for precipitation and panel 3 for the difference.
    fig.canvas.draw()
    p0, p1, p2 = [ax.get_position() for ax in axes]
    cbar_h_in = 0.13
    cbar_h = cbar_h_in / fig.get_figheight()
    gap_y = gap_in / fig.get_figheight()
    cbar_y = p0.y0 - gap_y - cbar_h
    cax_precip = fig.add_axes([p0.x0, cbar_y, p1.x1 - p0.x0, cbar_h])
    cax_diff = fig.add_axes([p2.x0, cbar_y, p2.x1 - p2.x0, cbar_h])

    cbar0 = fig.colorbar(
        im0, cax=cax_precip, orientation="horizontal", extend="max",
        ticks=PRECIP_LEVELS_IN,
    )
    cbar0.set_label("6-h precipitation (in)", fontsize=9.5, labelpad=2)
    cbar0.ax.tick_params(labelsize=7.0, pad=1)
    cbar0.ax.set_xticklabels([
        "0.01", "0.10", "0.25", "0.50", "0.75", "1.0", "1.25", "1.5",
        "1.75", "2.0", "2.5", "3", "4", "5", "7", "10", "15", "20",
    ])

    cbar2 = fig.colorbar(im2, cax=cax_diff, orientation="horizontal", extend="both")
    cbar2.set_label("HRRR − Stage IV (in)", fontsize=9.5, labelpad=2)
    cbar2.ax.tick_params(labelsize=8.0, pad=1)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    d = defaults()
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-csv", default=str(d["metrics"]))
    ap.add_argument("--archive-dir", default=str(d["archive"]))
    ap.add_argument("--output-dir", default=str(d["out"]))
    ap.add_argument("--count", type=int, default=5)
    ap.add_argument(
        "--min-wet-pixels", type=int, default=16,
        help="Require this many >=0.1-in pixels in BOTH HRRR and Stage IV.",
    )
    ap.add_argument("--dpi", type=int, default=180)
    args = ap.parse_args()

    metrics_path = Path(args.metrics_csv)
    arc = Path(args.archive_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(metrics_path)
    fcst = np.load(arc / "hrrr_qpf6_mm.npy", mmap_mode="r")
    obs = np.load(arc / "stageiv_budget_qpf6_mm.npy", mmap_mode="r")

    wet = df[
        (df["fcst_wet_pixels"] >= int(args.min_wet_pixels)) &
        (df["obs_wet_pixels"] >= int(args.min_wet_pixels))
    ].copy()
    if len(wet) < args.count:
        raise RuntimeError(f"Only {len(wet)} non-dry cases after >=0.1-in filter")

    specs = []
    g = wet.dropna(subset=["syn3"]).sort_values("syn3", ascending=False).head(args.count).copy()
    g["criterion_value"] = g["syn3"]
    specs.append(("01_highest_syn3", "Highest SYN-3 scores", "higher raw SYN-3 cosine", g))

    specs.append(("02_best_syn3_fss5_alignment", "Best SYN-3 / FSS-5 alignment", "smallest percentile-rank gap",
                  choose_rank_alignment(wet, "syn3", "fss5", args.count, True)))
    specs.append(("03_best_syn3_ets01_alignment", "Best SYN-3 / ETS-0.1 alignment", "smallest percentile-rank gap",
                  choose_rank_alignment(wet, "syn3", "ets_0p1", args.count, True)))
    specs.append(("04_bottom_syn3_fss5_alignment", "Bottom SYN-3 / FSS-5 alignment", "largest percentile-rank gap",
                  choose_rank_alignment(wet, "syn3", "fss5", args.count, False)))
    specs.append(("05_bottom_syn3_ets01_alignment", "Bottom SYN-3 / ETS-0.1 alignment", "largest percentile-rank gap",
                  choose_rank_alignment(wet, "syn3", "ets_0p1", args.count, False)))
    specs.append(("06_bottom_syn3_syn8_alignment", "Bottom SYN-3 / SYN-8 alignment", "largest percentile-rank gap",
                  choose_model_disagreement(wet, "syn3", "syn8", args.count)))
    specs.append(("07_bottom_syn8_fss5_alignment", "Bottom SYN-8 / FSS-5 alignment", "largest percentile-rank gap",
                  choose_rank_alignment(wet, "syn8", "fss5", args.count, False)))
    specs.append(("08_bottom_syn8_ets01_alignment", "Bottom SYN-8 / ETS-0.1 alignment", "largest percentile-rank gap",
                  choose_rank_alignment(wet, "syn8", "ets_0p1", args.count, False)))
    specs.append(("09_bottom_syn3_nfv_alignment", "Bottom SYN-3 / NFV alignment", "largest percentile-rank gap",
                  choose_model_disagreement(wet, "syn3", "nfv", args.count)))
    specs.append(("10_bottom_nfv_fss5_alignment", "Bottom NFV / FSS-5 alignment", "largest percentile-rank gap",
                  choose_rank_alignment(wet, "nfv", "fss5", args.count, False)))
    specs.append(("11_bottom_nfv_ets01_alignment", "Bottom NFV / ETS-0.1 alignment", "largest percentile-rank gap",
                  choose_rank_alignment(wet, "nfv", "ets_0p1", args.count, False)))

    all_rows = []
    for folder, title, criterion_desc, picks in specs:
        bdir = out / folder
        bdir.mkdir(parents=True, exist_ok=True)
        picks = picks.copy().reset_index(drop=True)
        picks["bucket"] = folder
        picks["bucket_title"] = title
        picks["criterion_description"] = criterion_desc
        picks["bucket_rank"] = np.arange(1, len(picks) + 1)
        picks.to_csv(bdir / "selections.csv", index=False)
        all_rows.append(picks)

        for _, row in picks.iterrows():
            rank = int(row["bucket_rank"])
            plot_case(
                row, fcst, obs,
                bdir / f"{rank:02d}_case{int(row.case_index):04d}.png",
                title, rank, args.dpi,
            )

        print(f"\n{folder}: {title}")
        print(picks[["bucket_rank", "case_index", "criterion_value", "syn3", "syn8", "nfv", "fss5", "ets_0p1"]].to_string(index=False))

    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(out / "all_bucket_selections.csv", index=False)

    print("\n" + "=" * 90)
    print("FINAL STAGE-IV GALLERIES COMPLETE")
    print("=" * 90)
    print(f"Non-dry candidate cases (>=16 pixels at >=0.1 in in BOTH fields): {len(wet)}")
    print(f"Buckets: {len(specs)}")
    print(f"Images : {sum(len(x[3]) for x in specs)}")
    print(f"Output : {out}")
    print("\nAlignment definition:")
    print("  learned vs FSS/ETS -> absolute percentile-rank gap")
    print("  learned vs learned -> absolute percentile-rank gap")
    print("  buckets are independent; overlaps across buckets are allowed")


if __name__ == "__main__":
    main()
