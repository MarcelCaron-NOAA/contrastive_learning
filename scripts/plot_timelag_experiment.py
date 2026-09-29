#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


LAGS = [1, 3, 6]
LAG_LABELS = ["1", "3", "6"]
XPOS = np.arange(len(LAGS))

LAG_COLORS = {
    1: "#1f77b4",   # blue
    3: "#d62728",   # red
    6: "#7f3fbf",   # purple
}
NEG_COLOR = "#6f6f6f"


def set_style():
    mpl.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 14,
            "axes.labelweight": "bold",
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "axes.linewidth": 1.2,
            "legend.fontsize": 12,
        }
    )


def bold_ticklabels(ax):
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontweight("bold")


def central_interval(values, p_lo=25, p_hi=75):
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    return np.percentile(x, p_lo), np.percentile(x, 50), np.percentile(x, p_hi)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--primary-window", type=int, default=15)
    parser.add_argument("--grid-spacing-km", type=float, default=3.0)
    parser.add_argument(
        "--title",
        default="HRRR time-lag positive-pair diagnostics",
    )
    args = parser.parse_args()

    set_style()

    exp = Path(args.experiment_dir)
    shard_paths = sorted((exp / "shards").glob("metrics_shard_*.csv.gz"))
    if not shard_paths:
        raise SystemExit("No shard files found.")

    df = pd.concat(
        [pd.read_csv(path, low_memory=False) for path in shard_paths],
        ignore_index=True,
    )
    df["lag_hours"] = pd.to_numeric(df["lag_hours"], errors="coerce")

    fcol = f"fss_w{args.primary_window}"
    if fcol not in df.columns:
        raise SystemExit(
            f"Required FSS column {fcol!r} not found. "
            f"Available FSS columns: {[c for c in df.columns if c.startswith('fss_w')]}"
        )

    cp = df[df["relation"] == "CP"].copy()
    cn = df[df["relation"] == "CN"].copy()
    co = df[df["relation"] == "CO"].copy()
    po = df[df["relation"] == "PO"].copy()
    no = df[df["relation"] == "NO"].copy()

    violation_rows = []
    for lag in LAGS:
        merged = cp.loc[cp["lag_hours"] == lag, ["sample_id", fcol]].merge(
            cn[["sample_id", fcol]],
            on="sample_id",
            suffixes=("_cp", "_cn"),
        ).dropna()

        violation_rows.append(
            {
                "lag_hours": lag,
                "ordering_violation_pct": 100.0 * np.mean(
                    merged[f"{fcol}_cp"] <= merged[f"{fcol}_cn"]
                ),
                "comparisons": len(merged),
            }
        )
    violation = pd.DataFrame(violation_rows)

    coref = co[["sample_id", fcol, "bcrmse_mm", "rmse_mm"]].rename(
        columns={
            fcol: "fss_co",
            "bcrmse_mm": "bcrmse_co",
            "rmse_mm": "rmse_co",
        }
    )

    pod = po.merge(coref, on="sample_id")
    pod["delta_obs_fss"] = pod[fcol] - pod["fss_co"]
    pod["delta_obs_bcrmse"] = pod["bcrmse_mm"] - pod["bcrmse_co"]

    nod = no.merge(coref, on="sample_id")
    nod["delta_obs_fss"] = nod[fcol] - nod["fss_co"]

    summary_rows = []
    for lag in LAGS:
        c = cp[cp["lag_hours"] == lag]
        q = pod[pod["lag_hours"] == lag]
        v = violation.loc[violation["lag_hours"] == lag].iloc[0]

        summary_rows.append(
            {
                "lag_hours": lag,
                "n_cp_pairs": len(c),
                "median_cp_fss": c[fcol].median(),
                "q25_cp_fss": c[fcol].quantile(0.25),
                "q75_cp_fss": c[fcol].quantile(0.75),
                "ordering_violation_pct": v["ordering_violation_pct"],
                "n_ordering_comparisons": int(v["comparisons"]),
                "n_stage4_positive_pairs": len(q),
                "median_obs_fss_delta_po_minus_co": q["delta_obs_fss"].median(),
                "median_abs_obs_fss_delta": q["delta_obs_fss"].abs().median(),
                "median_obs_bcrmse_delta_mm": q["delta_obs_bcrmse"].median(),
                "median_abs_obs_bcrmse_delta_mm": q["delta_obs_bcrmse"].abs().median(),
                "median_cp_bcrmse_mm": c["bcrmse_mm"].median(),
                "median_cp_mean_abs_diff_mm": c["mean_abs_diff_mm"].median(),
                "median_cp_corr": c["corr"].median(),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary_dir = exp / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_dir / "timelag_summary.csv", index=False)

    cp.groupby(["lag_hours", "direction"]).agg(
        n=("sample_id", "size"),
        median_fss=(fcol, "median"),
        median_bcrmse_mm=("bcrmse_mm", "median"),
        median_mean_abs_diff_mm=("mean_abs_diff_mm", "median"),
        median_corr=("corr", "median"),
    ).reset_index().to_csv(
        summary_dir / "timelag_summary_by_direction.csv",
        index=False,
    )

    fig = plt.figure(figsize=(16.3, 5.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.0])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    # Panel A
    offset = 0.13
    for lag, xpos in zip(LAGS, XPOS):
        cp_vals = cp.loc[cp["lag_hours"] == lag, fcol]
        q25, med, q75 = central_interval(cp_vals, 25, 75)
        ax1.errorbar(
            xpos - offset,
            med,
            yerr=[[med - q25], [q75 - med]],
            fmt="o",
            capsize=5,
            markersize=8,
            lw=2.0,
            color=LAG_COLORS[lag],
            zorder=3,
        )

        cn_vals = cn[fcol]
        q25n, medn, q75n = central_interval(cn_vals, 25, 75)
        ax1.errorbar(
            xpos + offset,
            medn,
            yerr=[[medn - q25n], [q75n - medn]],
            fmt="s",
            capsize=5,
            markersize=7,
            lw=2.0,
            color=NEG_COLOR,
            zorder=3,
        )

        vp = float(
            violation.loc[violation["lag_hours"] == lag, "ordering_violation_pct"].iloc[0]
        )
        ax1.text(
            xpos - offset + 0.10,
            med - 0.02,
            f"{vp:.2f}%\nviolation",
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    ax1.set_xticks(XPOS)
    ax1.set_xticklabels(LAG_LABELS)
    ax1.set_xlabel("Time lag (hours)")
    ax1.set_ylabel("FSS")
    ax1.set_title("(a)", pad=10)
    ax1.set_ylim(-0.02, 1.02)
    ax1.grid(alpha=0.20)
    ax1.legend(
        handles=[
            Line2D([0], [0], marker="o", color="black", linestyle="None",
                   markersize=8, label="C–P"),
            Line2D([0], [0], marker="s", color="black", linestyle="None",
                   markersize=7, label="C–N"),
        ],
        loc="upper right",
        frameon=False,
    )
    bold_ticklabels(ax1)

    # Panel B
    for lag, xpos in zip(LAGS, XPOS):
        vals = pod.loc[pod["lag_hours"] == lag, "delta_obs_fss"]
        q25, med, q75 = central_interval(vals, 25, 75)
        ax2.errorbar(
            xpos - offset / 2,
            med,
            yerr=[[med - q25], [q75 - med]],
            fmt="o",
            capsize=5,
            markersize=8,
            lw=2.0,
            color=LAG_COLORS[lag],
            zorder=3,
        )

        valsn = nod["delta_obs_fss"]
        q25n, medn, q75n = central_interval(valsn, 25, 75)
        ax2.errorbar(
            xpos + offset / 2,
            medn,
            yerr=[[medn - q25n], [q75n - medn]],
            fmt="s",
            capsize=5,
            markersize=7,
            lw=2.0,
            color=NEG_COLOR,
            zorder=3,
        )

    ax2.axhline(0.0, color="black", linestyle=":", linewidth=1.6, zorder=1)
    ax2.set_xticks(XPOS)
    ax2.set_xticklabels(LAG_LABELS)
    ax2.set_xlabel("Time lag (hours)")
    ax2.set_ylabel(r"$\Delta$ FSS")
    ax2.set_title("(b)", pad=10)
    ax2.set_ylim(-0.67, 0.15)
    ax2.grid(alpha=0.20)
    ax2.legend(
        handles=[
            Line2D([0], [0], marker="o", color="black", linestyle="None",
                   markersize=8, label="P–O minus C–O"),
            Line2D([0], [0], marker="s", color="black", linestyle="None",
                   markersize=7, label="N–O minus C–O*"),
        ],
        loc="upper right",
        frameon=False,
    )
    ax2.text(
        0.01,
        0.015,
        "*N–O is an unrelated-field baseline, not forecast skill.",
        transform=ax2.transAxes,
        fontsize=9.0,
        fontweight="bold",
        va="bottom",
    )
    bold_ticklabels(ax2)

    # Panel C
    fss_cols = sorted(
        [c for c in cp.columns if c.startswith("fss_w")],
        key=lambda c: int(c.split("w")[1]),
    )
    windows = [int(c.split("w")[1]) for c in fss_cols]
    scales_km = np.array(windows) * args.grid_spacing_km

    for lag in LAGS:
        subset = cp[cp["lag_hours"] == lag]
        medians = []
        for col in fss_cols:
            x = pd.to_numeric(subset[col], errors="coerce").dropna()
            medians.append(x.median() if len(x) else np.nan)

        ax3.plot(
            scales_km,
            medians,
            marker="o",
            ms=8,
            lw=2.5,
            color=LAG_COLORS[lag],
            label=f"{lag}-h lag",
            zorder=4,
        )

    ax3.set_xlabel("Neighborhood width (km)")
    ax3.set_ylabel("FSS")
    ax3.set_title("(c)", pad=10)
    ax3.set_ylim(0.0, 1.0)
    ax3.grid(alpha=0.20)
    ax3.legend(loc="lower right", frameon=False)
    bold_ticklabels(ax3)

    metas = sorted((exp / "shards").glob("metrics_shard_*.json"))
    threshold = None
    if metas:
        threshold = json.loads(metas[0].read_text()).get("fss_threshold_mm")

    title = args.title
    if threshold is not None:
        title += f"\nFSS threshold = {threshold:g} mm / 6 h"

    fig.suptitle(title, fontsize=18, fontweight="bold")

    fig_dir = exp / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    png_path = fig_dir / "timelag_lag_selection_3panel_v4.png"
    pdf_path = fig_dir / "timelag_lag_selection_3panel_v4.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(summary.to_string(index=False))
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()

