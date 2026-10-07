#!/usr/bin/env python3
"""
Publication-style 3-panel HRRR time-lag diagnostic.

Compatible with metrics_shard_*.csv.gz from run_timelag_experiment.py.

Panel (a)
    C-P FSS empirical median/IQR for lag 1/3/6 h, with C-N baseline and
    ordering-violation percentage P[FSS(C,P) <= FSS(C,N)].

Panel (b)
    Stage-IV consistency:
        FSS(P,O) - FSS(C,O)
    plus unrelated-negative baseline:
        FSS(N,O_C) - FSS(C,O)
    The latter is diagnostic only, not forecast skill.

Panel (c)
    Median FSS(C,P) versus neighborhood width.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LAGS = (1, 3, 6)
LAG_COLORS = {
    1: "#1f77b4",  # blue
    3: "#d62728",  # red
    6: "#7f3fbf",  # purple
}


def load_metrics(exp: Path) -> pd.DataFrame:
    paths = sorted((exp / "shards").glob("metrics_shard_*.csv.gz"))
    if not paths:
        raise RuntimeError(f"No metrics shards found under {exp / 'shards'}")
    return pd.concat(
        [pd.read_csv(p, low_memory=False) for p in paths],
        ignore_index=True,
    )


def qstats(x):
    x = pd.to_numeric(x, errors="coerce").dropna().to_numpy(float)
    if len(x) == 0:
        return np.nan, np.nan, np.nan, 0
    return (
        float(np.median(x)),
        float(np.quantile(x, 0.25)),
        float(np.quantile(x, 0.75)),
        len(x),
    )


def paired_stage4(cp, po, co, fcol, lag):
    """
    Build per-positive Stage-IV deltas by matching sample_id + lag + direction.
    """
    po_l = po.loc[
        po["lag_hours"] == lag,
        ["sample_id", "direction", fcol],
    ].rename(columns={fcol: "fss_po"})

    co_u = (
        co[["sample_id", fcol]]
        .drop_duplicates("sample_id")
        .rename(columns={fcol: "fss_co"})
    )

    d = po_l.merge(co_u, on="sample_id", how="inner")
    d["delta"] = d["fss_po"] - d["fss_co"]
    return d


def negative_stage4(no, co, fcol):
    no_u = no[["sample_id", "negative_index", fcol]].rename(
        columns={fcol: "fss_no"}
    )
    co_u = (
        co[["sample_id", fcol]]
        .drop_duplicates("sample_id")
        .rename(columns={fcol: "fss_co"})
    )
    d = no_u.merge(co_u, on="sample_id", how="inner")
    d["delta"] = d["fss_no"] - d["fss_co"]
    return d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-dir", required=True)
    p.add_argument("--primary-window", type=int, default=15)
    args = p.parse_args()

    exp = Path(args.experiment_dir)
    metrics = load_metrics(exp)

    fcol = f"fss_w{args.primary_window}"
    if fcol not in metrics.columns:
        raise RuntimeError(
            f"{fcol} not found. Available FSS columns: "
            f"{[c for c in metrics.columns if c.startswith('fss_w')]}"
        )

    cp = metrics.loc[metrics["relation"] == "CP"].copy()
    cn = metrics.loc[metrics["relation"] == "CN"].copy()
    co = metrics.loc[metrics["relation"] == "CO"].copy()
    po = metrics.loc[metrics["relation"] == "PO"].copy()
    no = metrics.loc[metrics["relation"] == "NO"].copy()

    # Read metadata for labels.
    meta_paths = sorted((exp / "shards").glob("metrics_shard_*.json"))
    meta = json.loads(meta_paths[0].read_text()) if meta_paths else {}
    negative_mode = meta.get("negative_mode", "unknown")
    climo_days = meta.get("climo_window_days")

    summary_rows = []

    for lag in LAGS:
        cpl = cp.loc[cp["lag_hours"] == lag].copy()

        med_cp, q25_cp, q75_cp, n_cp = qstats(cpl[fcol])
        med_cn, q25_cn, q75_cn, n_cn = qstats(cn[fcol])

        # Every C-P pair is compared against every sampled N from that same C.
        comp = (
            cpl[["sample_id", fcol]]
            .rename(columns={fcol: "fss_cp"})
            .merge(
                cn[["sample_id", "negative_index", fcol]].rename(
                    columns={fcol: "fss_cn"}
                ),
                on="sample_id",
                how="inner",
            )
            .dropna(subset=["fss_cp", "fss_cn"])
        )

        if len(comp):
            violation = 100.0 * np.mean(comp["fss_cp"] <= comp["fss_cn"])
            strict = 100.0 * np.mean(comp["fss_cp"] < comp["fss_cn"])
            ties = 100.0 * np.mean(
                np.isclose(comp["fss_cp"], comp["fss_cn"], rtol=0.0, atol=1e-12)
            )
        else:
            violation = strict = ties = np.nan

        po_delta = paired_stage4(cp, po, co, fcol, lag)
        no_delta = negative_stage4(no, co, fcol)

        po_med, po_q25, po_q75, n_stage4 = qstats(po_delta["delta"])
        no_med, no_q25, no_q75, n_no = qstats(no_delta["delta"])

        summary_rows.append(
            {
                "lag_hours": lag,
                "n_cp_pairs": n_cp,
                "median_cp_fss": med_cp,
                "q25_cp_fss": q25_cp,
                "q75_cp_fss": q75_cp,
                "median_cn_fss": med_cn,
                "q25_cn_fss": q25_cn,
                "q75_cn_fss": q75_cn,
                "ordering_violation_pct": violation,
                "strict_reversal_pct": strict,
                "tie_pct": ties,
                "n_ordering_comparisons": len(comp),
                "n_stage4_positive_pairs": n_stage4,
                "median_obs_fss_delta_po_minus_co": po_med,
                "q25_obs_fss_delta_po_minus_co": po_q25,
                "q75_obs_fss_delta_po_minus_co": po_q75,
                "median_obs_fss_delta_no_minus_co": no_med,
                "q25_obs_fss_delta_no_minus_co": no_q25,
                "q75_obs_fss_delta_no_minus_co": no_q75,
                "n_stage4_negative_pairs": n_no,
            }
        )

    summary = pd.DataFrame(summary_rows)

    summary_dir = exp / "summary"
    fig_dir = exp / "figures"
    summary_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    summary.to_csv(summary_dir / "timelag_summary.csv", index=False)

    # Directional summary retained because it is useful for later checks.
    direction_rows = []
    for lag in LAGS:
        for direction in ("earlier_init", "later_init"):
            d = cp.loc[
                (cp["lag_hours"] == lag) & (cp["direction"] == direction)
            ]
            med, q25, q75, n = qstats(d[fcol])
            direction_rows.append(
                {
                    "lag_hours": lag,
                    "direction": direction,
                    "median_fss": med,
                    "q25_fss": q25,
                    "q75_fss": q75,
                    "n": n,
                }
            )
    pd.DataFrame(direction_rows).to_csv(
        summary_dir / "timelag_summary_by_direction.csv",
        index=False,
    )

    mpl.rcParams.update(
        {
            "font.size": 12,
            "font.weight": "bold",
            "axes.labelsize": 13,
            "axes.labelweight": "bold",
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.linewidth": 1.2,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))
    fig.subplots_adjust(
        left=0.055,
        right=0.985,
        bottom=0.14,
        top=0.78,
        wspace=0.24,
    )

    xpos = np.arange(3, dtype=float)

    # ------------------------------------------------------------------
    # (a) C-P versus C-N
    # ------------------------------------------------------------------
    ax = axes[0]

    for i, lag in enumerate(LAGS):
        r = summary.loc[summary["lag_hours"] == lag].iloc[0]
        color = LAG_COLORS[lag]

        ax.errorbar(
            xpos[i] - 0.12,
            r["median_cp_fss"],
            yerr=[
                [r["median_cp_fss"] - r["q25_cp_fss"]],
                [r["q75_cp_fss"] - r["median_cp_fss"]],
            ],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.8,
            capsize=5,
            markersize=7,
            zorder=3,
        )

        ax.errorbar(
            xpos[i] + 0.14,
            r["median_cn_fss"],
            yerr=[
                [r["median_cn_fss"] - r["q25_cn_fss"]],
                [r["q75_cn_fss"] - r["median_cn_fss"]],
            ],
            fmt="s",
            color="0.45",
            ecolor="0.45",
            elinewidth=1.5,
            capsize=4,
            markersize=6,
            zorder=3,
        )

        ax.text(
            xpos[i] - 0.02,
            min(0.97, float(r["median_cp_fss"]) + 0.01),
            f"{r['ordering_violation_pct']:.2f}%\nviolation",
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    # black legend markers because color identifies lag, not relationship
    ax.plot([], [], "ko", label="C–P")
    ax.plot([], [], "ks", label="C–N")
    ax.legend(loc="upper right", frameon=False)

    ax.set_xticks(xpos, [str(x) for x in LAGS])
    ax.set_xlabel("Time lag (hours)")
    ax.set_ylabel("FSS")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.18)
    ax.set_title("(a)", pad=6)

    # ------------------------------------------------------------------
    # (b) Stage IV consistency
    # ------------------------------------------------------------------
    ax = axes[1]
    ax.axhline(0.0, color="black", linestyle=":", linewidth=1.4)

    for i, lag in enumerate(LAGS):
        r = summary.loc[summary["lag_hours"] == lag].iloc[0]
        color = LAG_COLORS[lag]

        ax.errorbar(
            xpos[i] - 0.08,
            r["median_obs_fss_delta_po_minus_co"],
            yerr=[
                [
                    r["median_obs_fss_delta_po_minus_co"]
                    - r["q25_obs_fss_delta_po_minus_co"]
                ],
                [
                    r["q75_obs_fss_delta_po_minus_co"]
                    - r["median_obs_fss_delta_po_minus_co"]
                ],
            ],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.7,
            capsize=5,
            markersize=7,
        )

        ax.errorbar(
            xpos[i] + 0.10,
            r["median_obs_fss_delta_no_minus_co"],
            yerr=[
                [
                    r["median_obs_fss_delta_no_minus_co"]
                    - r["q25_obs_fss_delta_no_minus_co"]
                ],
                [
                    r["q75_obs_fss_delta_no_minus_co"]
                    - r["median_obs_fss_delta_no_minus_co"]
                ],
            ],
            fmt="s",
            color="0.45",
            ecolor="0.45",
            elinewidth=1.5,
            capsize=4,
            markersize=6,
        )

    ax.plot([], [], "ko", label="P–O minus C–O")
    ax.plot([], [], "ks", label="N–O minus C–O*")
    ax.legend(loc="upper right", frameon=False)

    ax.set_xticks(xpos, [str(x) for x in LAGS])
    ax.set_xlabel("Time lag (hours)")
    ax.set_ylabel(r"$\Delta$ FSS")
    ax.set_ylim(-0.67, 0.15)
    ax.grid(alpha=0.18)
    ax.set_title("(b)", pad=6)

    ax.text(
        0.01,
        0.02,
        "*N–O is an unrelated-field baseline, not forecast skill.",
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

    # ------------------------------------------------------------------
    # (c) FSS versus scale for C-P
    # ------------------------------------------------------------------
    ax = axes[2]

    fss_cols = []
    for c in cp.columns:
        if c.startswith("fss_w"):
            try:
                w = int(c.split("w", 1)[1])
                fss_cols.append((w, c))
            except ValueError:
                pass
    fss_cols = sorted(fss_cols)

    for lag in LAGS:
        d = cp.loc[cp["lag_hours"] == lag]
        xs = []
        ys = []
        for w, col in fss_cols:
            xs.append(3.0 * w)
            vals = pd.to_numeric(d[col], errors="coerce").dropna()
            ys.append(float(np.median(vals)) if len(vals) else np.nan)

        ax.plot(
            xs,
            ys,
            marker="o",
            linewidth=2.4,
            markersize=6,
            color=LAG_COLORS[lag],
            label=f"{lag}-h lag",
        )

    ax.set_xlabel("Neighborhood width (km)")
    ax.set_ylabel("FSS")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.18)
    ax.legend(loc="lower right", frameon=False)
    ax.set_title("(c)", pad=6)

    for ax in axes:
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontweight("bold")

    neg_desc = (
        f"climo45 negatives (±{climo_days} climatological days)"
        if negative_mode == "climo45"
        else "calendar-unrestricted negatives"
    )

    fig.suptitle(
        "HRRR time-lag positive-pair diagnostics\n"
        f"FSS threshold = 5 mm / 6 h; {neg_desc}",
        fontsize=17,
        fontweight="bold",
        y=0.96,
    )

    png = fig_dir / "timelag_lag_selection_3panel.png"
    pdf = fig_dir / "timelag_lag_selection_3panel.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    print(summary.to_string(index=False))
    print(f"Wrote {summary_dir / 'timelag_summary.csv'}")
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()

