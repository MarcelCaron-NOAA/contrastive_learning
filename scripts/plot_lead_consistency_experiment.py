#!/usr/bin/env python3
"""
Plot +/-1-h positive-pair similarity versus control forecast lead.

The pooled line averages the two positive directions within each CONTROL first,
then computes the mean and 95% normal-approximation confidence interval across
controls. Directional lines are shown separately to expose any asymmetry.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = [
    ("fss", "FSS"),
    ("corr", "Spatial correlation"),
    ("ets", "ETS"),
]


def mean_ci(values):
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if len(x) == 0:
        return np.nan, np.nan, np.nan, 0

    mean = float(np.mean(x))
    if len(x) == 1:
        return mean, np.nan, np.nan, 1

    se = float(np.std(x, ddof=1) / np.sqrt(len(x)))
    return mean, mean - 1.96 * se, mean + 1.96 * se, len(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-dir", required=True)
    args = p.parse_args()

    exp = Path(args.experiment_dir)
    files = sorted(exp.glob("lead_f*.csv.gz"))
    if len(files) != 11:
        print(f"WARNING: found {len(files)} lead files; expected 11.")

    df = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)

    # One record per control per direction. Build a stable control key because
    # sample IDs are sufficient but explicit is clearer.
    df["control_key"] = (
        df["control_id"].astype(str)
        + "|"
        + df["patch_y"].astype(str)
        + "|"
        + df["patch_x"].astype(str)
    )

    leads = sorted(df["control_lead_hour"].dropna().astype(int).unique())

    summary_rows = []

    for metric, _ in METRICS:
        for lead in leads:
            d = df.loc[df["control_lead_hour"] == lead]

            for direction in ("earlier_init", "later_init"):
                vals = d.loc[d["direction"] == direction, metric]
                mean, lo, hi, n = mean_ci(vals)
                summary_rows.append(
                    {
                        "metric": metric,
                        "control_lead_hour": lead,
                        "group": direction,
                        "mean": mean,
                        "ci95_low": lo,
                        "ci95_high": hi,
                        "n": n,
                    }
                )

            # Pair-average within control first so the two directions do not
            # masquerade as independent pooled observations.
            pooled_by_control = (
                d.groupby("control_key", as_index=False)[metric]
                .mean(numeric_only=True)
            )
            mean, lo, hi, n = mean_ci(pooled_by_control[metric])
            summary_rows.append(
                {
                    "metric": metric,
                    "control_lead_hour": lead,
                    "group": "pooled",
                    "mean": mean,
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "n": n,
                }
            )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(exp / "lead_consistency_summary.csv", index=False)

    mpl.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 13,
            "axes.labelweight": "bold",
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.linewidth": 1.2,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)

    # Colors only distinguish positive direction; pooled is black.
    direction_style = {
        "earlier_init": dict(color="#1f77b4", marker="o", linestyle="--",
                             label="P initialized 1 h earlier"),
        "later_init": dict(color="#d62728", marker="s", linestyle="--",
                           label="P initialized 1 h later"),
    }

    for ax, (metric, ylabel) in zip(axes, METRICS):
        s = summary.loc[summary["metric"] == metric]

        pooled = s.loc[s["group"] == "pooled"].sort_values("control_lead_hour")
        x = pooled["control_lead_hour"].to_numpy()
        y = pooled["mean"].to_numpy()
        lo = pooled["ci95_low"].to_numpy()
        hi = pooled["ci95_high"].to_numpy()

        ax.plot(
            x, y, color="black", marker="o", linewidth=2.5,
            markersize=7, label="Pooled ±1 h"
        )
        ax.fill_between(x, lo, hi, color="black", alpha=0.12, linewidth=0)

        for group, style in direction_style.items():
            q = s.loc[s["group"] == group].sort_values("control_lead_hour")
            ax.plot(
                q["control_lead_hour"],
                q["mean"],
                linewidth=1.6,
                markersize=5,
                **style,
            )

        ax.set_xticks(range(7, 18))
        ax.set_xlabel("Control forecast lead (h)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.20)

        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontweight("bold")

    axes[0].set_title("(a)")
    axes[1].set_title("(b)")
    axes[2].set_title("(c)")

    # ETS can legitimately be negative.
    axes[0].set_ylim(0, 1)
    axes[1].set_ylim(-1, 1)

    handles, labels = axes[2].get_legend_handles_labels()
    axes[2].legend(handles, labels, loc="best", frameon=False)

    # Read threshold/window from first metadata file if available.
    meta_files = sorted(exp.glob("lead_f*.json"))
    subtitle = ""
    if meta_files:
        import json
        meta = json.loads(meta_files[0].read_text())
        subtitle = (
            f"\n5-mm / 6-h event threshold; "
            f"FSS window = {meta.get('fss_window', 15)} grid points"
        )

    fig.suptitle(
        "±1-h HRRR positive-pair consistency versus forecast lead" + subtitle,
        fontsize=17,
        fontweight="bold",
    )

    fig_dir = exp / "figures"
    fig_dir.mkdir(exist_ok=True)

    png = fig_dir / "lead_consistency_f07_f17.png"
    pdf = fig_dir / "lead_consistency_f07_f17.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    print(summary.to_string(index=False))
    print(f"Wrote {exp / 'lead_consistency_summary.csv'}")
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()

