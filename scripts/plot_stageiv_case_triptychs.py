#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize, TwoSlopeNorm
from scipy.ndimage import uniform_filter


MM_TO_IN = 1.0 / 25.4


def pick_col(df: pd.DataFrame, aliases, required=True):
    for c in aliases:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"Could not find any of these columns: {aliases}")
    return None


def load_array(path: str, var: str | None = None) -> np.ndarray:
    p = Path(path)
    suf = p.suffix.lower()

    if suf == ".npy":
        arr = np.load(p)
        return np.asarray(arr).squeeze()

    if suf == ".npz":
        z = np.load(p)
        if var is not None and var in z:
            return np.asarray(z[var]).squeeze()
        first = list(z.keys())[0]
        return np.asarray(z[first]).squeeze()

    # xarray fallback for netcdf/grib-ish inputs
    import xarray as xr
    ds = xr.open_dataset(p)
    if var is None:
        # pick first 2-D-ish variable
        candidates = []
        for name, da in ds.data_vars.items():
            if da.ndim >= 2:
                candidates.append(name)
        if not candidates:
            raise ValueError(f"No 2-D variable found in {path}")
        var = candidates[0]

    da = ds[var]
    while da.ndim > 2:
        da = da.isel({da.dims[0]: 0})
    return np.asarray(da).squeeze()


def ets_score(fcst_in: np.ndarray, obs_in: np.ndarray, thresh_in: float = 0.1) -> float:
    f = np.asarray(fcst_in) >= thresh_in
    o = np.asarray(obs_in) >= thresh_in
    hits = np.logical_and(f, o).sum()
    misses = np.logical_and(~f, o).sum()
    false_alarms = np.logical_and(f, ~o).sum()
    total = f.size

    random_hits = ((hits + misses) * (hits + false_alarms)) / total
    denom = hits + misses + false_alarms - random_hits
    if denom <= 0:
        return np.nan
    return (hits - random_hits) / denom


def fss_score(fcst_in: np.ndarray, obs_in: np.ndarray, thresh_in: float, width: int) -> float:
    f = (np.asarray(fcst_in) >= thresh_in).astype(float)
    o = (np.asarray(obs_in) >= thresh_in).astype(float)
    ff = uniform_filter(f, size=width, mode="nearest")
    oo = uniform_filter(o, size=width, mode="nearest")
    num = np.mean((ff - oo) ** 2)
    den = np.mean(ff ** 2 + oo ** 2)
    if den <= 0:
        return np.nan
    return 1.0 - num / den


def fmt(x, nd=3):
    try:
        if pd.isna(x):
            return "nan"
    except Exception:
        pass
    return f"{float(x):.{nd}f}"


def annotate(ax, lines, fontsize=11):
    txt = "\n".join(lines)
    ax.text(
        0.01, 0.99, txt,
        ha="left", va="top",
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="0.7")
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="Selected case manifest CSV")
    ap.add_argument("--output-dir", required=True, help="Directory for output figures")
    ap.add_argument("--forecast-var", default=None, help="Optional forecast variable name for non-npy inputs")
    ap.add_argument("--obs-var", default=None, help="Optional obs variable name for non-npy inputs")
    ap.add_argument("--assume-mm", action="store_true",
                    help="Convert loaded arrays from mm to inches")
    ap.add_argument("--dpi", type=int, default=170)
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Column discovery
    id_col = pick_col(manifest, ["case_id", "sample_id", "patch_id", "id"])
    bucket_col = pick_col(manifest, ["bucket"])
    brank_col = pick_col(manifest, ["bucket_rank"])
    fpath_col = pick_col(manifest, ["forecast_path", "fcst_path", "hrrr_path"])
    opath_col = pick_col(manifest, ["obs_path", "stageiv_path", "o_path"])

    v1_col = pick_col(manifest, ["v1_similarity", "learned_v1", "baseline_v1", "sim_v1"])
    v2_col = pick_col(manifest, ["v2_similarity", "learned_v2", "hard_ranking_v2", "sim_v2"])
    fss5_col = pick_col(manifest, ["fss5", "fss_5", "fss_w5"], required=False)
    fss10_col = pick_col(manifest, ["fss10", "fss_10", "fss_w10"], required=False)
    ets_col = pick_col(manifest, ["ets_0p1", "ets", "ets01", "ts_0p1"], required=False)
    corr_col = pick_col(manifest, ["corr", "correlation", "pearson_r"], required=False)
    init_col = pick_col(manifest, ["init_time", "init", "cycle_time"], required=False)
    valid_col = pick_col(manifest, ["valid_time", "valid"], required=False)
    lead_col = pick_col(manifest, ["lead_hours", "fhr", "forecast_hour"], required=False)

    for _, row in manifest.iterrows():
        cid = str(row[id_col])
        bucket = str(row[bucket_col])
        brank = int(row[brank_col])

        bucket_dir = out_dir / bucket
        bucket_dir.mkdir(parents=True, exist_ok=True)

        fcst = load_array(str(row[fpath_col]), var=args.forecast_var)
        obs = load_array(str(row[opath_col]), var=args.obs_var)

        if args.assume_mm:
            fcst = fcst * MM_TO_IN
            obs = obs * MM_TO_IN

        diff = fcst - obs

        # Prefer manifest values, compute if absent
        v1 = row[v1_col]
        v2 = row[v2_col]
        fss5 = row[fss5_col] if fss5_col else fss_score(fcst, obs, thresh_in=0.1, width=5)
        fss10 = row[fss10_col] if fss10_col else fss_score(fcst, obs, thresh_in=0.1, width=10)
        ets = row[ets_col] if ets_col else ets_score(fcst, obs, thresh_in=0.1)
        corr = row[corr_col] if corr_col else np.corrcoef(fcst.ravel(), obs.ravel())[0, 1]

        # Scales
        vmax = np.nanpercentile(np.concatenate([fcst.ravel(), obs.ravel()]), 99.5)
        vmax = max(vmax, 0.25)
        dmax = np.nanpercentile(np.abs(diff.ravel()), 99.5)
        dmax = max(dmax, 0.10)

        norm_precip = Normalize(vmin=0.0, vmax=vmax)
        norm_diff = TwoSlopeNorm(vcenter=0.0, vmin=-dmax, vmax=dmax)

        fig = plt.figure(figsize=(16.8, 4.7))
        gs = GridSpec(
            1, 5, figure=fig,
            width_ratios=[1.0, 1.0, 1.0, 0.05, 0.05],
            wspace=0.12
        )

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        cax1 = fig.add_subplot(gs[0, 3])  # precip colorbar
        cax2 = fig.add_subplot(gs[0, 4])  # diff colorbar

        im1 = ax1.imshow(fcst, origin="lower", cmap="Blues", norm=norm_precip, interpolation="nearest")
        im2 = ax2.imshow(obs, origin="lower", cmap="Blues", norm=norm_precip, interpolation="nearest")
        im3 = ax3.imshow(diff, origin="lower", cmap="RdBu_r", norm=norm_diff, interpolation="nearest")

        for ax in [ax1, ax2, ax3]:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(1.1)

        ax1.set_title("HRRR forecast", fontsize=15, fontweight="bold", pad=8)
        ax2.set_title("Stage IV", fontsize=15, fontweight="bold", pad=8)
        ax3.set_title("HRRR − Stage IV", fontsize=15, fontweight="bold", pad=8)

        cb1 = plt.colorbar(im2, cax=cax1)
        cb1.set_label("precipitation (in)", fontsize=11, fontweight="bold")
        cb1.ax.tick_params(labelsize=10)

        cb2 = plt.colorbar(im3, cax=cax2)
        cb2.set_label("difference (in)", fontsize=11, fontweight="bold")
        cb2.ax.tick_params(labelsize=10)

        header_parts = [f"{bucket}{brank:02d}", f"case={cid}"]
        if init_col and pd.notna(row[init_col]):
            header_parts.append(f"init={row[init_col]}")
        if valid_col and pd.notna(row[valid_col]):
            header_parts.append(f"valid={row[valid_col]}")
        if lead_col and pd.notna(row[lead_col]):
            header_parts.append(f"fhr={row[lead_col]}")

        fig.suptitle(
            " | ".join(header_parts),
            fontsize=16, fontweight="bold", y=0.98
        )

        stat_lines = [
            f"V1 sim   = {fmt(v1)}",
            f"V2 sim   = {fmt(v2)}",
            f"Δ(V2−V1) = {fmt(float(v2) - float(v1))}",
            f"FSS-5    = {fmt(fss5)}",
            f"FSS-10   = {fmt(fss10)}",
            f"ETS@0.1\" = {fmt(ets)}",
            f"corr     = {fmt(corr)}",
        ]
        annotate(ax3, stat_lines, fontsize=11)

        plt.subplots_adjust(left=0.02, right=0.985, top=0.88, bottom=0.04, wspace=0.10)

        out_png = bucket_dir / f"{bucket}{brank:02d}_{cid}.png"
        fig.savefig(out_png, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)

        print(f"Wrote {out_png}")

    print("=" * 80)
    print("DONE: Stage-IV case triptychs")
    print("=" * 80)
    print(f"Output dir: {out_dir}")


if __name__ == "__main__":
    main()
