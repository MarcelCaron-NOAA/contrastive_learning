#!/usr/bin/env python3
"""
Plot the three highest- and three lowest-similarity sampled negative examples.

Each rank figure has:

    top row:    high-similarity A/C | N | N-A
    bottom row: low-similarity  A/C | N | N-A

This is useful for inspecting how hard the climo45 negative population became.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import json
import os
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd
import zarr


PRECIP_BOUNDS_MM = [0.1, 0.5, 1, 2, 5, 10, 20, 30, 40, 60, 80, 100, 1.0e6]
PRECIP_COLORS = [
    "#ccffcc", "#7fff00", "#33cc33", "#009933",
    "#3399ff", "#00c8ff", "#9966ff", "#cc33ff",
    "#ff0000", "#ff7f00", "#ffd700", "#f4c2c2",
]
PRECIP_CMAP = ListedColormap(PRECIP_COLORS)
PRECIP_NORM = BoundaryNorm(PRECIP_BOUNDS_MM, PRECIP_CMAP.N)
DIFF_CMAP = plt.get_cmap("RdBu_r")


def as_bool(s):
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "t", "yes"])


def load_manifest(path):
    df = pd.read_csv(path, low_memory=False)
    req = ["sample_id", "store", "init_hour", "forecast_hour"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise RuntimeError(f"Manifest missing {missing}; found {list(df.columns)}")
    if "available" in df.columns:
        df = df.loc[as_bool(df["available"])].copy()
    return df.set_index("sample_id", drop=False)


class HRRRPatchReader:
    def __init__(self, max_open=24):
        self.max_open = max_open
        self.cache = OrderedDict()

    def _open(self, path):
        path = str(path)
        if path in self.cache:
            store, root = self.cache.pop(path)
            self.cache[path] = (store, root)
            return root

        store = zarr.ZipStore(path, mode="r")
        root = zarr.open_consolidated(store, mode="r")
        self.cache[path] = (store, root)

        while len(self.cache) > self.max_open:
            _, (old_store, _) = self.cache.popitem(last=False)
            old_store.close()

        return root

    def patch(self, field_id, y0, x0, size, manifest):
        row = manifest.loc[field_id]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        root = self._open(row["store"])
        arr = root["qpf6"][
            int(row["init_hour"]),
            int(row["forecast_hour"]) - 6,
            y0:y0 + size,
            x0:x0 + size,
        ]
        return np.asarray(arr, dtype=np.float32), row

    def close(self):
        for store, _ in self.cache.values():
            store.close()
        self.cache.clear()


def add_map_features(ax):
    ax.coastlines(resolution="50m", linewidth=0.7)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.5)
    ax.add_feature(
        cfeature.STATES.with_scale("50m"),
        linewidth=0.45,
        edgecolor="gray",
    )
    ax.set_xticks([])
    ax.set_yticks([])


def plot_patch(ax, lon, lat, data, title, cmap, norm):
    ax.pcolormesh(
        lon, lat, data,
        cmap=cmap, norm=norm, shading="auto",
        transform=ccrs.PlateCarree(),
    )
    add_map_features(ax)
    ax.set_extent(
        [
            float(np.nanmin(lon)), float(np.nanmax(lon)),
            float(np.nanmin(lat)), float(np.nanmax(lat)),
        ],
        crs=ccrs.PlateCarree(),
    )
    ax.set_title(title, fontweight="bold", pad=6)


def annotate(ax, text):
    ax.text(
        0.02, 0.02, text,
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=9.3, fontweight="bold",
        bbox=dict(
            facecolor="white",
            edgecolor="black",
            alpha=0.60,
            boxstyle="round,pad=0.35",
        ),
        zorder=20,
    )


def field_time(row):
    if "valid_end" in row.index and pd.notna(row["valid_end"]):
        v = row["valid_end"]
        if isinstance(v, (int, float, np.integer, np.floating)):
            return pd.to_datetime(v, unit="s", utc=True)
        return pd.to_datetime(v, utc=True)

    if "init_time" in row.index and pd.notna(row["init_time"]):
        return pd.to_datetime(row["init_time"], utc=True) + pd.Timedelta(
            hours=int(row["forecast_hour"])
        )
    return None


def field_label(row):
    if "init_time" in row.index and pd.notna(row["init_time"]):
        init = pd.to_datetime(row["init_time"], utc=True)
        return f"{init:%Y-%m-%d %HZ} F{int(row['forecast_hour']):02d}"

    vt = field_time(row)
    if vt is not None:
        init = vt - pd.Timedelta(hours=int(row["forecast_hour"]))
        return f"{init:%Y-%m-%d %HZ} F{int(row['forecast_hour']):02d}"

    return f"{int(row['init_hour']):02d}Z F{int(row['forecast_hour']):02d}"


def patch_size_from_meta(exp):
    paths = sorted((exp / "shards").glob("metrics_shard_*.json"))
    if not paths:
        return 256
    return int(json.loads(paths[0].read_text()).get("patch_size", 256))


def shared_diff_norm(arrays):
    vals = np.concatenate(
        [np.abs(np.asarray(x, dtype=np.float32)).ravel() for x in arrays]
    )
    vals = vals[np.isfinite(vals)]
    vmax = 1.0 if len(vals) == 0 else max(1.0, float(np.quantile(vals, 0.98)))
    return TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-dir", required=True)
    p.add_argument("--primary-window", type=int, default=15)
    p.add_argument("--hrrr-manifest", default=None)
    p.add_argument("--mapping", default=None)
    p.add_argument("--patch-size", type=int, default=None)
    args = p.parse_args()

    mpl.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "figure.titlesize": 16,
        }
    )

    exp = Path(args.experiment_dir)
    data_dir = Path(os.environ["DATA_DIR"])

    manifest_path = (
        Path(args.hrrr_manifest)
        if args.hrrr_manifest
        else data_dir / "forecasts/hrrr/qpf6_native/metadata/hrrr_qpf6_manifest.csv.gz"
    )
    mapping_path = (
        Path(args.mapping)
        if args.mapping
        else exp / "grid_mapping/stage4_to_hrrr_nearest.npz"
    )
    patch_size = args.patch_size or patch_size_from_meta(exp)

    shards = sorted((exp / "shards").glob("metrics_shard_*.csv.gz"))
    if not shards:
        raise RuntimeError("No metrics shards found.")

    metrics = pd.concat(
        [pd.read_csv(x, low_memory=False) for x in shards],
        ignore_index=True,
    )
    fcol = f"fss_w{args.primary_window}"

    if fcol not in metrics.columns:
        raise RuntimeError(f"{fcol} not found.")

    cn = metrics.loc[metrics["relation"] == "CN"].dropna(subset=[fcol]).copy()

    # Rank primarily by FSS, then correlation so tied FSS=0 cases are stable.
    high = (
        cn.sort_values([fcol, "corr"], ascending=[False, False])
        .head(3)
        .reset_index(drop=True)
    )
    low = (
        cn.sort_values([fcol, "corr"], ascending=[True, True])
        .head(3)
        .reset_index(drop=True)
    )

    if len(high) < 3 or len(low) < 3:
        raise RuntimeError("Fewer than three valid C-N rows were found.")

    manifest = load_manifest(manifest_path)

    mapping = np.load(mapping_path)
    lat_full = np.asarray(mapping["hrrr_lat"], dtype=np.float32)
    lon_full = np.asarray(mapping["hrrr_lon"], dtype=np.float32)

    meta_paths = sorted((exp / "shards").glob("metrics_shard_*.json"))
    meta = json.loads(meta_paths[0].read_text()) if meta_paths else {}
    mode = meta.get("negative_mode", "negative")

    reader = HRRRPatchReader()
    out_dir = exp / "figures/case_examples/negative_ranked"
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = []

    try:
        for rank in range(3):
            case_rows = [high.iloc[rank], low.iloc[rank]]
            case_data = []

            for row in case_rows:
                y0 = int(row["patch_y"])
                x0 = int(row["patch_x"])
                c, cmeta = reader.patch(row["a_id"], y0, x0, patch_size, manifest)
                n, nmeta = reader.patch(row["b_id"], y0, x0, patch_size, manifest)
                diff = n - c

                lat = lat_full[y0:y0 + patch_size, x0:x0 + patch_size]
                lon = lon_full[y0:y0 + patch_size, x0:x0 + patch_size]
                case_data.append((row, c, n, diff, cmeta, nmeta, lon, lat))

            diff_norm = shared_diff_norm([x[3] for x in case_data])

            fig, axes = plt.subplots(
                2, 3,
                figsize=(15.5, 9.0),
                subplot_kw={"projection": ccrs.PlateCarree()},
                constrained_layout=True,
            )

            title_mode = "climo45" if mode == "climo45" else mode
            fig.suptitle(
                f"{title_mode} negative-pair examples",
                fontweight="bold",
                fontsize=16,
            )

            for row_idx, data in enumerate(case_data):
                row, c, n, diff, cmeta, nmeta, lon, lat = data

                plot_patch(axes[row_idx, 0], lon, lat, c, "Anchor / Control (A)", PRECIP_CMAP, PRECIP_NORM)
                plot_patch(axes[row_idx, 1], lon, lat, n, "Negative (N)", PRECIP_CMAP, PRECIP_NORM)
                plot_patch(axes[row_idx, 2], lon, lat, diff, "N − A", DIFF_CMAP, diff_norm)

                cvalid = field_time(cmeta)
                nvalid = field_time(nmeta)

                descriptor = (
                    "High negative similarity"
                    if row_idx == 0
                    else "Low negative similarity"
                )

                lines = [descriptor]
                if cvalid is not None:
                    lines.append(f"A valid: {cvalid:%Y-%m-%d %HZ}")
                if nvalid is not None:
                    lines.append(f"N valid: {nvalid:%Y-%m-%d %HZ}")

                lines.extend(
                    [
                        f"A: {field_label(cmeta)}",
                        f"N: {field_label(nmeta)}",
                        f"FSS(A,N) = {float(row[fcol]):.3f}",
                        f"Corr = {float(row['corr']):.3f}",
                        f"MAD = {float(row['mean_abs_diff_mm']):.3f} mm",
                    ]
                )

                if "negative_time_distance_days" in row.index and pd.notna(
                    row["negative_time_distance_days"]
                ):
                    lines.append(
                        f"|Δt| = {float(row['negative_time_distance_days']):.0f} d"
                    )

                if "negative_climo_distance_days" in row.index and pd.notna(
                    row["negative_climo_distance_days"]
                ):
                    lines.append(
                        f"climo Δ = {int(row['negative_climo_distance_days'])} d"
                    )

                annotate(axes[row_idx, 2], "\n".join(lines))

            cbar1 = fig.colorbar(
                mpl.cm.ScalarMappable(norm=PRECIP_NORM, cmap=PRECIP_CMAP),
                ax=axes[:, :2],
                orientation="horizontal",
                fraction=0.045,
                pad=0.035,
                aspect=45,
            )
            cbar1.set_label("6-h accumulated precipitation (mm)", fontweight="bold")
            cbar1.set_ticks(PRECIP_BOUNDS_MM[:-1])

            cbar2 = fig.colorbar(
                mpl.cm.ScalarMappable(norm=diff_norm, cmap=DIFF_CMAP),
                ax=axes[:, 2],
                orientation="horizontal",
                fraction=0.045,
                pad=0.035,
                aspect=30,
            )
            cbar2.set_label("N − A (mm)", fontweight="bold")

            stem = f"negative_similarity_rank{rank + 1:02d}"
            png = out_dir / f"{stem}.png"
            pdf = out_dir / f"{stem}.pdf"
            fig.savefig(png, dpi=220, bbox_inches="tight")
            fig.savefig(pdf, bbox_inches="tight")
            plt.close(fig)

            selected.append(
                {
                    "rank": rank + 1,
                    "high_sample_id": high.iloc[rank]["sample_id"],
                    "high_negative_id": high.iloc[rank]["b_id"],
                    "high_fss": high.iloc[rank][fcol],
                    "low_sample_id": low.iloc[rank]["sample_id"],
                    "low_negative_id": low.iloc[rank]["b_id"],
                    "low_fss": low.iloc[rank][fcol],
                }
            )

            print(f"Wrote {png}")
            print(f"Wrote {pdf}")

    finally:
        reader.close()

    pd.DataFrame(selected).to_csv(
        out_dir / "selected_negative_examples.csv",
        index=False,
    )


if __name__ == "__main__":
    main()

