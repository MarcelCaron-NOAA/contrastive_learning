#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd
import xarray as xr
import zarr


LAGS = [1, 3, 6]
NY, NX = 1059, 1799

# Approximate precipitation palette for the case-study maps.
PRECIP_BOUNDS_MM = [0.1, 0.5, 1, 2, 5, 10, 20, 30, 40, 60, 80, 100, 1.0e6]
PRECIP_COLORS = [
    "#ccffcc", "#7fff00", "#33cc33", "#009933",
    "#3399ff", "#00c8ff", "#9966ff", "#cc33ff",
    "#ff0000", "#ff7f00", "#ffd700", "#f4c2c2",
]
PRECIP_CMAP = ListedColormap(PRECIP_COLORS)
PRECIP_NORM = BoundaryNorm(PRECIP_BOUNDS_MM, PRECIP_CMAP.N)
DIFF_CMAP = plt.get_cmap("RdBu_r")


def set_style():
    mpl.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "axes.labelweight": "bold",
            "figure.titlesize": 16,
        }
    )


def load_metrics(experiment_dir: Path) -> pd.DataFrame:
    paths = sorted((experiment_dir / "shards").glob("metrics_shard_*.csv.gz"))
    if not paths:
        raise FileNotFoundError(f"No metrics_shard_*.csv.gz files under {experiment_dir / 'shards'}")
    df = pd.concat([pd.read_csv(p, low_memory=False) for p in paths], ignore_index=True)
    df["lag_hours"] = pd.to_numeric(df["lag_hours"], errors="coerce")
    return df


def default_manifest(data_dir: Path) -> Path:
    canonical = data_dir / "forecasts/hrrr/qpf6_native/metadata/hrrr_qpf6_manifest.csv.gz"
    if canonical.exists():
        return canonical

    # Valid pathlib fallback: ** is supplied by rglob itself, not embedded
    # inside a path component.
    candidates = sorted(
        data_dir.joinpath("forecasts/hrrr").rglob("*manifest*.csv.gz")
    )
    if not candidates:
        raise FileNotFoundError(
            "Could not find an HRRR manifest. Expected:\n"
            f"  {canonical}"
        )
    return candidates[0]


def as_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "t", "yes"])


def load_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)

    required = ["sample_id", "store", "init_hour", "forecast_hour"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"HRRR manifest is missing expected columns {missing}.\n"
            f"Columns found: {list(df.columns)}"
        )

    if "available" in df.columns:
        df = df.loc[as_bool(df["available"])].copy()

    return df.set_index("sample_id", drop=False)


class HRRRPatchReader:
    """Read qpf6 patches from the canonical daily ZipStores."""

    def __init__(self, max_open: int = 24):
        self.max_open = max_open
        self.cache: OrderedDict[str, tuple] = OrderedDict()

    def _open(self, path: str):
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

    def patch(
        self,
        field_id: str,
        y0: int,
        x0: int,
        size: int,
        manifest: pd.DataFrame,
    ) -> Tuple[np.ndarray, pd.Series]:
        if field_id not in manifest.index:
            raise KeyError(f"Field ID not found in HRRR manifest: {field_id}")

        row = manifest.loc[field_id]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]

        root = self._open(row["store"])
        init_hour = int(row["init_hour"])
        fhr = int(row["forecast_hour"])
        lead_index = fhr - 6

        arr = root["qpf6"][
            init_hour,
            lead_index,
            y0:y0 + size,
            x0:x0 + size,
        ]
        return np.asarray(arr, dtype=np.float32), row

    def close(self):
        for store, _ in self.cache.values():
            store.close()
        self.cache.clear()


class Stage4Reader:
    """Read native Stage IV and remap to the HRRR grid using saved NN mapping."""

    def __init__(self, mapping_path: Path, stage4_root: Path):
        m = np.load(mapping_path)

        self.stage4_flat_index = np.asarray(m["stage4_flat_index"], dtype=np.int64).reshape(NY, NX)
        self.valid_mapping = np.asarray(m["valid_mapping"], dtype=bool).reshape(NY, NX)
        self.hrrr_lat = np.asarray(m["hrrr_lat"], dtype=np.float32)
        self.hrrr_lon = np.asarray(m["hrrr_lon"], dtype=np.float32)
        self.stage4_shape = tuple(int(x) for x in m["stage4_shape"])
        self.stage4_root = stage4_root
        self.cache: Dict[str, np.ndarray] = {}

    @staticmethod
    def _first_2d_var(ds: xr.Dataset) -> str:
        for name in ("tp", "APCP", "apcp"):
            if name in ds.data_vars and ds[name].ndim == 2:
                return name
        for name, da in ds.data_vars.items():
            if da.ndim == 2:
                return name
        raise RuntimeError("No 2-D Stage-IV data variable found")

    def file_path(self, valid_time: pd.Timestamp) -> Path:
        return (
            self.stage4_root
            / valid_time.strftime("%Y")
            / valid_time.strftime("%Y%m")
            / f"st4_conus.{valid_time:%Y%m%d%H}.06h.grb2"
        )

    def full_on_hrrr(self, valid_time: pd.Timestamp) -> np.ndarray:
        key = valid_time.strftime("%Y%m%d%H")
        if key in self.cache:
            return self.cache[key]

        path = self.file_path(valid_time)
        if not path.is_file():
            raise FileNotFoundError(f"Stage-IV file is missing: {path}")

        ds = xr.open_dataset(
            path,
            engine="cfgrib",
            backend_kwargs={"indexpath": ""},
        )
        try:
            name = self._first_2d_var(ds)
            native = np.asarray(ds[name].values, dtype=np.float32)
        finally:
            ds.close()

        if tuple(native.shape) != self.stage4_shape:
            raise RuntimeError(
                f"Stage-IV shape changed for {path}: "
                f"{native.shape} != {self.stage4_shape}"
            )

        out = np.full((NY, NX), np.nan, dtype=np.float32)
        valid = self.valid_mapping
        out[valid] = native.ravel()[self.stage4_flat_index[valid]]
        self.cache[key] = out
        return out

    def patch(
        self,
        valid_time: pd.Timestamp,
        y0: int,
        x0: int,
        size: int,
    ) -> np.ndarray:
        full = self.full_on_hrrr(valid_time)
        return full[y0:y0 + size, x0:x0 + size]


def patch_size_from_metadata(experiment_dir: Path) -> int:
    paths = sorted((experiment_dir / "shards").glob("metrics_shard_*.json"))
    if not paths:
        return 256
    try:
        return int(json.loads(paths[0].read_text()).get("patch_size", 256))
    except Exception:
        return 256


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
    mesh = ax.pcolormesh(
        lon,
        lat,
        data,
        cmap=cmap,
        norm=norm,
        shading="auto",
        transform=ccrs.PlateCarree(),
    )
    add_map_features(ax)
    ax.set_extent(
        [
            float(np.nanmin(lon)),
            float(np.nanmax(lon)),
            float(np.nanmin(lat)),
            float(np.nanmax(lat)),
        ],
        crs=ccrs.PlateCarree(),
    )
    ax.set_title(title, fontweight="bold", pad=6)
    return mesh


def annotation_box(ax, text: str):
    ax.text(
        0.02,
        0.02,
        text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.3,
        fontweight="bold",
        bbox=dict(
            facecolor="white",
            edgecolor="black",
            alpha=0.60,
            boxstyle="round,pad=0.35",
        ),
        zorder=20,
    )


def field_label(row: pd.Series) -> str:
    init_hour = int(row["init_hour"])
    fhr = int(row["forecast_hour"])
    if "init_time" in row.index and pd.notna(row["init_time"]):
        try:
            init = pd.to_datetime(row["init_time"], utc=True)
            return f"{init:%Y-%m-%d %HZ} F{fhr:02d}"
        except Exception:
            pass
    return f"{init_hour:02d}Z F{fhr:02d}"


def shared_diff_norm(diff_arrays: List[np.ndarray]):
    vals = np.concatenate(
        [np.abs(np.asarray(a, dtype=np.float32)).ravel() for a in diff_arrays]
    )
    vals = vals[np.isfinite(vals)]
    vmax = 1.0 if len(vals) == 0 else max(1.0, float(np.quantile(vals, 0.98)))
    return TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)


def select_cp_examples(cp: pd.DataFrame, fcol: str, n: int = 3):
    selected = {}
    for lag in LAGS:
        d = (
            cp.loc[cp["lag_hours"] == lag]
            .dropna(subset=[fcol])
            .sort_values(fcol)
            .reset_index(drop=True)
        )
        if len(d) < 2 * n:
            raise RuntimeError(f"Not enough C-P rows for {lag}-h lag.")
        selected[lag] = {
            "worst": d.head(n).reset_index(drop=True),
            "best": d.tail(n).sort_values(fcol, ascending=False).reset_index(drop=True),
        }
    return selected


def build_delta_table(
    cp: pd.DataFrame,
    po: pd.DataFrame,
    co: pd.DataFrame,
    fcol: str,
) -> pd.DataFrame:
    # PO row: a_id is the positive forecast; b_id is Stage IV path.
    po_use = po[
        [
            "sample_id", "valid_time", "patch_y", "patch_x",
            "lag_hours", "direction", "a_id", fcol,
        ]
    ].rename(
        columns={
            "a_id": "p_id",
            fcol: "fss_po",
        }
    )

    # CO row: a_id is the control forecast.
    co_use = co[
        ["sample_id", "a_id", fcol]
    ].rename(
        columns={
            "a_id": "c_id",
            fcol: "fss_co",
        }
    )

    # Bring in direct C-P FSS for annotation.
    cp_use = cp[
        ["sample_id", "lag_hours", "direction", "a_id", "b_id", fcol]
    ].rename(
        columns={
            "a_id": "cp_c_id",
            "b_id": "cp_p_id",
            fcol: "fss_cp",
        }
    )

    d = po_use.merge(co_use, on="sample_id", how="inner")
    d = d.merge(
        cp_use,
        on=["sample_id", "lag_hours", "direction"],
        how="left",
    )

    # These should agree; keeping both gives us a sanity check.
    bad_c = d["cp_c_id"].notna() & (d["cp_c_id"] != d["c_id"])
    bad_p = d["cp_p_id"].notna() & (d["cp_p_id"] != d["p_id"])
    if bad_c.any() or bad_p.any():
        raise RuntimeError("C/P IDs disagree while merging PO/CO/CP rows.")

    d["delta_fss"] = d["fss_po"] - d["fss_co"]
    return d


def select_delta_examples(delta: pd.DataFrame, n: int = 3):
    selected = {}
    for lag in LAGS:
        d = (
            delta.loc[delta["lag_hours"] == lag]
            .dropna(subset=["delta_fss"])
            .sort_values("delta_fss")
            .reset_index(drop=True)
        )
        if len(d) < 2 * n:
            raise RuntimeError(f"Not enough ΔFSS rows for {lag}-h lag.")
        selected[lag] = {
            "worst": d.head(n).reset_index(drop=True),
            "best": d.tail(n).sort_values("delta_fss", ascending=False).reset_index(drop=True),
        }
    return selected


def write_selected_cases(
    path: Path,
    cp_sel,
    delta_sel,
    fcol: str,
):
    rows = []
    for lag in LAGS:
        for rank in range(3):
            cb = cp_sel[lag]["best"].iloc[rank]
            cw = cp_sel[lag]["worst"].iloc[rank]
            db = delta_sel[lag]["best"].iloc[rank]
            dw = delta_sel[lag]["worst"].iloc[rank]

            rows.append(
                {
                    "family": "cp_fss",
                    "lag_hours": lag,
                    "rank": rank + 1,
                    "best_sample_id": cb["sample_id"],
                    "best_direction": cb["direction"],
                    "best_metric": cb[fcol],
                    "worst_sample_id": cw["sample_id"],
                    "worst_direction": cw["direction"],
                    "worst_metric": cw[fcol],
                }
            )
            rows.append(
                {
                    "family": "delta_fss",
                    "lag_hours": lag,
                    "rank": rank + 1,
                    "best_sample_id": db["sample_id"],
                    "best_direction": db["direction"],
                    "best_metric": db["delta_fss"],
                    "worst_sample_id": dw["sample_id"],
                    "worst_direction": dw["direction"],
                    "worst_metric": dw["delta_fss"],
                }
            )

    pd.DataFrame(rows).to_csv(path, index=False)


def make_cp_figure(
    lag: int,
    rank_idx: int,
    best_row: pd.Series,
    worst_row: pd.Series,
    fcol: str,
    patch_size: int,
    manifest: pd.DataFrame,
    hrrr: HRRRPatchReader,
    stage4: Stage4Reader,
    output_dir: Path,
):
    case_rows = [best_row, worst_row]
    case_data = []

    for row in case_rows:
        y0 = int(row["patch_y"])
        x0 = int(row["patch_x"])

        c, cmeta = hrrr.patch(row["a_id"], y0, x0, patch_size, manifest)
        p, pmeta = hrrr.patch(row["b_id"], y0, x0, patch_size, manifest)
        diff = p - c

        lon = stage4.hrrr_lon[y0:y0 + patch_size, x0:x0 + patch_size]
        lat = stage4.hrrr_lat[y0:y0 + patch_size, x0:x0 + patch_size]

        case_data.append((c, p, diff, cmeta, pmeta, lon, lat))

    diff_norm = shared_diff_norm([x[2] for x in case_data])

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15.5, 9.0),
        subplot_kw={"projection": ccrs.PlateCarree()},
        constrained_layout=True,
    )

    fig.suptitle(
        f"{lag}-h time-lag positive-pair examples",
        fontsize=16,
        fontweight="bold",
    )

    for row_idx, (metric_row, data) in enumerate(zip(case_rows, case_data)):
        c, p, diff, cmeta, pmeta, lon, lat = data

        plot_patch(axes[row_idx, 0], lon, lat, c, "Control (C)", PRECIP_CMAP, PRECIP_NORM)
        plot_patch(axes[row_idx, 1], lon, lat, p, "Positive (P)", PRECIP_CMAP, PRECIP_NORM)
        plot_patch(axes[row_idx, 2], lon, lat, diff, "P − C", DIFF_CMAP, diff_norm)

        valid = pd.to_datetime(metric_row["valid_time"], utc=True)
        descriptor = "High similarity" if row_idx == 0 else "Low similarity"
        text = (
            f"{descriptor}\n"
            f"Valid: {valid:%Y-%m-%d %HZ}\n"
            f"C: {field_label(cmeta)}\n"
            f"P: {field_label(pmeta)}\n"
            f"FSS = {float(metric_row[fcol]):.3f}"
        )
        annotation_box(axes[row_idx, 2], text)

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
    cbar1.ax.tick_params(labelsize=9)

    cbar2 = fig.colorbar(
        mpl.cm.ScalarMappable(norm=diff_norm, cmap=DIFF_CMAP),
        ax=axes[:, 2],
        orientation="horizontal",
        fraction=0.045,
        pad=0.035,
        aspect=30,
    )
    cbar2.set_label("P − C (mm)", fontweight="bold")
    cbar2.ax.tick_params(labelsize=9)

    stem = f"cp_similarity_lag{lag:02d}_rank{rank_idx + 1:02d}"
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


def make_delta_figure(
    lag: int,
    rank_idx: int,
    best_row: pd.Series,
    worst_row: pd.Series,
    patch_size: int,
    manifest: pd.DataFrame,
    hrrr: HRRRPatchReader,
    stage4: Stage4Reader,
    output_dir: Path,
):
    case_rows = [best_row, worst_row]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15.5, 9.0),
        subplot_kw={"projection": ccrs.PlateCarree()},
        constrained_layout=True,
    )

    fig.suptitle(
        f"{lag}-h time-lag Stage-IV consistency examples",
        fontsize=16,
        fontweight="bold",
    )

    for row_idx, row in enumerate(case_rows):
        y0 = int(row["patch_y"])
        x0 = int(row["patch_x"])

        c, cmeta = hrrr.patch(row["c_id"], y0, x0, patch_size, manifest)
        p, pmeta = hrrr.patch(row["p_id"], y0, x0, patch_size, manifest)

        valid = pd.to_datetime(row["valid_time"], utc=True)
        o = stage4.patch(valid, y0, x0, patch_size)

        lon = stage4.hrrr_lon[y0:y0 + patch_size, x0:x0 + patch_size]
        lat = stage4.hrrr_lat[y0:y0 + patch_size, x0:x0 + patch_size]

        plot_patch(axes[row_idx, 0], lon, lat, c, "Control (C)", PRECIP_CMAP, PRECIP_NORM)
        plot_patch(axes[row_idx, 1], lon, lat, p, "Positive (P)", PRECIP_CMAP, PRECIP_NORM)
        plot_patch(axes[row_idx, 2], lon, lat, o, "Stage IV (O)", PRECIP_CMAP, PRECIP_NORM)

        descriptor = "High ΔFSS" if row_idx == 0 else "Low ΔFSS"
        text = (
            f"{descriptor}\n"
            f"Valid: {valid:%Y-%m-%d %HZ}\n"
            f"C: {field_label(cmeta)}\n"
            f"P: {field_label(pmeta)}\n"
            f"FSS(C,P) = {float(row['fss_cp']):.3f}\n"
            f"FSS(C,O) = {float(row['fss_co']):.3f}\n"
            f"FSS(P,O) = {float(row['fss_po']):.3f}\n"
            f"ΔFSS = {float(row['delta_fss']):+.3f}"
        )
        annotation_box(axes[row_idx, 2], text)

    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=PRECIP_NORM, cmap=PRECIP_CMAP),
        ax=axes,
        orientation="horizontal",
        fraction=0.045,
        pad=0.035,
        aspect=50,
    )
    cbar.set_label("6-h accumulated precipitation (mm)", fontweight="bold")
    cbar.set_ticks(PRECIP_BOUNDS_MM[:-1])
    cbar.ax.tick_params(labelsize=9)

    stem = f"delta_fss_lag{lag:02d}_rank{rank_idx + 1:02d}"
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--primary-window", type=int, default=15)
    parser.add_argument("--hrrr-manifest", default=None)
    parser.add_argument("--stage4-root", default=None)
    parser.add_argument("--mapping", default=None)
    parser.add_argument("--patch-size", type=int, default=None)
    args = parser.parse_args()

    set_style()

    experiment_dir = Path(args.experiment_dir)
    data_dir = Path(os.environ["DATA_DIR"])

    manifest_path = (
        Path(args.hrrr_manifest)
        if args.hrrr_manifest
        else default_manifest(data_dir)
    )
    stage4_root = (
        Path(args.stage4_root)
        if args.stage4_root
        else data_dir / "observations/stage4/raw"
    )
    mapping_path = (
        Path(args.mapping)
        if args.mapping
        else experiment_dir / "grid_mapping/stage4_to_hrrr_nearest.npz"
    )
    patch_size = (
        args.patch_size
        if args.patch_size is not None
        else patch_size_from_metadata(experiment_dir)
    )

    print(f"Experiment dir : {experiment_dir}")
    print(f"HRRR manifest  : {manifest_path}")
    print(f"Stage IV root  : {stage4_root}")
    print(f"Grid mapping   : {mapping_path}")
    print(f"Patch size     : {patch_size}")

    metrics = load_metrics(experiment_dir)
    fcol = f"fss_w{args.primary_window}"
    if fcol not in metrics.columns:
        raise RuntimeError(
            f"{fcol} not found. Available FSS columns: "
            f"{[c for c in metrics.columns if c.startswith('fss_w')]}"
        )

    cp = metrics.loc[metrics["relation"] == "CP"].copy()
    po = metrics.loc[metrics["relation"] == "PO"].copy()
    co = metrics.loc[metrics["relation"] == "CO"].copy()

    cp_sel = select_cp_examples(cp, fcol, n=3)
    delta_table = build_delta_table(cp, po, co, fcol)
    delta_sel = select_delta_examples(delta_table, n=3)

    manifest = load_manifest(manifest_path)
    hrrr = HRRRPatchReader()
    stage4 = Stage4Reader(mapping_path, stage4_root)

    out_root = experiment_dir / "figures/case_examples"
    cp_dir = out_root / "cp_ranked"
    delta_dir = out_root / "delta_ranked"
    cp_dir.mkdir(parents=True, exist_ok=True)
    delta_dir.mkdir(parents=True, exist_ok=True)

    write_selected_cases(
        out_root / "selected_case_examples.csv",
        cp_sel,
        delta_sel,
        fcol,
    )

    try:
        for lag in LAGS:
            for rank_idx in range(3):
                make_cp_figure(
                    lag,
                    rank_idx,
                    cp_sel[lag]["best"].iloc[rank_idx],
                    cp_sel[lag]["worst"].iloc[rank_idx],
                    fcol,
                    patch_size,
                    manifest,
                    hrrr,
                    stage4,
                    cp_dir,
                )

        for lag in LAGS:
            for rank_idx in range(3):
                make_delta_figure(
                    lag,
                    rank_idx,
                    delta_sel[lag]["best"].iloc[rank_idx],
                    delta_sel[lag]["worst"].iloc[rank_idx],
                    patch_size,
                    manifest,
                    hrrr,
                    stage4,
                    delta_dir,
                )
    finally:
        hrrr.close()


if __name__ == "__main__":
    main()

