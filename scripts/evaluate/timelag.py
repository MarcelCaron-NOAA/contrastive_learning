#!/usr/bin/env python3
"""
HRRR time-lag positive/negative diagnostic experiment.

This keeps the existing lag-selection experiment interface/output shape, but
adds configurable negative sampling:

  calendar_any:
      N is HRRR F12, same UTC valid hour as A/C, |dt| > 90 days.

  climo45:
      same as calendar_any, plus a DIFFERENT calendar year and circular
      climatological day-of-year distance <= 45 days.

For a clean apples-to-apples comparison, --sample-plan-from can point to the
existing calendar_any experiment.  The same valid times and 256x256 crop
coordinates are then reused; only the negative fields change.

Control/anchor for this diagnostic:
    C/A = HRRR F12

Positive pairs:
    lag 1 h: earlier init F13, later init F11
    lag 3 h: earlier init F15, later init F09
    lag 6 h: earlier init F18, later init F06

Output relations:
    CP : control-positive
    CN : control-negative
    CO : control-Stage IV
    PO : positive-Stage IV
    NO : negative compared to Stage IV valid at C time
         (diagnostic baseline only; not forecast skill)
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import zarr
from scipy.ndimage import uniform_filter


NY, NX = 1059, 1799
LAGS = (1, 3, 6)
REQUIRED_FHRS = (6, 9, 11, 12, 13, 15, 18)
SEASONS = ("DJF", "MAM", "JJA", "SON")


# ---------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------

def as_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "t", "yes"])


def parse_time_series(s: pd.Series) -> pd.Series:
    """Parse strings or Unix numeric timestamps robustly."""
    if pd.api.types.is_numeric_dtype(s):
        vals = pd.to_numeric(s, errors="coerce")
        finite = vals[np.isfinite(vals)]
        if len(finite) == 0:
            return pd.to_datetime(s, errors="coerce", utc=True)

        med = float(np.nanmedian(np.abs(finite)))
        if med < 1e11:
            unit = "s"
        elif med < 1e14:
            unit = "ms"
        elif med < 1e17:
            unit = "us"
        else:
            unit = "ns"

        return pd.to_datetime(vals, unit=unit, errors="coerce", utc=True)

    return pd.to_datetime(s, errors="coerce", utc=True)


def season_from_month(month: int) -> str:
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"


def climo_day(ts: pd.Timestamp) -> int:
    """
    Leap-safe climatological day on a fixed 366-day calendar.
    Every month/day is mapped into leap year 2000.
    """
    t = pd.Timestamp(ts)
    return pd.Timestamp(year=2000, month=t.month, day=t.day).dayofyear


def circular_climo_distance(day_a: int, day_b: int) -> int:
    d = abs(int(day_a) - int(day_b))
    return min(d, 366 - d)


def load_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)

    required = ["sample_id", "store", "init_hour", "forecast_hour", "valid_end"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"Manifest is missing {missing}. Columns found: {list(df.columns)}"
        )

    if "available" in df.columns:
        df = df.loc[as_bool(df["available"])].copy()

    df["valid_end_dt"] = parse_time_series(df["valid_end"])
    df["forecast_hour"] = pd.to_numeric(df["forecast_hour"], errors="coerce")
    df["init_hour"] = pd.to_numeric(df["init_hour"], errors="coerce")

    df = df.dropna(
        subset=["sample_id", "store", "valid_end_dt", "forecast_hour", "init_hour"]
    ).copy()

    df["forecast_hour"] = df["forecast_hour"].astype(int)
    df["init_hour"] = df["init_hour"].astype(int)

    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    df["valid_unix_s"] = (
        (df["valid_end_dt"] - epoch).dt.total_seconds().round().astype(np.int64)
    )
    df["valid_hour_utc"] = df["valid_end_dt"].dt.hour.astype(int)
    df["valid_year"] = df["valid_end_dt"].dt.year.astype(int)
    df["climo_day"] = df["valid_end_dt"].map(climo_day)

    dup = df.duplicated(["valid_unix_s", "forecast_hour"], keep=False)
    if dup.any():
        raise RuntimeError(
            "Duplicate (valid_time, forecast_hour) rows in manifest:\n"
            + df.loc[
                dup,
                ["sample_id", "valid_end_dt", "forecast_hour", "init_hour"],
            ].head(20).to_string(index=False)
        )

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------
# HRRR reader
# ---------------------------------------------------------------------

class HRRRPatchReader:
    def __init__(self, manifest: pd.DataFrame, max_open: int = 24):
        self.manifest = manifest.set_index("sample_id", drop=False)
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

    def patch(self, field_id: str, y0: int, x0: int, size: int) -> np.ndarray:
        row = self.manifest.loc[field_id]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]

        root = self._open(row["store"])
        arr = root["qpf6"][
            int(row["init_hour"]),
            int(row["forecast_hour"]) - 6,
            y0:y0 + size,
            x0:x0 + size,
        ]
        return np.asarray(arr, dtype=np.float32)

    def close(self):
        for store, _ in self.cache.values():
            store.close()
        self.cache.clear()


# ---------------------------------------------------------------------
# Stage IV reader
# ---------------------------------------------------------------------

class Stage4Reader:
    def __init__(self, mapping_path: Path, stage4_root: Path, max_cache: int = 8):
        m = np.load(mapping_path)

        self.stage4_flat_index = np.asarray(
            m["stage4_flat_index"], dtype=np.int64
        ).reshape(NY, NX)
        self.valid_mapping = np.asarray(
            m["valid_mapping"], dtype=bool
        ).reshape(NY, NX)
        self.stage4_shape = tuple(int(x) for x in m["stage4_shape"])
        self.stage4_root = Path(stage4_root)
        self.max_cache = max_cache
        self.cache: OrderedDict[str, np.ndarray] = OrderedDict()

    @staticmethod
    def _first_2d_var(ds: xr.Dataset) -> str:
        for name in ("tp", "APCP", "apcp"):
            if name in ds.data_vars and ds[name].ndim == 2:
                return name
        for name, da in ds.data_vars.items():
            if da.ndim == 2:
                return name
        raise RuntimeError("No 2-D Stage-IV data variable found")

    def file_path(self, valid: pd.Timestamp) -> Path:
        return (
            self.stage4_root
            / valid.strftime("%Y")
            / valid.strftime("%Y%m")
            / f"st4_conus.{valid:%Y%m%d%H}.06h.grb2"
        )

    def available(self, valid: pd.Timestamp) -> bool:
        return self.file_path(valid).is_file()

    def full_on_hrrr(self, valid: pd.Timestamp) -> np.ndarray:
        key = valid.strftime("%Y%m%d%H")
        if key in self.cache:
            arr = self.cache.pop(key)
            self.cache[key] = arr
            return arr

        path = self.file_path(valid)
        if not path.is_file():
            raise FileNotFoundError(path)

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
        good = self.valid_mapping
        out[good] = native.ravel()[self.stage4_flat_index[good]]

        self.cache[key] = out
        while len(self.cache) > self.max_cache:
            self.cache.popitem(last=False)

        return out

    def patch(self, valid: pd.Timestamp, y0: int, x0: int, size: int) -> np.ndarray:
        full = self.full_on_hrrr(valid)
        return full[y0:y0 + size, x0:x0 + size]


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def spatial_corr(a: np.ndarray, b: np.ndarray) -> float:
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 3:
        return np.nan

    av = a[valid].astype(np.float64)
    bv = b[valid].astype(np.float64)

    if np.std(av) == 0.0 or np.std(bv) == 0.0:
        return np.nan

    return float(np.corrcoef(av, bv)[0, 1])


def fss(a: np.ndarray, b: np.ndarray, threshold: float, window: int) -> float:
    """
    Fractions Skill Score with finite-value masking.

    For all-HRRR pairs the finite mask is effectively all ones.  The mask
    handling matters for Stage IV near the mapping boundary.
    """
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() == 0:
        return np.nan

    ea = (np.isfinite(a) & (a >= threshold)).astype(np.float32)
    eb = (np.isfinite(b) & (b >= threshold)).astype(np.float32)
    vv = valid.astype(np.float32)

    fa_num = uniform_filter(ea, size=window, mode="constant", cval=0.0)
    fb_num = uniform_filter(eb, size=window, mode="constant", cval=0.0)
    fv = uniform_filter(vv, size=window, mode="constant", cval=0.0)

    good = fv > 0.0
    if not np.any(good):
        return np.nan

    fa = np.zeros_like(fa_num)
    fb = np.zeros_like(fb_num)
    fa[good] = fa_num[good] / fv[good]
    fb[good] = fb_num[good] / fv[good]

    den = np.mean((fa[good] ** 2) + (fb[good] ** 2))
    if den <= 0.0:
        return np.nan

    mse = np.mean((fa[good] - fb[good]) ** 2)
    return float(1.0 - mse / den)


def ets(a: np.ndarray, b: np.ndarray, threshold: float) -> float:
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() == 0:
        return np.nan

    aa = a[valid] >= threshold
    bb = b[valid] >= threshold

    hits = int(np.sum(aa & bb))
    fa = int(np.sum(~aa & bb))
    miss = int(np.sum(aa & ~bb))
    n = len(aa)

    hits_random = ((hits + miss) * (hits + fa)) / n
    den = hits + fa + miss - hits_random
    if den == 0:
        return np.nan

    return float((hits - hits_random) / den)


def pair_metrics(
    a: np.ndarray,
    b: np.ndarray,
    threshold: float,
    windows: tuple[int, ...],
) -> dict:
    valid = np.isfinite(a) & np.isfinite(b)

    out = {
        "rmse_mm": np.nan,
        "bcrmse_mm": np.nan,
        "mean_abs_diff_mm": np.nan,
        "mean_bias_b_minus_a_mm": np.nan,
        "corr": np.nan,
        "mean_a_mm": np.nan,
        "mean_b_mm": np.nan,
        "max_a_mm": np.nan,
        "max_b_mm": np.nan,
        "ets": np.nan,
    }
    for w in windows:
        out[f"fss_w{w}"] = np.nan

    if valid.sum() == 0:
        return out

    av = a[valid].astype(np.float64)
    bv = b[valid].astype(np.float64)
    diff = bv - av

    out["rmse_mm"] = float(np.sqrt(np.mean(diff ** 2)))
    bias = float(np.mean(diff))
    out["bcrmse_mm"] = float(np.sqrt(np.mean((diff - bias) ** 2)))
    out["mean_abs_diff_mm"] = float(np.mean(np.abs(diff)))
    out["mean_bias_b_minus_a_mm"] = bias
    out["corr"] = spatial_corr(a, b)
    out["mean_a_mm"] = float(np.mean(av))
    out["mean_b_mm"] = float(np.mean(bv))
    out["max_a_mm"] = float(np.max(av))
    out["max_b_mm"] = float(np.max(bv))
    out["ets"] = ets(a, b, threshold)

    for w in windows:
        out[f"fss_w{w}"] = fss(a, b, threshold, w)

    return out


# ---------------------------------------------------------------------
# Sample plan
# ---------------------------------------------------------------------

def load_sample_plan(experiment_dir: Path) -> pd.DataFrame:
    paths = sorted((Path(experiment_dir) / "shards").glob("metrics_shard_*.csv.gz"))
    if not paths:
        raise RuntimeError(
            f"--sample-plan-from was given, but no metrics shards were found under "
            f"{Path(experiment_dir) / 'shards'}"
        )

    parts = []
    for path in paths:
        d = pd.read_csv(
            path,
            usecols=lambda c: c in {
                "sample_id", "valid_time", "patch_y", "patch_x", "relation"
            },
            low_memory=False,
        )
        d = d.loc[d["relation"] == "CP"].copy()
        parts.append(d)

    plan = pd.concat(parts, ignore_index=True)
    plan = (
        plan[["sample_id", "valid_time", "patch_y", "patch_x"]]
        .drop_duplicates("sample_id")
        .copy()
    )
    plan["valid_time"] = pd.to_datetime(plan["valid_time"], utc=True)
    plan = plan.sort_values(["valid_time", "sample_id"]).reset_index(drop=True)

    if plan["sample_id"].duplicated().any():
        raise RuntimeError("Sample plan still contains duplicate sample IDs.")

    return plan


def build_seasonally_balanced_plan(
    eligible_valids: list[pd.Timestamp],
    total_target: int,
    seed: int,
) -> pd.DataFrame:
    pilot = pd.DataFrame({"valid_time": pd.DatetimeIndex(eligible_valids)})
    pilot["season"] = pilot["valid_time"].dt.month.map(season_from_month)

    base = total_target // 4
    remainder = total_target % 4
    targets = {
        season: base + (1 if i < remainder else 0)
        for i, season in enumerate(SEASONS)
    }

    rng = np.random.default_rng(seed)
    selected = []

    for season in SEASONS:
        vals = pilot.loc[pilot["season"] == season, "valid_time"].to_numpy()
        n_take = min(targets[season], len(vals))
        if n_take < targets[season]:
            print(
                f"WARNING: requested {targets[season]} {season} cases but only "
                f"{len(vals)} are available."
            )
        idx = rng.choice(len(vals), size=n_take, replace=False)
        selected.extend(vals[idx].tolist())

    selected = sorted(pd.to_datetime(selected, utc=True).tolist())
    counts = pd.Series([season_from_month(v.month) for v in selected]).value_counts()

    print(
        "Seasonally balanced pilot sample: "
        + ", ".join(f"{s}={int(counts.get(s, 0))}" for s in SEASONS)
    )

    # Crop coordinates are filled by the shard using a deterministic RNG.
    return pd.DataFrame(
        {
            "sample_id": [
                f"pilot_{pd.Timestamp(v):%Y%m%d%H}_{i:06d}"
                for i, v in enumerate(selected)
            ],
            "valid_time": selected,
            "patch_y": -1,
            "patch_x": -1,
        }
    )


# ---------------------------------------------------------------------
# Negative selection
# ---------------------------------------------------------------------

def make_negative_pool(manifest: pd.DataFrame) -> pd.DataFrame:
    pool = manifest.loc[manifest["forecast_hour"] == 12].copy()
    return pool[
        [
            "sample_id",
            "valid_end_dt",
            "valid_unix_s",
            "valid_hour_utc",
            "valid_year",
            "climo_day",
        ]
    ].reset_index(drop=True)


def eligible_negatives(
    pool: pd.DataFrame,
    valid: pd.Timestamp,
    min_days: float,
    mode: str,
    climo_window_days: int,
) -> pd.DataFrame:
    valid = pd.Timestamp(valid)
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    valid_s = int(round((valid - epoch).total_seconds()))
    min_s = int(round(min_days * 86400.0))

    m = (
        (pool["valid_hour_utc"] == valid.hour)
        & ((pool["valid_unix_s"] - valid_s).abs() > min_s)
    )

    d = pool.loc[m].copy()

    if mode == "climo45":
        # The hard-negative experiment intentionally uses a similar phase
        # of the annual cycle from a DIFFERENT calendar year.
        d = d.loc[d["valid_year"] != valid.year].copy()

        a_day = climo_day(valid)
        diff = (d["climo_day"] - a_day).abs()
        d["climo_distance_days"] = np.minimum(diff, 366 - diff).astype(int)
        d = d.loc[d["climo_distance_days"] <= climo_window_days].copy()
    else:
        d["climo_distance_days"] = [
            circular_climo_distance(climo_day(valid), x)
            for x in d["climo_day"].to_numpy()
        ]

    d["time_distance_days"] = (
        (d["valid_unix_s"] - valid_s).abs() / 86400.0
    )

    return d


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()

    p.add_argument("--experiment-dir", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--stage4-root", required=True)
    p.add_argument("--mapping", required=True)

    p.add_argument("--shard-index", type=int, required=True)
    p.add_argument("--n-shards", type=int, required=True)

    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--patches-per-valid", type=int, default=1)
    p.add_argument("--negative-count", type=int, default=8)
    p.add_argument("--negative-min-days", type=float, default=90.0)

    p.add_argument(
        "--negative-mode",
        choices=["calendar_any", "climo45"],
        default="climo45",
    )
    p.add_argument("--climo-window-days", type=int, default=45)

    p.add_argument("--wet-threshold-mm", type=float, default=1.0)
    p.add_argument("--min-wet-fraction", type=float, default=0.01)
    p.add_argument("--patch-tries", type=int, default=50)

    p.add_argument("--fss-threshold-mm", type=float, default=5.0)
    p.add_argument("--fss-windows", default="1,3,7,15,31,61")

    p.add_argument("--max-valid-times", type=int, default=None)
    p.add_argument("--sample-plan-from", default=None)
    p.add_argument("--seed", type=int, default=20260926)

    a = p.parse_args()

    if not (0 <= a.shard_index < a.n_shards):
        raise SystemExit("Invalid shard index.")
    if a.patch_size > NY or a.patch_size > NX:
        raise SystemExit("Patch size exceeds HRRR grid.")
    if a.negative_count < 1:
        raise SystemExit("--negative-count must be >= 1.")

    windows = tuple(int(x) for x in a.fss_windows.split(",") if x.strip())
    if any(w < 1 or w % 2 == 0 for w in windows):
        raise SystemExit("FSS windows must be positive odd integers.")

    exp = Path(a.experiment_dir)
    shard_dir = exp / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(Path(a.manifest))

    field_index = {
        (int(r.valid_unix_s), int(r.forecast_hour)): r.sample_id
        for r in manifest.itertuples(index=False)
    }

    f12 = manifest.loc[manifest["forecast_hour"] == 12].copy()
    eligible_valids = sorted(
        [
            r.valid_end_dt
            for r in f12.itertuples(index=False)
            if all((int(r.valid_unix_s), fhr) in field_index for fhr in REQUIRED_FHRS)
        ]
    )
    complete_case_total = len(eligible_valids)

    # ------------------------------------------------------------------
    # Use the old calendar_any experiment as the sample/crop plan when
    # supplied.  This is the preferred fair comparison.
    # ------------------------------------------------------------------
    if a.sample_plan_from:
        full_plan = load_sample_plan(Path(a.sample_plan_from))
        print(
            f"Reusing sample/crop plan from: {a.sample_plan_from}\n"
            f"Sample-plan cases: {len(full_plan):,}"
        )
    elif a.max_valid_times is not None:
        full_plan = build_seasonally_balanced_plan(
            eligible_valids,
            total_target=int(a.max_valid_times) * int(a.n_shards),
            seed=a.seed,
        )
    else:
        full_plan = pd.DataFrame(
            {
                "sample_id": [
                    f"full_{pd.Timestamp(v):%Y%m%d%H}_{i:06d}"
                    for i, v in enumerate(eligible_valids)
                ],
                "valid_time": eligible_valids,
                "patch_y": -1,
                "patch_x": -1,
            }
        )

    shard_plan = full_plan.iloc[a.shard_index :: a.n_shards].reset_index(drop=True)

    print(f"Complete-case valid times total: {complete_case_total:,}")
    print(f"Selected valid times total     : {len(full_plan):,}")
    print(f"Shard {a.shard_index}: {len(shard_plan):,} planned cases")
    print(f"Negative mode: {a.negative_mode}")
    if a.negative_mode == "climo45":
        print(f"Climatological window: +/-{a.climo_window_days} days")

    hrrr = HRRRPatchReader(manifest)
    stage4 = Stage4Reader(Path(a.mapping), Path(a.stage4_root))
    negative_pool = make_negative_pool(manifest)

    # Separate RNG streams so changing N rules does not alter crop selection.
    crop_rng = np.random.default_rng(a.seed + 10_000 + a.shard_index)
    neg_rng = np.random.default_rng(a.seed + 20_000 + a.shard_index)

    rows = []
    accepted_samples = 0
    skipped_no_wet_patch = 0
    skipped_no_negatives = 0
    stage4_samples = 0
    negative_candidate_counts = []

    try:
        for case_i, plan_row in shard_plan.iterrows():
            valid = pd.Timestamp(plan_row["valid_time"])
            epoch = pd.Timestamp("1970-01-01", tz="UTC")
            valid_s = int(round((valid - epoch).total_seconds()))

            ids = {
                fhr: field_index.get((valid_s, fhr))
                for fhr in REQUIRED_FHRS
            }
            if any(ids[fhr] is None for fhr in REQUIRED_FHRS):
                print(f"WARNING: sample-plan valid time no longer complete-case: {valid}")
                continue

            c_id = ids[12]

            # ----------------------------------------------------------
            # Crop.  If reusing an old plan, preserve its exact crop.
            # Otherwise draw until the C patch passes the original wet
            # criterion used in this diagnostic experiment.
            # ----------------------------------------------------------
            if int(plan_row["patch_y"]) >= 0 and int(plan_row["patch_x"]) >= 0:
                y0 = int(plan_row["patch_y"])
                x0 = int(plan_row["patch_x"])
                c = hrrr.patch(c_id, y0, x0, a.patch_size)
            else:
                c = None
                y0 = x0 = None
                for _ in range(a.patch_tries):
                    yy = int(crop_rng.integers(0, NY - a.patch_size + 1))
                    xx = int(crop_rng.integers(0, NX - a.patch_size + 1))
                    cc = hrrr.patch(c_id, yy, xx, a.patch_size)

                    wet_fraction = float(
                        np.mean(np.isfinite(cc) & (cc >= a.wet_threshold_mm))
                    )
                    if wet_fraction >= a.min_wet_fraction:
                        y0, x0, c = yy, xx, cc
                        break

                if c is None:
                    skipped_no_wet_patch += 1
                    continue

            # ----------------------------------------------------------
            # Negative candidates.
            # ----------------------------------------------------------
            cand = eligible_negatives(
                negative_pool,
                valid=valid,
                min_days=a.negative_min_days,
                mode=a.negative_mode,
                climo_window_days=a.climo_window_days,
            )
            negative_candidate_counts.append(len(cand))

            if len(cand) < a.negative_count:
                skipped_no_negatives += 1
                print(
                    f"WARNING: {valid} has only {len(cand)} eligible negatives "
                    f"for mode={a.negative_mode}; skipping."
                )
                continue

            pick = neg_rng.choice(
                len(cand),
                size=a.negative_count,
                replace=False,
            )
            nrows = cand.iloc[pick].reset_index(drop=True)

            neg_patches = []
            for nr in nrows.itertuples(index=False):
                neg_patches.append(
                    hrrr.patch(nr.sample_id, y0, x0, a.patch_size)
                )

            stage4_available = stage4.available(valid)
            obs = None
            obs_path = ""
            if stage4_available:
                try:
                    obs = stage4.patch(valid, y0, x0, a.patch_size)
                    obs_path = str(stage4.file_path(valid))
                    if np.isfinite(obs).sum() == 0:
                        obs = None
                        stage4_available = False
                except Exception as exc:
                    print(f"WARNING: Stage IV failed for {valid}: {exc}")
                    obs = None
                    stage4_available = False

            sample_id = str(plan_row["sample_id"])

            def append_row(
                relation,
                lag_hours,
                direction,
                negative_index,
                a_id,
                b_id,
                arr_a,
                arr_b,
                extra=None,
            ):
                rec = {
                    "sample_id": sample_id,
                    "valid_time": valid.isoformat(),
                    "patch_y": y0,
                    "patch_x": x0,
                    "relation": relation,
                    "lag_hours": lag_hours,
                    "direction": direction,
                    "negative_index": negative_index,
                    "a_id": a_id,
                    "b_id": b_id,
                    "stage4_available": bool(stage4_available),
                    "negative_mode": a.negative_mode,
                    "climo_window_days": (
                        a.climo_window_days
                        if a.negative_mode == "climo45"
                        else np.nan
                    ),
                }
                if extra:
                    rec.update(extra)
                rec.update(
                    pair_metrics(
                        arr_a,
                        arr_b,
                        threshold=a.fss_threshold_mm,
                        windows=windows,
                    )
                )
                rows.append(rec)

            # CP: six positive relationships.
            for lag in LAGS:
                pdefs = (
                    ("earlier_init", 12 + lag),
                    ("later_init", 12 - lag),
                )
                for direction, pfhr in pdefs:
                    p_id = ids[pfhr]
                    pp = hrrr.patch(p_id, y0, x0, a.patch_size)
                    append_row(
                        "CP", lag, direction, np.nan,
                        c_id, p_id, c, pp,
                    )

            # CN: same N set for all lag comparisons.
            for ni, (nr, nn) in enumerate(zip(nrows.itertuples(index=False), neg_patches)):
                append_row(
                    "CN", 0, "negative", ni,
                    c_id, nr.sample_id, c, nn,
                    extra={
                        "negative_valid_time": nr.valid_end_dt.isoformat(),
                        "negative_time_distance_days": float(nr.time_distance_days),
                        "negative_climo_distance_days": int(nr.climo_distance_days),
                    },
                )

            # Stage IV rows.
            if obs is not None:
                stage4_samples += 1

                append_row(
                    "CO", 0, "control", np.nan,
                    c_id, obs_path, c, obs,
                )

                for lag in LAGS:
                    pdefs = (
                        ("earlier_init", 12 + lag),
                        ("later_init", 12 - lag),
                    )
                    for direction, pfhr in pdefs:
                        p_id = ids[pfhr]
                        pp = hrrr.patch(p_id, y0, x0, a.patch_size)
                        append_row(
                            "PO", lag, direction, np.nan,
                            p_id, obs_path, pp, obs,
                        )

                for ni, (nr, nn) in enumerate(zip(nrows.itertuples(index=False), neg_patches)):
                    append_row(
                        "NO", 0, "negative", ni,
                        nr.sample_id, obs_path, nn, obs,
                        extra={
                            "negative_valid_time": nr.valid_end_dt.isoformat(),
                            "negative_time_distance_days": float(nr.time_distance_days),
                            "negative_climo_distance_days": int(nr.climo_distance_days),
                        },
                    )

            accepted_samples += 1

            if accepted_samples % 100 == 0:
                print(
                    f"{accepted_samples}/{len(shard_plan)} accepted; "
                    f"rows={len(rows):,}"
                )

    finally:
        hrrr.close()

    out_csv = shard_dir / f"metrics_shard_{a.shard_index:03d}.csv.gz"
    pd.DataFrame(rows).to_csv(out_csv, index=False, compression="gzip")

    meta = {
        "shard_index": a.shard_index,
        "n_shards": a.n_shards,
        "complete_case_valid_times_total": complete_case_total,
        "sample_plan_from": a.sample_plan_from,
        "selected_valid_times_total": len(full_plan),
        "shard_valid_times": len(shard_plan),
        "accepted_samples": accepted_samples,
        "stage4_samples": stage4_samples,
        "skipped_no_wet_patch": skipped_no_wet_patch,
        "skipped_no_negatives": skipped_no_negatives,
        "rows": len(rows),
        "negative_mode": a.negative_mode,
        "negative_count": a.negative_count,
        "negative_min_days": a.negative_min_days,
        "climo_window_days": (
            a.climo_window_days if a.negative_mode == "climo45" else None
        ),
        "negative_candidate_count_min": (
            int(np.min(negative_candidate_counts))
            if negative_candidate_counts else None
        ),
        "negative_candidate_count_median": (
            float(np.median(negative_candidate_counts))
            if negative_candidate_counts else None
        ),
        "negative_candidate_count_max": (
            int(np.max(negative_candidate_counts))
            if negative_candidate_counts else None
        ),
        "fss_threshold_mm": a.fss_threshold_mm,
        "fss_windows": list(windows),
        "patch_size": a.patch_size,
        "patches_per_valid": a.patches_per_valid,
        "wet_threshold_mm": a.wet_threshold_mm,
        "min_wet_fraction": a.min_wet_fraction,
        "seed": a.seed,
    }

    out_json = shard_dir / f"metrics_shard_{a.shard_index:03d}.json"
    out_json.write_text(json.dumps(meta, indent=2) + "\n")

    print(f"Wrote {out_csv}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

