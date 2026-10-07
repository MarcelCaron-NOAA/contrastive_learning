#!/usr/bin/env python3
"""
Evaluate +/-1-h same-valid-time HRRR positive-pair similarity as a function
of CONTROL forecast lead (F07 ... F17).

For a control forecast initialized at time t with lead L:
    positive_earlier_init: init t-1 h, lead L+1
    positive_later_init:   init t+1 h, lead L-1

Both positives therefore verify over the same 6-h valid interval as C/A.

The experiment:
  * requires BOTH +/-1-h positives to exist for a candidate control;
  * samples controls approximately equally from DJF/MAM/JJA/SON;
  * chooses one random 256x256 patch per accepted control;
  * requires a minimum wet fraction in the control patch;
  * computes FSS, continuous-field correlation, ETS, and supporting metrics.

This is a development diagnostic, not the final training sampler.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from scipy.ndimage import uniform_filter


SEASONS = ("DJF", "MAM", "JJA", "SON")


def season_from_month(month: int) -> str:
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"


def parse_time_series(series: pd.Series) -> pd.Series:
    """Parse strings or Unix seconds/nanoseconds into UTC timestamps."""
    if pd.api.types.is_numeric_dtype(series):
        vals = pd.to_numeric(series, errors="coerce")
        finite = vals[np.isfinite(vals)]
        if len(finite) == 0:
            return pd.to_datetime(series, errors="coerce", utc=True)

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

    return pd.to_datetime(series, errors="coerce", utc=True)


def find_col(df: pd.DataFrame, names, required=True):
    for name in names:
        if name in df.columns:
            return name
    if required:
        raise RuntimeError(
            f"Could not find any of {names}. Manifest columns:\n{list(df.columns)}"
        )
    return None


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "t", "yes"])


def load_manifest(path: Path):
    df = pd.read_csv(path, low_memory=False)

    sample_col = find_col(df, ["sample_id", "field_id", "id"])
    store_col = find_col(df, ["store", "store_path", "zarr_path", "path"])
    init_col = find_col(df, ["init_hour", "cycle_hour", "cycle"])
    fhr_col = find_col(df, ["forecast_hour", "lead_hour", "fhr", "lead"])
    valid_col = find_col(
        df,
        ["valid_end", "valid_end_time", "valid_time", "valid_end_unix"],
    )
    avail_col = find_col(df, ["available"], required=False)

    if avail_col is not None:
        df = df.loc[as_bool(df[avail_col])].copy()

    df["sample_id_std"] = df[sample_col].astype(str)
    df["store_std"] = df[store_col].astype(str)
    df["init_hour_std"] = pd.to_numeric(df[init_col], errors="coerce").astype("Int64")
    df["forecast_hour_std"] = pd.to_numeric(df[fhr_col], errors="coerce").astype("Int64")
    df["valid_end_dt"] = parse_time_series(df[valid_col])

    df = df.dropna(
        subset=["sample_id_std", "store_std", "init_hour_std",
                "forecast_hour_std", "valid_end_dt"]
    ).copy()

    df["init_hour_std"] = df["init_hour_std"].astype(int)
    df["forecast_hour_std"] = df["forecast_hour_std"].astype(int)
    df["valid_key"] = df["valid_end_dt"].astype("int64")

    # same valid time + forecast lead uniquely determines the initialization
    key_to_row = {
        (int(row.valid_key), int(row.forecast_hour_std)): idx
        for idx, row in df.iterrows()
    }

    return df, key_to_row


class ZarrCache:
    def __init__(self, max_open=24):
        self.max_open = max_open
        self.cache = OrderedDict()

    def open(self, path: str):
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

    def close(self):
        for store, _ in self.cache.values():
            store.close()
        self.cache.clear()


def read_patch(row, cache: ZarrCache, y0, x0, size):
    root = cache.open(row["store_std"])
    fhr = int(row["forecast_hour_std"])
    init_hour = int(row["init_hour_std"])
    lead_index = fhr - 6

    return np.asarray(
        root["qpf6"][
            init_hour,
            lead_index,
            y0:y0 + size,
            x0:x0 + size,
        ],
        dtype=np.float32,
    )


def fractions_skill_score(a, b, threshold, window):
    aa = np.isfinite(a) & (a >= threshold)
    bb = np.isfinite(b) & (b >= threshold)

    fa = uniform_filter(aa.astype(np.float32), size=window, mode="constant")
    fb = uniform_filter(bb.astype(np.float32), size=window, mode="constant")

    den = np.mean(fa * fa + fb * fb)
    if den <= 0:
        return np.nan

    mse = np.mean((fa - fb) ** 2)
    return float(1.0 - mse / den)


def spatial_corr(a, b):
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 3:
        return np.nan

    av = a[valid].astype(np.float64)
    bv = b[valid].astype(np.float64)

    if np.std(av) == 0 or np.std(bv) == 0:
        return np.nan

    return float(np.corrcoef(av, bv)[0, 1])


def equitable_threat_score(a, b, threshold):
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() == 0:
        return np.nan

    aa = a[valid] >= threshold
    bb = b[valid] >= threshold

    hits = np.sum(aa & bb)
    false_alarms = np.sum(~aa & bb)
    misses = np.sum(aa & ~bb)
    n = len(aa)

    yes_a = hits + misses
    yes_b = hits + false_alarms
    hits_random = (yes_a * yes_b) / n

    den = hits + false_alarms + misses - hits_random
    if den == 0:
        return np.nan

    return float((hits - hits_random) / den)


def pair_metrics(a, b, fss_threshold, fss_window):
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() == 0:
        return {
            "fss": np.nan,
            "corr": np.nan,
            "ets": np.nan,
            "rmse_mm": np.nan,
            "bcrmse_mm": np.nan,
            "mean_abs_diff_mm": np.nan,
            "mean_bias_p_minus_a_mm": np.nan,
            "mean_a_mm": np.nan,
            "mean_p_mm": np.nan,
            "max_a_mm": np.nan,
            "max_p_mm": np.nan,
        }

    av = a[valid].astype(np.float64)
    bv = b[valid].astype(np.float64)
    diff = bv - av

    rmse = np.sqrt(np.mean(diff**2))
    bcrmse = np.sqrt(np.mean((diff - np.mean(diff))**2))

    return {
        "fss": fractions_skill_score(a, b, fss_threshold, fss_window),
        "corr": spatial_corr(a, b),
        "ets": equitable_threat_score(a, b, fss_threshold),
        "rmse_mm": float(rmse),
        "bcrmse_mm": float(bcrmse),
        "mean_abs_diff_mm": float(np.mean(np.abs(diff))),
        "mean_bias_p_minus_a_mm": float(np.mean(diff)),
        "mean_a_mm": float(np.mean(av)),
        "mean_p_mm": float(np.mean(bv)),
        "max_a_mm": float(np.max(av)),
        "max_p_mm": float(np.max(bv)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--lead-hour", type=int, required=True, choices=range(7, 18))
    p.add_argument("--start", default="2020-12-02")
    p.add_argument("--end", default="2025-09-23")
    p.add_argument("--samples-per-season", type=int, default=250)
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--patch-tries", type=int, default=30)
    p.add_argument("--wet-threshold-mm", type=float, default=1.0)
    p.add_argument("--min-wet-fraction", type=float, default=0.01)
    p.add_argument("--fss-threshold-mm", type=float, default=5.0)
    p.add_argument("--fss-window", type=int, default=15)
    p.add_argument("--seed", type=int, default=20260929)
    args = p.parse_args()

    if args.fss_window % 2 == 0:
        raise SystemExit("--fss-window must be odd.")

    manifest_path = Path(args.manifest)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, key_to_row = load_manifest(manifest_path)

    start = pd.Timestamp(args.start, tz="UTC")
    # inclusive calendar-day end
    end = pd.Timestamp(args.end, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)

    L = args.lead_hour

    control = df.loc[
        (df["forecast_hour_std"] == L)
        & (df["valid_end_dt"] >= start)
        & (df["valid_end_dt"] <= end)
    ].copy()

    # Require both +/-1-h positive directions.
    eligible_records = []
    for idx, row in control.iterrows():
        vk = int(row["valid_key"])
        earlier_idx = key_to_row.get((vk, L + 1))  # init t-1
        later_idx = key_to_row.get((vk, L - 1))    # init t+1
        if earlier_idx is None or later_idx is None:
            continue

        eligible_records.append(
            {
                "control_idx": idx,
                "p_earlier_idx": earlier_idx,
                "p_later_idx": later_idx,
                "valid_time": row["valid_end_dt"],
                "season": season_from_month(row["valid_end_dt"].month),
            }
        )

    elig = pd.DataFrame(eligible_records)
    print(f"Control lead F{L:02d}")
    print(f"Eligible controls with both +/-1-h positives: {len(elig):,}")
    print(elig["season"].value_counts().reindex(SEASONS).fillna(0).astype(int).to_string())

    rng = np.random.default_rng(args.seed + L * 1000)
    cache = ZarrCache()
    rows = []
    accepted_by_season = {s: 0 for s in SEASONS}
    skipped_dry = 0

    try:
        for season in SEASONS:
            s = elig.loc[elig["season"] == season].copy()
            if s.empty:
                continue

            order = rng.permutation(len(s))
            s = s.iloc[order]

            for _, rec in s.iterrows():
                if accepted_by_season[season] >= args.samples_per_season:
                    break

                c_row = df.loc[int(rec["control_idx"])]
                p_earlier_row = df.loc[int(rec["p_earlier_idx"])]
                p_later_row = df.loc[int(rec["p_later_idx"])]

                accepted_patch = False
                c_patch = None
                y0 = x0 = None

                for _ in range(args.patch_tries):
                    y0 = int(rng.integers(0, 1059 - args.patch_size + 1))
                    x0 = int(rng.integers(0, 1799 - args.patch_size + 1))
                    c_patch = read_patch(c_row, cache, y0, x0, args.patch_size)

                    wet_fraction = float(
                        np.mean(np.isfinite(c_patch) & (c_patch >= args.wet_threshold_mm))
                    )
                    if wet_fraction >= args.min_wet_fraction:
                        accepted_patch = True
                        break

                if not accepted_patch:
                    skipped_dry += 1
                    continue

                wet_fraction = float(
                    np.mean(np.isfinite(c_patch) & (c_patch >= args.wet_threshold_mm))
                )

                for direction, p_row in (
                    ("earlier_init", p_earlier_row),
                    ("later_init", p_later_row),
                ):
                    p_patch = read_patch(p_row, cache, y0, x0, args.patch_size)
                    metrics = pair_metrics(
                        c_patch,
                        p_patch,
                        args.fss_threshold_mm,
                        args.fss_window,
                    )

                    rows.append(
                        {
                            "control_lead_hour": L,
                            "direction": direction,
                            "valid_time": pd.Timestamp(rec["valid_time"]).isoformat(),
                            "season": season,
                            "patch_y": y0,
                            "patch_x": x0,
                            "control_id": c_row["sample_id_std"],
                            "positive_id": p_row["sample_id_std"],
                            "control_init_hour": int(c_row["init_hour_std"]),
                            "positive_init_hour": int(p_row["init_hour_std"]),
                            "control_fhr": int(c_row["forecast_hour_std"]),
                            "positive_fhr": int(p_row["forecast_hour_std"]),
                            "wet_fraction_control": wet_fraction,
                            **metrics,
                        }
                    )

                accepted_by_season[season] += 1

    finally:
        cache.close()

    result = pd.DataFrame(rows)

    out_csv = out_dir / f"lead_f{L:02d}.csv.gz"
    result.to_csv(out_csv, index=False, compression="gzip")

    meta = {
        "control_lead_hour": L,
        "start": args.start,
        "end": args.end,
        "eligible_controls": int(len(elig)),
        "samples_per_season_requested": args.samples_per_season,
        "accepted_controls_by_season": accepted_by_season,
        "accepted_controls_total": int(sum(accepted_by_season.values())),
        "pair_rows": int(len(result)),
        "skipped_no_wet_patch": int(skipped_dry),
        "patch_size": args.patch_size,
        "patch_tries": args.patch_tries,
        "wet_threshold_mm": args.wet_threshold_mm,
        "min_wet_fraction": args.min_wet_fraction,
        "fss_threshold_mm": args.fss_threshold_mm,
        "fss_window": args.fss_window,
        "seed": args.seed,
    }

    out_json = out_dir / f"lead_f{L:02d}.json"
    out_json.write_text(json.dumps(meta, indent=2) + "\n")

    print("=" * 72)
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_json}")
    print(f"Accepted controls: {sum(accepted_by_season.values()):,}")
    print(f"Pair rows        : {len(result):,}")
    print(f"Skipped dry      : {skipped_dry:,}")
    print("=" * 72)


if __name__ == "__main__":
    main()

