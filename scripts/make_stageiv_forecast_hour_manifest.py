#!/usr/bin/env python3
"""Build the balanced HRRR/Stage-IV manifest for forecast-hour verification.

The experiment is indexed by *valid time*.  For every retained valid time we
require one available HRRR forecast at each requested lead.  The default set is
F06,F12,...,F48 from 00/06/12/18Z initializations.  The verification patch is
chosen later by the archive builder, but one deterministic patch seed is shared
by all leads verifying at a given valid time.

The validation split is defined by valid_end, matching docs/splits.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def _as_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "t", "yes", "y"])


def stable_seed(base_seed: int, key: str) -> int:
    raw = hashlib.blake2b(f"{base_seed}|{key}".encode(), digest_size=8).digest()
    # numpy default_rng accepts unsigned 64-bit values.
    return int.from_bytes(raw, "little", signed=False)


def stageiv_path(root: Path, valid: pd.Timestamp) -> Path:
    name = f"st4_conus.{valid.strftime('%Y%m%d%H')}.06h.grb2"
    return root / valid.strftime("%Y") / valid.strftime("%Y%m") / name


def main() -> None:
    data = Path(os.environ["DATA_DIR"])
    p = argparse.ArgumentParser()
    p.add_argument(
        "--hrrr-manifest",
        default=str(data / "forecasts/hrrr/qpf6_native/metadata/hrrr_qpf6_manifest.csv.gz"),
    )
    p.add_argument(
        "--output-dir",
        default=str(data / "derived/evaluation/stageiv_forecast_hour_v1"),
    )
    p.add_argument(
        "--stageiv-root",
        default=str(data / "observations/stage4/raw"),
    )
    p.add_argument("--start-date", default="2024-10-01")
    p.add_argument("--end-date", default="2025-09-20")
    p.add_argument("--cycles", type=int, nargs="+", default=[0, 6, 12, 18])
    p.add_argument("--forecast-hours", type=int, nargs="+", default=[6, 12, 18, 24, 30, 36, 42, 48])
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=20261008)
    p.add_argument(
        "--allow-missing-stageiv",
        action="store_true",
        help=(
            "Permit isolated missing Stage-IV valid times by dropping those valid "
            "times, and all eight associated HRRR lead cases, from the balanced "
            "verification manifest. Without this flag, any missing Stage-IV file "
            "is a preflight failure."
        ),
    )
    args = p.parse_args()

    requested_fhrs = sorted(set(int(x) for x in args.forecast_hours))
    requested_cycles = sorted(set(int(x) for x in args.cycles))
    if not requested_fhrs:
        raise SystemExit("No forecast hours requested")

    src = Path(args.hrrr_manifest)
    if not src.is_file():
        raise SystemExit(f"HRRR manifest not found: {src}")

    df = pd.read_csv(src, low_memory=False)
    required = [
        "sample_id", "init_hour", "forecast_hour", "init_time",
        "valid_start", "valid_end", "available",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"HRRR manifest missing required columns: {missing}")

    df = df.loc[_as_bool(df["available"])].copy()
    df["sample_id"] = df["sample_id"].astype(str)
    df["init_hour"] = pd.to_numeric(df["init_hour"], errors="raise").astype(int)
    df["forecast_hour"] = pd.to_numeric(df["forecast_hour"], errors="raise").astype(int)
    for col in ["init_time", "valid_start", "valid_end"]:
        df[col] = pd.to_datetime(df[col], utc=True, errors="raise")

    start = pd.Timestamp(args.start_date, tz="UTC")
    # Inclusive end date, irrespective of synoptic hour.
    end_exclusive = pd.Timestamp(args.end_date, tz="UTC") + pd.Timedelta(days=1)

    df = df[
        (df["valid_end"] >= start)
        & (df["valid_end"] < end_exclusive)
        & df["init_hour"].isin(requested_cycles)
        & df["forecast_hour"].isin(requested_fhrs)
    ].copy()

    if df.empty:
        raise SystemExit("No available HRRR fields remain after requested filters")

    # There should be one forecast for each (valid_end, forecast_hour).  Make
    # duplicates an error rather than silently picking one.
    dup = df.duplicated(["valid_end", "forecast_hour"], keep=False)
    if dup.any():
        cols = ["sample_id", "init_time", "valid_end", "forecast_hour"]
        raise SystemExit(
            "Duplicate (valid_end, forecast_hour) rows detected, e.g.\n"
            + df.loc[dup, cols].head(20).to_string(index=False)
        )

    expected = tuple(requested_fhrs)
    keep_valid = []
    rejected = []
    for valid, g in df.groupby("valid_end", sort=True):
        got = tuple(sorted(int(x) for x in g["forecast_hour"].tolist()))
        if got == expected:
            keep_valid.append(valid)
        else:
            rejected.append((valid, got))

    if not keep_valid:
        raise SystemExit(
            "No valid times contain the complete requested forecast-hour set "
            f"{list(expected)}"
        )

    df = df[df["valid_end"].isin(keep_valid)].copy()
    df = df.sort_values(["valid_end", "forecast_hour"]).reset_index(drop=True)

    root = Path(args.stageiv_root)
    valid_rows = []
    for valid in sorted(pd.unique(df["valid_end"])):
        valid = pd.Timestamp(valid)
        sid = valid.strftime("%Y%m%d%H")
        path = stageiv_path(root, valid)
        valid_rows.append(
            {
                "valid_id": sid,
                "valid_end": valid.isoformat(),
                "valid_start": (valid - pd.Timedelta(hours=6)).isoformat(),
                "valid_hour_utc": int(valid.hour),
                "patch_seed": stable_seed(args.seed, sid),
                "patch_size": int(args.patch_size),
                "stageiv_filename": path.name,
                "stageiv_path": str(path),
                "stageiv_exists": path.is_file(),
            }
        )

    valid_df = pd.DataFrame(valid_rows)
    complete_hrrr_valid_times = int(len(valid_df))
    missing_stageiv = valid_df.loc[~valid_df["stageiv_exists"]].copy()

    # A missing Stage-IV valid time must never leave an unbalanced or broken
    # archive.  When explicitly allowed, drop the whole valid time before
    # writing the pair manifest.  Because each valid time contributes exactly
    # one case at every requested forecast hour, balance is preserved.
    if len(missing_stageiv) and args.allow_missing_stageiv:
        valid_df = valid_df.loc[valid_df["stageiv_exists"]].copy().reset_index(drop=True)

    df["valid_id"] = df["valid_end"].dt.strftime("%Y%m%d%H")
    if len(missing_stageiv) and args.allow_missing_stageiv:
        keep_ids = set(valid_df["valid_id"].astype(str))
        df = df.loc[df["valid_id"].astype(str).isin(keep_ids)].copy()

    df = df.merge(
        valid_df[["valid_id", "patch_seed", "patch_size", "stageiv_filename", "stageiv_path", "stageiv_exists"]],
        on="valid_id",
        how="left",
        validate="many_to_one",
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pair_path = out / "pair_manifest.csv.gz"
    valid_path = out / "valid_times_manifest.csv.gz"
    df.to_csv(pair_path, index=False)
    valid_df.to_csv(valid_path, index=False)

    files_needed = valid_df[["valid_id", "valid_end", "stageiv_filename", "stageiv_path", "stageiv_exists"]].copy()
    files_needed.to_csv(out / "stageiv_files_needed.csv", index=False)
    (out / "stageiv_files_needed.txt").write_text("\n".join(files_needed["stageiv_filename"]) + "\n")
    missing_stageiv.to_csv(out / "missing_stageiv_files.csv", index=False)

    count_by_fhr = df["forecast_hour"].value_counts().sort_index()
    count_by_cycle = df["init_hour"].value_counts().sort_index()

    summary = {
        "hrrr_manifest": str(src),
        "stageiv_root": str(root),
        "validation_start": args.start_date,
        "validation_end": args.end_date,
        "cycles": requested_cycles,
        "forecast_hours": requested_fhrs,
        "patch_size": int(args.patch_size),
        "seed": int(args.seed),
        "n_complete_hrrr_valid_times": complete_hrrr_valid_times,
        "n_retained_valid_times": int(len(valid_df)),
        "n_cases": int(len(df)),
        "n_rejected_incomplete_hrrr_valid_times": int(len(rejected)),
        "n_missing_stageiv_files": int(len(missing_stageiv)),
        "n_dropped_missing_stageiv_valid_times": int(len(missing_stageiv)) if args.allow_missing_stageiv else 0,
        "cases_per_forecast_hour": {str(int(k)): int(v) for k, v in count_by_fhr.items()},
        "cases_per_init_hour": {str(int(k)): int(v) for k, v in count_by_cycle.items()},
    }
    (out / "manifest_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("=" * 88)
    print("STAGE-IV FORECAST-HOUR MANIFEST PREFLIGHT")
    print("=" * 88)
    print(f"Validation valid times : {args.start_date} through {args.end_date}")
    print(f"Requested cycles       : {requested_cycles}")
    print(f"Requested forecast hrs : {requested_fhrs}")
    print(f"Complete HRRR valid times : {complete_hrrr_valid_times:,}")
    print(f"Incomplete HRRR times     : {len(rejected):,} rejected")
    print(f"Missing Stage-IV times    : {len(missing_stageiv):,}")
    if len(missing_stageiv) and args.allow_missing_stageiv:
        print(f"Retained valid times      : {len(valid_df):,} after dropping missing Stage-IV times")
    else:
        print(f"Retained valid times      : {len(valid_df):,}")
    print(f"Balanced HRRR cases       : {len(df):,}")
    print("\nCases by forecast hour:")
    print(count_by_fhr.to_string())
    print("\nCases by initialization hour:")
    print(count_by_cycle.to_string())
    print(f"\nMissing Stage-IV files : {len(missing_stageiv):,}")
    if len(missing_stageiv):
        print(missing_stageiv[["valid_end", "stageiv_path"]].head(20).to_string(index=False))
        if args.allow_missing_stageiv:
            print("These valid times were dropped wholesale before writing the balanced pair manifest.")
    print(f"\nPair manifest          : {pair_path}")
    print(f"Valid-time manifest    : {valid_path}")

    if len(missing_stageiv) and not args.allow_missing_stageiv:
        raise SystemExit(
            "Preflight failed because required Stage-IV files are missing. "
            "Stage/retrieve them first, or rerun with --allow-missing-stageiv "
            "to drop those complete valid times while preserving lead balance."
        )

    # Strong balance assertion.
    if set(count_by_fhr.index.astype(int)) != set(requested_fhrs) or count_by_fhr.nunique() != 1:
        raise SystemExit("Internal error: output manifest is not balanced by forecast hour")


if __name__ == "__main__":
    main()
