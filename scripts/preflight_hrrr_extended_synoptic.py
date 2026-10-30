#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import os
from pathlib import Path
import pandas as pd


def as_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "t", "yes", "y"])


def date_strings(start: str, end: str):
    a = datetime.strptime(start, "%Y%m%d")
    b = datetime.strptime(end, "%Y%m%d")
    while a <= b:
        yield a.strftime("%Y%m%d")
        a += timedelta(days=1)


def main():
    data = Path(os.environ["DATA_DIR"])
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(data / "forecasts/hrrr/qpf6_extended_synoptic"))
    p.add_argument("--manifest", default=None)
    p.add_argument("--start", default="20240929")
    p.add_argument("--end", default="20250920")
    p.add_argument("--min-complete-fraction", type=float, default=0.90)
    args = p.parse_args()

    root = Path(args.root)
    manifest = Path(args.manifest) if args.manifest else root / "metadata/hrrr_qpf6_manifest.csv.gz"
    if not manifest.is_file():
        raise SystemExit(f"Manifest not found: {manifest}")

    df = pd.read_csv(manifest, low_memory=False)
    req = ["date", "init_hour", "forecast_hour", "available", "store"]
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise SystemExit(f"Manifest missing columns: {miss}")
    df["date"] = df["date"].astype(str).str.replace("-", "", regex=False)
    df["init_hour"] = pd.to_numeric(df["init_hour"], errors="raise").astype(int)
    df["forecast_hour"] = pd.to_numeric(df["forecast_hour"], errors="raise").astype(int)
    df = df[as_bool(df["available"])].copy()

    expected_dates = list(date_strings(args.start, args.end))
    expected_cycles = [0, 6, 12, 18]
    expected_fhrs = [6, 12, 18, 24, 30, 36, 42, 48]
    expected_pairs = {(c, f) for c in expected_cycles for f in expected_fhrs}

    bad_dates = []
    missing_dates = []
    for date in expected_dates:
        sub = df[df["date"] == date]
        if sub.empty:
            missing_dates.append(date)
            continue
        got = set(zip(sub["init_hour"].tolist(), sub["forecast_hour"].tolist()))
        if got != expected_pairs:
            bad_dates.append((date, sorted(expected_pairs - got), sorted(got - expected_pairs)))

    print("=" * 88)
    print("EXTENDED SYNOPTIC HRRR PREFLIGHT")
    print("=" * 88)
    print(f"Manifest           : {manifest}")
    print(f"Date range         : {args.start} through {args.end}")
    print(f"Expected dates     : {len(expected_dates):,}")
    print(f"Available rows     : {len(df):,}")
    print(f"Expected rows/date : 32")
    complete_dates = len(expected_dates) - len(missing_dates) - len(bad_dates)
    complete_fraction = complete_dates / max(1, len(expected_dates))
    print(f"Complete dates     : {complete_dates:,} ({100*complete_fraction:.2f}%)")
    print(f"Missing dates      : {len(missing_dates):,}")
    print(f"Incomplete dates   : {len(bad_dates):,}")
    print("\nAvailable counts by forecast hour:")
    print(df["forecast_hour"].value_counts().sort_index().to_string())
    print("\nAvailable counts by init hour:")
    print(df["init_hour"].value_counts().sort_index().to_string())

    if missing_dates:
        print("\nFirst missing dates:", missing_dates[:20])
    if bad_dates:
        print("\nFirst incomplete dates:")
        for item in bad_dates[:10]:
            print(item)

    if complete_fraction < float(args.min_complete_fraction):
        raise SystemExit(
            f"Extended HRRR archive preflight FAILED: only {100*complete_fraction:.2f}% of dates "
            f"are complete, below required {100*args.min_complete_fraction:.2f}%"
        )

    if sorted(df["forecast_hour"].unique().tolist()) != expected_fhrs:
        raise SystemExit("Forecast-hour set mismatch")
    if sorted(df["init_hour"].unique().tolist()) != expected_cycles:
        raise SystemExit("Initialization-hour set mismatch")

    print(
        "\nPASS: requested cycles/leads are represented and the archive exceeds the "
        "minimum complete-date fraction. Any isolated source gaps will be rejected later "
        "at the valid-time balancing step."
    )


if __name__ == "__main__":
    main()
