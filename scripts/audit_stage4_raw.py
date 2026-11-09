#!/usr/bin/env python3
"""Audit the raw native Stage IV 6-h archive after HPSS extraction."""

from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime, timedelta
import csv
import gzip
import sys


SYNOPTIC_HOURS = ("00", "06", "12", "18")


def date_range(start, end):
    cur = datetime.strptime(start, "%Y%m%d")
    stop = datetime.strptime(end, "%Y%m%d")
    while cur <= stop:
        yield cur.strftime("%Y%m%d")
        cur += timedelta(days=1)


def grib_light_check(path: Path):
    size = path.stat().st_size
    if size <= 8:
        return False, f"file too small ({size} bytes)"
    with path.open("rb") as f:
        head = f.read(4)
        f.seek(-4, 2)
        tail = f.read(4)
    if head != b"GRIB":
        return False, "missing GRIB magic"
    if tail != b"7777":
        return False, "missing GRIB end marker"
    return True, ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--manifest", default=None)
    args = p.parse_args()

    root = Path(args.root)
    manifest = (
        Path(args.manifest)
        if args.manifest
        else root.parent / "metadata" / "stage4_6h_manifest.csv.gz"
    )
    manifest.parent.mkdir(parents=True, exist_ok=True)

    expected = 0
    present = 0
    bad = []
    total_bytes = 0

    with gzip.open(manifest, "wt", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["valid_time", "path", "size_bytes", "ok"],
        )
        writer.writeheader()

        for date in date_range(args.start, args.end):
            year, ym = date[:4], date[:6]
            for hh in SYNOPTIC_HOURS:
                expected += 1
                path = (
                    root / year / ym /
                    f"st4_conus.{date}{hh}.06h.grb2"
                )
                row = {
                    "valid_time": f"{date}{hh}",
                    "path": str(path),
                    "size_bytes": "",
                    "ok": False,
                }

                if path.exists():
                    present += 1
                    size = path.stat().st_size
                    total_bytes += size
                    row["size_bytes"] = size
                    ok, why = grib_light_check(path)
                    row["ok"] = ok
                    if not ok:
                        bad.append(f"{path}: {why}")
                else:
                    bad.append(f"{path}: missing")

                writer.writerow(row)

    print("=" * 72)
    print("STAGE IV RAW ARCHIVE AUDIT")
    print("=" * 72)
    print(f"Expected files : {expected:,}")
    print(f"Present files  : {present:,}")
    print(f"Missing/bad    : {len(bad):,}")
    print(f"Total size     : {total_bytes / 1024**3:.3f} GiB")
    print(f"Manifest       : {manifest}")

    if bad:
        print("\nProblems:")
        for item in bad[:200]:
            print(" ", item)
        if len(bad) > 200:
            print(f" ... {len(bad)-200} more")
        sys.exit(1)


if __name__ == "__main__":
    main()

