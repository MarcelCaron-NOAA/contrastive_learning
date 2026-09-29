#!/usr/bin/env python3
"""
Audit the canonical HRRR daily QPF6 archive without rereading the full 2.2-TB dataset.

Checks every daily .zarr.zip store for:
  * readable ZIP central directory
  * consolidated Zarr metadata
  * expected qpf6 shape/chunks/dtype
  * expected init/lead metadata
  * valid-time consistency
  * finite, consistent native-grid x/y coordinates
  * availability flags and source precision metadata
  * one deterministic data sample per store (default)

A full ZIP CRC scan is optional because it would require reading essentially
the entire archive again.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path
import hashlib
import re
import sys
import zipfile

import numpy as np
import zarr

EXPECTED_QPF_SHAPE = (24, 13, 1059, 1799)
EXPECTED_QPF_CHUNKS = (1, 1, 256, 256)
EXPECTED_FHRS = np.arange(6, 19, dtype=np.int16)
EXPECTED_INIT_HOURS = np.arange(24, dtype=np.int8)
DATE_RE = re.compile(r"hrrr_qpf6_(\d{8})\.zarr\.zip$")


def sha256_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()


def expected_times(date: str):
    day = datetime.strptime(date, "%Y%m%d").replace(tzinfo=timezone.utc)
    init = np.array(
        [int((day + timedelta(hours=h)).timestamp()) for h in range(24)],
        dtype=np.int64,
    )
    start = np.empty((24, 13), dtype=np.int64)
    end = np.empty((24, 13), dtype=np.int64)
    for i, h in enumerate(range(24)):
        init_dt = day + timedelta(hours=h)
        for j, fhr in enumerate(range(6, 19)):
            end_dt = init_dt + timedelta(hours=fhr)
            start_dt = end_dt - timedelta(hours=6)
            start[i, j] = int(start_dt.timestamp())
            end[i, j] = int(end_dt.timestamp())
    return init, start, end


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        required=True,
        help="Canonical HRRR qpf6_native root containing daily/YYYY/YYYYMM",
    )
    p.add_argument(
        "--date-list",
        default=None,
        help="Optional expected YYYYMMDD date list; missing/extra dates are reported",
    )
    p.add_argument(
        "--sample-size",
        type=int,
        default=64,
        help="Square data sample per daily store; 0 disables data reads (default 64)",
    )
    p.add_argument(
        "--full-crc",
        action="store_true",
        help="Run ZipFile.testzip() on every store. WARNING: reads the full archive.",
    )
    p.add_argument(
        "--report",
        default=None,
        help="Optional text report path",
    )
    return p.parse_args()


class Reporter:
    def __init__(self, report=None):
        self.fh = open(report, "w") if report else None

    def say(self, msg=""):
        print(msg)
        if self.fh:
            print(msg, file=self.fh)
            self.fh.flush()

    def close(self):
        if self.fh:
            self.fh.close()


def main():
    args = parse_args()
    root = Path(args.root)
    rep = Reporter(args.report)

    stores = sorted(root.glob("daily/*/*/hrrr_qpf6_????????.zarr.zip"))
    rep.say(f"Stores discovered : {len(stores):,}")
    if not stores:
        raise SystemExit("No stores found")

    found_dates = []
    errors = []
    warnings = []
    false_fields_total = 0
    total_fields = 0
    total_bytes = 0
    sizes = []
    reference_x_hash = None
    reference_y_hash = None
    reference_x = None
    reference_y = None
    source_bits_seen = set()
    sampled_values = 0

    for n, path in enumerate(stores, 1):
        m = DATE_RE.search(path.name)
        if not m:
            errors.append(f"{path}: filename does not match expected convention")
            continue
        date = m.group(1)
        found_dates.append(date)

        try:
            size = path.stat().st_size
            total_bytes += size
            sizes.append(size)

            # Read the ZIP central directory.  This does not scan every data byte.
            if not zipfile.is_zipfile(path):
                raise RuntimeError("not recognized as a ZIP file")
            with zipfile.ZipFile(path, "r") as zf:
                names = set(zf.namelist())
                if ".zmetadata" not in names:
                    raise RuntimeError("ZIP is missing consolidated .zmetadata")
                if args.full_crc:
                    bad = zf.testzip()
                    if bad is not None:
                        raise RuntimeError(f"ZIP CRC failure in member {bad}")

            with zarr.ZipStore(str(path), mode="r") as zs:
                z = zarr.open_consolidated(zs, mode="r")

                attr_date = str(z.attrs.get("date_utc", ""))
                if attr_date != date:
                    raise RuntimeError(
                        f"date_utc attribute {attr_date!r} != filename date {date}"
                    )

                required = [
                    "qpf6", "init_hour", "forecast_hour",
                    "init_time_unix", "valid_start_unix", "valid_end_unix",
                    "projection_x_coordinate", "projection_y_coordinate",
                    "available", "source_dtype_bits",
                ]
                missing = [name for name in required if name not in z]
                if missing:
                    raise RuntimeError(f"missing arrays: {missing}")

                q = z["qpf6"]
                if tuple(q.shape) != EXPECTED_QPF_SHAPE:
                    raise RuntimeError(f"qpf6 shape {q.shape} != {EXPECTED_QPF_SHAPE}")
                if tuple(q.chunks) != EXPECTED_QPF_CHUNKS:
                    raise RuntimeError(
                        f"qpf6 chunks {q.chunks} != {EXPECTED_QPF_CHUNKS}"
                    )
                if np.dtype(q.dtype) != np.dtype("float32"):
                    raise RuntimeError(f"qpf6 dtype is {q.dtype}, expected float32")

                init_hours = np.asarray(z["init_hour"][:])
                fhrs = np.asarray(z["forecast_hour"][:])
                if not np.array_equal(init_hours, EXPECTED_INIT_HOURS):
                    raise RuntimeError(f"unexpected init_hour array: {init_hours}")
                if not np.array_equal(fhrs, EXPECTED_FHRS):
                    raise RuntimeError(f"unexpected forecast_hour array: {fhrs}")

                exp_init, exp_start, exp_end = expected_times(date)
                if not np.array_equal(np.asarray(z["init_time_unix"][:]), exp_init):
                    raise RuntimeError("init_time_unix does not match date/init_hour")
                if not np.array_equal(np.asarray(z["valid_start_unix"][:]), exp_start):
                    raise RuntimeError("valid_start_unix is inconsistent")
                if not np.array_equal(np.asarray(z["valid_end_unix"][:]), exp_end):
                    raise RuntimeError("valid_end_unix is inconsistent")

                x = np.asarray(z["projection_x_coordinate"][:], dtype=np.float64)
                y = np.asarray(z["projection_y_coordinate"][:], dtype=np.float64)
                if x.shape != (1799,) or y.shape != (1059,):
                    raise RuntimeError(f"coordinate shapes x={x.shape}, y={y.shape}")
                if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
                    raise RuntimeError("non-finite native-grid coordinate")
                xh, yh = sha256_array(x), sha256_array(y)
                if reference_x_hash is None:
                    reference_x_hash, reference_y_hash = xh, yh
                    reference_x, reference_y = x.copy(), y.copy()
                elif xh != reference_x_hash or yh != reference_y_hash:
                    # Hash mismatch gets an exact check before being called an error.
                    if not np.array_equal(x, reference_x) or not np.array_equal(y, reference_y):
                        raise RuntimeError("native HRRR grid differs from reference store")

                available = np.asarray(z["available"][:], dtype=bool)
                if available.shape != (24, 13):
                    raise RuntimeError(f"available shape is {available.shape}")
                false_count = int((~available).sum())
                false_fields_total += false_count
                total_fields += available.size
                if false_count:
                    warnings.append(f"{date}: {false_count} unavailable fields")

                bits = np.asarray(z["source_dtype_bits"][:])
                bad_bits = set(int(v) for v in np.unique(bits) if int(v) not in (0, 16, 32))
                if bad_bits:
                    raise RuntimeError(f"unexpected source dtype bits: {sorted(bad_bits)}")
                source_bits_seen.update(int(v) for v in np.unique(bits) if int(v) != 0)
                if np.any((bits == 0) & np.any(available, axis=1)):
                    raise RuntimeError("available cycle has source_dtype_bits == 0")

                if args.sample_size > 0:
                    s = args.sample_size
                    if s > 1059 or s > 1799:
                        raise RuntimeError("--sample-size exceeds domain")
                    candidates = np.argwhere(available)
                    if candidates.size:
                        # Deterministic location/field based on day number.
                        i, j = candidates[len(candidates) // 2]
                        seed = int(date)
                        y0 = seed % (1059 - s + 1)
                        x0 = (seed * 17) % (1799 - s + 1)
                        patch = np.asarray(
                            q[int(i), int(j), y0:y0+s, x0:x0+s],
                            dtype=np.float32,
                        )
                        sampled_values += patch.size
                        if not np.all(np.isfinite(patch)):
                            raise RuntimeError("sampled qpf6 patch contains non-finite data")
                        if float(np.min(patch)) < -1e-5:
                            raise RuntimeError(
                                f"sampled qpf6 patch contains negative value {float(np.min(patch))}"
                            )

        except Exception as exc:
            errors.append(f"{date} {path}: {type(exc).__name__}: {exc}")

        if n % 100 == 0 or n == len(stores):
            rep.say(f"Checked {n:4d}/{len(stores)} stores")

    if args.date_list:
        expected_dates = [
            line.strip()
            for line in Path(args.date_list).read_text().splitlines()
            if line.strip()
        ]
        expected_set = set(expected_dates)
        found_set = set(found_dates)
        missing = sorted(expected_set - found_set)
        extra = sorted(found_set - expected_set)
    else:
        missing, extra = [], []

    rep.say()
    rep.say("=" * 72)
    rep.say("HRRR ARCHIVE AUDIT SUMMARY")
    rep.say("=" * 72)
    rep.say(f"Stores found              : {len(stores):,}")
    rep.say(f"Unique dates              : {len(set(found_dates)):,}")
    rep.say(f"Total logical fields      : {total_fields:,}")
    rep.say(f"Unavailable fields        : {false_fields_total:,}")
    rep.say(f"Source dtype bits seen    : {sorted(source_bits_seen)}")
    rep.say(f"Sampled qpf6 values       : {sampled_values:,}")
    rep.say(f"Archive size              : {total_bytes / 1024**4:.3f} TiB")
    if sizes:
        rep.say(f"Smallest daily store      : {min(sizes) / 1024**3:.3f} GiB")
        rep.say(f"Largest daily store       : {max(sizes) / 1024**3:.3f} GiB")
        rep.say(f"Mean daily store          : {(sum(sizes)/len(sizes)) / 1024**3:.3f} GiB")
    rep.say(f"Missing expected dates    : {len(missing):,}")
    rep.say(f"Extra dates               : {len(extra):,}")
    rep.say(f"Warnings                  : {len(warnings):,}")
    rep.say(f"Errors                    : {len(errors):,}")

    if missing:
        rep.say("\nMissing dates:")
        for d in missing:
            rep.say(f"  {d}")
    if extra:
        rep.say("\nExtra dates:")
        for d in extra:
            rep.say(f"  {d}")
    if warnings:
        rep.say("\nWarnings:")
        for w in warnings[:200]:
            rep.say(f"  {w}")
        if len(warnings) > 200:
            rep.say(f"  ... {len(warnings)-200} more warnings omitted")
    if errors:
        rep.say("\nErrors:")
        for e in errors:
            rep.say(f"  {e}")

    rep.close()

    if errors or missing:
        sys.exit(1)


if __name__ == "__main__":
    main()

