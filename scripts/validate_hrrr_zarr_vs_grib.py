#!/usr/bin/env python3

import os
import tempfile
from pathlib import Path

import numpy as np
import s3fs
import xarray as xr
import zarr


RAW_BUCKET = "noaa-hrrr-bdp-pds"
ZARR_BUCKET = "hrrrzarr"

# Three deliberately different source-validation cases.
CASES = [
    ("20210105", 1,  "winter 2021 / float16"),
    ("20210901", 12, "warm-season 2021 / float16"),
    ("20250715", 12, "summer 2025 / float32"),
]


def read_idx(fs, grib_path):
    """Read a wgrib2-style .idx inventory."""
    idx_path = grib_path + ".idx"

    with fs.open(idx_path, "rt") as f:
        lines = [line.strip() for line in f if line.strip()]

    entries = []

    for line in lines:
        parts = line.split(":")
        if len(parts) < 6:
            continue

        entries.append(
            {
                "record": int(parts[0]),
                "start": int(parts[1]),
                "parts": parts,
                "line": line,
            }
        )

    return entries


def find_one_hour_apcp(entries, fhr):
    """
    Find the deterministic surface APCP record representing the
    single hour ending at forecast hour fhr.

    Example for F06:
        APCP:surface:5-6 hour acc fcst
    """
    wanted = f"{fhr - 1}-{fhr} hour acc fcst"

    matches = []

    for i, entry in enumerate(entries):
        p = entry["parts"]

        if (
            len(p) >= 6
            and p[3] == "APCP"
            and p[4] == "surface"
            and p[5] == wanted
        ):
            matches.append((i, entry))

    if len(matches) != 1:
        print(f"\nCould not uniquely identify APCP for F{fhr:02d}.")
        print(f"Wanted: APCP:surface:{wanted}")
        print("\nAPCP inventory lines found:")

        for entry in entries:
            if ":APCP:" in entry["line"]:
                print(" ", entry["line"])

        raise RuntimeError(
            f"Expected exactly 1 matching APCP record; found {len(matches)}"
        )

    return matches[0]


def fetch_grib_message(fs, grib_path, fhr, destination):
    """
    Fetch only the single 1-h APCP GRIB2 message using an S3 byte range.
    """
    entries = read_idx(fs, grib_path)

    i, entry = find_one_hour_apcp(entries, fhr)

    start = entry["start"]

    if i + 1 < len(entries):
        end = entries[i + 1]["start"] - 1
    else:
        info = fs.info(grib_path)
        size = info.get("Size", info.get("size"))
        end = size - 1

    nbytes = end - start + 1

    print(
        f"    F{fhr:02d}: bytes {start}-{end} "
        f"({nbytes / 1024**2:.3f} MiB)"
    )
    print(f"         {entry['line']}")

    with fs.open(grib_path, "rb") as src:
        src.seek(start)
        data = src.read(nbytes)

    if len(data) != nbytes:
        raise RuntimeError(
            f"Expected {nbytes} bytes but received {len(data)}"
        )

    with open(destination, "wb") as f:
        f.write(data)


def read_single_grib(path):
    """
    Decode a single-message GRIB2 file with cfgrib.
    """
    ds = xr.open_dataset(
        path,
        engine="cfgrib",
        backend_kwargs={"indexpath": ""},
    )

    vars_ = list(ds.data_vars)

    if len(vars_) != 1:
        raise RuntimeError(
            f"Expected one GRIB data variable, found {vars_}"
        )

    var = vars_[0]
    da = ds[var]

    values = np.asarray(da.values).squeeze().astype(np.float32)

    units = da.attrs.get("units", "unknown")
    long_name = da.attrs.get("long_name", "unknown")

    ds.close()

    return values, var, units, long_name


def basic_stats(raw, zz):
    mask = np.isfinite(raw) & np.isfinite(zz)

    r = raw[mask]
    z = zz[mask]
    d = z - r
    ad = np.abs(d)

    return {
        "n": r.size,
        "raw_mean": np.mean(r),
        "raw_max": np.max(r),
        "mae": np.mean(ad),
        "rmse": np.sqrt(np.mean(d * d)),
        "max_abs": np.max(ad),
        "bias": np.mean(d),
        "exact_pct": 100.0 * np.mean(d == 0),
        "gt_001_pct": 100.0 * np.mean(ad > 0.01),
        "gt_005_pct": 100.0 * np.mean(ad > 0.05),
        "gt_010_pct": 100.0 * np.mean(ad > 0.10),
    }


def print_stats(label, stats):
    print(label)
    print(f"    N                  : {stats['n']:,}")
    print(f"    raw mean           : {stats['raw_mean']:.6f} mm")
    print(f"    raw max            : {stats['raw_max']:.6f} mm")
    print(f"    bias (Zarr - GRIB) : {stats['bias']:.8f} mm")
    print(f"    MAE                : {stats['mae']:.8f} mm")
    print(f"    RMSE               : {stats['rmse']:.8f} mm")
    print(f"    max abs diff       : {stats['max_abs']:.8f} mm")
    print(f"    exactly equal      : {stats['exact_pct']:.4f}%")
    print(f"    |diff| > 0.01 mm   : {stats['gt_001_pct']:.6f}%")
    print(f"    |diff| > 0.05 mm   : {stats['gt_005_pct']:.6f}%")
    print(f"    |diff| > 0.10 mm   : {stats['gt_010_pct']:.6f}%")


def precip_bin_stats(raw_cube, zarr_cube):
    """
    Quantify error conditional on the magnitude of raw hourly precipitation.
    """
    bins = [
        (0.0, 0.01, "<0.01"),
        (0.01, 0.1, "0.01-0.1"),
        (0.1, 1.0, "0.1-1"),
        (1.0, 5.0, "1-5"),
        (5.0, 10.0, "5-10"),
        (10.0, np.inf, ">=10"),
    ]

    print("\nERROR BY RAW 1-H PRECIPITATION MAGNITUDE")
    print(
        f"{'bin (mm)':>12} "
        f"{'N':>12} "
        f"{'MAE (mm)':>14} "
        f"{'max abs (mm)':>16}"
    )

    finite = np.isfinite(raw_cube) & np.isfinite(zarr_cube)

    for low, high, label in bins:
        mask = finite & (raw_cube >= low) & (raw_cube < high)

        if not np.any(mask):
            continue

        diff = np.abs(zarr_cube[mask] - raw_cube[mask])

        print(
            f"{label:>12} "
            f"{mask.sum():12,d} "
            f"{np.mean(diff):14.8f} "
            f"{np.max(diff):16.8f}"
        )


def validate_case(date, cycle, description):
    print("\n")
    print("=" * 88)
    print(f"{date} {cycle:02d}Z -- {description}")
    print("=" * 88)

    raw_fs = s3fs.S3FileSystem(anon=True)
    zarr_fs = s3fs.S3FileSystem(anon=True)

    # --------------------------------------------------------
    # Load the complete 18-hour Zarr APCP cube once.
    # --------------------------------------------------------
    zbase = (
        f"{ZARR_BUCKET}/sfc/{date}/"
        f"{date}_{cycle:02d}z_fcst.zarr"
    )

    store = s3fs.S3Map(root=zbase, s3=zarr_fs, check=False)
    root = zarr.open_consolidated(store, mode="r")

    za = (
        root["surface"]
        ["APCP_1hr_acc_fcst"]
        ["surface"]
        ["APCP_1hr_acc_fcst"]
    )

    print("\nZarr:")
    print("  path   :", zbase)
    print("  shape  :", za.shape)
    print("  chunks :", za.chunks)
    print("  dtype  :", za.dtype)
    print("  using  : F01-F18")


    # Some HRRR cycles (00/06/12/18Z) are extended forecasts
    # and contain more than 18 forecast hours.  This project only
    # requires F01-F18.
    if za.shape[0] < 18:
        raise RuntimeError(
            f"Zarr contains only {za.shape[0]} forecast hours"
        )

    if za.shape[1:] != (1059, 1799):
        raise RuntimeError(
            f"Unexpected Zarr spatial shape {za.shape[1:]}"
        )

    zarr_cube = za[:18].astype(np.float32)

    # --------------------------------------------------------
    # Retrieve and decode the corresponding 18 raw GRIB fields.
    # --------------------------------------------------------
    raw_fields = []

    with tempfile.TemporaryDirectory() as tmp:
        print("\nFetching raw NOAA APCP messages:")

        for fhr in range(1, 19):
            grib_path = (
                f"{RAW_BUCKET}/hrrr.{date}/conus/"
                f"hrrr.t{cycle:02d}z.wrfsfcf{fhr:02d}.grib2"
            )

            subset = os.path.join(
                tmp,
                f"hrrr.{date}.t{cycle:02d}z."
                f"f{fhr:02d}.apcp1h.grib2"
            )

            fetch_grib_message(
                raw_fs,
                grib_path,
                fhr,
                subset,
            )

            values, var, units, long_name = read_single_grib(subset)

            if fhr == 1:
                print("\nDecoded raw GRIB:")
                print("  variable :", var)
                print("  units    :", units)
                print("  long_name:", long_name)
                print("  shape    :", values.shape)

            if values.shape != (1059, 1799):
                raise RuntimeError(
                    f"Unexpected raw GRIB shape at F{fhr:02d}: "
                    f"{values.shape}"
                )

            raw_fields.append(values)

    raw_cube = np.stack(raw_fields, axis=0)

    # --------------------------------------------------------
    # All hourly fields together.
    # --------------------------------------------------------
    print("\n" + "-" * 88)
    print("ALL F01-F18 1-H APCP VALUES")
    print("-" * 88)

    print_stats(
        "Whole-domain aggregate:",
        basic_stats(raw_cube, zarr_cube),
    )

    precip_bin_stats(raw_cube, zarr_cube)

    # --------------------------------------------------------
    # Per-hour summary.
    # --------------------------------------------------------
    print("\nPER-FORECAST-HOUR 1-H APCP")
    print(
        f"{'FHR':>4} "
        f"{'raw mean':>11} "
        f"{'raw max':>11} "
        f"{'MAE':>12} "
        f"{'RMSE':>12} "
        f"{'max abs':>12}"
    )

    for i in range(18):
        s = basic_stats(raw_cube[i], zarr_cube[i])

        print(
            f"{i+1:4d} "
            f"{s['raw_mean']:11.5f} "
            f"{s['raw_max']:11.5f} "
            f"{s['mae']:12.7f} "
            f"{s['rmse']:12.7f} "
            f"{s['max_abs']:12.7f}"
        )

    # --------------------------------------------------------
    # Directly validate the final quantity we care about:
    # rolling 6-h precipitation.
    # --------------------------------------------------------
    print("\n" + "-" * 88)
    print("DERIVED 6-H QPF")
    print("-" * 88)

    for end_fhr in [6, 12, 18]:
        start = end_fhr - 6
        stop = end_fhr

        raw6 = np.sum(
            raw_cube[start:stop],
            axis=0,
            dtype=np.float32,
        )

        zarr6 = np.sum(
            zarr_cube[start:stop],
            axis=0,
            dtype=np.float32,
        )

        print()
        print_stats(
            f"F{end_fhr:02d} 6-h accumulation:",
            basic_stats(raw6, zarr6),
        )


def main():
    for date, cycle, description in CASES:
        validate_case(date, cycle, description)

    print("\n")
    print("=" * 88)
    print("SOURCE VALIDATION COMPLETE")
    print("=" * 88)


if __name__ == "__main__":
    main()
