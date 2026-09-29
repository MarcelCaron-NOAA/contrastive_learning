#!/usr/bin/env python3
"""Prepare an experiment-specific Stage-IV -> HRRR nearest-neighbor mapping.

HRRR latitude/longitude are read from the MesoWest HRRR Zarr archive's
dedicated static grid store:

    s3://hrrrzarr/grid/HRRR_chunk_index.zarr

Stage-IV latitude/longitude are decoded from one local 6-h GRIB2 analysis.

Each HRRR grid point is mapped to the nearest Stage-IV grid point on the
sphere.  HRRR points farther than --max-distance-km from any Stage-IV point
are masked so Stage IV is not extrapolated far beyond its native domain.

This mapping is intended for the exploratory time-lag selection experiment,
not as the final verification-grade remapping choice.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import s3fs
import xarray as xr
from scipy.spatial import cKDTree


NY = 1059
NX = 1799
EARTH_RADIUS_KM = 6371.0
HRRR_GRID_STORE = "s3://hrrrzarr/grid/HRRR_chunk_index.zarr"


def xyz_from_latlon(lat_deg, lon_deg):
    """Convert latitude/longitude to Cartesian coordinates on unit sphere."""
    lat = np.deg2rad(np.asarray(lat_deg, dtype=np.float64).ravel())
    lon = np.deg2rad(np.asarray(lon_deg, dtype=np.float64).ravel())

    cos_lat = np.cos(lat)

    return np.column_stack(
        (
            cos_lat * np.cos(lon),
            cos_lat * np.sin(lon),
            np.sin(lat),
        )
    )


def read_hrrr_latlon():
    """Read the static native HRRR latitude/longitude grid from AWS."""
    fs = s3fs.S3FileSystem(anon=True)

    print(f"Opening HRRR static grid:")
    print(f"  {HRRR_GRID_STORE}")

    store = s3fs.S3Map(
        root=HRRR_GRID_STORE,
        s3=fs,
        check=False,
    )

    ds = xr.open_zarr(store)

    try:
        if "latitude" not in ds or "longitude" not in ds:
            raise RuntimeError(
                "HRRR static grid store does not contain latitude/longitude. "
                f"Variables found: {list(ds.variables)}"
            )

        lat = np.asarray(ds["latitude"].values, dtype=np.float32)
        lon = np.asarray(ds["longitude"].values, dtype=np.float32)

    finally:
        ds.close()

    if lat.shape != (NY, NX):
        raise RuntimeError(
            f"Unexpected HRRR latitude shape {lat.shape}; "
            f"expected {(NY, NX)}"
        )

    if lon.shape != (NY, NX):
        raise RuntimeError(
            f"Unexpected HRRR longitude shape {lon.shape}; "
            f"expected {(NY, NX)}"
        )

    if not np.all(np.isfinite(lat)) or not np.all(np.isfinite(lon)):
        raise RuntimeError("HRRR latitude/longitude contains non-finite values")

    print(f"HRRR grid shape: {lat.shape}")
    print(
        f"HRRR latitude range : {float(np.min(lat)):.3f} "
        f"to {float(np.max(lat)):.3f}"
    )
    print(
        f"HRRR longitude range: {float(np.min(lon)):.3f} "
        f"to {float(np.max(lon)):.3f}"
    )

    return lat, lon


def first_2d_data_var(ds: xr.Dataset) -> str:
    """Return the Stage-IV 2-D precipitation variable name."""
    for name in ("tp", "APCP", "apcp"):
        if name in ds.data_vars and ds[name].ndim == 2:
            return name

    for name, data_array in ds.data_vars.items():
        if data_array.ndim == 2:
            return name

    raise RuntimeError(
        "No 2-D Stage-IV data variable found. "
        f"Data variables: {list(ds.data_vars)}"
    )


def read_stage4_grid(path: str):
    """Read Stage-IV latitude/longitude from one local GRIB2 file."""
    print(f"Opening Stage IV sample:")
    print(f"  {path}")

    ds = xr.open_dataset(
        path,
        engine="cfgrib",
        backend_kwargs={"indexpath": ""},
    )

    try:
        if "latitude" not in ds.coords or "longitude" not in ds.coords:
            raise RuntimeError(
                "Stage-IV cfgrib dataset lacks latitude/longitude coordinates. "
                f"Coordinates: {list(ds.coords)}"
            )

        var_name = first_2d_data_var(ds)

        lat = np.asarray(ds["latitude"].values, dtype=np.float32)
        lon = np.asarray(ds["longitude"].values, dtype=np.float32)
        data_shape = tuple(ds[var_name].shape)

    finally:
        ds.close()

    if lat.shape != data_shape or lon.shape != data_shape:
        raise RuntimeError(
            "Stage-IV coordinate/data shapes disagree: "
            f"lat={lat.shape}, lon={lon.shape}, data={data_shape}"
        )

    if not np.all(np.isfinite(lat)) or not np.all(np.isfinite(lon)):
        raise RuntimeError("Stage-IV latitude/longitude contains non-finite values")

    print(f"Stage-IV variable: {var_name}")
    print(f"Stage-IV grid shape: {data_shape}")
    print(
        f"Stage-IV latitude range : {float(np.min(lat)):.3f} "
        f"to {float(np.max(lat)):.3f}"
    )
    print(
        f"Stage-IV longitude range: {float(np.min(lon)):.3f} "
        f"to {float(np.max(lon)):.3f}"
    )

    return lat, lon, data_shape, var_name


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stage4-file",
        required=True,
        help="One local Stage-IV 6-h GRIB2 file",
    )

    parser.add_argument(
        "--output",
        default=os.path.join(
            os.environ.get("DATA_DIR", "."),
            "derived/experiments/timelag_lag_selection_v1/"
            "grid_mapping/stage4_to_hrrr_nearest.npz",
        ),
    )

    parser.add_argument(
        "--max-distance-km",
        type=float,
        default=10.0,
        help=(
            "Maximum distance from an HRRR point to its nearest Stage-IV "
            "point before the mapping is masked (default: 10 km)"
        ),
    )

    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    hrrr_lat, hrrr_lon = read_hrrr_latlon()

    stage4_lat, stage4_lon, stage4_shape, stage4_var = read_stage4_grid(
        args.stage4_file
    )

    print()
    print("Building nearest-neighbor Stage-IV -> HRRR mapping...")

    stage4_xyz = xyz_from_latlon(stage4_lat, stage4_lon)
    hrrr_xyz = xyz_from_latlon(hrrr_lat, hrrr_lon)

    tree = cKDTree(stage4_xyz)

    chord_distance, flat_index = tree.query(
        hrrr_xyz,
        k=1,
        workers=-1,
    )

    # Convert chord distance on unit sphere to great-circle arc distance.
    chord_distance = np.clip(chord_distance, 0.0, 2.0)
    arc_distance = 2.0 * np.arcsin(chord_distance / 2.0)

    distance_km = (
        EARTH_RADIUS_KM * arc_distance
    ).astype(np.float32)

    flat_index = flat_index.astype(np.int32)

    valid_mapping = distance_km <= args.max_distance_km

    np.savez_compressed(
        output,
        stage4_flat_index=flat_index,
        distance_km=distance_km,
        valid_mapping=valid_mapping,
        hrrr_lat=hrrr_lat.astype(np.float32),
        hrrr_lon=hrrr_lon.astype(np.float32),
        stage4_shape=np.asarray(stage4_shape, dtype=np.int32),
    )

    distance_percentiles = np.percentile(
        distance_km,
        [0, 25, 50, 75, 90, 95, 99, 100],
    )

    metadata = {
        "mapping_method": (
            "nearest Stage-IV grid point to each native HRRR grid point"
        ),
        "distance_geometry": "great-circle distance via unit-sphere cKDTree",
        "max_distance_km": args.max_distance_km,
        "hrrr_grid_source": HRRR_GRID_STORE,
        "hrrr_shape": [NY, NX],
        "stage4_sample_file": str(Path(args.stage4_file).resolve()),
        "stage4_variable": stage4_var,
        "stage4_shape": list(stage4_shape),
        "valid_hrrr_fraction": float(np.mean(valid_mapping)),
        "distance_percentiles_km": {
            "p00": float(distance_percentiles[0]),
            "p25": float(distance_percentiles[1]),
            "p50": float(distance_percentiles[2]),
            "p75": float(distance_percentiles[3]),
            "p90": float(distance_percentiles[4]),
            "p95": float(distance_percentiles[5]),
            "p99": float(distance_percentiles[6]),
            "p100": float(distance_percentiles[7]),
        },
    }

    metadata_path = output.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    print()
    print("=" * 72)
    print("GRID MAPPING COMPLETE")
    print("=" * 72)
    print(f"Output                  : {output}")
    print(f"Metadata                : {metadata_path}")
    print(
        f"Valid HRRR mapping      : "
        f"{np.mean(valid_mapping) * 100:.2f}%"
    )
    print(
        "Distance percentiles km:",
        " ".join(
            f"{p:.3f}"
            for p in distance_percentiles
        ),
    )
    print("=" * 72)


if __name__ == "__main__":
    main()

