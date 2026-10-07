#!/usr/bin/env python3
"""
Create a deterministic candidate manifest for HRRR-vs-Stage-IV evaluation.

This is intentionally validation-only by default. It does not read
precipitation fields; it selects HRRR anchors, exact 6-h Stage-IV valid times,
and deterministic 256x256 crop coordinates. The downstream archive builder
rejects crops that are not fully covered by valid Stage-IV data.

Stage-IV 6-h products are restricted to valid-end hours 00/06/12/18 UTC.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd


GRID_SHAPE = (1059, 1799)


def stable_seed(base_seed: int, key: str) -> int:
    d = hashlib.blake2b(
        f"{base_seed}|{key}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(d, "little", signed=False)


def parse_time(series):
    return pd.to_datetime(series, utc=True, errors="raise")


def main():
    data = Path(os.environ["DATA_DIR"])
    p = argparse.ArgumentParser()
    p.add_argument(
        "--anchors",
        default=str(data / "metadata/dataset_v1/eligible_anchors_v1.csv.gz"),
    )
    p.add_argument(
        "--manifest",
        default=str(
            data / "forecasts/hrrr/qpf6_native/metadata/hrrr_qpf6_manifest.csv.gz"
        ),
    )
    p.add_argument(
        "--output",
        default=str(
            data / "derived/evaluation/stageiv_reference_v1/candidate_pairs.csv"
        ),
    )
    p.add_argument("--split", default="validation")
    p.add_argument("--fhr-min", type=int, default=7)
    p.add_argument("--fhr-max", type=int, default=17)
    p.add_argument(
        "--valid-hours", type=int, nargs="+", default=[0,6,12,18]
    )
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument(
        "--candidate-count",
        type=int,
        default=3000,
        help="Oversample candidates; archive builder stops after target count.",
    )
    p.add_argument("--seed", type=int, default=20261004)
    p.add_argument(
        "--stageiv-root",
        default="",
        help=(
            "Optional local directory containing staged 6-h Stage-IV GRIB2. "
            "If omitted, stageiv_path contains only the expected filename."
        ),
    )
    args = p.parse_args()

    anchors = pd.read_csv(args.anchors, low_memory=False)
    manifest = pd.read_csv(args.manifest, low_memory=False)

    aid_col = next(
        (c for c in ("anchor_id","sample_id","field_id") if c in anchors.columns),
        None,
    )
    if aid_col is None:
        raise RuntimeError("Could not identify anchor ID column.")
    if "split" not in anchors:
        raise RuntimeError("Eligible anchors must contain split.")

    anchors = anchors[[aid_col, "split"]].copy()
    anchors["sample_id"] = anchors[aid_col].astype(str)

    required = ["sample_id","forecast_hour","valid_end"]
    missing = [c for c in required if c not in manifest]
    if missing:
        raise RuntimeError(f"HRRR manifest missing {missing}")

    m = manifest[required].copy()
    m["sample_id"] = m["sample_id"].astype(str)
    m["valid_end"] = parse_time(m["valid_end"])
    m["forecast_hour"] = pd.to_numeric(
        m["forecast_hour"], errors="raise"
    ).astype(int)

    df = anchors.merge(m, on="sample_id", how="inner", validate="one_to_one")
    df = df[
        (df["split"] == args.split)
        & (df["forecast_hour"].between(args.fhr_min, args.fhr_max))
        & (df["valid_end"].dt.hour.isin(args.valid_hours))
    ].copy()

    if len(df) == 0:
        raise RuntimeError("No candidate anchors after filtering.")

    # Deterministically shuffle the candidate anchor set.
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(df))
    df = df.iloc[order].reset_index(drop=True)
    df = df.iloc[: min(args.candidate_count, len(df))].copy()

    ny, nx = GRID_SHAPE
    ps = int(args.patch_size)
    if ps > ny or ps > nx:
        raise ValueError("patch size exceeds HRRR grid")

    y0s, x0s, paths, names = [], [], [], []
    root = Path(args.stageiv_root) if args.stageiv_root else None

    for row in df.itertuples():
        local_rng = np.random.default_rng(
            stable_seed(args.seed, str(row.sample_id))
        )
        y0 = int(local_rng.integers(0, ny - ps + 1))
        x0 = int(local_rng.integers(0, nx - ps + 1))
        y0s.append(y0)
        x0s.append(x0)

        ts = pd.Timestamp(row.valid_end)
        name = f"st4_conus.{ts.strftime('%Y%m%d%H')}.06h.grb2"
        names.append(name)
        paths.append(str(root / name) if root else name)

    df["y0"] = y0s
    df["x0"] = x0s
    df["patch_size"] = ps
    df["valid_start"] = df["valid_end"] - pd.Timedelta(hours=6)
    df["valid_hour_utc"] = df["valid_end"].dt.hour
    df["stageiv_filename"] = names
    df["stageiv_path"] = paths
    df["stageiv_exists"] = [
        Path(x).is_file() if root is not None else False
        for x in paths
    ]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    unique_files = (
        df[["stageiv_filename","stageiv_path"]]
        .drop_duplicates()
        .sort_values("stageiv_filename")
    )
    unique_files.to_csv(
        out.with_name("stageiv_files_needed.csv"), index=False
    )
    out.with_name("stageiv_files_needed.txt").write_text(
        "\n".join(unique_files["stageiv_filename"]) + "\n"
    )

    print(f"Eligible after filters : {len(df):,}")
    print(f"Unique Stage-IV files : {len(unique_files):,}")
    print(f"Candidate manifest    : {out}")
    print(
        "NOTE: downstream archive construction will reject crops without "
        "complete valid Stage-IV coverage."
    )


if __name__ == "__main__":
    main()

