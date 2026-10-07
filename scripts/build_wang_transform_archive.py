#!/usr/bin/env python3
"""
Build a deterministic held-out VALIDATION archive for Wang-style transformations.

This intentionally does NOT reuse the 512 fixed-validation cases that selected
the training checkpoint. Instead it draws a separate deterministic subset from
the larger validation split, preserving the sealed test period.

Only the original HRRR patches are stored. Wang-style transformed fields are
generated deterministically on demand from transform_spec.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import HRRRArchive


GRID_SHAPE = (1059, 1799)


def stable_seed(base_seed, key):
    d = hashlib.blake2b(
        f"{base_seed}|{key}".encode(), digest_size=8
    ).digest()
    return int.from_bytes(d, "little")


def wet_stats(x, threshold):
    wet = np.asarray(x) >= float(threshold)
    vals = np.asarray(x)[wet]
    return {
        "wet_fraction": float(wet.mean()),
        "wet_pixel_count": int(wet.sum()),
        "wet_p90_mm": float(np.percentile(vals,90)) if vals.size else np.nan,
        "max_mm": float(np.max(x)),
        "mean_mm": float(np.mean(x)),
    }


def main():
    data = Path(os.environ["DATA_DIR"])
    p = argparse.ArgumentParser()
    p.add_argument(
        "--anchors",
        default=str(data/"metadata/dataset_v1/eligible_anchors_v1.csv.gz"),
    )
    p.add_argument(
        "--manifest",
        default=str(
            data/"forecasts/hrrr/qpf6_native/metadata/hrrr_qpf6_manifest.csv.gz"
        ),
    )
    p.add_argument(
        "--output-dir",
        default=str(data/"derived/evaluation/wang_transform_v1"),
    )
    p.add_argument("--split", default="validation")
    p.add_argument("--fhr-min", type=int, default=7)
    p.add_argument("--fhr-max", type=int, default=17)
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--target-cases", type=int, default=2000)
    p.add_argument("--wet-threshold-mm", type=float, default=0.1)
    p.add_argument("--include-dry", action="store_true")
    p.add_argument("--seed", type=int, default=20261005)
    args = p.parse_args()

    a = pd.read_csv(args.anchors, low_memory=False)
    m = pd.read_csv(args.manifest, low_memory=False)

    aid = next(
        (c for c in ("anchor_id","sample_id","field_id") if c in a), None
    )
    if aid is None or "split" not in a:
        raise RuntimeError("Could not identify anchor ID/split columns.")

    a = a[[aid,"split"]].copy()
    a["sample_id"] = a[aid].astype(str)

    need = ["sample_id","forecast_hour","valid_end"]
    missing = [c for c in need if c not in m]
    if missing:
        raise RuntimeError(f"manifest missing {missing}")

    m = m[need].copy()
    m["sample_id"] = m["sample_id"].astype(str)
    m["forecast_hour"] = pd.to_numeric(m["forecast_hour"]).astype(int)
    m["valid_end"] = pd.to_datetime(m["valid_end"], utc=True)

    df = a.merge(m,on="sample_id",how="inner",validate="one_to_one")
    df = df[
        (df["split"] == args.split)
        & df["forecast_hour"].between(args.fhr_min,args.fhr_max)
    ].copy()

    rng = np.random.default_rng(args.seed)
    df = df.iloc[rng.permutation(len(df))].reset_index(drop=True)

    archive = HRRRArchive(args.manifest)
    ps = int(args.patch_size)
    ny,nx = GRID_SHAPE
    patches, rows = [], []

    for row in df.itertuples():
        local = np.random.default_rng(
            stable_seed(args.seed, str(row.sample_id))
        )
        y0 = int(local.integers(0, ny-ps+1))
        x0 = int(local.integers(0, nx-ps+1))
        x = archive.read_patch(row.sample_id,y0,x0,ps)
        stats = wet_stats(x,args.wet_threshold_mm)
        dry = stats["wet_pixel_count"] == 0
        if dry and not args.include_dry:
            continue

        rows.append({
            "case_index": len(rows),
            "sample_id": row.sample_id,
            "valid_end": row.valid_end,
            "forecast_hour": int(row.forecast_hour),
            "y0": y0, "x0": x0, "patch_size": ps,
            "is_dry": bool(dry),
            **stats,
        })
        patches.append(np.asarray(x,dtype=np.float32))
        if len(rows) >= args.target_cases:
            break

    archive.close()
    if len(rows) < args.target_cases:
        raise RuntimeError(
            f"Only found {len(rows)} qualifying patches; "
            f"requested {args.target_cases}."
        )

    out = Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=True)
    arr = np.stack(patches)
    np.savez_compressed(out/"original_patches.npz",qpf6_mm=arr)
    pd.DataFrame(rows).to_csv(out/"cases.csv",index=False)

    spec = {
        "version": 1,
        "selection": {
            "split": args.split,
            "fhr_min": args.fhr_min,
            "fhr_max": args.fhr_max,
            "target_cases": args.target_cases,
            "patch_size": ps,
            "seed": args.seed,
            "dry_policy": "include" if args.include_dry else "exclude",
            "wet_threshold_mm": args.wet_threshold_mm,
            "checkpoint_selection_panel_reused": False,
            "sealed_test_used": False,
        },
        "displacement_biases_grid_cells": list(range(-10,11)),
        "intensity_factors": [
            .50,.55,.60,.65,.70,.75,.80,.85,.90,.95,1.00,
            1.10,1.20,1.30,1.40,1.50,1.60,1.70,1.80,1.90,2.00
        ],
        "area_size_factors": [
            .50,.55,.60,.65,.70,.75,.80,.85,.90,.95,1.00,
            1.10,1.20,1.30,1.40,1.50,1.60,1.70,1.80,1.90,2.00
        ],
        "physical_scale_note": (
            "The primary comparison preserves Wang's exact grid-cell bias "
            "numbers. Wang used 0.125-degree IFS data, whereas these patches "
            "are native ~3-km HRRR. A km-matched displacement sensitivity "
            "may be added separately and must not be conflated with the "
            "paper-index replication."
        ),
    }
    (out/"transform_spec.json").write_text(json.dumps(spec,indent=2)+"\n")

    print(f"Archive cases : {len(rows)}")
    print(f"Array shape   : {arr.shape}")
    print(f"Output        : {out}")
    print("Separate from the 512-case checkpoint-selection panel: YES")
    print("Sealed test used: NO")


if __name__ == "__main__":
    main()

