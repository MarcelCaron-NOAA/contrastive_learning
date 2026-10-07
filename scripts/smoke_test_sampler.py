#!/usr/bin/env python3
"""
Smoke test the real HRRR contrastive data pipeline before model training.

Two stages:

1) metadata-only contract test
   - fast; no Zarr reads
   - samples many recipes and mechanically verifies A/P/N rules

2) materialization test
   - reads a few real HRRR minibatches
   - verifies shapes/dtypes/finiteness
   - measures exact-duplicate negative masking
   - reports dry A/P occurrence and I/O throughput

This script does NOT train a network.
"""

from __future__ import annotations

import argparse
from collections import Counter
import itertools
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data import (
    ContrastiveIndex,
    NegativeRules,
    ContrastiveBatchSampler,
    HRRRArchive,
    ContrastiveDataset,
    PrecipTransform,
    contrastive_collate,
)
from src.data.contrastive_index import circular_climo_distance


def default_paths():
    data = Path(os.environ["DATA_DIR"])
    return {
        "anchors": data / "metadata/dataset_v1/eligible_anchors_v1.csv.gz",
        "negatives": data / "metadata/dataset_v1/negative_pool_v1.csv.gz",
        "manifest": (
            data
            / "forecasts/hrrr/qpf6_native/metadata/hrrr_qpf6_manifest.csv.gz"
        ),
    }


def validate_recipe(index, recipe, rules, pool_split_by_id):
    a = index.field_metadata(recipe.anchor_id)
    p = index.field_metadata(recipe.positive_id)

    assert recipe.split == "train"
    assert 7 <= int(a["forecast_hour"]) <= 17
    assert int(a["valid_unix_s"]) == int(p["valid_unix_s"])

    index.validate_positive_relation(
        recipe.anchor_id,
        recipe.positive_id,
        recipe.positive_direction,
    )

    assert 0 <= recipe.y0 <= 1059 - recipe.patch_size
    assert 0 <= recipe.x0 <= 1799 - recipe.patch_size

    for nid in recipe.negative_ids:
        n = index.field_metadata(nid)

        assert pool_split_by_id[nid] == recipe.split
        assert int(n["forecast_hour"]) == int(a["forecast_hour"])
        assert int(n["valid_hour_utc"]) == int(a["valid_hour_utc"])

        dt_days = abs(
            int(n["valid_unix_s"]) - int(a["valid_unix_s"])
        ) / 86400.0
        assert dt_days > rules.min_days

        if rules.mode == "climo45":
            assert int(n["valid_year"]) != int(a["valid_year"])
            cd = int(
                circular_climo_distance(
                    int(n["climo_day"]),
                    int(a["climo_day"]),
                )
            )
            assert cd <= rules.climo_window_days


def collect_recipe_sample(sampler, n_recipes):
    out = []
    for batch in sampler:
        out.extend(batch)
        if len(out) >= n_recipes:
            break
    return out[:n_recipes]


def main():
    defaults = default_paths()

    p = argparse.ArgumentParser()
    p.add_argument("--anchors", default=str(defaults["anchors"]))
    p.add_argument("--negative-pool", default=str(defaults["negatives"]))
    p.add_argument("--manifest", default=str(defaults["manifest"]))

    p.add_argument("--negative-mode", choices=["calendar_any", "climo45"],
                   default="climo45")
    p.add_argument("--negative-count", type=int, default=8)
    p.add_argument("--negative-min-days", type=float, default=90.0)
    p.add_argument("--climo-window-days", type=int, default=45)

    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=20260930)

    p.add_argument("--metadata-recipes", type=int, default=10000)
    p.add_argument("--materialize-batches", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=0)

    p.add_argument(
        "--transform",
        choices=["none", "log1p", "raw_zscore", "log1p_zscore"],
        default="log1p",
    )
    p.add_argument("--stats-json", default=None)
    p.add_argument("--summary-json", default=None)

    a = p.parse_args()

    if "zscore" in a.transform and not a.stats_json:
        raise SystemExit(
            f"--transform {a.transform} requires --stats-json. "
            "For the first smoke test, use --transform log1p."
        )

    print("=" * 76)
    print("CONTRASTIVE SAMPLER SMOKE TEST")
    print("=" * 76)
    print(f"Anchors       : {a.anchors}")
    print(f"Negative pool : {a.negative_pool}")
    print(f"Manifest      : {a.manifest}")
    print(f"Negative mode : {a.negative_mode}")
    print(f"K negatives   : {a.negative_count}")
    print(f"Batch size    : {a.batch_size}")
    print(f"Patch size    : {a.patch_size}")
    print(f"Transform     : {a.transform}")
    print()

    index = ContrastiveIndex(
        a.anchors,
        a.negative_pool,
        a.manifest,
    )

    rules = NegativeRules(
        mode=a.negative_mode,
        min_days=a.negative_min_days,
        climo_window_days=a.climo_window_days,
    )

    sampler = ContrastiveBatchSampler(
        index,
        split="train",
        batch_size=a.batch_size,
        patch_size=a.patch_size,
        negative_count=a.negative_count,
        negative_rules=rules,
        seed=a.seed,
    )

    print(f"Training anchors: {sampler.n_anchors:,}")
    print(f"Batches / epoch : {len(sampler):,}")
    print()

    # ------------------------------------------------------------------
    # Metadata-only contract test
    # ------------------------------------------------------------------
    print("-" * 76)
    print(f"1) METADATA CONTRACT TEST ({a.metadata_recipes:,} recipes)")
    print("-" * 76)

    pool_split_by_id = dict(
        zip(
            index.negative_pool["negative_id"].astype(str),
            index.negative_pool["split"].astype(str),
        )
    )

    sampler.set_epoch(0)

    n_checked = 0
    seen_anchors = set()
    p_direction = Counter()
    lead_counts = Counter()
    hour_counts = Counter()
    candidate_counts = []
    n_climo = []
    n_dt = []
    crop_x = []
    crop_y = []

    t0 = time.perf_counter()

    for batch in sampler:
        # Negative IDs must be unique throughout this minibatch.
        batch_negs = [nid for r in batch for nid in r.negative_ids]
        assert len(batch_negs) == len(set(batch_negs))

        # No N field is also A or P in this same optimization step.
        batch_ap = {x for r in batch for x in (r.anchor_id, r.positive_id)}
        assert set(batch_negs).isdisjoint(batch_ap)

        for recipe in batch:
            assert recipe.anchor_id not in seen_anchors
            seen_anchors.add(recipe.anchor_id)

            validate_recipe(index, recipe, rules, pool_split_by_id)

            candidate_counts.append(recipe.negative_candidate_count)

            p_direction[recipe.positive_direction] += 1
            lead_counts[recipe.anchor_forecast_hour] += 1
            hour_counts[recipe.anchor_valid_hour_utc] += 1
            crop_x.append(recipe.x0)
            crop_y.append(recipe.y0)

            am = index.field_metadata(recipe.anchor_id)
            for nid in recipe.negative_ids:
                nm = index.field_metadata(nid)
                n_dt.append(
                    abs(
                        int(nm["valid_unix_s"])
                        - int(am["valid_unix_s"])
                    ) / 86400.0
                )
                n_climo.append(
                    int(
                        circular_climo_distance(
                            int(nm["climo_day"]),
                            int(am["climo_day"]),
                        )
                    )
                )

            n_checked += 1
            if n_checked >= a.metadata_recipes:
                break

        if n_checked >= a.metadata_recipes:
            break

    metadata_seconds = time.perf_counter() - t0

    cc = np.asarray(candidate_counts)
    print(f"Recipes checked            : {n_checked:,}")
    print(f"Unique anchors             : {len(seen_anchors):,}")
    print(f"Positive directions        : {dict(p_direction)}")
    print(
        "Candidate N count         : "
        f"min={cc.min():,}, median={np.median(cc):,.0f}, max={cc.max():,}"
    )
    print(
        "Sampled |Δt| days         : "
        f"min={np.min(n_dt):.1f}, median={np.median(n_dt):.1f}, "
        f"max={np.max(n_dt):.1f}"
    )
    if a.negative_mode == "climo45":
        print(
            "Sampled climo Δ days      : "
            f"min={np.min(n_climo)}, median={np.median(n_climo):.0f}, "
            f"max={np.max(n_climo)}"
        )
    print(
        f"Metadata throughput        : "
        f"{n_checked / metadata_seconds:,.1f} recipes/s"
    )
    print("Metadata contract          : PASS")
    print()

    # ------------------------------------------------------------------
    # Epoch stochasticity
    # ------------------------------------------------------------------
    print("-" * 76)
    print("2) EPOCH STOCHASTICITY CHECK")
    print("-" * 76)

    compare_n = min(5000, a.metadata_recipes)
    sampler.set_epoch(0)
    r0 = collect_recipe_sample(sampler, compare_n)
    sampler.set_epoch(1)
    r1 = collect_recipe_sample(sampler, compare_n)

    d0 = {r.anchor_id: r for r in r0}
    d1 = {r.anchor_id: r for r in r1}
    overlap = sorted(set(d0) & set(d1))

    changed_p = changed_crop = changed_n = 0
    for aid in overlap:
        x0 = d0[aid]
        x1 = d1[aid]
        changed_p += x0.positive_id != x1.positive_id
        changed_crop += (x0.y0, x0.x0) != (x1.y0, x1.x0)
        changed_n += x0.negative_ids != x1.negative_ids

    print(f"Compared overlapping A     : {len(overlap):,}")
    if overlap:
        print(
            f"P changed                 : "
            f"{100 * changed_p / len(overlap):.1f}%"
        )
        print(
            f"Crop changed              : "
            f"{100 * changed_crop / len(overlap):.1f}%"
        )
        print(
            f"N set changed             : "
            f"{100 * changed_n / len(overlap):.1f}%"
        )
    assert not overlap or changed_crop > 0
    assert not overlap or changed_n > 0
    print("Epoch stochasticity         : PASS")
    print()

    # ------------------------------------------------------------------
    # Materialization / I/O test
    # ------------------------------------------------------------------
    print("-" * 76)
    print(f"3) MATERIALIZATION TEST ({a.materialize_batches} batches)")
    print("-" * 76)

    if a.stats_json:
        transform = PrecipTransform.from_stats_json(
            a.transform,
            a.stats_json,
        )
    else:
        transform = PrecipTransform(a.transform)

    archive = HRRRArchive(a.manifest)
    dataset = ContrastiveDataset(
        archive,
        negative_count=a.negative_count,
        transform=transform,
    )

    sampler.set_epoch(0)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=contrastive_collate,
        num_workers=a.num_workers,
        pin_memory=False,
    )

    t0 = time.perf_counter()

    n_batches = 0
    n_samples = 0
    n_negative_slots = 0
    n_masked = 0
    n_same_a = 0
    n_same_p = 0
    n_dry_a = 0
    n_dry_p = 0
    n_dry_both = 0

    first_shapes = None

    try:
        for batch in itertools.islice(loader, a.materialize_batches):
            B = batch["anchor"].shape[0]

            assert batch["anchor"].shape == (
                B, 1, a.patch_size, a.patch_size
            )
            assert batch["positive"].shape == (
                B, 1, a.patch_size, a.patch_size
            )
            assert batch["negatives"].shape == (
                B, a.negative_count, 1, a.patch_size, a.patch_size
            )
            assert batch["negative_mask"].shape == (
                B, a.negative_count
            )

            assert batch["anchor"].dtype == torch.float32
            assert batch["positive"].dtype == torch.float32
            assert batch["negatives"].dtype == torch.float32

            assert torch.isfinite(batch["anchor"]).all()
            assert torch.isfinite(batch["positive"]).all()
            assert torch.isfinite(batch["negatives"]).all()

            expected_mask = ~(
                batch["negative_identical_to_anchor"]
                | batch["negative_identical_to_positive"]
            )
            assert torch.equal(batch["negative_mask"], expected_mask)

            if first_shapes is None:
                first_shapes = {
                    "A": tuple(batch["anchor"].shape),
                    "P": tuple(batch["positive"].shape),
                    "N": tuple(batch["negatives"].shape),
                    "mask": tuple(batch["negative_mask"].shape),
                }

            max_a = batch["anchor_raw_max_mm"]
            max_p = batch["positive_raw_max_mm"]
            dry_a = max_a == 0.0
            dry_p = max_p == 0.0

            n_dry_a += int(dry_a.sum())
            n_dry_p += int(dry_p.sum())
            n_dry_both += int((dry_a & dry_p).sum())

            n_negative_slots += B * a.negative_count
            n_masked += int((~batch["negative_mask"]).sum())
            n_same_a += int(batch["negative_identical_to_anchor"].sum())
            n_same_p += int(batch["negative_identical_to_positive"].sum())

            n_samples += B
            n_batches += 1
    finally:
        archive.close()

    io_seconds = time.perf_counter() - t0

    print(f"Batch tensor shapes         : {first_shapes}")
    print(f"Samples materialized        : {n_samples:,}")
    print(f"Negative slots              : {n_negative_slots:,}")
    print(
        f"Masked exact duplicates     : {n_masked:,} "
        f"({100*n_masked/max(n_negative_slots,1):.3f}%)"
    )
    print(f"  identical to A            : {n_same_a:,}")
    print(f"  identical to P            : {n_same_p:,}")
    print(f"Dry A patches               : {n_dry_a:,}")
    print(f"Dry P patches               : {n_dry_p:,}")
    print(f"Dry A=P occurrence count    : {n_dry_both:,}")
    print(
        f"Materialization throughput  : "
        f"{n_samples / io_seconds:,.2f} samples/s "
        f"({io_seconds:.2f} s)"
    )
    print("Materialization contract     : PASS")
    print()

    summary = {
        "negative_mode": a.negative_mode,
        "negative_count": a.negative_count,
        "negative_min_days": a.negative_min_days,
        "climo_window_days": a.climo_window_days,
        "batch_size": a.batch_size,
        "patch_size": a.patch_size,
        "transform": a.transform,
        "training_anchors": sampler.n_anchors,
        "batches_per_epoch": len(sampler),
        "metadata_recipes_checked": n_checked,
        "candidate_count_min": int(cc.min()),
        "candidate_count_median": float(np.median(cc)),
        "candidate_count_max": int(cc.max()),
        "positive_directions": dict(p_direction),
        "materialized_batches": n_batches,
        "materialized_samples": n_samples,
        "negative_slots": n_negative_slots,
        "masked_negative_slots": n_masked,
        "masked_negative_fraction": (
            n_masked / n_negative_slots if n_negative_slots else None
        ),
        "dry_anchor_patches": n_dry_a,
        "dry_positive_patches": n_dry_p,
        "dry_both_count": n_dry_both,
        "materialization_samples_per_second": (
            n_samples / io_seconds if io_seconds > 0 else None
        ),
    }

    if a.summary_json:
        path = Path(a.summary_json)
    else:
        path = (
            Path(os.environ["DATA_DIR"])
            / "metadata/dataset_v1/sampler_smoke_test_summary.json"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n")

    print("=" * 76)
    print("SMOKE TEST PASS")
    print("=" * 76)
    print(f"Summary: {path}")


if __name__ == "__main__":
    main()

