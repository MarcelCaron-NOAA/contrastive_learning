#!/usr/bin/env python3
"""
CPU-only metadata preflight for the first real train/validation experiment.

No Zarr reads and no GPU use. Safe to run before entering the H100 queue.

Checks:
  - literal train/validation split labels exist in both indexes
  - expected anchor/pool counts are nonzero
  - +/-1-h positives resolve correctly
  - sampled training anchors have >= K climo45 negatives
  - sampled validation anchors have >= K calendar_any negatives
  - validation negative rules satisfy same split/lead/UTC/>90-day constraints
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import numpy as np

from src.data import ContrastiveIndex, NegativeRules


def defaults():
    data = Path(os.environ["DATA_DIR"])
    return {
        "anchors": data / "metadata/dataset_v1/eligible_anchors_v1.csv.gz",
        "negative_pool": data / "metadata/dataset_v1/negative_pool_v1.csv.gz",
        "manifest": (
            data
            / "forecasts/hrrr/qpf6_native/metadata/hrrr_qpf6_manifest.csv.gz"
        ),
    }


def check_split(index, split, rules, k, n_check, seed):
    anchors = index.anchors_for_split(split)
    pool = index.negative_pool.loc[
        index.negative_pool["split"].astype(str) == str(split)
    ]

    print(f"{split!r}: anchors={len(anchors):,}, negative_pool={len(pool):,}")

    rng = np.random.default_rng(seed)
    take = min(n_check, len(anchors))
    ids = rng.choice(len(anchors), size=take, replace=False)

    candidate_counts = []
    earlier = 0
    later = 0

    for ii in ids:
        row = anchors.iloc[int(ii)]

        positives = index.positive_ids(row)
        index.validate_positive_relation(
            str(row["anchor_id"]),
            str(positives["earlier_init"]),
            "earlier_init",
        )
        index.validate_positive_relation(
            str(row["anchor_id"]),
            str(positives["later_init"]),
            "later_init",
        )
        earlier += 1
        later += 1

        c = index.negative_candidates(row, rules)
        candidate_counts.append(len(c))

        if len(c) < k:
            raise RuntimeError(
                f"{split} anchor {row['anchor_id']} has only {len(c)} "
                f"negative candidates; need K={k}"
            )

        # Spot-check candidate metadata contract on up to the first 10.
        for _, nrow in c.head(10).iterrows():
            nm = index.field_metadata(str(nrow["negative_id"]))

            assert int(nm["forecast_hour"]) == int(row["forecast_hour"])
            assert int(nm["valid_hour_utc"]) == int(row["valid_hour_utc"])

            dt_days = abs(
                int(nm["valid_unix_s"]) - int(row["valid_unix_s"])
            ) / 86400.0
            assert dt_days > rules.min_days

            if rules.mode == "climo45":
                assert int(nm["valid_year"]) != int(row["valid_year"])
                assert int(nrow["climo_distance_days"]) <= rules.climo_window_days

    arr = np.asarray(candidate_counts)
    print(
        f"  checked={take:,} | positives OK | "
        f"N candidates min={arr.min():,}, "
        f"median={np.median(arr):,.0f}, max={arr.max():,}"
    )


def main():
    d = defaults()
    p = argparse.ArgumentParser()
    p.add_argument("--anchors", default=str(d["anchors"]))
    p.add_argument("--negative-pool", default=str(d["negative_pool"]))
    p.add_argument("--manifest", default=str(d["manifest"]))
    p.add_argument("--train-split", default="train")
    p.add_argument("--validation-split", default="validation")
    p.add_argument("--negative-count", type=int, default=8)
    p.add_argument("--check-per-split", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20260930)
    args = p.parse_args()

    print("=" * 78)
    print("FIRST REAL TRAIN/VALIDATION METADATA PREFLIGHT")
    print("=" * 78)

    index = ContrastiveIndex(
        args.anchors,
        args.negative_pool,
        args.manifest,
    )

    train_rules = NegativeRules(
        mode="climo45",
        min_days=90.0,
        climo_window_days=45,
    )
    validation_rules = NegativeRules(
        mode="calendar_any",
        min_days=90.0,
        climo_window_days=45,
    )

    check_split(
        index,
        args.train_split,
        train_rules,
        args.negative_count,
        args.check_per_split,
        args.seed,
    )
    check_split(
        index,
        args.validation_split,
        validation_rules,
        args.negative_count,
        args.check_per_split,
        args.seed + 1,
    )

    print("=" * 78)
    print("PREFLIGHT PASS")
    print("=" * 78)


if __name__ == "__main__":
    main()

