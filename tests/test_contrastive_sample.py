#!/usr/bin/env python3
from __future__ import annotations

import pandas as pd
import numpy as np

from src.data.contrastive_index import (
    ContrastiveIndex,
    NegativeRules,
    climo_day,
    circular_climo_distance,
)
from src.data.contrastive_sampler import ContrastiveBatchSampler
from src.data.contrastive_dataset import PrecipTransform


def _field(valid, fhr, sample_id):
    valid = pd.Timestamp(valid, tz="UTC")
    init = valid - pd.Timedelta(hours=fhr)
    return {
        "sample_id": sample_id,
        "init_hour": init.hour,
        "forecast_hour": fhr,
        "valid_end": int(valid.timestamp()),
        "available": True,
        "store": "/tmp/fake.zarr.zip",
    }


def _synthetic_index():
    manifest_rows = []

    # Two eligible anchors plus both +/-1-h positives.
    anchors_spec = [
        ("2024-01-15 12:00", "A1"),
        ("2024-07-15 18:00", "A2"),
    ]
    for valid, aid in anchors_spec:
        manifest_rows.append(_field(valid, 12, aid))
        manifest_rows.append(_field(valid, 13, aid + "_EARLY"))
        manifest_rows.append(_field(valid, 11, aid + "_LATE"))

    # Good climo45 negatives for A1 (same hour, F12, other years).
    for j, valid in enumerate([
        "2021-01-05 12:00",
        "2021-01-20 12:00",
        "2022-12-20 12:00",
        "2022-02-10 12:00",
        "2023-01-10 12:00",
        "2023-02-20 12:00",
    ]):
        manifest_rows.append(_field(valid, 12, f"N1_{j}"))

    # Good climo45 negatives for A2.
    for j, valid in enumerate([
        "2021-06-20 18:00",
        "2021-07-10 18:00",
        "2022-07-20 18:00",
        "2022-08-15 18:00",
        "2023-06-10 18:00",
        "2023-07-25 18:00",
    ]):
        manifest_rows.append(_field(valid, 12, f"N2_{j}"))

    # Deliberate non-candidates.
    manifest_rows += [
        _field("2023-01-10 18:00", 12, "WRONG_HOUR"),
        _field("2023-01-10 12:00", 10, "WRONG_LEAD"),
        _field("2023-07-15 12:00", 12, "WRONG_SEASON"),
        _field("2024-01-01 12:00", 12, "SAME_YEAR"),
    ]

    manifest = pd.DataFrame(manifest_rows)

    anchors = pd.DataFrame(
        {
            "anchor_id": ["A1", "A2"],
            "split": ["train", "train"],
        }
    )

    negative_ids = [
        x["sample_id"]
        for x in manifest_rows
        if x["sample_id"].startswith("N")
        or x["sample_id"].startswith("WRONG")
        or x["sample_id"] == "SAME_YEAR"
    ]
    pool = pd.DataFrame(
        {
            "sample_id": negative_ids,
            "split": ["train"] * len(negative_ids),
        }
    )

    return ContrastiveIndex.from_frames(anchors, pool, manifest)


def test_climo_distance_wraps_new_year():
    d1 = climo_day(pd.Timestamp("2024-12-31"))
    d2 = climo_day(pd.Timestamp("2024-01-05"))
    assert int(circular_climo_distance(d1, d2)) == 5


def test_positive_and_climo45_contract():
    idx = _synthetic_index()
    a = idx.anchors_for_split("train").iloc[0]

    pos = idx.positive_ids(a)
    idx.validate_positive_relation("A1", pos["earlier_init"], "earlier_init")
    idx.validate_positive_relation("A1", pos["later_init"], "later_init")

    rules = NegativeRules(mode="climo45", min_days=90, climo_window_days=45)
    cands = idx.negative_candidates(a, rules)

    assert len(cands) >= 4
    assert set(cands["negative_id"]).isdisjoint(
        {"WRONG_HOUR", "WRONG_LEAD", "WRONG_SEASON", "SAME_YEAR"}
    )
    assert (cands["time_distance_days"] > 90).all()
    assert (cands["climo_distance_days"] <= 45).all()
    assert (cands["valid_year"] != 2024).all()


def test_sampler_epoch_and_minibatch_uniqueness():
    idx = _synthetic_index()
    rules = NegativeRules(mode="climo45", min_days=90, climo_window_days=45)

    sampler = ContrastiveBatchSampler(
        idx,
        split="train",
        batch_size=2,
        patch_size=64,
        negative_count=2,
        negative_rules=rules,
        seed=1234,
    )

    sampler.set_epoch(0)
    batches0 = list(iter(sampler))
    recipes0 = [r for b in batches0 for r in b]

    assert sorted(r.anchor_id for r in recipes0) == ["A1", "A2"]

    for batch in batches0:
        negs = [nid for r in batch for nid in r.negative_ids]
        assert len(negs) == len(set(negs))

        forbidden = {
            x for r in batch for x in (r.anchor_id, r.positive_id)
        }
        assert set(negs).isdisjoint(forbidden)

        for r in batch:
            assert 0 <= r.y0 <= 1059 - 64
            assert 0 <= r.x0 <= 1799 - 64
            idx.validate_positive_relation(
                r.anchor_id, r.positive_id, r.positive_direction
            )

    sampler.set_epoch(1)
    recipes1 = [r for b in sampler for r in b]
    by0 = {r.anchor_id: r for r in recipes0}
    by1 = {r.anchor_id: r for r in recipes1}

    changed = 0
    for aid in by0:
        r0 = by0[aid]
        r1 = by1[aid]
        if (
            r0.positive_id != r1.positive_id
            or r0.y0 != r1.y0
            or r0.x0 != r1.x0
            or r0.negative_ids != r1.negative_ids
        ):
            changed += 1
    assert changed >= 1


def test_precip_transform_shapes_and_values():
    x = np.array([[0.0, 1.0], [9.0, 99.0]], dtype=np.float32)

    t = PrecipTransform("log1p")
    y = t(x)
    assert tuple(y.shape) == (1, 2, 2)
    np.testing.assert_allclose(
        y.numpy()[0],
        np.log1p(x),
        rtol=1e-6,
        atol=1e-6,
    )

    tz = PrecipTransform("log1p_zscore", mean=1.0, std=2.0)
    yz = tz(x)
    np.testing.assert_allclose(
        yz.numpy()[0],
        (np.log1p(x) - 1.0) / 2.0,
        rtol=1e-6,
        atol=1e-6,
    )

