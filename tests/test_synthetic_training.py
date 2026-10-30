#!/usr/bin/env python3
from __future__ import annotations

import pandas as pd

from src.training.synthetic import (
    SyntheticBatchSampler,
    configure_experiment,
    synthetic_bias,
)


class _FakeIndex:
    def __init__(self):
        self._anchors = pd.DataFrame(
            [
                {
                    "anchor_id": "2025040212_f18",
                    "split": "train",
                    "valid_time": "2025-04-03T00:00:00Z",
                    "forecast_hour": 18,
                }
            ]
        )

    def anchors_for_split(self, split):
        assert split == "train"
        return self._anchors


def test_syn3_and_syn8_family_definitions():
    syn3 = configure_experiment("syn3")
    assert syn3.families == ("displacement", "intensity", "area_size")

    syn8 = configure_experiment("syn8")
    assert syn8.families[:3] == syn3.families
    assert len(syn8.families) == 8
    assert "smoothing" in syn8.families
    assert "rotation_deg" in syn8.families


def test_wang_rank_mappings():
    assert synthetic_bias("displacement", 10, -1) == -10.0
    assert synthetic_bias("displacement", 10, +1) == +10.0
    assert synthetic_bias("intensity", 1, -1) == 0.95
    assert synthetic_bias("intensity", 10, -1) == 0.50
    assert synthetic_bias("intensity", 1, +1) == 1.10
    assert synthetic_bias("intensity", 10, +1) == 2.00
    assert synthetic_bias("area_size", 10, +1) == 2.00


def test_syn8_added_rank_mappings():
    assert synthetic_bias("smoothing", 10, +1) == 6.0
    assert synthetic_bias("object_dispersion_cells", 10, -1) == -16.0
    assert synthetic_bias("object_dispersion_cells", 10, +1) == +16.0
    assert synthetic_bias("principal_axis_log2_scale", 10, -1) == -1.0
    assert synthetic_bias("rotation_deg", 10, +1) == 90.0
    assert synthetic_bias("tile_fragmentation_cells", 10, +1) == 12.0


def test_shared_sampler_preserves_syn3_stream_for_shared_families():
    index = _FakeIndex()

    configure_experiment("syn3")
    syn3 = SyntheticBatchSampler(
        index,
        split="train",
        batch_size=2,
        patch_size=256,
        seed=20261014,
    )._recipe(index._anchors.iloc[0])

    configure_experiment("syn8")
    syn8 = SyntheticBatchSampler(
        index,
        split="train",
        batch_size=2,
        patch_size=256,
        seed=20261014,
    )._recipe(index._anchors.iloc[0])

    assert (syn3.y0, syn3.x0) == (syn8.y0, syn8.x0)
    assert syn3.ranks == syn8.ranks[:3]
    assert syn3.directions == syn8.directions[:3]
    assert syn3.biases == syn8.biases[:3]
