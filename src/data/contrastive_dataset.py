#!/usr/bin/env python3
"""
Materialize contrastive recipes into HRRR tensors.

Dry/correct-negative policy:
  - A/P are NEVER discarded merely because they are dry.
  - N is masked if its raw patch is exactly identical to A or exactly
    identical to P.
  - We do not replace that N; the loss receives negative_mask=False for that
    slot. This preserves the sampled climatological distribution while
    preventing contradictory "push identical fields apart" supervision.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .hrrr_archive import HRRRArchive
from .contrastive_sampler import ContrastiveRecipe


class PrecipTransform:
    """
    Supported modes:
      none
      log1p
      raw_zscore
      log1p_zscore

    Z-score mean/std must be derived from TRAINING data only.
    """

    def __init__(
        self,
        mode: str = "log1p",
        *,
        mean: float | None = None,
        std: float | None = None,
    ):
        valid = ("none", "log1p", "raw_zscore", "log1p_zscore")
        if mode not in valid:
            raise ValueError(f"mode must be one of {valid}, got {mode!r}")

        self.mode = mode
        self.mean = None if mean is None else float(mean)
        self.std = None if std is None else float(std)

        if "zscore" in mode:
            if self.mean is None or self.std is None:
                raise ValueError(f"{mode} requires mean and std")
            if not np.isfinite(self.std) or self.std <= 0:
                raise ValueError("std must be finite and > 0")

    @classmethod
    def from_stats_json(cls, mode: str, path: str | Path):
        spec = json.loads(Path(path).read_text())

        if mode == "raw_zscore":
            key = "raw"
        elif mode == "log1p_zscore":
            key = "log1p"
        else:
            return cls(mode)

        if key not in spec:
            raise RuntimeError(
                f"{path} does not contain a {key!r} normalization entry"
            )

        return cls(
            mode,
            mean=float(spec[key]["mean"]),
            std=float(spec[key]["std"]),
        )

    def __call__(self, raw: np.ndarray) -> torch.Tensor:
        x = np.asarray(raw, dtype=np.float32)

        finite = np.isfinite(x)
        if finite.any():
            minval = float(np.nanmin(x))
            if minval < -1.0e-5:
                raise RuntimeError(
                    f"Found materially negative QPF value {minval:.6g} mm"
                )

        x = np.where(np.isfinite(x), x, 0.0).astype(np.float32, copy=False)
        x = np.maximum(x, 0.0)

        if self.mode in ("log1p", "log1p_zscore"):
            x = np.log1p(x).astype(np.float32, copy=False)

        if self.mode in ("raw_zscore", "log1p_zscore"):
            x = ((x - self.mean) / self.std).astype(np.float32, copy=False)

        return torch.from_numpy(x[None, ...].copy())


def _array_equal(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape:
        return False
    same = (a == b) | (np.isnan(a) & np.isnan(b))
    return bool(np.all(same))


class ContrastiveDataset(Dataset):
    """Dataset whose indices are ContrastiveRecipe objects."""

    def __init__(
        self,
        archive: HRRRArchive,
        *,
        negative_count: int = 8,
        transform: PrecipTransform | None = None,
    ):
        self.archive = archive
        self.negative_count = int(negative_count)
        self.transform = transform or PrecipTransform("log1p")

        if self.negative_count < 1:
            raise ValueError("negative_count must be >= 1")

    def __len__(self):
        return self.archive.n_fields

    def __getitem__(self, recipe: ContrastiveRecipe):
        if not isinstance(recipe, ContrastiveRecipe):
            raise TypeError(
                "ContrastiveDataset expects ContrastiveRecipe objects from "
                "ContrastiveBatchSampler."
            )

        if len(recipe.negative_ids) != self.negative_count:
            raise RuntimeError(
                f"Recipe has {len(recipe.negative_ids)} negatives; "
                f"dataset expects {self.negative_count}."
            )

        a_raw = self.archive.read_patch(
            recipe.anchor_id, recipe.y0, recipe.x0, recipe.patch_size
        )
        p_raw = self.archive.read_patch(
            recipe.positive_id, recipe.y0, recipe.x0, recipe.patch_size
        )

        negatives = []
        negative_mask = []
        identical_to_anchor = []
        identical_to_positive = []

        for nid in recipe.negative_ids:
            n_raw = self.archive.read_patch(
                nid, recipe.y0, recipe.x0, recipe.patch_size
            )

            same_a = _array_equal(n_raw, a_raw)
            same_p = _array_equal(n_raw, p_raw)
            valid = not (same_a or same_p)

            negatives.append(self.transform(n_raw))
            negative_mask.append(valid)
            identical_to_anchor.append(same_a)
            identical_to_positive.append(same_p)

        return {
            "anchor": self.transform(a_raw),             # [1,H,W]
            "positive": self.transform(p_raw),           # [1,H,W]
            "negatives": torch.stack(negatives, dim=0),  # [K,1,H,W]
            "negative_mask": torch.tensor(
                negative_mask, dtype=torch.bool
            ),
            "negative_identical_to_anchor": torch.tensor(
                identical_to_anchor, dtype=torch.bool
            ),
            "negative_identical_to_positive": torch.tensor(
                identical_to_positive, dtype=torch.bool
            ),
            "anchor_raw_min_mm": float(np.nanmin(a_raw)),
            "anchor_raw_max_mm": float(np.nanmax(a_raw)),
            "positive_raw_min_mm": float(np.nanmin(p_raw)),
            "positive_raw_max_mm": float(np.nanmax(p_raw)),
            "metadata": asdict(recipe),
        }


def contrastive_collate(items):
    if not items:
        raise ValueError("Cannot collate an empty batch")

    return {
        "anchor": torch.stack([x["anchor"] for x in items], dim=0),
        "positive": torch.stack([x["positive"] for x in items], dim=0),
        "negatives": torch.stack([x["negatives"] for x in items], dim=0),
        "negative_mask": torch.stack(
            [x["negative_mask"] for x in items], dim=0
        ),
        "negative_identical_to_anchor": torch.stack(
            [x["negative_identical_to_anchor"] for x in items], dim=0
        ),
        "negative_identical_to_positive": torch.stack(
            [x["negative_identical_to_positive"] for x in items], dim=0
        ),
        "metadata": [x["metadata"] for x in items],
        "anchor_raw_min_mm": torch.tensor(
            [x["anchor_raw_min_mm"] for x in items], dtype=torch.float32
        ),
        "anchor_raw_max_mm": torch.tensor(
            [x["anchor_raw_max_mm"] for x in items], dtype=torch.float32
        ),
        "positive_raw_min_mm": torch.tensor(
            [x["positive_raw_min_mm"] for x in items], dtype=torch.float32
        ),
        "positive_raw_max_mm": torch.tensor(
            [x["positive_raw_max_mm"] for x in items], dtype=torch.float32
        ),
    }

