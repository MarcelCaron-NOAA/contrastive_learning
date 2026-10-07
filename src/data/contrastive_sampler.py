#!/usr/bin/env python3
"""
Epoch-aware sampler for natural HRRR contrastive relationships.

One epoch:
  1. takes every eligible anchor exactly once,
  2. globally shuffles anchors,
  3. independently chooses +/-1-h P direction per anchor,
  4. independently chooses one 256x256 crop per anchor,
  5. samples K explicit N field IDs obeying the configured negative rules,
     without replacement across the minibatch.

Different samples in the same minibatch may use different leads, UTC hours,
seasons, and geographic crops. A/P/N *within one sample* use the same crop.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Iterator, Tuple

import numpy as np

from .contrastive_index import ContrastiveIndex, NegativeRules


DEFAULT_GRID_SHAPE = (1059, 1799)


def _stable_u32(text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "little", signed=False)


def _rng_for(base_seed: int, epoch: int, stream: int, key: str = ""):
    ss = np.random.SeedSequence(
        [
            int(base_seed),
            int(epoch),
            int(stream),
            int(_stable_u32(key)),
        ]
    )
    return np.random.default_rng(ss)


@dataclass(frozen=True)
class ContrastiveRecipe:
    anchor_id: str
    positive_id: str
    positive_direction: str
    negative_ids: Tuple[str, ...]

    y0: int
    x0: int
    patch_size: int

    split: str
    anchor_valid_unix_s: int
    anchor_valid_time: str
    anchor_forecast_hour: int
    anchor_valid_hour_utc: int
    anchor_climo_day: int
    negative_candidate_count: int
    epoch: int


class ContrastiveBatchSampler:
    """
    Yield lists of ContrastiveRecipe objects.

    Compatible with:
        DataLoader(dataset, batch_sampler=sampler, collate_fn=...)
    because Dataset.__getitem__ accepts a ContrastiveRecipe as its "index".
    """

    def __init__(
        self,
        index: ContrastiveIndex,
        *,
        split: str = "train",
        batch_size: int = 16,
        patch_size: int = 256,
        negative_count: int = 8,
        negative_rules: NegativeRules | None = None,
        seed: int = 20260930,
        grid_shape=DEFAULT_GRID_SHAPE,
        drop_last: bool = False,
    ):
        self.index = index
        self.split = str(split)
        self.batch_size = int(batch_size)
        self.patch_size = int(patch_size)
        self.negative_count = int(negative_count)
        self.rules = negative_rules or NegativeRules()
        self.seed = int(seed)
        self.grid_shape = tuple(int(x) for x in grid_shape)
        self.drop_last = bool(drop_last)
        self.epoch = 0

        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.negative_count < 1:
            raise ValueError("negative_count must be >= 1")

        ny, nx = self.grid_shape
        if self.patch_size < 1 or self.patch_size > ny or self.patch_size > nx:
            raise ValueError(
                f"Invalid patch_size={self.patch_size} for grid {self.grid_shape}"
            )

        self._anchors = self.index.anchors_for_split(self.split)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        n = len(self._anchors)
        if self.drop_last:
            return n // self.batch_size
        return math.ceil(n / self.batch_size)

    @property
    def n_anchors(self) -> int:
        return len(self._anchors)

    def _choose_positive_and_crop(self, row):
        aid = str(row["anchor_id"])
        positives = self.index.positive_ids(row)

        prng = _rng_for(self.seed, self.epoch, 11, aid)
        direction = (
            "earlier_init"
            if int(prng.integers(0, 2)) == 0
            else "later_init"
        )
        pid = str(positives[direction])

        crng = _rng_for(self.seed, self.epoch, 17, aid)
        ny, nx = self.grid_shape
        y0 = int(crng.integers(0, ny - self.patch_size + 1))
        x0 = int(crng.integers(0, nx - self.patch_size + 1))

        return direction, pid, y0, x0

    def __iter__(self) -> Iterator[list[ContrastiveRecipe]]:
        order_rng = _rng_for(self.seed, self.epoch, 3, "anchor_order")
        order = order_rng.permutation(len(self._anchors))

        for start in range(0, len(order), self.batch_size):
            idxs = order[start:start + self.batch_size]
            if len(idxs) < self.batch_size and self.drop_last:
                continue

            rows = [self._anchors.iloc[int(i)] for i in idxs]

            # First determine A/P/crops for the complete minibatch so negative
            # fields cannot duplicate any A or P used in this optimization step.
            prepared = []
            forbidden = set()

            for row in rows:
                direction, pid, y0, x0 = self._choose_positive_and_crop(row)
                aid = str(row["anchor_id"])

                prepared.append((row, aid, pid, direction, y0, x0))
                forbidden.add(aid)
                forbidden.add(pid)

            used_negative_ids = set()
            recipes = []

            for row, aid, pid, direction, y0, x0 in prepared:
                candidates = self.index.negative_candidates(row, self.rules)

                if len(candidates) < self.negative_count:
                    raise RuntimeError(
                        f"Anchor {aid} has only {len(candidates)} candidates "
                        f"under {self.rules}; need {self.negative_count}."
                    )

                nrng = _rng_for(self.seed, self.epoch, 23, aid)
                perm = nrng.permutation(len(candidates))

                chosen = []
                for ii in perm:
                    nid = str(candidates.iloc[int(ii)]["negative_id"])
                    if nid in forbidden or nid in used_negative_ids:
                        continue

                    chosen.append(nid)
                    used_negative_ids.add(nid)

                    if len(chosen) == self.negative_count:
                        break

                if len(chosen) < self.negative_count:
                    raise RuntimeError(
                        f"Could not draw {self.negative_count} unique minibatch "
                        f"negatives for anchor {aid}; got {len(chosen)}. "
                        "Reduce batch_size/negative_count or inspect pool size."
                    )

                recipes.append(
                    ContrastiveRecipe(
                        anchor_id=aid,
                        positive_id=pid,
                        positive_direction=direction,
                        negative_ids=tuple(chosen),
                        y0=y0,
                        x0=x0,
                        patch_size=self.patch_size,
                        split=str(row["split"]),
                        anchor_valid_unix_s=int(row["valid_unix_s"]),
                        anchor_valid_time=str(row["valid_time"]),
                        anchor_forecast_hour=int(row["forecast_hour"]),
                        anchor_valid_hour_utc=int(row["valid_hour_utc"]),
                        anchor_climo_day=int(row["climo_day"]),
                        negative_candidate_count=int(len(candidates)),
                        epoch=self.epoch,
                    )
                )

            if recipes:
                yield recipes

