#!/usr/bin/env python3
"""
Hard-ranking HRRR sampling for v2 contrastive fine-tuning.

Scientific design
-----------------
For each anchor A (HRRR QPF6, F07-F17), choose one deterministic/randomized
"branch" that is feasible all the way to a 6-h offset.

For low-lead anchors (F07-F11):
    branch_sign = +1
    P / TL3 / TL6 use the same valid time with larger forecast leads
      (earlier initializations).
    E1 / E2 / E3 / E6 use the same initialization with later valid times.

For high-lead anchors (F13-F17):
    branch_sign = -1
    P / TL3 / TL6 use the same valid time with smaller forecast leads
      (later initializations).
    E1 / E2 / E3 / E6 use the same initialization with earlier valid times.

For F12, if both branches are complete, choose +/- with 50/50 probability from
an anchor-specific RNG. If only one branch is complete (for example near a
split boundary), use the feasible branch.

Thus every logical sample contains:
    A, P, TL3, TL6, E1, E2, E3, E6, N1..NK

The TL chain is a same-valid-time initialization-lag ranking target:
    sim(A,P) > sim(A,TL3) > sim(A,TL6)

The E chain is same-initialization forecast-evolution ranking target:
    sim(A,E1) > sim(A,E2) > sim(A,E3) > sim(A,E6)

Global N fields retain the v1 negative contract. Training normally uses
climo45; validation normally uses calendar_any.

This module only defines recipes and materialization. Losses live separately.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from typing import Iterator, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data import ContrastiveIndex, NegativeRules, HRRRArchive, PrecipTransform


DEFAULT_GRID_SHAPE = (1059, 1799)
TL_OFFSETS = (1, 3, 6)
EVOLUTION_OFFSETS = (1, 2, 3, 6)


def _stable_u32(text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "little", signed=False)


def _rng_for(base_seed: int, epoch: int, stream: int, key: str = ""):
    ss = np.random.SeedSequence(
        [int(base_seed), int(epoch), int(stream), int(_stable_u32(key))]
    )
    return np.random.default_rng(ss)


def _array_equal(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape:
        return False
    same = (a == b) | (np.isnan(a) & np.isnan(b))
    return bool(np.all(same))


@dataclass(frozen=True)
class HardRankingRecipe:
    anchor_id: str
    positive_id: str
    tl3_id: str
    tl6_id: str
    e1_id: str
    e2_id: str
    e3_id: str
    e6_id: str
    negative_ids: Tuple[str, ...]

    branch_sign: int
    branch_name: str
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


class HardRankingBatchSampler:
    """Yield deterministic batches of HardRankingRecipe objects."""

    def __init__(
        self,
        index: ContrastiveIndex,
        *,
        split: str,
        batch_size: int = 8,
        patch_size: int = 256,
        negative_count: int = 4,
        negative_rules: NegativeRules | None = None,
        seed: int = 20261005,
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

        self._field_by_valid_fhr = {
            (int(r.valid_unix_s), int(r.forecast_hour)): str(r.sample_id)
            for r in self.index.manifest.itertuples(index=False)
        }

        anchors = self.index.anchors_for_split(self.split)
        self._split_min_valid_s = int(anchors["valid_unix_s"].min())
        self._split_max_valid_s = int(anchors["valid_unix_s"].max())

        self._branch_options = {}
        keep_rows = []
        for row in anchors.itertuples(index=False):
            row_dict = row._asdict()
            aid = str(row_dict["anchor_id"])
            options = []
            fhr = int(row_dict["forecast_hour"])

            candidate_signs = []
            if fhr < 12:
                candidate_signs = [+1]
            elif fhr > 12:
                candidate_signs = [-1]
            else:
                candidate_signs = [-1, +1]

            for sign in candidate_signs:
                ids = self._hard_ids_if_complete(row_dict, sign)
                if ids is not None:
                    options.append((int(sign), ids))

            if options:
                self._branch_options[aid] = tuple(options)
                keep_rows.append(row_dict)

        if not keep_rows:
            raise RuntimeError(
                f"No hard-ranking anchors remain for split={self.split!r}."
            )

        # Reconstruct as the same DataFrame type/columns without depending on
        # pandas in Dataset workers.
        import pandas as pd
        self._anchors = pd.DataFrame(keep_rows).reset_index(drop=True)

    def _hard_ids_if_complete(self, row, sign: int):
        aid = str(row["anchor_id"])
        valid_s = int(row["valid_unix_s"])
        fhr = int(row["forecast_hour"])

        ids = {"anchor_id": aid}

        # Same-valid-time initialization lags.
        for offset, name in ((1, "positive_id"), (3, "tl3_id"), (6, "tl6_id")):
            target_fhr = fhr + sign * offset
            sid = self._field_by_valid_fhr.get((valid_s, target_fhr))
            if sid is None:
                return None
            ids[name] = str(sid)

        # Same-initialization evolution fields. Keeping the init fixed means
        # valid time and forecast lead move together by the same signed offset.
        for offset, name in ((1, "e1_id"), (2, "e2_id"), (3, "e3_id"), (6, "e6_id")):
            target_valid_s = valid_s + sign * offset * 3600
            target_fhr = fhr + sign * offset

            # Strictly prevent evolution views from crossing the requested
            # split into a guard period or another split.
            if not (
                self._split_min_valid_s
                <= target_valid_s
                <= self._split_max_valid_s
            ):
                return None

            sid = self._field_by_valid_fhr.get((target_valid_s, target_fhr))
            if sid is None:
                return None
            ids[name] = str(sid)

        return ids

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @property
    def n_anchors(self) -> int:
        return len(self._anchors)

    @property
    def branch_counts(self):
        counts = {"forced_plus": 0, "forced_minus": 0, "f12_both": 0, "f12_one": 0}
        for row in self._anchors.itertuples(index=False):
            aid = str(row.anchor_id)
            opts = self._branch_options[aid]
            fhr = int(row.forecast_hour)
            if fhr < 12:
                counts["forced_plus"] += 1
            elif fhr > 12:
                counts["forced_minus"] += 1
            elif len(opts) == 2:
                counts["f12_both"] += 1
            else:
                counts["f12_one"] += 1
        return counts

    def __len__(self) -> int:
        n = len(self._anchors)
        if self.drop_last:
            return n // self.batch_size
        return math.ceil(n / self.batch_size)

    def _choose_branch_and_crop(self, row):
        aid = str(row["anchor_id"])
        options = self._branch_options[aid]

        if len(options) == 1:
            sign, ids = options[0]
        else:
            brng = _rng_for(self.seed, self.epoch, 31, aid)
            sign, ids = options[int(brng.integers(0, len(options)))]

        crng = _rng_for(self.seed, self.epoch, 37, aid)
        ny, nx = self.grid_shape
        y0 = int(crng.integers(0, ny - self.patch_size + 1))
        x0 = int(crng.integers(0, nx - self.patch_size + 1))
        return int(sign), dict(ids), y0, x0

    def __iter__(self) -> Iterator[list[HardRankingRecipe]]:
        order_rng = _rng_for(self.seed, self.epoch, 29, "anchor_order")
        order = order_rng.permutation(len(self._anchors))

        for start in range(0, len(order), self.batch_size):
            idxs = order[start:start + self.batch_size]
            if len(idxs) < self.batch_size and self.drop_last:
                continue

            rows = [self._anchors.iloc[int(i)] for i in idxs]
            prepared = []
            forbidden = set()

            for row in rows:
                sign, ids, y0, x0 = self._choose_branch_and_crop(row)
                hard_ids = {
                    ids["anchor_id"], ids["positive_id"], ids["tl3_id"], ids["tl6_id"],
                    ids["e1_id"], ids["e2_id"], ids["e3_id"], ids["e6_id"],
                }
                forbidden.update(hard_ids)
                prepared.append((row, sign, ids, y0, x0))

            used_negative_ids = set()
            recipes = []

            for row, sign, ids, y0, x0 in prepared:
                aid = str(ids["anchor_id"])
                candidates = self.index.negative_candidates(row, self.rules)
                if len(candidates) < self.negative_count:
                    raise RuntimeError(
                        f"Anchor {aid} has only {len(candidates)} global negative "
                        f"candidates under {self.rules}; need {self.negative_count}."
                    )

                nrng = _rng_for(self.seed, self.epoch, 41, aid)
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
                        f"Could not draw {self.negative_count} unique minibatch global "
                        f"negatives for anchor {aid}; got {len(chosen)}."
                    )

                recipes.append(
                    HardRankingRecipe(
                        anchor_id=aid,
                        positive_id=str(ids["positive_id"]),
                        tl3_id=str(ids["tl3_id"]),
                        tl6_id=str(ids["tl6_id"]),
                        e1_id=str(ids["e1_id"]),
                        e2_id=str(ids["e2_id"]),
                        e3_id=str(ids["e3_id"]),
                        e6_id=str(ids["e6_id"]),
                        negative_ids=tuple(chosen),
                        branch_sign=int(sign),
                        branch_name=("higher_lead" if sign > 0 else "lower_lead"),
                        y0=int(y0),
                        x0=int(x0),
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


class HardRankingDataset(Dataset):
    """Materialize HardRankingRecipe objects into transformed HRRR tensors."""

    TL_PAIRS = ((0, 1), (0, 2), (1, 2))
    E_PAIRS = (
        (0, 1), (0, 2), (0, 3),
        (1, 2), (1, 3),
        (2, 3),
    )

    def __init__(
        self,
        archive: HRRRArchive,
        *,
        negative_count: int = 4,
        transform: PrecipTransform | None = None,
    ):
        self.archive = archive
        self.negative_count = int(negative_count)
        self.transform = transform or PrecipTransform("log1p")

    def __len__(self):
        return self.archive.n_fields

    def __getitem__(self, recipe: HardRankingRecipe):
        if not isinstance(recipe, HardRankingRecipe):
            raise TypeError("HardRankingDataset expects HardRankingRecipe indices")
        if len(recipe.negative_ids) != self.negative_count:
            raise RuntimeError(
                f"Recipe has {len(recipe.negative_ids)} negatives; expected "
                f"{self.negative_count}."
            )

        def read(sid):
            return self.archive.read_patch(
                sid, recipe.y0, recipe.x0, recipe.patch_size
            )

        a_raw = read(recipe.anchor_id)
        p_raw = read(recipe.positive_id)
        tl3_raw = read(recipe.tl3_id)
        tl6_raw = read(recipe.tl6_id)
        e1_raw = read(recipe.e1_id)
        e2_raw = read(recipe.e2_id)
        e3_raw = read(recipe.e3_id)
        e6_raw = read(recipe.e6_id)

        tl_raw = (p_raw, tl3_raw, tl6_raw)
        e_raw = (e1_raw, e2_raw, e3_raw, e6_raw)

        tl_pair_mask = torch.tensor(
            [not _array_equal(tl_raw[i], tl_raw[j]) for i, j in self.TL_PAIRS],
            dtype=torch.bool,
        )
        evolution_pair_mask = torch.tensor(
            [not _array_equal(e_raw[i], e_raw[j]) for i, j in self.E_PAIRS],
            dtype=torch.bool,
        )

        negatives = []
        negative_mask = []
        for nid in recipe.negative_ids:
            n_raw = read(nid)
            valid = not (_array_equal(n_raw, a_raw) or _array_equal(n_raw, p_raw))
            negatives.append(self.transform(n_raw))
            negative_mask.append(valid)

        return {
            "anchor": self.transform(a_raw),
            "positive": self.transform(p_raw),
            "tl3": self.transform(tl3_raw),
            "tl6": self.transform(tl6_raw),
            "e1": self.transform(e1_raw),
            "e2": self.transform(e2_raw),
            "e3": self.transform(e3_raw),
            "e6": self.transform(e6_raw),
            "negatives": torch.stack(negatives, dim=0),
            "negative_mask": torch.tensor(negative_mask, dtype=torch.bool),
            "tl_pair_mask": tl_pair_mask,
            "evolution_pair_mask": evolution_pair_mask,
            "metadata": asdict(recipe),
        }


def hard_ranking_collate(items):
    if not items:
        raise ValueError("Cannot collate an empty batch")

    tensor_keys = (
        "anchor", "positive", "tl3", "tl6",
        "e1", "e2", "e3", "e6",
        "negatives", "negative_mask", "tl_pair_mask", "evolution_pair_mask",
    )
    out = {key: torch.stack([x[key] for x in items], dim=0) for key in tensor_keys}
    out["metadata"] = [x["metadata"] for x in items]
    return out

