#!/usr/bin/env python3
"""
Metadata/index layer for HRRR contrastive sampling.

Scientific contract implemented here:
  A: eligible HRRR anchor from the requested split, F07-F17.
  P: same valid time, initialization +/-1 h from A.
     - P initialized 1 h earlier -> forecast lead A+1
     - P initialized 1 h later   -> forecast lead A-1
  N:
     - same split
     - same forecast lead as A
     - same valid UTC hour as A
     - strictly more than 90 days from A valid time
     - for climo45: different calendar year and within +/-45 circular
       climatological days of A.

No precipitation values are read here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


ANCHOR_FHR_MIN = 7
ANCHOR_FHR_MAX = 17


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin(
        ["true", "1", "t", "yes", "y"]
    )


def _parse_time_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        vals = pd.to_numeric(series, errors="coerce")
        finite = vals[np.isfinite(vals)]
        if len(finite) == 0:
            return pd.to_datetime(series, errors="coerce", utc=True)

        med = float(np.nanmedian(np.abs(finite)))
        if med < 1e11:
            unit = "s"
        elif med < 1e14:
            unit = "ms"
        elif med < 1e17:
            unit = "us"
        else:
            unit = "ns"

        return pd.to_datetime(vals, unit=unit, errors="coerce", utc=True)

    return pd.to_datetime(series, errors="coerce", utc=True)


def climo_day(ts: pd.Timestamp) -> int:
    """Map month/day to fixed leap year 2000 for annual-cycle distance."""
    t = pd.Timestamp(ts)
    return pd.Timestamp(year=2000, month=t.month, day=t.day).dayofyear


def circular_climo_distance(day_a, day_b):
    a = np.asarray(day_a, dtype=np.int16)
    b = np.asarray(day_b, dtype=np.int16)
    d = np.abs(a - b)
    return np.minimum(d, 366 - d)


def _detect_anchor_id_column(df: pd.DataFrame) -> str:
    for name in ("anchor_id", "sample_id", "field_id"):
        if name in df.columns:
            return name
    raise RuntimeError(
        "Could not identify anchor ID column. Expected one of "
        "'anchor_id', 'sample_id', or 'field_id'."
    )


def _detect_pool_id_column(df: pd.DataFrame) -> str:
    for name in ("sample_id", "field_id", "anchor_id", "negative_id"):
        if name in df.columns:
            return name
    raise RuntimeError(
        "Could not identify negative-pool ID column. Expected one of "
        "'sample_id', 'field_id', 'anchor_id', or 'negative_id'."
    )


def _normalize_manifest(df: pd.DataFrame) -> pd.DataFrame:
    required = ["sample_id", "init_hour", "forecast_hour", "valid_end"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"Manifest missing required columns {missing}. "
            f"Found: {list(df.columns)}"
        )

    d = df.copy()
    if "available" in d.columns:
        d = d.loc[_as_bool(d["available"])].copy()

    d["sample_id"] = d["sample_id"].astype(str)
    d["init_hour"] = pd.to_numeric(d["init_hour"], errors="raise").astype(int)
    d["forecast_hour"] = pd.to_numeric(
        d["forecast_hour"], errors="raise"
    ).astype(int)
    d["valid_time"] = _parse_time_series(d["valid_end"])

    d = d.dropna(subset=["sample_id", "valid_time"]).copy()

    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    d["valid_unix_s"] = (
        (d["valid_time"] - epoch).dt.total_seconds().round().astype(np.int64)
    )
    d["valid_hour_utc"] = d["valid_time"].dt.hour.astype(np.int8)
    d["valid_year"] = d["valid_time"].dt.year.astype(np.int16)
    d["climo_day"] = d["valid_time"].map(climo_day).astype(np.int16)

    if d["sample_id"].duplicated().any():
        raise RuntimeError("Manifest contains duplicate sample_id values.")

    return d


@dataclass(frozen=True)
class NegativeRules:
    mode: str = "climo45"
    min_days: float = 90.0
    climo_window_days: int = 45
    match_forecast_lead: bool = True
    match_valid_hour: bool = True
    different_year: bool = False

    def __post_init__(self):
        if self.mode not in ("calendar_any", "climo45"):
            raise ValueError(
                f"negative mode must be calendar_any or climo45, got {self.mode}"
            )
        if self.min_days < 0:
            raise ValueError("min_days must be nonnegative")
        if not (0 <= self.climo_window_days <= 183):
            raise ValueError("climo_window_days must be in [0, 183]")


class ContrastiveIndex:
    """Fast metadata lookup for anchors, positives, and negative candidates."""

    def __init__(
        self,
        eligible_anchors_path: str | Path,
        negative_pool_path: str | Path,
        manifest_path: str | Path,
    ):
        anchors_raw = pd.read_csv(eligible_anchors_path, low_memory=False)
        pool_raw = pd.read_csv(negative_pool_path, low_memory=False)
        manifest_raw = pd.read_csv(manifest_path, low_memory=False)
        self._initialize(anchors_raw, pool_raw, manifest_raw)

    @classmethod
    def from_frames(
        cls,
        anchors: pd.DataFrame,
        negative_pool: pd.DataFrame,
        manifest: pd.DataFrame,
    ):
        obj = cls.__new__(cls)
        obj._initialize(anchors.copy(), negative_pool.copy(), manifest.copy())
        return obj

    def _initialize(
        self,
        anchors_raw: pd.DataFrame,
        pool_raw: pd.DataFrame,
        manifest_raw: pd.DataFrame,
    ):
        manifest = _normalize_manifest(manifest_raw)
        meta_cols = [
            "sample_id",
            "init_hour",
            "forecast_hour",
            "valid_time",
            "valid_unix_s",
            "valid_hour_utc",
            "valid_year",
            "climo_day",
        ]

        aid_col = _detect_anchor_id_column(anchors_raw)
        a = anchors_raw.copy()
        a["anchor_id"] = a[aid_col].astype(str)

        if "split" not in a.columns:
            raise RuntimeError("Eligible-anchor index must contain a 'split' column.")

        keep_a = ["anchor_id", "split"]
        for optional in ("negative_candidate_count",):
            if optional in a.columns:
                keep_a.append(optional)
        a = a[keep_a].copy()

        m = manifest[meta_cols].rename(columns={"sample_id": "anchor_id"})
        a = a.merge(m, on="anchor_id", how="left", validate="many_to_one")

        if a["valid_time"].isna().any():
            bad = a.loc[a["valid_time"].isna(), "anchor_id"].head(10).tolist()
            raise RuntimeError(
                "Eligible anchors missing/unavailable in manifest, e.g. "
                + ", ".join(bad)
            )

        a = a.loc[
            a["forecast_hour"].between(ANCHOR_FHR_MIN, ANCHOR_FHR_MAX)
        ].copy()

        nid_col = _detect_pool_id_column(pool_raw)
        n = pool_raw.copy()
        n["negative_id"] = n[nid_col].astype(str)

        if "split" not in n.columns:
            raise RuntimeError("Negative-pool index must contain a 'split' column.")

        n = n[["negative_id", "split"]].copy()

        nm = manifest[meta_cols].rename(columns={"sample_id": "negative_id"})
        n = n.merge(nm, on="negative_id", how="left", validate="many_to_one")

        if n["valid_time"].isna().any():
            bad = n.loc[n["valid_time"].isna(), "negative_id"].head(10).tolist()
            raise RuntimeError(
                "Negative-pool fields missing/unavailable in manifest, e.g. "
                + ", ".join(bad)
            )

        n = n.loc[
            n["forecast_hour"].between(ANCHOR_FHR_MIN, ANCHOR_FHR_MAX)
        ].copy()

        self.manifest = manifest.set_index("sample_id", drop=False)
        self.anchors = a.reset_index(drop=True)
        self.negative_pool = n.reset_index(drop=True)

        self._field_by_valid_fhr: Dict[Tuple[int, int], str] = {
            (int(r.valid_unix_s), int(r.forecast_hour)): str(r.sample_id)
            for r in manifest.itertuples(index=False)
        }

        self._negative_groups = {}
        for key, group in self.negative_pool.groupby(
            ["split", "forecast_hour", "valid_hour_utc"],
            sort=False,
        ):
            self._negative_groups[
                (str(key[0]), int(key[1]), int(key[2]))
            ] = group[
                [
                    "negative_id",
                    "valid_unix_s",
                    "valid_year",
                    "climo_day",
                    "valid_time",
                ]
            ].reset_index(drop=True)

    def anchors_for_split(self, split: str) -> pd.DataFrame:
        out = self.anchors.loc[self.anchors["split"].astype(str) == str(split)].copy()
        if out.empty:
            raise RuntimeError(f"No eligible anchors found for split={split!r}")
        return out.reset_index(drop=True)

    def field_metadata(self, sample_id: str) -> pd.Series:
        sid = str(sample_id)
        if sid not in self.manifest.index:
            raise KeyError(sid)
        return self.manifest.loc[sid]

    def positive_ids(self, anchor_row) -> dict:
        valid_s = int(anchor_row["valid_unix_s"])
        fhr = int(anchor_row["forecast_hour"])

        out = {
            "earlier_init": self._field_by_valid_fhr.get((valid_s, fhr + 1)),
            "later_init": self._field_by_valid_fhr.get((valid_s, fhr - 1)),
        }

        if out["earlier_init"] is None or out["later_init"] is None:
            raise RuntimeError(
                f"Anchor {anchor_row['anchor_id']} was marked eligible but one "
                f"of its +/-1-h positives is missing: {out}"
            )
        return out

    def negative_candidates(
        self,
        anchor_row,
        rules: NegativeRules,
    ) -> pd.DataFrame:
        split = str(anchor_row["split"])
        fhr = int(anchor_row["forecast_hour"])
        hour = int(anchor_row["valid_hour_utc"])

        if not rules.match_forecast_lead or not rules.match_valid_hour:
            raise NotImplementedError(
                "This v1 index intentionally locks same forecast lead and "
                "same valid UTC hour. Relaxing either should be a separate "
                "explicit experiment."
            )

        key = (split, fhr, hour)
        group = self._negative_groups.get(key)

        empty_cols = [
            "negative_id",
            "valid_unix_s",
            "valid_year",
            "climo_day",
            "valid_time",
            "time_distance_days",
            "climo_distance_days",
        ]
        if group is None or group.empty:
            return pd.DataFrame(columns=empty_cols)

        a_s = int(anchor_row["valid_unix_s"])
        min_s = int(round(rules.min_days * 86400.0))

        dt_s = np.abs(group["valid_unix_s"].to_numpy(np.int64) - a_s)
        keep = dt_s > min_s

        # climo45 is explicitly an interannual hard-negative experiment.
        # calendar_any retains the original >90-day experiment unless
        # different_year=True is requested explicitly.
        if rules.mode == "climo45" or rules.different_year:
            keep &= (
                group["valid_year"].to_numpy(np.int16)
                != int(anchor_row["valid_year"])
            )

        cdist = circular_climo_distance(
            group["climo_day"].to_numpy(np.int16),
            int(anchor_row["climo_day"]),
        )

        if rules.mode == "climo45":
            keep &= cdist <= int(rules.climo_window_days)

        out = group.loc[keep].copy()
        out["time_distance_days"] = dt_s[keep] / 86400.0
        out["climo_distance_days"] = cdist[keep].astype(int)

        out = out.loc[
            out["negative_id"].astype(str) != str(anchor_row["anchor_id"])
        ].copy()

        return out.reset_index(drop=True)

    def validate_positive_relation(
        self,
        anchor_id: str,
        positive_id: str,
        direction: str,
    ) -> None:
        a = self.field_metadata(anchor_id)
        p = self.field_metadata(positive_id)

        if int(a["valid_unix_s"]) != int(p["valid_unix_s"]):
            raise AssertionError("A and P do not share valid time.")

        if direction == "earlier_init":
            if int(p["forecast_hour"]) != int(a["forecast_hour"]) + 1:
                raise AssertionError("earlier_init P does not have A lead + 1.")
            if (int(p["init_hour"]) - int(a["init_hour"])) % 24 != 23:
                raise AssertionError("earlier_init P is not one UTC hour earlier.")
        elif direction == "later_init":
            if int(p["forecast_hour"]) != int(a["forecast_hour"]) - 1:
                raise AssertionError("later_init P does not have A lead - 1.")
            if (int(p["init_hour"]) - int(a["init_hour"])) % 24 != 1:
                raise AssertionError("later_init P is not one UTC hour later.")
        else:
            raise AssertionError(f"Unknown positive direction {direction!r}")

