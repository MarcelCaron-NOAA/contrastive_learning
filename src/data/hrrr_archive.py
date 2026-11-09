#!/usr/bin/env python3
"""
Read native HRRR QPF6 patches from the daily ZipStore archive.

The reader is intentionally ignorant of the contrastive-learning experiment.
It only knows:
  sample_id -> daily store / init hour / forecast hour -> 2-D patch.

Each worker/process maintains its own small LRU cache of open ZipStores.
"""

from __future__ import annotations

from collections import OrderedDict
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


DEFAULT_GRID_SHAPE = (1059, 1799)


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin(
        ["true", "1", "t", "yes", "y"]
    )


class HRRRArchive:
    """Random-access reader for the canonical daily HRRR QPF6 ZipStores."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        array_name: str = "qpf6",
        first_forecast_hour: int = 6,
        grid_shape: Tuple[int, int] = DEFAULT_GRID_SHAPE,
        max_open_stores: int = 16,
    ):
        self.manifest_path = Path(manifest_path)
        self.array_name = array_name
        self.first_forecast_hour = int(first_forecast_hour)
        self.grid_shape = tuple(int(x) for x in grid_shape)
        self.max_open_stores = int(max_open_stores)

        df = pd.read_csv(self.manifest_path, low_memory=False)

        required = ["sample_id", "store", "init_hour", "forecast_hour"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise RuntimeError(
                f"HRRR manifest is missing required columns {missing}. "
                f"Found: {list(df.columns)}"
            )

        if "available" in df.columns:
            df = df.loc[_as_bool(df["available"])].copy()

        df["sample_id"] = df["sample_id"].astype(str)
        df["init_hour"] = pd.to_numeric(df["init_hour"], errors="raise").astype(int)
        df["forecast_hour"] = pd.to_numeric(
            df["forecast_hour"], errors="raise"
        ).astype(int)

        if df["sample_id"].duplicated().any():
            dup = df.loc[df["sample_id"].duplicated(False), "sample_id"].head(10)
            raise RuntimeError(
                "Duplicate sample_id values in HRRR manifest, e.g. "
                + ", ".join(dup.astype(str))
            )

        self._meta = df.set_index("sample_id", drop=False)
        self._pid = os.getpid()
        self._cache: OrderedDict[str, tuple] = OrderedDict()

    @property
    def n_fields(self) -> int:
        return len(self._meta)

    def has_field(self, sample_id: str) -> bool:
        return str(sample_id) in self._meta.index

    def metadata(self, sample_id: str) -> pd.Series:
        sid = str(sample_id)
        if sid not in self._meta.index:
            raise KeyError(f"Unknown/unavailable HRRR sample_id: {sid}")
        return self._meta.loc[sid]

    def _ensure_process_local_cache(self) -> None:
        pid = os.getpid()
        if pid != self._pid:
            self._cache = OrderedDict()
            self._pid = pid

    def _open_root(self, store_path: str):
        import zarr

        self._ensure_process_local_cache()

        store_path = str(store_path)
        if store_path in self._cache:
            store, root = self._cache.pop(store_path)
            self._cache[store_path] = (store, root)
            return root

        store = zarr.ZipStore(store_path, mode="r")
        try:
            root = zarr.open_consolidated(store, mode="r")
        except Exception:
            root = zarr.open(store, mode="r")

        self._cache[store_path] = (store, root)

        while len(self._cache) > self.max_open_stores:
            _, (old_store, _) = self._cache.popitem(last=False)
            old_store.close()

        return root

    def read_patch(
        self,
        sample_id: str,
        y0: int,
        x0: int,
        size: int,
    ) -> np.ndarray:
        """Return one native HRRR patch as float32 [H, W]."""
        y0 = int(y0)
        x0 = int(x0)
        size = int(size)

        ny, nx = self.grid_shape
        if y0 < 0 or x0 < 0 or y0 + size > ny or x0 + size > nx:
            raise ValueError(
                f"Patch {(y0, x0, size)} is outside HRRR grid {self.grid_shape}"
            )

        row = self.metadata(sample_id)
        fhr = int(row["forecast_hour"])
        fidx = fhr - self.first_forecast_hour
        if fidx < 0:
            raise ValueError(
                f"forecast_hour={fhr} cannot be indexed with "
                f"first_forecast_hour={self.first_forecast_hour}"
            )

        root = self._open_root(row["store"])
        arr = root[self.array_name][
            int(row["init_hour"]),
            fidx,
            y0:y0 + size,
            x0:x0 + size,
        ]
        out = np.asarray(arr, dtype=np.float32)

        if out.shape != (size, size):
            raise RuntimeError(
                f"Unexpected patch shape for {sample_id}: "
                f"{out.shape}, expected {(size, size)}"
            )

        return out

    def close(self) -> None:
        for store, _ in self._cache.values():
            try:
                store.close()
            except Exception:
                pass
        self._cache.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cache"] = OrderedDict()
        state["_pid"] = os.getpid()
        return state

