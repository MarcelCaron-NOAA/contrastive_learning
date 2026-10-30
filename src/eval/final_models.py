"""Canonical model/checkpoint registry for the final SYN-3/SYN-8/NFV study."""
from __future__ import annotations

import os
from pathlib import Path

MM_PER_IN = 25.4
THRESHOLD_0P1_IN_MM = 0.1 * MM_PER_IN
THRESHOLD_0P5_IN_MM = 0.5 * MM_PER_IN

FINAL_EXPERIMENT_IDS = {
    "syn3": "synthetic_perturb_syn3_v2",
    "syn8": "synthetic_perturb_syn8_v2",
    # Historical on-disk ID retained so existing trained checkpoints remain valid.
    "nfv": "hard_ranking_v3_full_v1",
}

DISPLAY_NAMES = {
    "syn3": "SYN-3",
    "syn8": "SYN-8",
    "nfv": "NFV",
}


def data_dir() -> Path:
    try:
        return Path(os.environ["DATA_DIR"])
    except KeyError as exc:
        raise RuntimeError("DATA_DIR is not set; source load_env.sh first") from exc


def checkpoint_paths(root: Path | None = None) -> dict[str, Path]:
    base = (root or data_dir()) / "derived" / "experiments"
    return {
        key: base / experiment_id / "checkpoint_best.pt"
        for key, experiment_id in FINAL_EXPERIMENT_IDS.items()
    }


def normalization_stats(root: Path | None = None) -> Path:
    base = root or data_dir()
    return base / "metadata" / "dataset_v1" / "precip_normalization_stats_v1.json"
