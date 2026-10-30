#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.eval.final_models import checkpoint_paths

MM_PER_IN = 25.4
WET_THRESHOLD_IN = 0.10
WET_THRESHOLD_MM = WET_THRESHOLD_IN * MM_PER_IN

PRECIP_LEVELS_IN = np.asarray([
    0.01, 0.10, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50,
    1.75, 2.00, 2.50, 3.00, 4.00, 5.00, 7.00, 10.00,
    15.00, 20.00,
], dtype=float)
PRECIP_COLORS_IN = [
    '#7fff00', '#00cd00', '#008b00', '#104e8b',
    '#1e90ff', '#00b2ee', '#00eeee', '#8968cd',
    '#912cee', '#8b008b', '#8b0000', '#cd0000',
    '#ee4000', '#ff7f00', '#cd8500', '#ffd700',
    '#ffff00', '#ffff02',
]
PRECIP_OVER_IN = '#ffaeb9'


def data_dir() -> Path:
    return Path(os.environ['DATA_DIR'])


def default_archive_dir() -> Path:
    return data_dir() / 'derived/evaluation/stageiv_reference_v1'


def default_stats_path() -> Path:
    return data_dir() / 'metadata/dataset_v1/precip_normalization_stats_v1.json'


def default_checkpoints() -> dict[str, Path]:
    return checkpoint_paths(data_dir())


def load_stageiv_archive(archive_dir: Path):
    cases = pd.read_csv(archive_dir / 'cases.csv')
    stageiv = np.load(archive_dir / 'stageiv_budget_qpf6_mm.npy', mmap_mode='r')
    if len(cases) != len(stageiv):
        raise RuntimeError(f'cases.csv has {len(cases)} rows but Stage-IV archive has {len(stageiv)} fields')
    return cases, stageiv


def _case_label(row: pd.Series) -> str:
    for col in ('valid_end', 'valid_time', 'valid'):
        if col in row.index and pd.notna(row[col]):
            try:
                ts = pd.to_datetime(row[col], utc=True)
                return ts.strftime('%HZ %b %d, %Y')
            except Exception:
                return str(row[col])
    return ''


def case_metadata(cases: pd.DataFrame, idx: int) -> dict[str, Any]:
    row = cases.iloc[int(idx)]
    out = {'case_index': int(idx), 'label': _case_label(row)}
    for col in ('forecast_hour', 'lead_hours', 'fhr', 'init_hour', 'valid_end', 'valid_time', 'valid'):
        if col in row.index and pd.notna(row[col]):
            v = row[col]
            if isinstance(v, np.generic):
                v = v.item()
            out[col] = v
    return out


def candidate_metrics(field_mm: np.ndarray) -> dict[str, float]:
    x = np.asarray(field_mm, dtype=np.float64)
    finite = np.isfinite(x)
    x = np.where(finite, np.maximum(x, 0.0), 0.0)
    h, w = x.shape
    wet = x >= WET_THRESHOLD_MM
    wet_count = int(wet.sum())
    wet_frac = wet_count / float(x.size)
    total = float(x.sum())
    max_mm = float(x.max())
    p99_mm = float(np.percentile(x[x > 0], 99)) if np.any(x > 0) else 0.0

    weights = np.where(wet, x, 0.0)
    sw = float(weights.sum())
    if sw > 0:
        yy, xx = np.indices(x.shape, dtype=np.float64)
        cy = float((weights * yy).sum() / sw)
        cx = float((weights * xx).sum() / sw)
        dy = (cy - (h - 1) / 2.0) / (h / 2.0)
        dx = (cx - (w - 1) / 2.0) / (w / 2.0)
        center_dist = float(min(1.5, math.hypot(dx, dy)))

        y = yy - cy
        xx0 = xx - cx
        cxx = float((weights * xx0 * xx0).sum() / sw)
        cyy = float((weights * y * y).sum() / sw)
        cxy = float((weights * xx0 * y).sum() / sw)
        vals = np.linalg.eigvalsh(np.array([[cxx, cxy], [cxy, cyy]], dtype=float))
        vals = np.maximum(vals, 1.0e-8)
        axis_ratio = float(np.sqrt(vals[-1] / vals[0]))
    else:
        center_dist = 9.0
        axis_ratio = 1.0

    by = max(1, int(round(0.12 * h)))
    bx = max(1, int(round(0.12 * w)))
    edge_mask = np.zeros_like(x, dtype=bool)
    edge_mask[:by, :] = True
    edge_mask[-by:, :] = True
    edge_mask[:, :bx] = True
    edge_mask[:, -bx:] = True
    edge_mass_fraction = float(x[edge_mask].sum() / total) if total > 0 else 1.0

    # Heuristic for a high-impact, centered, reasonably coherent/elongated first pilot.
    # It is only a case-discovery score, not a scientific metric.
    impact = np.log1p(max_mm / WET_THRESHOLD_MM) + 0.6 * np.log1p(p99_mm / WET_THRESHOLD_MM)
    center_term = max(0.0, 1.0 - center_dist)
    edge_term = max(0.0, 1.0 - edge_mass_fraction)
    line_term = min(1.0, max(0.0, (axis_ratio - 1.0) / 3.0))
    area_term = 1.0 if 0.01 <= wet_frac <= 0.30 else max(0.0, 1.0 - abs(wet_frac - 0.12) / 0.25)
    score = float(impact * (0.45 + 0.25 * center_term + 0.20 * edge_term + 0.10 * line_term) * area_term)

    return {
        'wet_pixels_ge_0p1in': wet_count,
        'wet_fraction': wet_frac,
        'max_mm': max_mm,
        'max_in': max_mm / MM_PER_IN,
        'p99_mm': p99_mm,
        'p99_in': p99_mm / MM_PER_IN,
        'total_mm_sum': total,
        'center_distance_normalized': center_dist,
        'edge_mass_fraction': edge_mass_fraction,
        'axis_ratio': axis_ratio,
        'selection_score': score,
    }


def rank_candidates(cases: pd.DataFrame, stageiv, min_wet_pixels: int = 64) -> pd.DataFrame:
    rows = []
    for i in range(len(stageiv)):
        m = candidate_metrics(stageiv[i])
        if m['wet_pixels_ge_0p1in'] < int(min_wet_pixels):
            continue
        rows.append({**case_metadata(cases, i), **m})
    if not rows:
        raise RuntimeError('No Stage-IV cases satisfy candidate criteria')
    return pd.DataFrame(rows).sort_values('selection_score', ascending=False).reset_index(drop=True)


def resolve_case_index(archive_dir: Path, case_index: int | None, min_wet_pixels: int = 64):
    cases, stageiv = load_stageiv_archive(archive_dir)
    if case_index is None or int(case_index) < 0:
        ranked = rank_candidates(cases, stageiv, min_wet_pixels=min_wet_pixels)
        idx = int(ranked.iloc[0]['case_index'])
    else:
        idx = int(case_index)
        if idx < 0 or idx >= len(stageiv):
            raise IndexError(f'case_index {idx} outside [0,{len(stageiv)-1}]')
        ranked = None
    field = np.asarray(stageiv[idx], dtype=np.float32)
    meta = {**case_metadata(cases, idx), **candidate_metrics(field)}
    return idx, field, meta, ranked


def read_log1p_stats(path: Path) -> tuple[float, float]:
    spec = json.loads(path.read_text())
    if 'log1p' not in spec:
        raise RuntimeError(f'{path} does not contain log1p normalization statistics')
    return float(spec['log1p']['mean']), float(spec['log1p']['std'])
