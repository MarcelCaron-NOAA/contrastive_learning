#!/usr/bin/env python3
"""Evaluate SYN-3 v2, SYN-8 v2, and NFV on the common Stage-IV reference archive.

Writes one per-case table used by all gallery bucket selections. Learned scores are
raw cosine similarities of the normalized projection embeddings. Traditional
metrics are computed directly from HRRR forecast and Stage-IV BUDGET patches.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.data import PrecipTransform
from src.eval.verification_metrics import contingency, fss
from src.models.checkpoint_loader import load_model_checkpoint
from src.eval.final_models import (
    checkpoint_paths, normalization_stats, THRESHOLD_0P1_IN_MM, THRESHOLD_0P5_IN_MM, MM_PER_IN,
)


def defaults():
    data = Path(os.environ["DATA_DIR"])
    ckpt = checkpoint_paths(data)
    return {
        "archive": data / "derived/evaluation/stageiv_reference_v1",
        "syn3": ckpt["syn3"],
        "syn8": ckpt["syn8"],
        "nfv": ckpt["nfv"],
        "stats": normalization_stats(data),
        "out": data / "derived/evaluation/final_syn_nfv_stageiv_v1",
    }


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def checkpoint_step(ckpt: dict) -> int:
    for key in ("fine_tune_step", "step", "global_step"):
        if key in ckpt:
            try:
                return int(ckpt[key])
            except Exception:
                pass
    return -1


@torch.inference_mode()
def encode(model, raw: np.ndarray, ptransform, device: torch.device, batch_size: int) -> np.ndarray:
    pieces = []
    for s in range(0, len(raw), batch_size):
        e = min(len(raw), s + batch_size)
        x = torch.stack([ptransform(v) for v in raw[s:e]], dim=0).to(device, non_blocking=True)
        _, z = model(x)
        pieces.append(F.normalize(z.float(), dim=-1).cpu().numpy())
        if s == 0 or e == len(raw) or (s // batch_size) % 50 == 0:
            print(f"    encoded {e:6d}/{len(raw):6d}", flush=True)
    return np.concatenate(pieces, axis=0)


def infer_similarity(label, checkpoint, fcst, obs, ptransform, device, batch_size):
    print(f"\nLoading {label}: {checkpoint}", flush=True)
    model, ckpt = load_model_checkpoint(checkpoint, device=device)
    model.eval()
    print(f"  {label}: HRRR embeddings", flush=True)
    zf = encode(model, fcst, ptransform, device, batch_size)
    print(f"  {label}: Stage-IV embeddings", flush=True)
    zo = encode(model, obs, ptransform, device, batch_size)
    sim = np.sum(zf * zo, axis=1).astype(np.float32)
    step = checkpoint_step(ckpt)
    del model, ckpt, zf, zo
    gc.collect()
    torch.cuda.empty_cache()
    return sim, step


def categorical_counts(f, o, threshold):
    h, m, fa, cn = contingency(f, o, threshold)
    return float(h), float(m), float(fa), float(cn)


def ets_from_counts(h, m, fa, cn):
    total = h + m + fa + cn
    if total <= 0:
        return np.nan
    href = ((h + fa) * (h + m)) / total
    den = h + m + fa - href
    return np.nan if den == 0 else float((h - href) / den)


def fbias_from_counts(h, m, fa):
    obs_yes = h + m
    return np.nan if obs_yes == 0 else float((h + fa) / obs_yes)


def pod_from_counts(h, m):
    den = h + m
    return np.nan if den == 0 else float(h / den)


def sr_from_counts(h, fa):
    den = h + fa
    return np.nan if den == 0 else float(h / den)


def main():
    d = defaults()
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive-dir", default=str(d["archive"]))
    ap.add_argument("--syn3-checkpoint", default=str(d["syn3"]))
    ap.add_argument("--syn8-checkpoint", default=str(d["syn8"]))
    ap.add_argument("--nfv-checkpoint", default=str(d["nfv"]))
    ap.add_argument("--stats-json", default=str(d["stats"]))
    ap.add_argument("--output-dir", default=str(d["out"]))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--fss-threshold-mm", type=float, default=THRESHOLD_0P1_IN_MM)
    ap.add_argument("--wet-threshold-mm", type=float, default=THRESHOLD_0P1_IN_MM)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required")

    arc = Path(args.archive_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    required = [
        arc / "cases.csv",
        arc / "hrrr_qpf6_mm.npy",
        arc / "stageiv_budget_qpf6_mm.npy",
        Path(args.syn3_checkpoint), Path(args.syn8_checkpoint), Path(args.nfv_checkpoint), Path(args.stats_json),
    ]
    for p in required:
        if not p.is_file():
            raise SystemExit(f"Required file missing: {p}")

    cases = pd.read_csv(arc / "cases.csv")
    fcst = np.load(arc / "hrrr_qpf6_mm.npy", mmap_mode="r")
    obs = np.load(arc / "stageiv_budget_qpf6_mm.npy", mmap_mode="r")
    if not (len(cases) == len(fcst) == len(obs)):
        raise RuntimeError(f"Archive length mismatch: cases={len(cases)} fcst={len(fcst)} obs={len(obs)}")

    device = torch.device("cuda")
    ptransform = PrecipTransform.from_stats_json("log1p_zscore", args.stats_json)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Stage-IV cases: {len(cases):,}")

    sim_syn3, step_syn3 = infer_similarity("SYN-3", Path(args.syn3_checkpoint), fcst, obs, ptransform, device, args.batch_size)
    sim_syn8, step_syn8 = infer_similarity("SYN-8", Path(args.syn8_checkpoint), fcst, obs, ptransform, device, args.batch_size)
    sim_nfv, step_nfv = infer_similarity("NFV", Path(args.nfv_checkpoint), fcst, obs, ptransform, device, args.batch_size)

    wt = float(args.wet_threshold_mm)
    fss_t = float(args.fss_threshold_mm)
    rows = []
    for i in range(len(cases)):
        f = np.asarray(fcst[i], dtype=np.float32)
        o = np.asarray(obs[i], dtype=np.float32)
        base = cases.iloc[i].to_dict()
        h01, m01, fa01, cn01 = categorical_counts(f, o, THRESHOLD_0P1_IN_MM)
        h05, m05, fa05, cn05 = categorical_counts(f, o, THRESHOLD_0P5_IN_MM)
        fwp = int(np.sum(f >= wt))
        owp = int(np.sum(o >= wt))

        rows.append({
            **base,
            "case_index": int(base.get("case_index", i)),
            "archive_row": int(i),
            "syn3": float(sim_syn3[i]),
            "syn8": float(sim_syn8[i]),
            "nfv": float(sim_nfv[i]),
            "fss5": fss(f, o, fss_t, 5),
            "fss10": fss(f, o, fss_t, 10),
            "ets_0p1": ets_from_counts(h01, m01, fa01, cn01),
            "ets_0p5": ets_from_counts(h05, m05, fa05, cn05),
            "fbias_0p1": fbias_from_counts(h01, m01, fa01),
            "fbias_0p5": fbias_from_counts(h05, m05, fa05),
            "pod_0p1": pod_from_counts(h01, m01),
            "sr_0p1": sr_from_counts(h01, fa01),
            "pod_0p5": pod_from_counts(h05, m05),
            "sr_0p5": sr_from_counts(h05, fa05),
            "fcst_wet_pixels": fwp,
            "obs_wet_pixels": owp,
            "fcst_wet_fraction": float(fwp / f.size),
            "obs_wet_fraction": float(owp / o.size),
            "fcst_mean_in": float(np.mean(f) / MM_PER_IN),
            "obs_mean_in": float(np.mean(o) / MM_PER_IN),
            "fcst_max_in": float(np.max(f) / MM_PER_IN),
            "obs_max_in": float(np.max(o) / MM_PER_IN),
        })
        if (i + 1) % 100 == 0 or i + 1 == len(cases):
            print(f"  traditional metrics {i+1:5d}/{len(cases):5d}", flush=True)

    df = pd.DataFrame(rows)
    csv_path = out / "stageiv_per_case_metrics_inches.csv.gz"
    df.to_csv(csv_path, index=False)

    meta = {
        "archive": str(arc),
        "learned_score": "raw cosine similarity of L2-normalized projection z embeddings",
        "fss_threshold_mm": fss_t,
        "gallery_wet_threshold_mm": wt,
        "traditional_thresholds_in": [0.1, 0.5],
        "traditional_thresholds_mm_internal": [THRESHOLD_0P1_IN_MM, THRESHOLD_0P5_IN_MM],
        "syn3": {"checkpoint": str(args.syn3_checkpoint), "sha256": sha256_file(args.syn3_checkpoint), "step": step_syn3},
        "syn8": {"checkpoint": str(args.syn8_checkpoint), "sha256": sha256_file(args.syn8_checkpoint), "step": step_syn8},
        "nfv": {"checkpoint": str(args.nfv_checkpoint), "sha256": sha256_file(args.nfv_checkpoint), "step": step_nfv},
        "n_cases": int(len(df)),
        "fbias_definition": "(hits + false alarms) / (hits + misses)",
    }
    (out / "run_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")

    print("\nFINAL STAGE-IV MODEL EVALUATION COMPLETE")
    print(f"SYN-3 mean: {df.syn3.mean():+.4f}")
    print(f"SYN-8 mean: {df.syn8.mean():+.4f}")
    print(f"NFV mean  : {df.nfv.mean():+.4f}")
    print(f"Output    : {csv_path}")


if __name__ == "__main__":
    main()

