#!/usr/bin/env python3
"""Evaluate a synthetic-perturbation model and natural-variability V3 on Stage IV.

Defaults deliberately reuse the original stageiv_reference_v1 archive so gallery
comparisons are apples-to-apples with the earlier V1/V2/V3 Stage-IV diagnostics.
The learned scores written here are raw cosine similarities of L2-normalized
projection-space embeddings; they are not rescaled skill scores.
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
from src.eval.verification_metrics import contingency, fss, pearson_r, rmse
from src.models.checkpoint_loader import load_model_checkpoint


def defaults():
    data = Path(os.environ["DATA_DIR"])
    return {
        "archive": data / "derived/evaluation/stageiv_reference_v1",
        "v3": data / "derived/experiments/hard_ranking_v3_full_v1/checkpoint_best.pt",
        "synthetic": data / "derived/experiments/synthetic_perturb_all8_v1/checkpoint_best.pt",
        "stats": data / "metadata/dataset_v1/precip_normalization_stats_v1.json",
        "out": data / "derived/experiments/synthetic_perturb_all8_v1/evaluation/stageiv_vs_v3",
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
def encode(model, raw, ptransform, device, batch_size):
    out = []
    for s in range(0, len(raw), batch_size):
        e = min(len(raw), s + batch_size)
        x = torch.stack([ptransform(v) for v in raw[s:e]], dim=0)
        x = x.to(device, non_blocking=True)
        _, z = model(x)
        out.append(F.normalize(z.float(), dim=-1).cpu().numpy())
        if s == 0 or e == len(raw) or (s // batch_size) % 50 == 0:
            print(f"    encoded {e:6d}/{len(raw):6d}", flush=True)
    return np.concatenate(out, axis=0)


def infer_similarity(name, checkpoint, fcst, obs, ptransform, device, batch_size):
    print(f"\nLoading {name}: {checkpoint}", flush=True)
    model, ckpt = load_model_checkpoint(checkpoint, device=device)
    model.eval()
    print(f"  {name}: forecast embeddings", flush=True)
    zf = encode(model, fcst, ptransform, device, batch_size)
    print(f"  {name}: Stage-IV embeddings", flush=True)
    zo = encode(model, obs, ptransform, device, batch_size)
    sim = np.sum(zf * zo, axis=1)
    step = checkpoint_step(ckpt)
    del model, ckpt, zf, zo
    gc.collect()
    torch.cuda.empty_cache()
    return sim, step


def ets_score(f, o, threshold):
    h, m, fa, cn = contingency(f, o, threshold)
    total = h + m + fa + cn
    if total <= 0:
        return np.nan
    href = ((h + fa) * (h + m)) / float(total)
    denom = h + m + fa - href
    if denom == 0:
        return np.nan
    return float((h - href) / denom)


def success_ratio(f, o, threshold):
    h, _, fa, _ = contingency(f, o, threshold)
    den = h + fa
    return np.nan if den == 0 else float(h / den)


def main():
    d = defaults()
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive-dir", default=str(d["archive"]))
    ap.add_argument("--v3-checkpoint", default=str(d["v3"]))
    ap.add_argument("--synthetic-checkpoint", default=str(d["synthetic"]))
    ap.add_argument("--stats-json", default=str(d["stats"]))
    ap.add_argument("--output-dir", default=str(d["out"]))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--metric-threshold-mm", type=float, default=2.54,
                    help="Threshold for ETS/Success Ratio/FSS; 2.54 mm = 0.10 inch.")
    ap.add_argument("--wet-threshold-mm", type=float, default=0.1,
                    help="Threshold used only to flag non-dry gallery cases.")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required")

    arc = Path(args.archive_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cases_path = arc / "cases.csv"
    fcst_path = arc / "hrrr_qpf6_mm.npy"
    obs_path = arc / "stageiv_budget_qpf6_mm.npy"
    for p in (cases_path, fcst_path, obs_path, Path(args.v3_checkpoint), Path(args.synthetic_checkpoint), Path(args.stats_json)):
        if not p.is_file():
            raise SystemExit(f"Required file missing: {p}")

    cases = pd.read_csv(cases_path)
    fcst = np.load(fcst_path, mmap_mode="r")
    obs = np.load(obs_path, mmap_mode="r")
    if not (len(cases) == len(fcst) == len(obs)):
        raise RuntimeError(f"Archive length mismatch: cases={len(cases)}, fcst={len(fcst)}, obs={len(obs)}")

    print(f"Stage-IV cases: {len(cases):,}")
    ptransform = PrecipTransform.from_stats_json("log1p_zscore", args.stats_json)
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    sim_v3, v3_step = infer_similarity(
        "V3 natural-variability", Path(args.v3_checkpoint), fcst, obs,
        ptransform, device, args.batch_size,
    )
    sim_syn, syn_step = infer_similarity(
        "synthetic model", Path(args.synthetic_checkpoint), fcst, obs,
        ptransform, device, args.batch_size,
    )

    recs = []
    t = float(args.metric_threshold_mm)
    wt = float(args.wet_threshold_mm)
    for i in range(len(cases)):
        f = np.asarray(fcst[i], dtype=np.float32)
        o = np.asarray(obs[i], dtype=np.float32)
        r = cases.iloc[i].to_dict()
        fwp = int(np.sum(f >= wt))
        owp = int(np.sum(o >= wt))
        f5 = fss(f, o, t, 5)
        f10 = fss(f, o, t, 10)
        recs.append({
            **r,
            "case_index": int(r.get("case_index", i)),
            "learned_v3": float(sim_v3[i]),
            "learned_synthetic": float(sim_syn[i]),
            "delta_synthetic_minus_v3": float(sim_syn[i] - sim_v3[i]),
            "ets": ets_score(f, o, t),
            "success_ratio": success_ratio(f, o, t),
            "fss5": f5,
            "fss10": f10,
            "fss_mean": float(np.nanmean([f5, f10])),
            "corr": pearson_r(f, o),
            "rmse_mm": rmse(f, o),
            "fcst_wet_pixels": fwp,
            "obs_wet_pixels": owp,
            "fcst_wet_fraction": float(fwp / f.size),
            "obs_wet_fraction": float(owp / o.size),
            "fcst_mean_mm": float(np.mean(f)),
            "obs_mean_mm": float(np.mean(o)),
            "fcst_max_mm": float(np.max(f)),
            "obs_max_mm": float(np.max(o)),
        })
        if (i + 1) % 100 == 0 or i + 1 == len(cases):
            print(f"  traditional metrics {i+1:5d}/{len(cases):5d}", flush=True)

    df = pd.DataFrame(recs)
    df.to_csv(out / "per_case_metrics.csv.gz", index=False)

    meta = {
        "archive": str(arc),
        "v3_checkpoint": str(args.v3_checkpoint),
        "v3_checkpoint_sha256": sha256_file(args.v3_checkpoint),
        "v3_step": v3_step,
        "synthetic_checkpoint": str(args.synthetic_checkpoint),
        "synthetic_checkpoint_sha256": sha256_file(args.synthetic_checkpoint),
        "synthetic_step": syn_step,
        "metric_threshold_mm": t,
        "wet_threshold_mm": wt,
        "learned_score_definition": "raw cosine similarity of L2-normalized projection embeddings",
        "stageiv_regrid": "BUDGET",
        "n_cases": int(len(df)),
    }
    (out / "run_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")

    print("\nSTAGE-IV SYNTHETIC/V3 EVALUATION COMPLETE")
    print(f"V3 mean       : {df.learned_v3.mean():+.4f}")
    print(f"Synthetic mean: {df.learned_synthetic.mean():+.4f}")
    print(f"Output        : {out}")


if __name__ == "__main__":
    main()

