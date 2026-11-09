#!/usr/bin/env python3
"""
Read-only audit of the contrastive-learning precipitation project's data holdings.

Default behavior:
  * inventories important/preprocessed data areas
  * checks known critical Stage-IV and model artifacts
  * checks broken symlinks and zero-byte files
  * looks for manifests / file lists that reference missing local files
  * optionally validates NPY/NPZ/Zarr stores with --deep
  * writes a timestamped report under $WORK_DIR/audits/

It does NOT move, delete, rename, or modify project data.
"""

from __future__ import annotations
import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DEFAULT_PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_WORK_DIR = DEFAULT_PROJECT_DIR

IMPORTANT_TOPLEVEL_NAMES = {
    "processed", "preprocessed", "prepared", "derived",
    "features", "archives", "archive",
}
TEXT_MANIFEST_PATTERNS = (
    "manifest", "files_needed", "sample_plan", "sample-plan",
    "cases", "metadata",
)
KNOWN_CRITICAL = [
    ("stageiv_reference_dir", "derived/evaluation/stageiv_reference_v1"),
    ("stageiv_candidate_pairs", "derived/evaluation/stageiv_reference_v1/candidate_pairs.csv"),
    ("stageiv_files_needed_txt", "derived/evaluation/stageiv_reference_v1/stageiv_files_needed.txt"),
    ("stageiv_files_needed_csv", "derived/evaluation/stageiv_reference_v1/stageiv_files_needed.csv"),
    ("stageiv_hrrr_qpf6", "derived/evaluation/stageiv_reference_v1/hrrr_qpf6_mm.npy"),
    ("stageiv_budget_qpf6", "derived/evaluation/stageiv_reference_v1/stageiv_budget_qpf6_mm.npy"),
    ("stageiv_bilin_qpf6", "derived/evaluation/stageiv_reference_v1/stageiv_bilin_qpf6_mm.npy"),
    ("stageiv_cases", "derived/evaluation/stageiv_reference_v1/cases.csv"),
    ("syn3_checkpoint", "derived/experiments/synthetic_perturb_syn3_v2/checkpoint_best.pt"),
    ("syn8_checkpoint", "derived/experiments/synthetic_perturb_syn8_v2/checkpoint_best.pt"),
    ("nfv_checkpoint", "derived/experiments/hard_ranking_v3_full_v1/checkpoint_best.pt"),
    ("timelag_selection", "derived/experiments/timelag_lag_selection_v1"),
    ("timelag_selection_climo45", "derived/experiments/timelag_lag_selection_climo45_v1"),
]
PREPROCESSED_SUFFIXES = {
    ".npy", ".npz", ".nc", ".nc4", ".h5", ".hdf5",
    ".parquet", ".pt", ".pth", ".csv", ".json",
}
LOCAL_ABS_PATH_RE = re.compile(r"(?P<path>/scratch4/[^\s,'\";\]\[\)\(}{}<>]+)")

@dataclass
class CriticalStatus:
    label: str
    path: str
    exists: bool
    kind: str
    size_bytes: Optional[int]
    mtime_utc: Optional[str]
    note: str = ""

def fmt_bytes(n):
    if n is None:
        return "-"
    val = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if val < 1024.0 or unit == "PiB":
            return f"{val:,.2f} {unit}"
        val /= 1024.0

def utc_mtime(p: Path):
    try:
        return datetime.fromtimestamp(
            p.stat().st_mtime, tz=timezone.utc
        ).strftime("%Y-%m-%d %H:%M:%S UTC")
    except OSError:
        return None

def count_csv_rows(path: Path):
    try:
        with path.open("r", newline="", errors="replace") as f:
            return max(0, sum(1 for _ in f) - 1)
    except OSError:
        return None

def dir_summary(root: Path):
    total_bytes = 0
    n_files = 0
    n_dirs = 0
    zero_files = []
    broken_symlinks = []
    oldest = None
    newest = None
    suffix_counts = Counter()
    preprocessed_bytes = 0
    preprocessed_files = 0

    if not root.exists():
        return {
            "root": str(root), "exists": False, "files": 0, "dirs": 0,
            "bytes": 0, "zero_files": [], "broken_symlinks": [],
            "oldest_mtime": None, "newest_mtime": None,
            "suffix_counts": {}, "preprocessed_files": 0,
            "preprocessed_bytes": 0,
        }

    for cur, dirs, files in os.walk(root, followlinks=False):
        n_dirs += len(dirs)
        curp = Path(cur)
        for d in list(dirs):
            dp = curp / d
            if dp.is_symlink() and not dp.exists():
                broken_symlinks.append(str(dp))

        for name in files:
            p = curp / name
            try:
                if p.is_symlink() and not p.exists():
                    broken_symlinks.append(str(p))
                    continue
                st = p.stat()
            except OSError:
                continue

            n_files += 1
            total_bytes += st.st_size
            suf = p.suffix.lower()
            suffix_counts[suf or "<none>"] += 1
            if suf in PREPROCESSED_SUFFIXES or ".zarr/" in str(p):
                preprocessed_files += 1
                preprocessed_bytes += st.st_size
            if st.st_size == 0:
                zero_files.append(str(p))
            mt = st.st_mtime
            oldest = mt if oldest is None else min(oldest, mt)
            newest = mt if newest is None else max(newest, mt)

    def dt(x):
        return datetime.fromtimestamp(x, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        ) if x is not None else None

    return {
        "root": str(root), "exists": True, "files": n_files, "dirs": n_dirs,
        "bytes": total_bytes, "zero_files": zero_files,
        "broken_symlinks": broken_symlinks,
        "oldest_mtime": dt(oldest), "newest_mtime": dt(newest),
        "suffix_counts": dict(suffix_counts),
        "preprocessed_files": preprocessed_files,
        "preprocessed_bytes": preprocessed_bytes,
    }

def check_critical(data_dir: Path):
    rows = []
    for label, rel in KNOWN_CRITICAL:
        p = data_dir / rel
        exists = p.exists()
        kind = "missing"
        size = None
        note = ""
        if exists:
            if p.is_dir():
                kind = "dir"
            elif p.is_file():
                kind = "file"
                try:
                    size = p.stat().st_size
                except OSError:
                    pass
                if p.suffix.lower() == ".csv":
                    n = count_csv_rows(p)
                    if n is not None:
                        note = f"{n:,} data rows"
        rows.append(CriticalStatus(
            label, str(p), exists, kind, size,
            utc_mtime(p) if exists else None, note
        ))
    return rows

def validate_numpy(path: Path):
    try:
        import numpy as np
    except Exception as e:
        return False, f"numpy unavailable: {e}"
    try:
        if path.suffix.lower() == ".npy":
            arr = np.load(path, mmap_mode="r", allow_pickle=False)
            return True, f"shape={tuple(arr.shape)}, dtype={arr.dtype}"
        if path.suffix.lower() == ".npz":
            with np.load(path, allow_pickle=False) as z:
                keys = list(z.files)
            return True, f"{len(keys)} arrays: {keys[:8]}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    return True, "not a NumPy file"

def validate_zarr(path: Path):
    try:
        import zarr
    except Exception as e:
        meta = [path / ".zgroup", path / "zarr.json", path / ".zmetadata"]
        if any(p.exists() for p in meta):
            return True, f"metadata present; zarr import unavailable: {e}"
        return False, f"no obvious Zarr metadata; zarr import unavailable: {e}"
    try:
        obj = zarr.open(str(path), mode="r")
        if hasattr(obj, "shape"):
            return True, f"array shape={tuple(obj.shape)}, dtype={obj.dtype}"
        return True, f"group keys={list(obj.keys())[:12]}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

def deep_validate(root: Path, max_files=10000):
    results = []
    candidates = []
    for cur, dirs, files in os.walk(root, followlinks=False):
        curp = Path(cur)
        zarr_dirs = [d for d in dirs if d.endswith(".zarr")]
        candidates.extend(curp / d for d in zarr_dirs)
        dirs[:] = [d for d in dirs if not d.endswith(".zarr")]
        for name in files:
            p = curp / name
            if p.suffix.lower() in {".npy", ".npz"}:
                candidates.append(p)
        if len(candidates) >= max_files:
            break

    for p in candidates[:max_files]:
        if p.name.endswith(".zarr") and p.is_dir():
            ok, detail = validate_zarr(p)
            typ = "zarr"
        else:
            ok, detail = validate_numpy(p)
            typ = p.suffix.lower().lstrip(".")
        results.append((str(p), typ, ok, detail))
    return results

def find_manifest_files(search_roots):
    out = []
    seen = set()
    for root in search_roots:
        if not root.exists():
            continue
        for cur, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = [
                d for d in dirs
                if d not in {".git", "__pycache__", ".pytest_cache", ".mypy_cache"}
                and not d.startswith("env")
            ]
            for name in files:
                low = name.lower()
                if not any(tok in low for tok in TEXT_MANIFEST_PATTERNS):
                    continue
                p = Path(cur) / name
                try:
                    if p.stat().st_size > 20 * 1024 * 1024:
                        continue
                except OSError:
                    continue
                rp = str(p.resolve()) if p.exists() else str(p)
                if rp not in seen:
                    seen.add(rp)
                    out.append(p)
    return out

def extract_local_references(path: Path):
    refs = set()
    try:
        text = path.read_text(errors="replace")
    except Exception:
        return refs
    for m in LOCAL_ABS_PATH_RE.finditer(text):
        refs.add(m.group("path").rstrip(".,:"))
    return refs

def check_manifest_references(project_dir, data_dir, stageiv_root):
    manifest_files = find_manifest_files([project_dir, data_dir])
    rows = []
    for mf in manifest_files:
        for ref in sorted(extract_local_references(mf)):
            if any(tok in ref for tok in ("${", "{", "}", "$(")):
                continue
            p = Path(ref)
            rows.append((str(mf), ref, p.exists()))
        if (
            stageiv_root and
            "stageiv_files_needed" in mf.name.lower() and
            mf.suffix.lower() == ".txt"
        ):
            try:
                for line in mf.read_text(errors="replace").splitlines():
                    item = line.strip().split()[0] if line.strip() else ""
                    if not item or item.startswith("#") or item.startswith("/"):
                        continue
                    if item.startswith(("s3://", "http://", "https://")):
                        continue
                    p = stageiv_root / item
                    rows.append((str(mf), str(p), p.exists()))
            except Exception:
                pass
    return manifest_files, rows

def compare_stageiv_shapes(data_dir: Path):
    base = data_dir / "derived/evaluation/stageiv_reference_v1"
    cases = base / "cases.csv"
    arrays = [
        base / "hrrr_qpf6_mm.npy",
        base / "stageiv_budget_qpf6_mm.npy",
        base / "stageiv_bilin_qpf6_mm.npy",
    ]
    n_cases = count_csv_rows(cases) if cases.exists() else None
    result = []
    try:
        import numpy as np
    except Exception as e:
        return n_cases, [(str(p), None, f"numpy unavailable: {e}") for p in arrays]
    for p in arrays:
        if not p.exists():
            result.append((str(p), None, "MISSING"))
            continue
        try:
            a = np.load(p, mmap_mode="r", allow_pickle=False)
            shape = tuple(a.shape)
            if n_cases is None:
                note = "cases.csv unavailable"
            elif not shape:
                note = "scalar array (unexpected)"
            elif shape[0] != n_cases:
                note = f"WARNING: first dimension {shape[0]} != cases rows {n_cases}"
            else:
                note = "OK: first dimension matches cases.csv"
            result.append((str(p), shape, note))
        except Exception as e:
            result.append((str(p), None, f"ERROR: {type(e).__name__}: {e}"))
    return n_cases, result

def write_csv(path, header, rows):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

def main():
    ap = argparse.ArgumentParser(
        description="Audit contrastive-learning project data for missing/scrubbed files."
    )
    ap.add_argument("--project-dir", default=os.environ.get("PROJECT_DIR", str(DEFAULT_PROJECT_DIR)))
    ap.add_argument("--work-dir", default=os.environ.get("WORK_DIR", str(DEFAULT_WORK_DIR)))
    ap.add_argument("--data-dir", default=os.environ.get("DATA_DIR", ""))
    ap.add_argument("--scope", choices=("important", "all"), default="important")
    ap.add_argument("--deep", action="store_true")
    ap.add_argument("--output-dir", default="")
    args = ap.parse_args()

    project_dir = Path(args.project_dir)
    work_dir = Path(args.work_dir)
    data_dir = Path(args.data_dir) if args.data_dir else work_dir / "data"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.output_dir) if args.output_dir else work_dir / "audits" / f"data_audit_{ts}"
    outdir.mkdir(parents=True, exist_ok=False)
    stageiv_root = Path(os.environ["STAGEIV_ROOT"]) if os.environ.get("STAGEIV_ROOT") else None

    if args.scope == "all":
        scan_roots = [data_dir]
    else:
        scan_roots = []
        if data_dir.exists():
            for p in sorted(data_dir.iterdir()):
                if p.is_dir() and p.name.lower() in IMPORTANT_TOPLEVEL_NAMES:
                    scan_roots.append(p)
        derived = data_dir / "derived"
        if derived.exists() and derived not in scan_roots:
            scan_roots.append(derived)

    critical = check_critical(data_dir)
    inventory = []
    for root in scan_roots:
        print(f"Scanning metadata: {root}", flush=True)
        inventory.append(dir_summary(root))

    top_level_rows = []
    if data_dir.exists():
        for p in sorted(data_dir.iterdir()):
            try:
                st = p.lstat()
                top_level_rows.append((
                    p.name, "dir" if p.is_dir() else "file",
                    st.st_size, utc_mtime(p), p.is_symlink(), p.exists()
                ))
            except OSError:
                top_level_rows.append((p.name, "ERROR", "", "", "", ""))

    write_csv(outdir / "top_level_data_dir.csv",
              ["name","kind","entry_size_bytes","mtime_utc","is_symlink","exists"],
              top_level_rows)
    write_csv(outdir / "critical_status.csv",
              ["label","path","exists","kind","size_bytes","mtime_utc","note"],
              [[x.label,x.path,x.exists,x.kind,x.size_bytes,x.mtime_utc,x.note] for x in critical])

    zero_files = []
    broken_links = []
    for inv in inventory:
        zero_files.extend(inv["zero_files"])
        broken_links.extend(inv["broken_symlinks"])
    (outdir / "zero_byte_files.txt").write_text("\n".join(zero_files) + ("\n" if zero_files else ""))
    (outdir / "broken_symlinks.txt").write_text("\n".join(broken_links) + ("\n" if broken_links else ""))

    manifest_files, ref_rows = check_manifest_references(project_dir, data_dir, stageiv_root)
    write_csv(outdir / "manifest_reference_status.csv",
              ["manifest","referenced_local_path","exists"], ref_rows)
    (outdir / "manifest_files_examined.txt").write_text(
        "\n".join(map(str, manifest_files)) + ("\n" if manifest_files else "")
    )

    n_cases, stageiv_shapes = compare_stageiv_shapes(data_dir)
    write_csv(outdir / "stageiv_shape_check.csv",
              ["path","shape","status"],
              [[p,repr(shape),note] for p,shape,note in stageiv_shapes])

    deep_rows = []
    if args.deep:
        for root in scan_roots:
            print(f"Deep-validating array metadata: {root}", flush=True)
            deep_rows.extend(deep_validate(root))
        write_csv(outdir / "deep_validation.csv",
                  ["path","type","ok","detail"], deep_rows)

    with (outdir / "inventory.json").open("w") as f:
        json.dump({
            "timestamp": ts, "project_dir": str(project_dir),
            "work_dir": str(work_dir), "data_dir": str(data_dir),
            "scope": args.scope, "deep": args.deep, "inventory": inventory,
        }, f, indent=2)

    missing_critical = [x for x in critical if not x.exists]
    missing_refs = [(m,p) for m,p,ok in ref_rows if not ok]
    bad_deep = [r for r in deep_rows if not r[2]]

    lines = [
        "="*88,
        "CONTRASTIVE-LEARNING DATA AUDIT",
        "="*88,
        f"Project: {project_dir}",
        f"Work   : {work_dir}",
        f"Data   : {data_dir}",
        f"Scope  : {args.scope}",
        f"Deep   : {args.deep}",
        "",
        "CRITICAL ARTIFACTS",
        "-"*88,
    ]
    for x in critical:
        mark = "OK" if x.exists else "MISSING"
        extra = f" [{x.note}]" if x.note else ""
        sz = f" {fmt_bytes(x.size_bytes)}" if x.size_bytes is not None else ""
        lines.append(f"{mark:7s} {x.label:30s} {x.path}{sz}{extra}")

    lines += ["", "RECURSIVE INVENTORY", "-"*88]
    if not inventory:
        lines.append("No important/preprocessed roots were found to scan.")
    for inv in inventory:
        lines.append(
            f"{inv['root']}: {inv['files']:,} files, {inv['dirs']:,} dirs, "
            f"{fmt_bytes(inv['bytes'])}; preprocessed-like="
            f"{inv['preprocessed_files']:,} files/{fmt_bytes(inv['preprocessed_bytes'])}"
        )
        lines.append(
            f"    oldest mtime={inv['oldest_mtime']}  newest mtime={inv['newest_mtime']}"
        )

    lines += ["", "STAGE-IV CONSISTENCY", "-"*88]
    lines.append(f"cases.csv rows: {n_cases if n_cases is not None else 'unavailable'}")
    for p,shape,note in stageiv_shapes:
        lines.append(f"{p}: shape={shape} -> {note}")

    lines += ["", "MISSING / SUSPICIOUS SUMMARY", "-"*88]
    lines.append(f"Missing known critical artifacts : {len(missing_critical)}")
    lines.append(f"Broken symlinks                 : {len(broken_links)}")
    lines.append(f"Zero-byte files                 : {len(zero_files)}")
    lines.append(f"Missing local manifest refs     : {len(missing_refs)}")
    if args.deep:
        lines.append(f"Failed deep validations         : {len(bad_deep)}")

    if missing_refs:
        lines += ["", "FIRST 50 MISSING MANIFEST REFERENCES", "-"*88]
        for mf,p in missing_refs[:50]:
            lines.append(p)
            lines.append(f"    referenced by: {mf}")

    if args.deep and bad_deep:
        lines += ["", "FIRST 50 DEEP-VALIDATION FAILURES", "-"*88]
        for p,typ,ok,detail in bad_deep[:50]:
            lines.append(f"{p} [{typ}]: {detail}")

    lines += ["", "REPORT FILES", "-"*88]
    lines.extend(str(p) for p in sorted(outdir.iterdir()))
    summary = "\n".join(lines) + "\n"
    (outdir / "SUMMARY.txt").write_text(summary)
    print("\n" + summary)
    print(f"Audit report written to: {outdir}")

    if missing_critical or broken_links or bad_deep:
        sys.exit(2)

if __name__ == "__main__":
    main()

