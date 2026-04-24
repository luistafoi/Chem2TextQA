"""Create an Option-A zip bundle of the ChemQA workspace.

Includes: scripts, eval/scripts, eval/results, eval/results_full, logs, data,
          checkpoints/*/final (final LoRA adapters only).
Excludes: hf_cache (large + contains HF token), intermediate trainer ckpts,
          .venv, cuda_home (symlinks), tensorboard runs.
"""
from __future__ import annotations
import os
import zipfile
from pathlib import Path
import fnmatch

ROOT = Path("/mnt/data_lab/ChemQA")
OUT = Path("/mnt/data_lab/bundles/ChemQA_bundle.zip")
OUT.parent.mkdir(parents=True, exist_ok=True)

INCLUDE = [
    "scripts",
    "logs",
    "data",
    "eval/scripts",
    "eval/results",
    "eval/results_full",
    "checkpoints/qwen2_5_14b_chemqa/final",
    "checkpoints/llama3_1_8b_chemqa/final",
    "checkpoints/gemma3_12b_chemqa/final",
]
EXCLUDE_GLOBS = [
    "*/hf_cache/*",
    "*/.venv/*",
    "*/cuda_home/*",
    "*/runs/*",
    "*/tb/*",
    "*/checkpoint-*",
    "*/hf_cache",
    "*token*",  # defensive: never ship the HF token
]


def skip(p: Path) -> bool:
    sp = str(p)
    for pat in EXCLUDE_GLOBS:
        if fnmatch.fnmatch(sp, pat):
            return True
    # Never ship the HF token file (name exactly "token" under any hf cache)
    if p.name == "token" and "hf_cache" in sp:
        return True
    return False


total_bytes = 0
total_files = 0
print(f"Writing {OUT}", flush=True)
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as zf:
    for rel in INCLUDE:
        base = ROOT / rel
        if not base.exists():
            print(f"  skip (missing): {rel}")
            continue
        if base.is_file():
            if skip(base):
                continue
            arcname = f"ChemQA/{rel}"
            zf.write(base, arcname=arcname)
            total_bytes += base.stat().st_size
            total_files += 1
        else:
            for root, dirs, files in os.walk(base, followlinks=False):
                # Prune dirs in place
                dirs[:] = [d for d in dirs if not skip(Path(root) / d)]
                for fn in files:
                    fp = Path(root) / fn
                    if skip(fp):
                        continue
                    arc = "ChemQA/" + str(fp.relative_to(ROOT))
                    try:
                        zf.write(fp, arcname=arc)
                        total_bytes += fp.stat().st_size
                        total_files += 1
                    except (OSError, ValueError) as e:
                        print(f"  warn skip {fp}: {e}")
        print(f"  added {rel}", flush=True)

print(f"\nfiles={total_files}  raw_bytes={total_bytes/1e9:.2f} GB  -> {OUT}")
print(f"zip size: {OUT.stat().st_size/1e9:.2f} GB")
