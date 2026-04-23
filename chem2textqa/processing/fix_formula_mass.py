"""One-shot utility to retrofit molecular_formula + molecular_weight into
an already-built JSONL dataset.

The initial build used CID-Mass.gz incorrectly (grabbed formula string and
tried to parse as float, so both fields came out empty). Re-running the full
build is ~7 hours. This script does a single streaming pass over a CID-Mass
lookup and patches the JSONL in-place.
"""
from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path

from tqdm import tqdm

logger = logging.getLogger(__name__)


def load_cid_mass_map(cid_mass_path: Path) -> dict[int, tuple[str, float | None]]:
    """Build {cid: (formula, mw)} from CID-Mass.gz.

    File format: CID<TAB>Formula<TAB>ExactMass<TAB>MonoIsotopicMass
    """
    logger.info("Loading %s...", cid_mass_path)
    result: dict[int, tuple[str, float | None]] = {}

    with gzip.open(cid_mass_path, "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2 or not parts[0].isdigit():
                continue
            cid = int(parts[0])
            formula = parts[1] if len(parts) > 1 else ""
            mw: float | None = None
            if len(parts) > 2:
                try:
                    mw = float(parts[2])
                except ValueError:
                    mw = None
            result[cid] = (formula, mw)

    logger.info("  Loaded %d CID→(formula, mw) entries", len(result))
    return result


def patch_jsonl(
    input_path: Path,
    output_path: Path,
    cid_mass_map: dict[int, tuple[str, float | None]],
) -> tuple[int, int]:
    """Stream records, fill in molecular_formula + molecular_weight for
    each linked_compound, write to output.

    Returns (records_written, compound_entries_patched).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = 0
    patched = 0

    with input_path.open("r", encoding="utf-8") as fin, \
         output_path.open("w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc="Patching", unit=" records"):
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            for c in rec.get("linked_compounds", []) or []:
                cid = c.get("cid")
                if cid is None:
                    continue
                info = cid_mass_map.get(int(cid))
                if not info:
                    continue
                formula, mw = info
                # Only fill if missing
                if not c.get("molecular_formula") and formula:
                    c["molecular_formula"] = formula
                    patched += 1
                if c.get("molecular_weight") in (None, 0) and mw is not None:
                    c["molecular_weight"] = mw

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            records += 1

    return records, patched
