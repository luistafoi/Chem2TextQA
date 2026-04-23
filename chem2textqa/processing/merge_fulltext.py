"""Merge PMC full-text records into an existing filtered article JSONL.

Uses a two-pass byte-offset index so we don't have to load the 60 GB
pmc_fulltext.jsonl into memory.

Pass 1: stream pmc_fulltext.jsonl, build {pmid: byte_offset} index (~50 MB).
Pass 2: stream the input dataset. For each record missing `full_text`,
        seek() to the offset in pmc_fulltext.jsonl and read one line.

If a PMID appears multiple times in pmc_fulltext.jsonl (e.g. the same article
across oa_comm / oa_noncomm / oa_other subsets), we keep the offset that
points to the record with the longest `full_text`.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class MergeStats:
    total: int = 0
    already_had_fulltext: int = 0
    newly_merged: int = 0
    no_pmc_match: int = 0


def build_pmid_offset_index(pmc_path: Path) -> dict[int, int]:
    """Build a {pmid: byte_offset} index over pmc_fulltext.jsonl.

    For duplicate PMIDs across PMC subsets, keep the offset whose record has
    the longest full_text (best chance of actually useful content).
    """
    logger.info("Building PMID→offset index over %s...", pmc_path)
    best: dict[int, tuple[int, int]] = {}  # pmid -> (offset, len_fulltext)
    offset = 0

    with pmc_path.open("rb") as f:
        for line in tqdm(f, desc="Indexing PMC", unit=" lines"):
            try:
                rec = json.loads(line)
                pmid_raw = rec.get("pmid")
                if pmid_raw is not None:
                    pmid = int(pmid_raw)
                    ft_len = len(rec.get("full_text") or "")
                    prev = best.get(pmid)
                    if prev is None or ft_len > prev[1]:
                        best[pmid] = (offset, ft_len)
            except (json.JSONDecodeError, ValueError, TypeError):
                pass
            offset += len(line)

    offsets = {pmid: off for pmid, (off, _) in best.items()}
    logger.info("Indexed %d unique PMIDs", len(offsets))
    return offsets


def merge_pmc_fulltext(
    input_path: Path,
    output_path: Path,
    pmc_path: Path,
) -> MergeStats:
    """Stream input JSONL, fill missing full_text/sections from pmc_fulltext.

    Preserves existing full_text when already populated.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    pmc_path = Path(pmc_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    offsets = build_pmid_offset_index(pmc_path)
    stats = MergeStats()

    with pmc_path.open("rb") as pmc_f, \
         input_path.open("r", encoding="utf-8") as fin, \
         output_path.open("w", encoding="utf-8") as fout:

        for line in tqdm(fin, desc="Merging", unit=" records"):
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            stats.total += 1

            # Already has full text — skip merge
            if rec.get("full_text"):
                stats.already_had_fulltext += 1
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue

            # Look up in PMC index
            pmid_raw = rec.get("pmid")
            try:
                pmid = int(pmid_raw) if pmid_raw is not None else None
            except (ValueError, TypeError):
                pmid = None

            if pmid is None or pmid not in offsets:
                stats.no_pmc_match += 1
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue

            pmc_f.seek(offsets[pmid])
            pmc_line = pmc_f.readline()
            try:
                pmc_rec = json.loads(pmc_line)
            except json.JSONDecodeError:
                stats.no_pmc_match += 1
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue

            full_text = pmc_rec.get("full_text") or ""
            if not full_text:
                stats.no_pmc_match += 1
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue

            rec["full_text"] = full_text
            rec["sections"] = pmc_rec.get("sections") or {}
            # Fill in pmcid / doi if the input record was missing them
            if not rec.get("pmcid") and pmc_rec.get("pmcid"):
                rec["pmcid"] = pmc_rec["pmcid"]
            if not rec.get("doi") and pmc_rec.get("doi"):
                rec["doi"] = pmc_rec["doi"]

            stats.newly_merged += 1
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return stats
