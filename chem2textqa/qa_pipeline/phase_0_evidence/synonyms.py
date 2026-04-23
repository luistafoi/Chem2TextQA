"""Collect and filter per-compound synonyms for redaction.

For each CID we want a set of strings that reliably refer to the compound
in prose (common name, brand names, IUPAC name, MeSH terms, synonyms from
PubChem's curated CID-Synonym-filtered.gz). We then filter out tokens that
would cause false-positive matches (too short, too common, pure numbers
without hyphens, etc).

The output is used to (a) find sentences mentioning the compound and
(b) redact every mention to [COMPOUND] before handing text to an LLM.
"""
from __future__ import annotations

import gzip
import logging
import re
from pathlib import Path

from chem2textqa.qa_pipeline.config import (
    COMMON_WORD_BLOCKLIST,
    MAX_SYNONYMS_PER_COMPOUND,
    MIN_SYNONYM_LENGTH,
)

logger = logging.getLogger(__name__)


# Matches strings that are meaningful chemical names:
# must contain at least one alphabetic character.
_HAS_LETTER = re.compile(r"[A-Za-z]")


def is_usable_synonym(
    synonym: str,
    blocklist: frozenset[str] = COMMON_WORD_BLOCKLIST,
    min_length: int = MIN_SYNONYM_LENGTH,
) -> bool:
    """Return True if this synonym is safe for whole-word redaction.

    Drops tokens that would cause too many false positive matches:
      - too short
      - all digits / all punctuation
      - common English/chemistry words
      - pure structural strings (SMILES/InChI-like)
    """
    if not synonym:
        return False
    s = synonym.strip()
    if len(s) < min_length:
        return False
    if not _HAS_LETTER.search(s):
        return False
    if s.lower() in blocklist:
        return False
    # SMILES / InChI strings typically contain brackets, equals, slashes;
    # we want to keep names but not raw structures.
    if s.startswith("InChI="):
        return False
    # Raw SMILES heuristic: mostly non-alpha chars, lots of brackets/digits
    non_alpha = sum(1 for ch in s if not ch.isalpha())
    if len(s) >= 5 and non_alpha / len(s) > 0.5 and ("(" in s or "=" in s or "#" in s):
        return False
    return True


def collect_compound_synonyms(
    compound: dict,
    extra: set[str] | None = None,
) -> set[str]:
    """Gather synonyms from a single compound record (linked_compounds entry).

    Includes: name, iupac_name, mesh_terms, plus any extras passed in
    (typically pulled from CID-Synonym-filtered.gz).
    """
    out: set[str] = set()
    for field in ("name", "iupac_name"):
        v = (compound.get(field) or "").strip()
        if v:
            out.add(v)
    for t in compound.get("mesh_terms") or []:
        t = (t or "").strip()
        if t:
            out.add(t)
    if extra:
        out.update(extra)
    return out


def filter_synonyms(
    synonyms: set[str],
    blocklist: frozenset[str] = COMMON_WORD_BLOCKLIST,
    min_length: int = MIN_SYNONYM_LENGTH,
    max_count: int = MAX_SYNONYMS_PER_COMPOUND,
) -> list[str]:
    """Drop unusable synonyms; sort longest-first so redaction matches
    the most specific form first. Cap at max_count.
    """
    kept = [s for s in synonyms if is_usable_synonym(s, blocklist, min_length)]
    # Longest first so alternation in regex prefers specific → general
    kept.sort(key=lambda s: (-len(s), s.lower()))
    if len(kept) > max_count:
        kept = kept[:max_count]
    return kept


def load_pubchem_synonyms(
    path: Path,
    target_cids: set[int],
) -> dict[int, set[str]]:
    """Stream CID-Synonym-filtered.gz, keeping only synonyms for the target
    CIDs. File format: `CID<TAB>synonym` (one synonym per line, multiple
    lines per CID).
    """
    logger.info("Loading PubChem synonyms for %d target CIDs...", len(target_cids))
    result: dict[int, set[str]] = {}

    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            tab = line.find("\t")
            if tab < 0:
                continue
            cid_str = line[:tab]
            if not cid_str.isdigit():
                continue
            cid = int(cid_str)
            if cid not in target_cids:
                continue
            syn = line[tab + 1:].rstrip("\n").strip()
            if not syn:
                continue
            result.setdefault(cid, set()).add(syn)

    logger.info("Collected synonyms for %d / %d compounds",
                 len(result), len(target_cids))
    return result
