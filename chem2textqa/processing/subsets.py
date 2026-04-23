"""Create quality tiers from the merged dataset.

Relevance signals per article:
  - major_topic: at least one linked compound (by name or MeSH synonym) is
    tagged with the MeSH major-topic marker (*) in this article
  - in_title:    at least one linked compound name/synonym appears in the title

Tiers:
  - premium:  major_topic = True
  - standard: major_topic = True OR in_title = True
  - broad:    pass-through (no additional filter)
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class SubsetStats:
    total: int = 0
    premium: int = 0     # major_topic = True
    standard: int = 0    # major_topic OR in_title
    broad: int = 0       # all records (pass-through)


VALID_TIERS = ("premium", "standard", "broad")


def _major_topic_descriptors(mesh_headings: list[str]) -> set[str]:
    """Extract lowercased MeSH descriptors tagged as major topic (* prefix)."""
    out: set[str] = set()
    for h in mesh_headings or []:
        if not h or not h.startswith("*"):
            continue
        clean = h.replace("*", "")
        desc = clean.split("/", 1)[0].strip().lower()
        if desc:
            out.add(desc)
    return out


def _compound_names(compound: dict) -> set[str]:
    """Return lowercased name + MeSH synonyms for a compound."""
    names: set[str] = set()
    n = (compound.get("name") or "").strip().lower()
    if n:
        names.add(n)
    for syn in compound.get("mesh_terms", []) or []:
        s = (syn or "").strip().lower()
        if s:
            names.add(s)
    return names


def classify_record(record: dict) -> tuple[bool, bool]:
    """Return (is_major_topic, is_in_title) for a record."""
    compounds = record.get("linked_compounds", []) or []
    if not compounds:
        return False, False

    title = (record.get("title") or "").lower()
    major_descriptors = _major_topic_descriptors(
        record.get("mesh_headings", []) or []
    )

    is_major = False
    is_in_title = False

    for c in compounds:
        names = _compound_names(c)
        if not names:
            continue
        if names & major_descriptors:
            is_major = True
        for name in names:
            if len(name) < 3:
                continue
            if re.search(r"\b" + re.escape(name) + r"\b", title):
                is_in_title = True
                break
        if is_major and is_in_title:
            break

    return is_major, is_in_title


def make_subsets(
    input_path: Path,
    premium_path: Path,
    standard_path: Path,
) -> SubsetStats:
    """Stream input JSONL and write two derived subsets.

    Premium:  major_topic
    Standard: major_topic OR in_title

    (The 'broad' tier is the input file itself; we don't duplicate it.)
    """
    input_path = Path(input_path)
    premium_path = Path(premium_path)
    standard_path = Path(standard_path)
    premium_path.parent.mkdir(parents=True, exist_ok=True)
    standard_path.parent.mkdir(parents=True, exist_ok=True)

    stats = SubsetStats()

    with input_path.open("r", encoding="utf-8") as fin, \
         premium_path.open("w", encoding="utf-8") as fp, \
         standard_path.open("w", encoding="utf-8") as fs:
        for line in tqdm(fin, desc="Subsetting", unit=" records"):
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            stats.total += 1
            stats.broad += 1

            is_major, is_title = classify_record(rec)

            if is_major or is_title:
                fs.write(line + "\n")
                stats.standard += 1

            if is_major:
                fp.write(line + "\n")
                stats.premium += 1

    return stats
