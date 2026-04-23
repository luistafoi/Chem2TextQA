"""Filter the built dataset to keep only high-quality, drug-relevant articles.

Input:  data/drug_articles.jsonl       (6.3M records from build-dataset)
Output: data/filtered/drug_articles_filtered.jsonl

Filters applied (all configurable):
  1. Language = English
  2. Abstract length >= 500 chars
  3. Not a retraction / erratum
  4. Not editorial / letter / news / comment
  5. At least one "specific" (non-generic) linked compound
  6. Compound actually mentioned in title/abstract OR as MeSH major topic
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from tqdm import tqdm

logger = logging.getLogger(__name__)


# Compounds that appear in >~1% of articles — ubiquitous physiological
# or cellular components that rarely make an article "about" them.
# An article linked ONLY to these is probably incidental.
GENERIC_COMPOUNDS: frozenset[str] = frozenset({
    # Electrolytes
    "Calcium", "Potassium", "Sodium", "Magnesium", "Iron",
    "Chloride", "Phosphate", "Bicarbonate",
    # Sugars / energy
    "D-Glucose", "Glucose Monohydrate", "Glucose", "Fructose",
    "Sucrose", "Lactose",
    # Ubiquitous small molecules
    "Water", "Oxygen", "Hydrogen", "Nitrogen",
    "Carbon Dioxide", "Carbon Monoxide", "Ammonia",
    # Hormones (abundant endogenous)
    "Insulin", "Corticotropin", "Cortisol", "Glucagon",
    "Norepinephrine", "Norepinephrine hydrochloride, (A+-)-",
    "Norepinephrine Bitartrate", "Epinephrine", "Dopamine",
    "Serotonin", "Thyroxine", "Testosterone", "Estradiol",
    "Progesterone",
    # Nucleotides / cofactors
    "Camp", "5'-Atp", "Adenosine Triphosphate", "ATP",
    "Adenosine Diphosphate", "Adenosine Monophosphate",
    "NADH", "NAD", "NADP", "NADPH",
    "Guanosine Triphosphate", "5'-Gtp", "GTP",
    # Nucleobases
    "Adenine", "Guanine", "Cytosine", "Thymine", "Uracil",
    "DNA", "RNA",
    # Common metabolites / cellular components
    "Cholesterol", "Ascorbic Acid", "Urea", "Creatinine",
    "Pyruvic Acid", "Lactic Acid", "Citric Acid",
    "Succinic Acid", "Acetic Acid", "Ethanol", "Methanol",
    # Amino acids (all 20)
    "Glycine", "Alanine", "Valine", "Leucine", "Isoleucine",
    "Proline", "Phenylalanine", "Tryptophan", "Methionine",
    "Serine", "Threonine", "Cysteine", "Tyrosine", "Asparagine",
    "Glutamine", "Lysine", "Arginine", "Histidine",
    "Aspartic Acid", "Glutamic Acid",
    "L-Tyrosine", "L-Tryptophan", "L-Leucine", "L-Lysine",
    "L-Methionine", "L-Alanine", "L-Arginine", "L-Serine",
})

# Publication types to filter out (not primary research content)
NON_PRIMARY_TYPES: frozenset[str] = frozenset({
    "Editorial", "Letter", "Comment", "Published Erratum",
    "Retraction of Publication", "Retracted Publication",
    "News", "Biography", "Autobiography", "Historical Article",
    "Interview", "Legal Case", "Newspaper Article",
    "Retraction Notice",
})


@dataclass
class FilterConfig:
    """Configuration for the cleanup pass."""
    language: str | None = "eng"
    min_abstract_chars: int = 500
    exclude_retracted: bool = True
    exclude_non_primary: bool = True
    require_specific_compound: bool = True
    require_compound_mention: bool = True
    generic_compounds: frozenset[str] = GENERIC_COMPOUNDS


@dataclass
class FilterStats:
    total: int = 0
    kept: int = 0
    dropped_language: int = 0
    dropped_short_abstract: int = 0
    dropped_retracted: int = 0
    dropped_non_primary: int = 0
    dropped_generic_only: int = 0
    dropped_no_mention: int = 0
    dropped_no_compounds: int = 0
    config: dict = field(default_factory=dict)

    def summary(self) -> str:
        if self.total == 0:
            return "No records processed."

        def pct(n: int) -> str:
            return f"{100*n/self.total:.1f}%"

        lines = [
            f"  Total records:            {self.total:>10,}",
            f"  Kept:                     {self.kept:>10,}  ({pct(self.kept)})",
            f"  Dropped (language):       {self.dropped_language:>10,}  ({pct(self.dropped_language)})",
            f"  Dropped (short abstract): {self.dropped_short_abstract:>10,}  ({pct(self.dropped_short_abstract)})",
            f"  Dropped (retracted):      {self.dropped_retracted:>10,}  ({pct(self.dropped_retracted)})",
            f"  Dropped (non-primary):    {self.dropped_non_primary:>10,}  ({pct(self.dropped_non_primary)})",
            f"  Dropped (no compounds):   {self.dropped_no_compounds:>10,}  ({pct(self.dropped_no_compounds)})",
            f"  Dropped (only generic):   {self.dropped_generic_only:>10,}  ({pct(self.dropped_generic_only)})",
            f"  Dropped (not mentioned):  {self.dropped_no_mention:>10,}  ({pct(self.dropped_no_mention)})",
        ]
        return "\n".join(lines)


# ------------------------------------------------------------------
# Filter predicates
# ------------------------------------------------------------------


def _passes_language(record: dict, config: FilterConfig) -> bool:
    if config.language is None:
        return True
    return record.get("language", "") == config.language


def _passes_abstract_length(record: dict, config: FilterConfig) -> bool:
    return len(record.get("abstract", "")) >= config.min_abstract_chars


def _is_retracted(record: dict) -> bool:
    pub_types = record.get("publication_types", []) or []
    return any("Retract" in pt or "Erratum" in pt for pt in pub_types)


def _is_non_primary(record: dict) -> bool:
    pub_types = set(record.get("publication_types", []) or [])
    return bool(pub_types & NON_PRIMARY_TYPES)


def _specific_compounds(record: dict, generics: frozenset[str]) -> list[dict]:
    """Return compounds that are not in the generic set."""
    compounds = record.get("linked_compounds", []) or []
    return [c for c in compounds if c.get("name", "") not in generics]


def _text_to_search(record: dict) -> str:
    """Lowercased title + abstract for compound mention matching."""
    title = record.get("title", "") or ""
    abstract = record.get("abstract", "") or ""
    return (title + " " + abstract).lower()


def _mesh_major_terms(record: dict) -> set[str]:
    """Lowercase set of MeSH descriptors tagged as major topic (* prefix)."""
    out: set[str] = set()
    for heading in record.get("mesh_headings", []) or []:
        if not heading.startswith("*"):
            continue
        # Strip '*' from both descriptor and subheading; take descriptor part
        clean = heading.replace("*", "")
        descriptor = clean.split("/", 1)[0].strip().lower()
        if descriptor:
            out.add(descriptor)
    return out


def _compound_is_mentioned(
    compound: dict,
    text: str,
    major_mesh: set[str],
) -> bool:
    """Check whether a compound is 'really discussed' in the article.

    True if:
      - the compound's name (or any MeSH synonym) appears in title/abstract, OR
      - the compound appears as a MeSH major topic (* marker)
    """
    # Gather candidate names: primary name + MeSH synonyms
    candidates = set()
    name = (compound.get("name") or "").strip()
    if name:
        candidates.add(name.lower())
    for syn in compound.get("mesh_terms", []) or []:
        syn = (syn or "").strip()
        if syn:
            candidates.add(syn.lower())

    # Major MeSH match
    if candidates & major_mesh:
        return True

    # Text match — require whole-word boundary to avoid substring hits
    for cand in candidates:
        if len(cand) < 3:
            continue
        pattern = r"\b" + re.escape(cand) + r"\b"
        if re.search(pattern, text):
            return True

    return False


# ------------------------------------------------------------------
# Main filter
# ------------------------------------------------------------------


def passes_filters(
    record: dict,
    config: FilterConfig,
    stats: FilterStats,
) -> bool:
    """Apply all configured filters to a single record. Updates stats."""
    if not _passes_language(record, config):
        stats.dropped_language += 1
        return False

    if not _passes_abstract_length(record, config):
        stats.dropped_short_abstract += 1
        return False

    if config.exclude_retracted and _is_retracted(record):
        stats.dropped_retracted += 1
        return False

    if config.exclude_non_primary and _is_non_primary(record):
        stats.dropped_non_primary += 1
        return False

    compounds = record.get("linked_compounds", []) or []
    if not compounds:
        stats.dropped_no_compounds += 1
        return False

    # Specific (non-generic) compounds
    specific = _specific_compounds(record, config.generic_compounds)
    if config.require_specific_compound and not specific:
        stats.dropped_generic_only += 1
        return False

    # Compound actually mentioned in text or MeSH major topic
    if config.require_compound_mention:
        text = _text_to_search(record)
        major_mesh = _mesh_major_terms(record)
        # At least one of the non-generic compounds must be mentioned
        check_set = specific if specific else compounds
        if not any(_compound_is_mentioned(c, text, major_mesh) for c in check_set):
            stats.dropped_no_mention += 1
            return False

    stats.kept += 1
    return True


def cleanup_dataset(
    input_path: Path,
    output_path: Path,
    config: FilterConfig | None = None,
    stats_path: Path | None = None,
) -> FilterStats:
    """Stream input JSONL, write filtered records to output."""
    config = config or FilterConfig()
    stats = FilterStats(config={
        k: (sorted(v) if isinstance(v, (set, frozenset)) else v)
        for k, v in asdict(config).items()
        if k != "generic_compounds"  # don't serialize the big set
    })

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Filtering %s → %s", input_path, output_path)

    with input_path.open("r", encoding="utf-8") as fin, \
         output_path.open("w", encoding="utf-8") as fout:

        for line in tqdm(fin, desc="Filtering", unit=" records"):
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            stats.total += 1

            if passes_filters(record, config, stats):
                fout.write(line + "\n")

    if stats_path:
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with stats_path.open("w", encoding="utf-8") as f:
            data = asdict(stats)
            json.dump(data, f, indent=2)

    return stats
