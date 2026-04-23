"""Phase 0: build per-compound evidence bundles.

For each target compound (CID) we produce:
    {
      "cid": 3672,
      "name": "Ibuprofen",          # withheld from LLM
      "iupac_name": "...",          # withheld from LLM
      "smiles": "CC(C)CC1=...",     # SHOWN to LLM
      "molecular_formula": "C13H18O2",
      "molecular_weight": 206.28,
      "inchi_key": "...",           # withheld from LLM
      "num_synonyms": 415,
      "num_pmids": 20,
      "evidence_sentences": [
         {"id": 1, "pmid": "12345", "text": "[COMPOUND] inhibits COX-1..."},
         ...
      ],
    }

Algorithm:
  1. Stream the tier JSONL once to build CID → list[PMID, byte_offset] index.
  2. Build CID → set[synonyms] from compound.name, compound.mesh_terms, plus
     PubChem CID-Synonym-filtered.gz filtered for the target set.
  3. For each target CID:
       - compile redaction regex
       - for each of its PMIDs, seek to the article record, split its abstract
         + full_text into sentences, keep those that match, redact, dedupe
       - cap at MAX_EVIDENCE_SENTENCES_PER_COMPOUND
       - write one output JSONL record
"""
from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from tqdm import tqdm

from chem2textqa.qa_pipeline.config import (
    CID_SYNONYM_FILE,
    MAX_EVIDENCE_SENTENCES_PER_COMPOUND,
    MAX_SENTENCE_LENGTH,
)
from chem2textqa.qa_pipeline.phase_0_evidence.redact import (
    compile_redaction_regex,
    count_hits,
    redact,
)
from chem2textqa.qa_pipeline.phase_0_evidence.sentences import split_into_sentences
from chem2textqa.qa_pipeline.phase_0_evidence.synonyms import (
    collect_compound_synonyms,
    filter_synonyms,
    load_pubchem_synonyms,
)

logger = logging.getLogger(__name__)


@dataclass
class CompoundIndexEntry:
    """Where a compound appears in the dataset and its structural profile."""
    cid: int
    name: str = ""
    iupac_name: str = ""
    smiles: str = ""
    molecular_formula: str = ""
    molecular_weight: float | None = None
    inchi_key: str = ""
    mesh_terms: list[str] = field(default_factory=list)
    # List of (pmid, byte_offset_in_dataset)
    article_positions: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class ExtractStats:
    compounds_scanned: int = 0
    compounds_with_evidence: int = 0
    compounds_without_evidence: int = 0
    total_evidence_sentences: int = 0
    total_articles_scanned: int = 0


def build_compound_index(dataset_path: Path) -> dict[int, CompoundIndexEntry]:
    """Stream the dataset once, building a CID → CompoundIndexEntry map.

    Captures structure fields from the first time we see each compound and
    records the byte offset of every article the compound appears in.
    """
    logger.info("Building compound index from %s...", dataset_path)
    index: dict[int, CompoundIndexEntry] = {}

    offset = 0
    with dataset_path.open("rb") as f:
        for line in tqdm(f, desc="Indexing", unit=" articles"):
            raw = line
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                offset += len(raw)
                continue

            pmid = rec.get("pmid")
            try:
                pmid_int = int(pmid) if pmid is not None else None
            except (TypeError, ValueError):
                pmid_int = None

            for c in rec.get("linked_compounds", []) or []:
                cid_raw = c.get("cid")
                try:
                    cid = int(cid_raw) if cid_raw is not None else None
                except (TypeError, ValueError):
                    cid = None
                if cid is None:
                    continue

                entry = index.get(cid)
                if entry is None:
                    entry = CompoundIndexEntry(
                        cid=cid,
                        name=c.get("name") or "",
                        iupac_name=c.get("iupac_name") or "",
                        smiles=c.get("smiles") or "",
                        molecular_formula=c.get("molecular_formula") or "",
                        molecular_weight=c.get("molecular_weight"),
                        inchi_key=c.get("inchi_key") or "",
                        mesh_terms=list(c.get("mesh_terms") or []),
                    )
                    index[cid] = entry

                if pmid_int is not None:
                    entry.article_positions.append((pmid_int, offset))

            offset += len(raw)

    logger.info("Index: %d unique compounds across %s", len(index), dataset_path)
    return index


def iter_compound_evidence(
    dataset_path: Path,
    entry: CompoundIndexEntry,
    redaction_regex,
    rng: random.Random | None = None,
) -> Iterator[dict]:
    """Yield up to MAX_EVIDENCE_SENTENCES_PER_COMPOUND redacted evidence
    sentences for one compound, sampled uniformly across all matching
    sentences from all of the compound's articles.

    Two-pass: (1) scan every article, collect every matching redacted
    sentence (deduped by redacted text), (2) random.sample() down to the cap.
    This replaces the old greedy-by-byte-offset approach, which biased
    hints toward whichever articles happened to appear first in the tier
    file.

    Scans BOTH abstract and full_text. Dedup key is the redacted sentence
    itself, so abstract-repeats-in-body collapse.

    `rng` is a random.Random instance for reproducible sampling. If None,
    defaults to random.Random(entry.cid) so reruns of the same compound
    produce the same sample.
    """
    if rng is None:
        rng = random.Random(entry.cid)

    seen_sentences: set[str] = set()
    candidates: list[dict] = []

    with dataset_path.open("r", encoding="utf-8") as f:
        for pmid, byte_offset in entry.article_positions:
            f.seek(byte_offset)
            line = f.readline()
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            sources = []
            abstract = (rec.get("abstract") or "").strip()
            if abstract:
                sources.append(("abstract", abstract))
            full_text = (rec.get("full_text") or "").strip()
            if full_text:
                sources.append(("full_text", full_text))

            for source_name, body in sources:
                for s in split_into_sentences(body):
                    if len(s) < 40 or len(s) > MAX_SENTENCE_LENGTH:
                        continue
                    if count_hits(s, redaction_regex) == 0:
                        continue
                    redacted = redact(s, redaction_regex)
                    if redacted in seen_sentences:
                        continue
                    seen_sentences.add(redacted)
                    candidates.append({
                        "pmid": str(pmid),
                        "source": source_name,
                        "text": redacted,
                    })

    if len(candidates) > MAX_EVIDENCE_SENTENCES_PER_COMPOUND:
        candidates = rng.sample(candidates, MAX_EVIDENCE_SENTENCES_PER_COMPOUND)

    for i, c in enumerate(candidates, start=1):
        yield {"id": i, **c}


def build_evidence_record(
    entry: CompoundIndexEntry,
    synonyms: list[str],
    evidence: list[dict],
) -> dict:
    """Assemble the final per-compound evidence record."""
    pmids_used = sorted({e["pmid"] for e in evidence})
    return {
        "cid": entry.cid,
        "name": entry.name,
        "iupac_name": entry.iupac_name,
        "inchi_key": entry.inchi_key,
        "smiles": entry.smiles,
        "molecular_formula": entry.molecular_formula,
        "molecular_weight": entry.molecular_weight,
        "num_synonyms": len(synonyms),
        "num_pmids": len(pmids_used),
        "pmids": pmids_used,
        "synonyms_sample": synonyms[:10],  # for auditing
        "evidence_sentences": evidence,
    }


def run_phase_0(
    dataset_path: Path,
    output_path: Path,
    cid_synonym_file: Path = CID_SYNONYM_FILE,
    target_cids: set[int] | None = None,
) -> ExtractStats:
    """Run the full Phase 0 pipeline.

    Args:
        dataset_path: tier JSONL (e.g. drug_articles_v2_premium.jsonl)
        output_path: destination JSONL (one record per compound)
        cid_synonym_file: PubChem synonyms file (optional — if missing, we
            rely on name + mesh_terms only)
        target_cids: if provided, restrict to these CIDs (useful for pilots)
    """
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Pass 1: index
    index = build_compound_index(dataset_path)
    if target_cids is not None:
        index = {cid: e for cid, e in index.items() if cid in target_cids}
        logger.info("Filtered to %d target CIDs", len(index))

    # Pass 2: PubChem synonyms (streaming)
    pubchem_syns: dict[int, set[str]] = {}
    if cid_synonym_file.exists():
        pubchem_syns = load_pubchem_synonyms(cid_synonym_file, set(index.keys()))
    else:
        logger.warning("PubChem synonym file not found at %s — using only "
                       "name + mesh_terms", cid_synonym_file)

    stats = ExtractStats()
    stats.compounds_scanned = len(index)

    # Pass 3: per-compound evidence extraction
    with output_path.open("w", encoding="utf-8") as out:
        for cid, entry in tqdm(index.items(), desc="Extracting evidence",
                                unit=" compounds"):
            all_syns = collect_compound_synonyms(
                {
                    "name": entry.name,
                    "iupac_name": entry.iupac_name,
                    "mesh_terms": entry.mesh_terms,
                },
                extra=pubchem_syns.get(cid),
            )
            filtered = filter_synonyms(all_syns)
            if not filtered:
                stats.compounds_without_evidence += 1
                continue

            regex = compile_redaction_regex(filtered)
            evidence = list(iter_compound_evidence(dataset_path, entry, regex))
            stats.total_articles_scanned += len(entry.article_positions)

            if not evidence:
                stats.compounds_without_evidence += 1
                continue

            record = build_evidence_record(entry, filtered, evidence)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats.compounds_with_evidence += 1
            stats.total_evidence_sentences += len(evidence)

    logger.info("Phase 0 done: %d compounds with evidence, %d without",
                 stats.compounds_with_evidence, stats.compounds_without_evidence)
    return stats
