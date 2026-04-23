"""Build a clean drug/metabolite dataset from bulk PubChem + PubMed files.

Pipeline:
  1. Load compound index (filter to drug/metabolite compounds via CID-MeSH)
  2. Build PMID→[CIDs] reverse index
  3. Stream PubMed baseline XML files, extract matching articles
  4. Apply local QA category filter
  5. Tag each article with linked compounds
  6. Optionally merge in PMC full text
  7. Write to JSONL
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from chem2textqa.processing.compounds import (
    build_pmid_to_cids,
    load_compound_index,
)
from chem2textqa.processing.mesh_local import categorize_mesh_headings
from chem2textqa.processing.pubmed_xml import iter_pubmed_articles

logger = logging.getLogger(__name__)


@dataclass
class BuildStats:
    compounds_loaded: int = 0
    pmids_targeted: int = 0
    xml_files_processed: int = 0
    articles_seen: int = 0
    articles_matched: int = 0
    articles_written: int = 0
    articles_dropped_no_category: int = 0
    articles_dropped_no_abstract: int = 0
    fulltext_merged: int = 0


def build_dataset(
    bulk_dir: Path,
    output_path: Path,
    require_abstract: bool = True,
    require_qa_category: bool = True,
    fulltext_path: Path | None = None,
    max_articles: int | None = None,
    extra_sources: list[str] | None = None,
) -> BuildStats:
    """Run the full local processing pipeline.

    Args:
        bulk_dir: Directory containing CID-* files and pubmed_baseline/
        output_path: Where to write the final JSONL dataset
        require_abstract: Drop articles with no abstract text
        require_qa_category: Drop articles that don't match any QA category
        fulltext_path: Optional pmc_fulltext.jsonl to merge in
        max_articles: Optional cap for testing
        extra_sources: List of curated source aliases (DrugBank, HMDB, KEGG,
            ChEBI, BindingDB, ChEMBL) whose compound CIDs should be unioned
            with the MeSH-classified set. Requires that CID lists have been
            cached under <bulk_dir>/sources/<alias>.txt (see download_sources).

    Returns:
        BuildStats with counts.
    """
    bulk_dir = Path(bulk_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = BuildStats()

    # Phase 1: Compound index
    extra_cids: set[int] = set()
    if extra_sources:
        from chem2textqa.processing.sources import load_source_cids
        logger.info("Loading CIDs from extra sources: %s", extra_sources)
        extra_cids = load_source_cids(bulk_dir, extra_sources)
        logger.info("Extra source CIDs (union): %d", len(extra_cids))

    compounds = load_compound_index(
        bulk_dir, require_mesh=True, extra_cids=extra_cids,
    )
    stats.compounds_loaded = len(compounds)

    # Phase 2: PMID → [CIDs] reverse index
    target_cids = set(compounds.keys())
    pmid_to_cids, _ = build_pmid_to_cids(bulk_dir, target_cids)
    stats.pmids_targeted = len(pmid_to_cids)

    # Phase 3: Optional full-text index (PMID → full_text record)
    fulltext_index: dict[int, dict] = {}
    if fulltext_path and Path(fulltext_path).exists():
        fulltext_index = _load_fulltext_index(Path(fulltext_path), pmid_to_cids)
        logger.info("Loaded full text for %d PMIDs", len(fulltext_index))

    # Phase 4: Stream PubMed XML files
    pubmed_dir = bulk_dir / "pubmed_baseline"
    xml_files = sorted(pubmed_dir.glob("*.xml.gz"))
    if not xml_files:
        raise FileNotFoundError(f"No PubMed XML files in {pubmed_dir}")

    logger.info("Processing %d PubMed XML files...", len(xml_files))

    target_pmids = set(pmid_to_cids.keys())

    with output_path.open("w", encoding="utf-8") as out:
        with tqdm(total=len(xml_files), desc="PubMed XML") as pbar:
            for xml_file in xml_files:
                for article in iter_pubmed_articles(xml_file, target_pmids):
                    stats.articles_seen += 1

                    if require_abstract and not article.abstract:
                        stats.articles_dropped_no_abstract += 1
                        continue

                    qa_categories = categorize_mesh_headings(article.mesh_headings)
                    if require_qa_category and not qa_categories:
                        stats.articles_dropped_no_category += 1
                        continue

                    stats.articles_matched += 1

                    # Build the joined record
                    linked_cids = pmid_to_cids.get(article.pmid, [])
                    linked_compounds = [
                        compounds[cid].to_dict() for cid in linked_cids if cid in compounds
                    ]

                    record = article.to_dict()
                    record["qa_categories"] = qa_categories
                    record["linked_compounds"] = linked_compounds

                    # Merge full text if available
                    if article.pmid in fulltext_index:
                        ft = fulltext_index[article.pmid]
                        record["full_text"] = ft.get("full_text", "")
                        record["sections"] = ft.get("sections", {})
                        stats.fulltext_merged += 1

                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stats.articles_written += 1

                    if max_articles and stats.articles_written >= max_articles:
                        logger.info("Reached max_articles limit (%d). Stopping.",
                                     max_articles)
                        stats.xml_files_processed += 1
                        pbar.update(1)
                        return stats

                stats.xml_files_processed += 1
                pbar.update(1)

    return stats


def _load_fulltext_index(
    fulltext_path: Path,
    pmid_to_cids: dict[int, list[int]],
) -> dict[int, dict]:
    """Load PMC full-text JSONL into a {pmid: record} dict.

    Only keeps records whose PMID is in our target set (saves memory).
    """
    target_pmids_str = {str(p) for p in pmid_to_cids}
    index: dict[int, dict] = {}

    with fulltext_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            pmid_str = str(rec.get("pmid", ""))
            if pmid_str not in target_pmids_str:
                continue
            try:
                index[int(pmid_str)] = rec
            except ValueError:
                continue

    return index
