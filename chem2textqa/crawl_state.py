"""
Persistent crawl state that tracks progress across runs.

Stores:
  - seen_pmids:      all PMIDs already downloaded (avoids duplicates)
  - seen_cids:       all compound CIDs already processed
  - discovered_terms: MeSH headings, chemical entities, and keywords
                      extracted from downloaded articles (used for expansion)
  - queried_terms:   terms already used as PubMed search queries
                      (avoids re-searching the same term)
  - stats:           running counters

The state is saved as a single JSON file so it survives between runs.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CrawlState:
    seen_pmids: set[int] = field(default_factory=set)
    seen_cids: set[int] = field(default_factory=set)
    discovered_terms: set[str] = field(default_factory=set)
    queried_terms: set[str] = field(default_factory=set)
    total_compounds: int = 0
    total_abstracts: int = 0
    total_full_texts: int = 0
    rounds_completed: int = 0

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "seen_pmids": sorted(self.seen_pmids),
            "seen_cids": sorted(self.seen_cids),
            "discovered_terms": sorted(self.discovered_terms),
            "queried_terms": sorted(self.queried_terms),
            "total_compounds": self.total_compounds,
            "total_abstracts": self.total_abstracts,
            "total_full_texts": self.total_full_texts,
            "rounds_completed": self.rounds_completed,
        }
        path.write_text(json.dumps(data, indent=2))
        logger.info("Saved crawl state to %s (%d PMIDs, %d terms)",
                     path, len(self.seen_pmids), len(self.discovered_terms))

    @classmethod
    def load(cls, path: Path) -> CrawlState:
        if not path.exists():
            logger.info("No existing crawl state at %s, starting fresh", path)
            return cls()

        data = json.loads(path.read_text())
        state = cls(
            seen_pmids=set(data.get("seen_pmids", [])),
            seen_cids=set(data.get("seen_cids", [])),
            discovered_terms=set(data.get("discovered_terms", [])),
            queried_terms=set(data.get("queried_terms", [])),
            total_compounds=data.get("total_compounds", 0),
            total_abstracts=data.get("total_abstracts", 0),
            total_full_texts=data.get("total_full_texts", 0),
            rounds_completed=data.get("rounds_completed", 0),
        )
        logger.info("Loaded crawl state: %d PMIDs, %d CIDs, %d terms, %d queried, %d rounds",
                     len(state.seen_pmids), len(state.seen_cids),
                     len(state.discovered_terms), len(state.queried_terms),
                     state.rounds_completed)
        return state

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def filter_new_pmids(self, pmids: list[int]) -> list[int]:
        """Return only PMIDs we haven't seen before."""
        return [p for p in pmids if p not in self.seen_pmids]

    def filter_new_cids(self, cids: list[int]) -> list[int]:
        """Return only CIDs we haven't processed before."""
        return [c for c in cids if c not in self.seen_cids]

    def mark_pmids(self, pmids: list[int]) -> None:
        self.seen_pmids.update(pmids)

    def mark_cids(self, cids: list[int]) -> None:
        self.seen_cids.update(cids)

    def new_terms(self) -> set[str]:
        """Return discovered terms not yet used as search queries."""
        return self.discovered_terms - self.queried_terms

    def extract_terms_from_document(self, doc_metadata: dict) -> set[str]:
        """Extract searchable MeSH terms from a document's metadata.

        Focuses on chemical/pharmacological headings that are useful
        for discovering more drug-related articles.
        """
        terms: set[str] = set()

        for heading in doc_metadata.get("mesh_headings", []):
            # Keep the main heading (before the /subheading)
            main = heading.strip("*").split("/")[0].strip()
            if not main:
                continue

            # Only keep headings that look like chemical/drug classes
            heading_lower = heading.replace("*", "").lower()
            if any(indicator in heading_lower for indicator in [
                "/pharmacology",
                "/therapeutic use",
                "/metabolism",
                "/toxicity",
                "/chemistry",
                "/adverse effects",
            ]):
                terms.add(main)

        return terms

    def summary(self) -> str:
        new = len(self.new_terms())
        return (
            f"Crawl state: {len(self.seen_pmids):,} PMIDs, "
            f"{len(self.seen_cids):,} CIDs, "
            f"{len(self.discovered_terms):,} terms "
            f"({new:,} unused), "
            f"{self.rounds_completed} rounds"
        )
