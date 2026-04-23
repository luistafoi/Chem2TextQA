"""
MeSH-subheading filters for categorizing PubMed articles into
structure-relevant QA categories.

After collecting PMIDs linked from PubChem compounds, we use these
filters to keep only articles that contain information answerable
from a chemical structure and to tag them by QA topic.

The filtering happens server-side on NCBI: we post our PMIDs to
the Entrez history server, then intersect with each category's
MeSH query using `#QueryKey AND <filter>`.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from urllib.error import HTTPError

from Bio import Entrez
from tqdm import tqdm

from chem2textqa.config.settings import Settings
from chem2textqa.utils.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

POST_BATCH = 5000  # max PMIDs per epost call


@dataclass(frozen=True)
class QACategory:
    """A single QA category with its PubMed MeSH filter."""

    name: str
    description: str
    pubmed_filter: str


# Structure-relevant QA categories and their MeSH subheading filters.
# These are appended to a history-server query key, e.g.:
#   #1 AND "pharmacology"[sh]
# to intersect a set of PMIDs with articles tagged by that subheading.
QA_CATEGORIES: list[QACategory] = [
    QACategory(
        name="mechanism_of_action",
        description="How the compound acts at molecular/cellular level",
        pubmed_filter='"pharmacology"[sh]',
    ),
    QACategory(
        name="therapeutic_use",
        description="Diseases and conditions the compound treats",
        pubmed_filter='"therapeutic use"[sh]',
    ),
    QACategory(
        name="toxicity",
        description="Side effects, adverse reactions, LD50",
        pubmed_filter='("toxicity"[sh] OR "adverse effects"[sh] OR "poisoning"[sh])',
    ),
    QACategory(
        name="metabolism",
        description="Metabolic pathways, CYP enzymes, metabolites",
        pubmed_filter='("metabolism"[sh] OR "pharmacokinetics"[sh])',
    ),
    QACategory(
        name="drug_interactions",
        description="Interactions with other drugs or substances",
        pubmed_filter='"Drug Interactions"[mh]',
    ),
    QACategory(
        name="chemistry",
        description="Chemical synthesis, SAR, structural analysis",
        pubmed_filter='("chemistry"[sh] OR "chemical synthesis"[sh])',
    ),
    QACategory(
        name="structure_activity",
        description="Structure-activity relationship studies",
        pubmed_filter='("structure-activity relationship"[tiab] OR "SAR"[tiab])',
    ),
]


class MeSHCategoryFilter:
    """
    Categorize a set of PMIDs by QA-relevant MeSH subheadings.

    Usage:
        filter = MeSHCategoryFilter(settings)
        mapping = filter.categorize(pmids)
        # mapping = {12345: ["mechanism_of_action", "metabolism"], ...}
    """

    def __init__(
        self,
        settings: Settings,
        categories: list[QACategory] | None = None,
    ):
        Entrez.email = settings.ncbi_email
        if settings.ncbi_api_key:
            Entrez.api_key = settings.ncbi_api_key
        self._rate_limiter = RateLimiter(settings.ncbi_rate_limit)
        self._categories = categories or QA_CATEGORIES

    def categorize(
        self,
        pmids: list[int],
    ) -> dict[int, list[str]]:
        """
        Post PMIDs to Entrez history, then for each QA category run a
        server-side intersection query. Returns {pmid: [category_names]}.
        """
        if not pmids:
            return {}

        logger.info("Categorizing %d PMIDs across %d QA categories",
                     len(pmids), len(self._categories))

        # Step 1: Post all PMIDs to history server
        web_env, query_key = self._post_pmids(pmids)

        # Step 2: For each category, intersect with the posted set
        pmid_to_cats: dict[int, list[str]] = {}

        for cat in tqdm(self._categories, desc="MeSH filtering"):
            matching = self._filter_category(web_env, query_key, cat)
            logger.info("  %s: %d / %d articles", cat.name, len(matching), len(pmids))
            for pmid in matching:
                pmid_to_cats.setdefault(pmid, []).append(cat.name)

        # Count articles with at least one category
        categorized = len(pmid_to_cats)
        uncategorized = len(pmids) - categorized
        logger.info("Categorized: %d, uncategorized (filtered out): %d",
                     categorized, uncategorized)

        return pmid_to_cats

    def categorize_with_counts(
        self,
        pmids: list[int],
    ) -> tuple[dict[int, list[str]], dict[str, int]]:
        """
        Like categorize(), but also returns per-category counts.
        Returns (pmid_to_cats, category_counts).
        """
        pmid_to_cats = self.categorize(pmids)

        counts: dict[str, int] = {}
        for cats in pmid_to_cats.values():
            for cat in cats:
                counts[cat] = counts.get(cat, 0) + 1

        return pmid_to_cats, counts

    def _post_pmids(self, pmids: list[int]) -> tuple[str, str]:
        """Post PMIDs to Entrez history. Returns (WebEnv, QueryKey)."""
        # epost supports large lists; send all at once
        self._rate_limiter.wait()
        handle = Entrez.epost(
            db="pubmed",
            id=",".join(str(p) for p in pmids),
        )
        result = Entrez.read(handle)
        handle.close()
        return result["WebEnv"], result["QueryKey"]

    def _filter_category(
        self,
        web_env: str,
        query_key: str,
        category: QACategory,
        max_retries: int = 3,
    ) -> list[int]:
        """
        Run an esearch that intersects the posted PMIDs (#QueryKey)
        with the category's MeSH filter. Returns matching PMIDs.
        """
        # The #N syntax references a previous query in the history
        term = f"#{query_key} AND {category.pubmed_filter}"

        for attempt in range(max_retries):
            self._rate_limiter.wait()
            try:
                handle = Entrez.esearch(
                    db="pubmed",
                    term=term,
                    webenv=web_env,
                    retmax=100_000,
                    usehistory="n",
                )
                result = Entrez.read(handle)
                handle.close()
                return [int(uid) for uid in result.get("IdList", [])]
            except HTTPError as e:
                if e.code == 429 and attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    logger.warning("Rate limited (429), waiting %ds...", wait)
                    time.sleep(wait)
                else:
                    raise

        return []
