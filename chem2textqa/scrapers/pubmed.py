from __future__ import annotations

import time
from datetime import date
from io import StringIO
from urllib.error import HTTPError

from Bio import Entrez, Medline
from tqdm import tqdm

from chem2textqa.config.settings import Settings
from chem2textqa.models.document import (
    Author,
    Identifier,
    ScientificDocument,
    SourceType,
)
from chem2textqa.scrapers.base import BaseScraper
from chem2textqa.utils.rate_limiter import RateLimiter

BATCH_SIZE = 500


class PubMedScraper(BaseScraper):
    def __init__(self, settings: Settings):
        super().__init__(settings)
        Entrez.email = settings.ncbi_email
        if settings.ncbi_api_key:
            Entrez.api_key = settings.ncbi_api_key
        self._rate_limiter = RateLimiter(settings.ncbi_rate_limit)

    @property
    def name(self) -> str:
        return "pubmed"

    # ------------------------------------------------------------------
    # Search by query string
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        max_results: int = 100,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[ScientificDocument]:
        """Search PubMed and return normalized documents."""
        self.logger.info("Searching PubMed: query=%r max_results=%d", query, max_results)

        search_kwargs: dict = {
            "db": "pubmed",
            "term": query,
            "retmax": 0,
            "usehistory": "y",
        }
        if date_from:
            search_kwargs["mindate"] = date_from
            search_kwargs["datetype"] = "pdat"
        if date_to:
            search_kwargs["maxdate"] = date_to
            search_kwargs["datetype"] = "pdat"

        search_handle = self._entrez_call(Entrez.esearch, **search_kwargs)
        search_results = Entrez.read(search_handle)
        search_handle.close()

        total_count = int(search_results["Count"])
        web_env = search_results["WebEnv"]
        query_key = search_results["QueryKey"]
        fetch_count = min(total_count, max_results)

        self.logger.info("Found %d results, fetching %d", total_count, fetch_count)
        return self._fetch_batches(web_env, query_key, fetch_count)

    # ------------------------------------------------------------------
    # Fetch by PMID list (from PubChem links)
    # ------------------------------------------------------------------

    def fetch_by_pmids(
        self,
        pmids: list[int],
        category_map: dict[int, list[str]] | None = None,
    ) -> list[ScientificDocument]:
        """Fetch abstracts for a list of PMIDs.

        Args:
            pmids: PubMed IDs to fetch.
            category_map: Optional {pmid: [category_name, ...]} from
                MeSHCategoryFilter. When provided, each document gets its
                qa_categories field populated.
        """
        if not pmids:
            return []

        self.logger.info("Fetching %d articles by PMID", len(pmids))

        # Post PMIDs to history server for batch retrieval
        post_handle = self._entrez_call(
            Entrez.epost, db="pubmed", id=",".join(str(p) for p in pmids),
        )
        post_results = Entrez.read(post_handle)
        post_handle.close()

        web_env = post_results["WebEnv"]
        query_key = post_results["QueryKey"]

        docs = self._fetch_batches(web_env, query_key, len(pmids))

        # Inject category tags if provided
        if category_map:
            for doc in docs:
                pmid_val = next(
                    (i.value for i in doc.identifiers if i.type == "pmid"), None
                )
                if pmid_val and int(pmid_val) in category_map:
                    doc.qa_categories = category_map[int(pmid_val)]

        return docs

    # ------------------------------------------------------------------
    # Shared batch fetcher
    # ------------------------------------------------------------------

    def _fetch_batches(
        self, web_env: str, query_key: str, fetch_count: int
    ) -> list[ScientificDocument]:
        """Fetch MEDLINE records in batches using history server."""
        documents: list[ScientificDocument] = []

        for start in tqdm(range(0, fetch_count, BATCH_SIZE), desc="PubMed fetch"):
            batch_size = min(BATCH_SIZE, fetch_count - start)
            try:
                fetch_handle = self._entrez_call(
                    Entrez.efetch,
                    db="pubmed",
                    rettype="medline",
                    retmode="text",
                    retstart=start,
                    retmax=batch_size,
                    webenv=web_env,
                    query_key=query_key,
                )
                records = Medline.parse(StringIO(fetch_handle.read()))
                for record in records:
                    doc = self._record_to_document(record)
                    if doc:
                        documents.append(doc)
                fetch_handle.close()
            except Exception as e:
                self.logger.warning("Failed to fetch batch at offset %d: %s", start, e)

        self.logger.info("Parsed %d documents from PubMed", len(documents))
        return documents

    # ------------------------------------------------------------------
    # Count (for estimation without downloading)
    # ------------------------------------------------------------------

    def count(self, query: str) -> int:
        """Return total number of results for a query without fetching."""
        handle = self._entrez_call(Entrez.esearch, db="pubmed", term=query, retmax=0)
        result = Entrez.read(handle)
        handle.close()
        return int(result["Count"])

    def search_pmids(self, query: str, max_results: int = 10000) -> list[int]:
        """Search PubMed and return just the PMIDs (no record download)."""
        handle = self._entrez_call(
            Entrez.esearch, db="pubmed", term=query, retmax=max_results,
        )
        result = Entrez.read(handle)
        handle.close()
        return [int(uid) for uid in result.get("IdList", [])]

    # ------------------------------------------------------------------
    # Record conversion
    # ------------------------------------------------------------------

    def _record_to_document(self, record: dict) -> ScientificDocument | None:
        title = record.get("TI", "")
        if not title:
            return None

        identifiers = []
        pmid = record.get("PMID")
        if pmid:
            identifiers.append(Identifier(type="pmid", value=pmid))
        for aid in record.get("AID", []):
            if aid.endswith("[doi]"):
                identifiers.append(Identifier(type="doi", value=aid.replace(" [doi]", "")))

        # Check for PMC ID
        pmc = record.get("PMC")
        if pmc:
            identifiers.append(Identifier(type="pmcid", value=pmc))

        authors = [Author(name=name) for name in record.get("AU", [])]
        pub_date = self._parse_date(record.get("DP", ""))

        mesh_headings = record.get("MH", [])
        chemical_entities = [
            mh.strip("*").split("/")[0] for mh in mesh_headings if self._is_chemical_mesh(mh)
        ]

        return ScientificDocument(
            source=SourceType.PUBMED,
            title=title,
            abstract=record.get("AB"),
            authors=authors,
            publication_date=pub_date,
            identifiers=identifiers,
            chemical_entities=chemical_entities,
            keywords=record.get("OT", []),
            journal=record.get("JT"),
            full_text_url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None,
            metadata={
                "mesh_headings": mesh_headings,
                "publication_type": record.get("PT", []),
                "language": record.get("LA", []),
                "pmc_id": record.get("PMC"),
            },
        )

    @staticmethod
    def _parse_date(date_str: str) -> date | None:
        if not date_str:
            return None
        parts = date_str.split()
        try:
            year = int(parts[0])
            month_map = {
                "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
                "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
            }
            month = month_map.get(parts[1], 1) if len(parts) > 1 else 1
            day = int(parts[2]) if len(parts) > 2 else 1
            return date(year, month, day)
        except (ValueError, IndexError):
            return None

    @staticmethod
    def _is_chemical_mesh(heading: str) -> bool:
        chemical_indicators = [
            "/pharmacology",
            "/chemistry",
            "/metabolism",
            "/chemical synthesis",
            "/therapeutic use",
            "/toxicity",
        ]
        heading_lower = heading.replace("*", "").lower()
        return any(ind in heading_lower for ind in chemical_indicators)

    def _entrez_call(self, func, *args, max_retries=3, **kwargs):
        """Wrapper around Entrez calls with 429 retry and backoff."""
        for attempt in range(max_retries):
            self._rate_limiter.wait()
            try:
                return func(*args, **kwargs)
            except HTTPError as e:
                if e.code == 429 and attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    self.logger.warning("Rate limited (429), waiting %ds...", wait)
                    time.sleep(wait)
                else:
                    raise
