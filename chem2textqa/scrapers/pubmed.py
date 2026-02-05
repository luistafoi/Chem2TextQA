from __future__ import annotations

from datetime import date
from io import StringIO

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

# Default queries targeting drug structures, mechanisms, and metabolites
DEFAULT_QUERIES = [
    '"drug mechanism of action"[Title/Abstract]',
    '"chemical structure"[Title/Abstract] AND "drug"[Title/Abstract]',
    '"metabolites"[MeSH Terms] AND "drug development"[Title/Abstract]',
    '"chemical compounds"[MeSH Terms] AND "pharmacology"[Subheading]',
]

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

    def search(
        self,
        query: str,
        max_results: int = 100,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[ScientificDocument]:
        """Search PubMed and return normalized documents."""
        self.logger.info("Searching PubMed: query=%r max_results=%d", query, max_results)

        # Build date range parameters
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

        # Initial search to get count and history keys
        self._rate_limiter.wait()
        search_handle = Entrez.esearch(**search_kwargs)
        search_results = Entrez.read(search_handle)
        search_handle.close()

        total_count = int(search_results["Count"])
        web_env = search_results["WebEnv"]
        query_key = search_results["QueryKey"]
        fetch_count = min(total_count, max_results)

        self.logger.info("Found %d results, fetching %d", total_count, fetch_count)

        # Fetch in batches using history
        documents: list[ScientificDocument] = []
        for start in tqdm(range(0, fetch_count, BATCH_SIZE), desc="PubMed fetch"):
            batch_size = min(BATCH_SIZE, fetch_count - start)
            self._rate_limiter.wait()
            fetch_handle = Entrez.efetch(
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

        self.logger.info("Parsed %d documents from PubMed", len(documents))
        return documents

    def _record_to_document(self, record: dict) -> ScientificDocument | None:
        """Convert a MEDLINE record dict to a ScientificDocument."""
        title = record.get("TI", "")
        if not title:
            return None

        # Identifiers
        identifiers = []
        pmid = record.get("PMID")
        if pmid:
            identifiers.append(Identifier(type="pmid", value=pmid))
        # DOI is in the AID field (article identifiers)
        for aid in record.get("AID", []):
            if aid.endswith("[doi]"):
                identifiers.append(Identifier(type="doi", value=aid.replace(" [doi]", "")))

        # Authors
        authors = [Author(name=name) for name in record.get("AU", [])]

        # Publication date
        pub_date = self._parse_date(record.get("DP", ""))

        # Chemical entities from MeSH headings
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
            journal_or_office=record.get("JT"),
            full_text_url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None,
            metadata={
                "mesh_headings": mesh_headings,
                "publication_type": record.get("PT", []),
                "language": record.get("LA", []),
            },
        )

    @staticmethod
    def _parse_date(date_str: str) -> date | None:
        """Parse MEDLINE date strings like '2024 Jan 15' or '2024 Mar'."""
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
        """Heuristic: MeSH headings with pharmacological subheadings are likely chemicals."""
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
