from __future__ import annotations

from datetime import date

import requests
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
from chem2textqa.utils.retry import with_retry

PATENTSVIEW_BASE_URL = "https://search.patentsview.org/api/v1/patent/"
PAGE_SIZE = 100


class USPTOScraper(BaseScraper):
    def __init__(self, settings: Settings):
        super().__init__(settings)
        # PatentsView: 45 requests/minute = 0.75 req/sec
        self._rate_limiter = RateLimiter(0.75)
        self._session = requests.Session()
        if settings.uspto_api_key:
            self._session.headers["X-Api-Key"] = settings.uspto_api_key

    @property
    def name(self) -> str:
        return "uspto"

    def search(
        self,
        query: str,
        max_results: int = 100,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[ScientificDocument]:
        """Search USPTO patents via PatentsView API."""
        self.logger.info("Searching USPTO: query=%r max_results=%d", query, max_results)

        documents: list[ScientificDocument] = []
        cursor = None
        fetched = 0

        with tqdm(total=max_results, desc="USPTO fetch") as pbar:
            while fetched < max_results:
                batch_size = min(PAGE_SIZE, max_results - fetched)
                query_body = self._build_query(query, batch_size, date_from, date_to, cursor)

                data = self._fetch_page(query_body)
                if not data or "patents" not in data:
                    break

                patents = data["patents"]
                if not patents:
                    break

                for patent in patents:
                    doc = self._patent_to_document(patent)
                    if doc:
                        documents.append(doc)
                        fetched += 1
                        pbar.update(1)
                        if fetched >= max_results:
                            break

                # Cursor-based pagination
                cursor = data.get("cursor")
                if not cursor:
                    break

        self.logger.info("Parsed %d documents from USPTO", len(documents))
        return documents

    def _build_query(
        self,
        query: str,
        size: int,
        date_from: str | None,
        date_to: str | None,
        cursor: str | None,
    ) -> dict:
        q_filter: dict = {
            "_or": [
                {"_text_phrase": {"patent_abstract": query}},
                {"_text_phrase": {"patent_title": query}},
            ]
        }

        # Add date range filter if specified
        if date_from or date_to:
            date_filter: dict = {}
            if date_from:
                date_filter["_gte"] = {"patent_date": date_from}
            if date_to:
                date_filter["_lte"] = {"patent_date": date_to}
            q_filter = {"_and": [q_filter, date_filter]}

        body: dict = {
            "q": q_filter,
            "f": [
                "patent_id",
                "patent_title",
                "patent_abstract",
                "patent_date",
                "inventors.inventor_name_last",
                "inventors.inventor_name_first",
                "assignees.assignee_organization",
                "cpc_current.cpc_group_id",
            ],
            "o": {"size": size},
        }

        if cursor:
            body["o"]["after"] = cursor

        return body

    @with_retry()
    def _fetch_page(self, query_body: dict) -> dict | None:
        self._rate_limiter.wait()
        resp = self._session.post(PATENTSVIEW_BASE_URL, json=query_body, timeout=30)
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _patent_to_document(patent: dict) -> ScientificDocument | None:
        title = patent.get("patent_title", "")
        if not title:
            return None

        patent_id = patent.get("patent_id", "")
        identifiers = []
        if patent_id:
            identifiers.append(Identifier(type="patent_number", value=patent_id))

        # Parse inventors
        authors = []
        for inv in patent.get("inventors", []):
            first = inv.get("inventor_name_first", "")
            last = inv.get("inventor_name_last", "")
            if first or last:
                authors.append(Author(name=f"{first} {last}".strip()))

        # Parse date
        pub_date = None
        date_str = patent.get("patent_date")
        if date_str:
            try:
                pub_date = date.fromisoformat(date_str)
            except ValueError:
                pass

        # CPC codes
        cpc_codes = [
            cpc.get("cpc_group_id", "")
            for cpc in patent.get("cpc_current", [])
            if cpc.get("cpc_group_id")
        ]

        # Assignees
        assignees = [
            a.get("assignee_organization", "")
            for a in patent.get("assignees", [])
            if a.get("assignee_organization")
        ]

        return ScientificDocument(
            source=SourceType.USPTO,
            title=title,
            abstract=patent.get("patent_abstract"),
            authors=authors,
            publication_date=pub_date,
            identifiers=identifiers,
            full_text_url=f"https://patents.google.com/patent/US{patent_id}" if patent_id else None,
            journal_or_office="USPTO",
            metadata={
                "cpc_codes": cpc_codes,
                "assignees": assignees,
            },
        )
