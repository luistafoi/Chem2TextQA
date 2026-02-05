from __future__ import annotations

from datetime import date

import requests
from bs4 import BeautifulSoup
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

GOOGLE_PATENTS_URL = "https://patents.google.com/"
SERPAPI_ENGINE = "google_patents"


class GooglePatentsScraper(BaseScraper):
    def __init__(self, settings: Settings):
        super().__init__(settings)
        self._serpapi_key = settings.serpapi_key
        # Be conservative with Google scraping: 1 req/sec
        self._rate_limiter = RateLimiter(1.0)
        self._session = requests.Session()
        self._session.headers["User-Agent"] = (
            "Mozilla/5.0 (compatible; Chem2TextQA/0.1; research)"
        )

    @property
    def name(self) -> str:
        return "google_patents"

    def search(
        self,
        query: str,
        max_results: int = 100,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[ScientificDocument]:
        """Search Google Patents via SerpAPI or HTTP fallback."""
        if self._serpapi_key:
            return self._search_serpapi(query, max_results, date_from, date_to)
        return self._search_http(query, max_results, date_from, date_to)

    def _search_serpapi(
        self,
        query: str,
        max_results: int,
        date_from: str | None,
        date_to: str | None,
    ) -> list[ScientificDocument]:
        """Search using SerpAPI's Google Patents engine."""
        try:
            from serpapi import GoogleSearch
        except ImportError:
            self.logger.error(
                "serpapi package not installed. Install with: pip install 'chem2textqa[serpapi]'"
            )
            return []

        self.logger.info("Searching Google Patents (SerpAPI): query=%r", query)

        documents: list[ScientificDocument] = []
        page = 0

        with tqdm(total=max_results, desc="Google Patents (SerpAPI)") as pbar:
            while len(documents) < max_results:
                params: dict = {
                    "engine": SERPAPI_ENGINE,
                    "q": query,
                    "api_key": self._serpapi_key,
                    "page": page,
                }
                if date_from:
                    params["before"] = date_to or ""
                    params["after"] = date_from

                self._rate_limiter.wait()
                search = GoogleSearch(params)
                results = search.get_dict()

                organic = results.get("organic_results", [])
                if not organic:
                    break

                for result in organic:
                    doc = self._serpapi_result_to_document(result)
                    if doc:
                        documents.append(doc)
                        pbar.update(1)
                        if len(documents) >= max_results:
                            break

                page += 1

        self.logger.info("Parsed %d documents from Google Patents (SerpAPI)", len(documents))
        return documents[:max_results]

    def _search_http(
        self,
        query: str,
        max_results: int,
        date_from: str | None,
        date_to: str | None,
    ) -> list[ScientificDocument]:
        """Fallback: scrape Google Patents HTML directly. Best-effort, fragile."""
        self.logger.info("Searching Google Patents (HTTP fallback): query=%r", query)
        self.logger.warning("HTTP scraping is fragile and may break if Google changes their HTML.")

        documents: list[ScientificDocument] = []
        page = 0

        with tqdm(total=max_results, desc="Google Patents (HTTP)") as pbar:
            while len(documents) < max_results:
                html = self._fetch_search_page(query, page, date_from, date_to)
                if not html:
                    break

                soup = BeautifulSoup(html, "lxml")
                results = soup.select("search-result-item, article.result")
                if not results:
                    # Try alternate selectors
                    results = soup.select("[data-result]")
                if not results:
                    self.logger.warning("No results found on page %d — HTML structure may have changed", page)
                    break

                for result in results:
                    doc = self._html_result_to_document(result)
                    if doc:
                        documents.append(doc)
                        pbar.update(1)
                        if len(documents) >= max_results:
                            break

                page += 1

        self.logger.info("Parsed %d documents from Google Patents (HTTP)", len(documents))
        return documents[:max_results]

    @with_retry(max_attempts=3)
    def _fetch_search_page(
        self,
        query: str,
        page: int,
        date_from: str | None,
        date_to: str | None,
    ) -> str | None:
        self._rate_limiter.wait()
        params: dict = {"q": query, "page": page}
        if date_from:
            params["after"] = f"priority:{date_from}"
        if date_to:
            params["before"] = f"priority:{date_to}"

        resp = self._session.get(GOOGLE_PATENTS_URL, params=params, timeout=15)
        if resp.status_code != 200:
            self.logger.warning("Google Patents returned status %d", resp.status_code)
            return None
        return resp.text

    @staticmethod
    def _serpapi_result_to_document(result: dict) -> ScientificDocument | None:
        title = result.get("title", "")
        if not title:
            return None

        patent_id = result.get("patent_id", "")
        identifiers = []
        if patent_id:
            identifiers.append(Identifier(type="patent_number", value=patent_id))

        # SerpAPI provides inventor and assignee info
        authors = []
        inventor = result.get("inventor", "")
        if inventor:
            authors.append(Author(name=inventor))

        pub_date = None
        date_str = result.get("priority_date") or result.get("filing_date")
        if date_str:
            try:
                pub_date = date.fromisoformat(date_str)
            except ValueError:
                pass

        return ScientificDocument(
            source=SourceType.GOOGLE_PATENTS,
            title=title,
            abstract=result.get("snippet"),
            authors=authors,
            publication_date=pub_date,
            identifiers=identifiers,
            full_text_url=result.get("pdf", result.get("link")),
            journal_or_office=result.get("assignee", ""),
            metadata={
                "grant_date": result.get("grant_date"),
                "thumbnail": result.get("thumbnail"),
            },
        )

    @staticmethod
    def _html_result_to_document(element) -> ScientificDocument | None:
        """Parse a single HTML result element. Best-effort."""
        # Try common selectors for title
        title_elem = element.select_one("h3, .result-title, [data-title]")
        title = title_elem.get_text(strip=True) if title_elem else ""
        if not title:
            title = element.get_text(strip=True)[:200]
        if not title:
            return None

        # Try to extract patent number from link
        link_elem = element.select_one("a[href*='/patent/']")
        patent_id = ""
        full_text_url = None
        if link_elem:
            href = link_elem.get("href", "")
            parts = href.split("/patent/")
            if len(parts) > 1:
                patent_id = parts[1].split("/")[0]
            full_text_url = f"https://patents.google.com{href}" if href.startswith("/") else href

        identifiers = []
        if patent_id:
            identifiers.append(Identifier(type="patent_number", value=patent_id))

        # Extract snippet as abstract
        snippet_elem = element.select_one(".result-snippet, .abstract, p")
        abstract = snippet_elem.get_text(strip=True) if snippet_elem else None

        return ScientificDocument(
            source=SourceType.GOOGLE_PATENTS,
            title=title,
            abstract=abstract,
            identifiers=identifiers,
            full_text_url=full_text_url,
            journal_or_office="Google Patents",
        )
