from __future__ import annotations

import time
from datetime import date
from io import BytesIO
from urllib.error import HTTPError

from lxml import etree
from Bio import Entrez
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

BATCH_SIZE = 25  # PMC full-text XML is large; keep batches small


class PMCScraper(BaseScraper):
    """Fetches full-text articles from PubMed Central (open access)."""

    def __init__(self, settings: Settings):
        super().__init__(settings)
        Entrez.email = settings.ncbi_email
        if settings.ncbi_api_key:
            Entrez.api_key = settings.ncbi_api_key
        self._rate_limiter = RateLimiter(settings.ncbi_rate_limit)

    @property
    def name(self) -> str:
        return "pmc"

    # ------------------------------------------------------------------
    # Search PMC directly
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        max_results: int = 100,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[ScientificDocument]:
        """Search PMC and fetch full-text XML for open-access articles."""
        self.logger.info("Searching PMC: query=%r max_results=%d", query, max_results)

        search_kwargs: dict = {
            "db": "pmc",
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

        handle = self._entrez_call(Entrez.esearch, **search_kwargs)
        results = Entrez.read(handle)
        handle.close()

        total = int(results["Count"])
        web_env = results["WebEnv"]
        query_key = results["QueryKey"]
        fetch_count = min(total, max_results)

        self.logger.info("Found %d results, fetching %d", total, fetch_count)
        return self._fetch_batches_xml(web_env, query_key, fetch_count)

    # ------------------------------------------------------------------
    # Fetch by PMCID list
    # ------------------------------------------------------------------

    def fetch_by_pmcids(self, pmcids: list[str]) -> list[ScientificDocument]:
        """Fetch full-text for a list of PMC IDs (e.g. 'PMC1234567')."""
        if not pmcids:
            return []

        self.logger.info("Fetching %d full-text articles from PMC", len(pmcids))

        # Strip "PMC" prefix for the API
        numeric_ids = [pid.replace("PMC", "") for pid in pmcids]

        post_handle = self._entrez_call(Entrez.epost, db="pmc", id=",".join(numeric_ids))
        post_results = Entrez.read(post_handle)
        post_handle.close()

        web_env = post_results["WebEnv"]
        query_key = post_results["QueryKey"]

        return self._fetch_batches_xml(web_env, query_key, len(pmcids))

    def fetch_by_pmids(self, pmids: list[int]) -> list[ScientificDocument]:
        """Fetch full-text from PMC using PubMed IDs (finds PMC versions)."""
        if not pmids:
            return []

        self.logger.info("Looking up PMC articles for %d PMIDs", len(pmids))

        # Use elink to convert PMIDs to PMCIDs (batches of 100 to avoid dropped connections)
        pmcids: list[str] = []
        for i in tqdm(range(0, len(pmids), 100), desc="PMID→PMC lookup"):
            batch = pmids[i : i + 100]
            try:
                handle = self._entrez_call(
                    Entrez.elink,
                    dbfrom="pubmed",
                    db="pmc",
                    id=[str(p) for p in batch],
                    linkname="pubmed_pmc",
                )
                link_results = Entrez.read(handle)
                handle.close()

                for record in link_results:
                    for linkset in record.get("LinkSetDb", []):
                        for link in linkset.get("Link", []):
                            pmcids.append(link["Id"])
            except Exception as e:
                self.logger.warning("elink batch at offset %d failed: %s", i, e)

        self.logger.info("Found %d PMC articles for %d PMIDs", len(pmcids), len(pmids))

        if not pmcids:
            return []

        # Now fetch the full text
        post_handle = self._entrez_call(Entrez.epost, db="pmc", id=",".join(pmcids))
        post_results = Entrez.read(post_handle)
        post_handle.close()

        return self._fetch_batches_xml(
            post_results["WebEnv"], post_results["QueryKey"], len(pmcids)
        )

    # ------------------------------------------------------------------
    # Count
    # ------------------------------------------------------------------

    def count(self, query: str) -> int:
        """Return total PMC results for a query."""
        handle = self._entrez_call(Entrez.esearch, db="pmc", term=query, retmax=0)
        result = Entrez.read(handle)
        handle.close()
        return int(result["Count"])

    # ------------------------------------------------------------------
    # Batch XML fetcher
    # ------------------------------------------------------------------

    def _fetch_batches_xml(
        self, web_env: str, query_key: str, fetch_count: int
    ) -> list[ScientificDocument]:
        documents: list[ScientificDocument] = []

        for start in tqdm(range(0, fetch_count, BATCH_SIZE), desc="PMC fetch"):
            batch_size = min(BATCH_SIZE, fetch_count - start)
            try:
                handle = self._entrez_call(
                    Entrez.efetch,
                    db="pmc",
                    rettype="xml",
                    retstart=start,
                    retmax=batch_size,
                    webenv=web_env,
                    query_key=query_key,
                )
                xml_bytes = handle.read()
                handle.close()

                if isinstance(xml_bytes, str):
                    xml_bytes = xml_bytes.encode("utf-8")

                docs = self._parse_pmc_xml(xml_bytes)
                documents.extend(docs)
            except Exception as e:
                self.logger.warning("Failed to fetch PMC batch at offset %d: %s", start, e)

        self.logger.info("Parsed %d full-text documents from PMC", len(documents))
        return documents

    # ------------------------------------------------------------------
    # PMC XML parser
    # ------------------------------------------------------------------

    def _parse_pmc_xml(self, xml_bytes: bytes) -> list[ScientificDocument]:
        """Parse PMC XML (which may contain multiple <article> elements)."""
        docs: list[ScientificDocument] = []

        try:
            root = etree.parse(BytesIO(xml_bytes))
        except etree.XMLSyntaxError as e:
            self.logger.warning("XML parse error: %s", e)
            return docs

        # Handle both single article and article-set
        articles = root.findall(".//article")
        if not articles:
            # Try parsing as a single article
            article = root.getroot()
            if article.tag == "article":
                articles = [article]

        for article in articles:
            doc = self._article_to_document(article)
            if doc:
                docs.append(doc)

        return docs

    def _article_to_document(self, article) -> ScientificDocument | None:
        """Convert a single PMC <article> XML element to a ScientificDocument."""
        # Title
        title_elem = article.find(".//article-title")
        title = self._elem_text(title_elem) if title_elem is not None else ""
        if not title:
            return None

        # Abstract
        abstract_elem = article.find(".//abstract")
        abstract = self._elem_text(abstract_elem) if abstract_elem is not None else None

        # Full text from body
        body_elem = article.find(".//body")
        full_text = self._elem_text(body_elem) if body_elem is not None else None

        # Sections
        sections: dict[str, str] = {}
        for sec in article.findall(".//body/sec"):
            sec_title_elem = sec.find("title")
            if sec_title_elem is not None:
                sec_title = self._elem_text(sec_title_elem).strip()
                sec_text = self._elem_text(sec)
                if sec_title and sec_text:
                    sections[sec_title] = sec_text

        # Identifiers
        identifiers: list[Identifier] = []
        for article_id in article.findall(".//article-id"):
            id_type = article_id.get("pub-id-type", "")
            id_value = (article_id.text or "").strip()
            if id_type == "pmid" and id_value:
                identifiers.append(Identifier(type="pmid", value=id_value))
            elif id_type == "pmc" and id_value:
                identifiers.append(Identifier(type="pmcid", value=f"PMC{id_value}"))
            elif id_type == "doi" and id_value:
                identifiers.append(Identifier(type="doi", value=id_value))

        # Authors
        authors: list[Author] = []
        for contrib in article.findall(".//contrib[@contrib-type='author']"):
            surname = contrib.findtext("name/surname", default="")
            given = contrib.findtext("name/given-names", default="")
            if surname:
                name = f"{given} {surname}".strip()
                aff_elem = contrib.find(".//aff")
                aff = self._elem_text(aff_elem).strip() if aff_elem is not None else None
                authors.append(Author(name=name, affiliation=aff))

        # Publication date
        pub_date = self._parse_pub_date(article)

        # Journal
        journal = article.findtext(".//journal-title", default=None)

        # Keywords
        keywords: list[str] = []
        for kw in article.findall(".//kwd"):
            text = (kw.text or "").strip()
            if text:
                keywords.append(text)

        # PMC ID for URL
        pmcid = None
        for ident in identifiers:
            if ident.type == "pmcid":
                pmcid = ident.value
                break

        return ScientificDocument(
            source=SourceType.PMC,
            title=title,
            abstract=abstract,
            full_text=full_text,
            authors=authors,
            publication_date=pub_date,
            identifiers=identifiers,
            keywords=keywords,
            journal=journal,
            sections=sections,
            full_text_url=f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/" if pmcid else None,
        )

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

    @staticmethod
    def _elem_text(elem) -> str:
        """Extract all text content from an XML element, including children."""
        return "".join(elem.itertext())

    @staticmethod
    def _parse_pub_date(article) -> date | None:
        """Parse publication date from PMC article XML."""
        for pub_date in article.findall(".//pub-date"):
            year = pub_date.findtext("year")
            if not year:
                continue
            try:
                y = int(year)
                m = int(pub_date.findtext("month", default="1"))
                d = int(pub_date.findtext("day", default="1"))
                return date(y, m, d)
            except (ValueError, TypeError):
                continue
        return None
