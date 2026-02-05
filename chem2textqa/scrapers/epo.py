from __future__ import annotations

from datetime import date

from lxml import etree
from tqdm import tqdm

from chem2textqa.config.settings import Settings
from chem2textqa.models.document import (
    Author,
    Identifier,
    ScientificDocument,
    SourceType,
)
from chem2textqa.scrapers.base import BaseScraper

PAGE_SIZE = 100

# IPC class A61K = "Preparations for medical, dental or toilet purposes"
DEFAULT_CQL_TEMPLATE = 'ta = "{query}" AND cl = "A61K"'


class EPOScraper(BaseScraper):
    def __init__(self, settings: Settings):
        super().__init__(settings)
        self._client = None
        if settings.epo_key and settings.epo_secret:
            import epo_ops

            self._client = epo_ops.Client(
                key=settings.epo_key,
                secret=settings.epo_secret,
                middlewares=[epo_ops.middlewares.Throttler()],
            )

    @property
    def name(self) -> str:
        return "epo"

    def search(
        self,
        query: str,
        max_results: int = 100,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[ScientificDocument]:
        """Search EPO patents via Open Patent Services."""
        if not self._client:
            self.logger.warning(
                "EPO scraper skipped: set EPO_KEY and EPO_SECRET in .env"
            )
            return []

        self.logger.info("Searching EPO: query=%r max_results=%d", query, max_results)

        cql = DEFAULT_CQL_TEMPLATE.format(query=query)
        if date_from:
            cql += f' AND pd >= "{date_from}"'
        if date_to:
            cql += f' AND pd <= "{date_to}"'

        documents: list[ScientificDocument] = []
        range_begin = 1

        with tqdm(total=max_results, desc="EPO fetch") as pbar:
            while len(documents) < max_results:
                range_end = min(range_begin + PAGE_SIZE - 1, max_results)
                try:
                    response = self._client.published_data_search(
                        cql=cql,
                        range_begin=range_begin,
                        range_end=range_end,
                    )
                except Exception as e:
                    self.logger.error("EPO search failed at range %d-%d: %s", range_begin, range_end, e)
                    break

                if response.status_code != 200:
                    self.logger.error("EPO returned status %d", response.status_code)
                    break

                batch_docs = self._parse_search_response(response.content)
                if not batch_docs:
                    break

                documents.extend(batch_docs)
                pbar.update(len(batch_docs))
                range_begin = range_end + 1

                if len(batch_docs) < PAGE_SIZE:
                    break

        documents = documents[:max_results]
        self.logger.info("Parsed %d documents from EPO", len(documents))
        return documents

    def _parse_search_response(self, xml_content: bytes) -> list[ScientificDocument]:
        """Parse EPO OPS XML search response into documents."""
        docs = []
        try:
            root = etree.fromstring(xml_content)
        except etree.XMLSyntaxError as e:
            self.logger.error("Failed to parse EPO XML: %s", e)
            return docs

        ns = {
            "ops": "http://ops.epo.org",
            "ex": "http://www.epo.org/exchange",
        }

        for result in root.findall(".//ops:search-result/ex:exchange-documents/ex:exchange-document", ns):
            doc = self._element_to_document(result, ns)
            if doc:
                docs.append(doc)

        return docs

    def _element_to_document(self, elem, ns: dict) -> ScientificDocument | None:
        """Convert a single EPO exchange-document XML element to a ScientificDocument."""
        # Title
        title_elem = elem.find(".//ex:bibliographic-data/ex:invention-title[@lang='en']", ns)
        if title_elem is None:
            title_elem = elem.find(".//ex:bibliographic-data/ex:invention-title", ns)
        title = title_elem.text if title_elem is not None and title_elem.text else ""
        if not title:
            return None

        # Abstract
        abstract_elem = elem.find(".//ex:abstract[@lang='en']/ex:p", ns)
        if abstract_elem is None:
            abstract_elem = elem.find(".//ex:abstract/ex:p", ns)
        abstract = abstract_elem.text if abstract_elem is not None else None

        # Document ID / patent number
        identifiers = []
        doc_id_elem = elem.find(".//ex:bibliographic-data/ex:publication-reference/ex:document-id", ns)
        if doc_id_elem is not None:
            country = doc_id_elem.findtext("ex:country", default="", namespaces=ns)
            doc_number = doc_id_elem.findtext("ex:doc-number", default="", namespaces=ns)
            kind = doc_id_elem.findtext("ex:kind", default="", namespaces=ns)
            if doc_number:
                patent_num = f"{country}{doc_number}{kind}"
                identifiers.append(Identifier(type="patent_number", value=patent_num))

        # Applicants as authors
        authors = []
        for applicant in elem.findall(".//ex:bibliographic-data/ex:parties/ex:applicants/ex:applicant/ex:applicant-name/ex:name", ns):
            if applicant.text:
                authors.append(Author(name=applicant.text))

        # Publication date
        pub_date = None
        date_elem = doc_id_elem.findtext("ex:date", default="", namespaces=ns) if doc_id_elem is not None else ""
        if date_elem and len(date_elem) == 8:
            try:
                pub_date = date(int(date_elem[:4]), int(date_elem[4:6]), int(date_elem[6:8]))
            except ValueError:
                pass

        # IPC codes
        ipc_codes = []
        for ipc in elem.findall(".//ex:bibliographic-data/ex:classifications-ipcr/ex:classification-ipcr/ex:text", ns):
            if ipc.text:
                ipc_codes.append(ipc.text.strip())

        return ScientificDocument(
            source=SourceType.EPO,
            title=title,
            abstract=abstract,
            authors=authors,
            publication_date=pub_date,
            identifiers=identifiers,
            journal_or_office="EPO",
            metadata={
                "ipc_codes": ipc_codes,
            },
        )
