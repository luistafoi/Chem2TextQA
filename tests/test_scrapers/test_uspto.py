from unittest.mock import MagicMock, patch

from chem2textqa.config.settings import Settings
from chem2textqa.scrapers.uspto import USPTOScraper

MOCK_RESPONSE = {
    "patents": [
        {
            "patent_id": "11234567",
            "patent_title": "Novel compound for treating hypertension",
            "patent_abstract": "A pharmaceutical composition comprising...",
            "patent_date": "2024-01-15",
            "inventors": [
                {"inventor_name_first": "Jane", "inventor_name_last": "Doe"},
                {"inventor_name_first": "John", "inventor_name_last": "Smith"},
            ],
            "assignees": [
                {"assignee_organization": "Pharma Corp"},
            ],
            "cpc_current": [
                {"cpc_group_id": "A61K31/00"},
                {"cpc_group_id": "A61P9/12"},
            ],
        }
    ],
    "cursor": None,
}


def _make_settings() -> Settings:
    return Settings(ncbi_email="test@test.com", uspto_api_key="fake-key")


@patch("chem2textqa.scrapers.uspto.requests.Session")
def test_uspto_search_basic(mock_session_cls):
    """USPTO scraper correctly maps patent data to ScientificDocument."""
    mock_session = MagicMock()
    mock_session_cls.return_value = mock_session

    mock_resp = MagicMock()
    mock_resp.json.return_value = MOCK_RESPONSE
    mock_resp.raise_for_status = MagicMock()
    mock_session.post.return_value = mock_resp

    scraper = USPTOScraper(_make_settings())
    scraper._session = mock_session

    docs = scraper.search("hypertension", max_results=5)

    assert len(docs) == 1
    doc = docs[0]
    assert doc.source == "uspto"
    assert "hypertension" in doc.title.lower()
    assert len(doc.authors) == 2
    assert doc.authors[0].name == "Jane Doe"
    assert doc.identifiers[0].type == "patent_number"
    assert doc.identifiers[0].value == "11234567"
    assert doc.publication_date is not None
    assert doc.metadata["cpc_codes"] == ["A61K31/00", "A61P9/12"]
    assert doc.metadata["assignees"] == ["Pharma Corp"]


def test_patent_to_document_empty_title():
    """Patents without titles are skipped."""
    doc = USPTOScraper._patent_to_document({"patent_title": ""})
    assert doc is None
