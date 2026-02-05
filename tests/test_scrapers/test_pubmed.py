from io import StringIO
from unittest.mock import MagicMock, patch

from chem2textqa.config.settings import Settings
from chem2textqa.scrapers.pubmed import PubMedScraper

MOCK_MEDLINE_RECORD = """\
PMID- 12345678
TI  - Mechanism of action of aspirin on platelet aggregation
AB  - Aspirin inhibits cyclooxygenase-1 (COX-1) enzyme irreversibly...
AU  - Smith J
AU  - Doe A
DP  - 2024 Mar 15
JT  - Journal of Pharmacology
MH  - Aspirin/*pharmacology
MH  - Platelet Aggregation/drug effects
OT  - aspirin
OT  - COX-1
AID - 10.1234/test.2024.001 [doi]
PT  - Journal Article
LA  - eng

"""


def _make_settings() -> Settings:
    return Settings(ncbi_email="test@test.com")


@patch("chem2textqa.scrapers.pubmed.Entrez")
def test_pubmed_search_basic(mock_entrez):
    """PubMed scraper correctly maps MEDLINE records to ScientificDocument."""
    # Mock esearch
    mock_search_handle = MagicMock()
    mock_entrez.esearch.return_value = mock_search_handle
    mock_entrez.read.return_value = {
        "Count": "1",
        "WebEnv": "fake_webenv",
        "QueryKey": "1",
    }

    # Mock efetch
    mock_fetch_handle = MagicMock()
    mock_fetch_handle.read.return_value = MOCK_MEDLINE_RECORD
    mock_entrez.efetch.return_value = mock_fetch_handle

    scraper = PubMedScraper(_make_settings())
    docs = scraper.search("aspirin", max_results=5)

    assert len(docs) == 1
    doc = docs[0]
    assert doc.source == "pubmed"
    assert "aspirin" in doc.title.lower()
    assert doc.abstract is not None
    assert len(doc.authors) == 2
    assert doc.authors[0].name == "Smith J"

    # Check identifiers
    id_types = {i.type for i in doc.identifiers}
    assert "pmid" in id_types
    assert "doi" in id_types

    # Check chemical entities
    assert any("Aspirin" in c for c in doc.chemical_entities)

    # Check metadata
    assert "mesh_headings" in doc.metadata


@patch("chem2textqa.scrapers.pubmed.Entrez")
def test_pubmed_sets_entrez_email(mock_entrez):
    """Scraper sets Entrez.email from settings."""
    PubMedScraper(Settings(ncbi_email="myemail@example.com"))
    assert mock_entrez.email == "myemail@example.com"


@patch("chem2textqa.scrapers.pubmed.Entrez")
def test_pubmed_sets_api_key_when_available(mock_entrez):
    PubMedScraper(Settings(ncbi_email="test@test.com", ncbi_api_key="my-key"))
    assert mock_entrez.api_key == "my-key"


def test_parse_date():
    assert PubMedScraper._parse_date("2024 Mar 15").isoformat() == "2024-03-15"
    assert PubMedScraper._parse_date("2024 Jan").isoformat() == "2024-01-01"
    assert PubMedScraper._parse_date("2024").isoformat() == "2024-01-01"
    assert PubMedScraper._parse_date("") is None
    assert PubMedScraper._parse_date("invalid") is None
