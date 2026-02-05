from unittest.mock import MagicMock, patch

from chem2textqa.config.settings import Settings
from chem2textqa.scrapers.epo import EPOScraper

MOCK_EPO_XML = b"""\
<?xml version="1.0" encoding="UTF-8"?>
<ops:world-patent-data xmlns:ops="http://ops.epo.org"
                       xmlns:ex="http://www.epo.org/exchange">
  <ops:biblio-search>
    <ops:search-result>
      <ex:exchange-documents>
        <ex:exchange-document>
          <ex:bibliographic-data>
            <ex:invention-title lang="en">Novel drug delivery system</ex:invention-title>
            <ex:publication-reference>
              <ex:document-id>
                <ex:country>EP</ex:country>
                <ex:doc-number>1234567</ex:doc-number>
                <ex:kind>A1</ex:kind>
                <ex:date>20240315</ex:date>
              </ex:document-id>
            </ex:publication-reference>
            <ex:parties>
              <ex:applicants>
                <ex:applicant>
                  <ex:applicant-name>
                    <ex:name>Pharma Inc.</ex:name>
                  </ex:applicant-name>
                </ex:applicant>
              </ex:applicants>
            </ex:parties>
            <ex:classifications-ipcr>
              <ex:classification-ipcr>
                <ex:text>A61K  9/00</ex:text>
              </ex:classification-ipcr>
            </ex:classifications-ipcr>
          </ex:bibliographic-data>
          <ex:abstract lang="en">
            <ex:p>A novel system for targeted drug delivery...</ex:p>
          </ex:abstract>
        </ex:exchange-document>
      </ex:exchange-documents>
    </ops:search-result>
  </ops:biblio-search>
</ops:world-patent-data>
"""


def _make_settings() -> Settings:
    return Settings(
        ncbi_email="test@test.com",
        epo_key="fake-key",
        epo_secret="fake-secret",
    )


def test_epo_skipped_without_credentials():
    """EPO scraper returns empty list when credentials are missing."""
    settings = Settings(ncbi_email="test@test.com")
    scraper = EPOScraper(settings)
    docs = scraper.search("drug delivery")
    assert docs == []


@patch("chem2textqa.scrapers.epo.epo_ops", create=True)
def test_epo_parse_xml(_mock_epo_ops):
    """EPO XML response is correctly parsed into documents."""
    settings = _make_settings()
    scraper = EPOScraper.__new__(EPOScraper)
    scraper.settings = settings
    scraper.logger = MagicMock()
    scraper._client = MagicMock()

    docs = scraper._parse_search_response(MOCK_EPO_XML)

    assert len(docs) == 1
    doc = docs[0]
    assert doc.source == "epo"
    assert "drug delivery" in doc.title.lower()
    assert doc.abstract is not None
    assert doc.identifiers[0].value == "EP1234567A1"
    assert doc.authors[0].name == "Pharma Inc."
    assert doc.publication_date is not None
    assert doc.publication_date.year == 2024
