from datetime import date
from unittest.mock import MagicMock, patch

from chem2textqa.config.settings import Settings
from chem2textqa.scrapers.pmc import PMCScraper

MOCK_PMC_XML = b"""\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE pmc-articleset PUBLIC "-//NLM//DTD ARTICLE SET 2.0//EN" "https://dtd.nlm.nih.gov/ncbi/pmc/articleset/nlm-articleset-2.0.dtd">
<pmc-articleset>
<article>
  <front>
    <journal-meta>
      <journal-title>Journal of Pharmacology</journal-title>
    </journal-meta>
    <article-meta>
      <article-id pub-id-type="pmid">12345678</article-id>
      <article-id pub-id-type="pmc">9876543</article-id>
      <article-id pub-id-type="doi">10.1234/test.2024</article-id>
      <title-group>
        <article-title>Aspirin mechanism of action</article-title>
      </title-group>
      <contrib-group>
        <contrib contrib-type="author">
          <name><surname>Smith</surname><given-names>John</given-names></name>
        </contrib>
      </contrib-group>
      <pub-date pub-type="epub">
        <year>2024</year><month>3</month><day>15</day>
      </pub-date>
      <kwd-group>
        <kwd>aspirin</kwd>
        <kwd>COX-1</kwd>
      </kwd-group>
      <abstract><p>Aspirin inhibits COX-1 irreversibly.</p></abstract>
    </article-meta>
  </front>
  <body>
    <sec>
      <title>Introduction</title>
      <p>Aspirin is a widely used drug.</p>
    </sec>
    <sec>
      <title>Methods</title>
      <p>We studied the mechanism in vitro.</p>
    </sec>
  </body>
</article>
</pmc-articleset>
"""


def _make_settings() -> Settings:
    return Settings(ncbi_email="test@test.com")


@patch("chem2textqa.scrapers.pmc.Entrez")
def test_pmc_search(mock_entrez):
    mock_entrez.esearch.return_value = MagicMock()
    mock_entrez.read.return_value = {
        "Count": "1",
        "WebEnv": "fake_webenv",
        "QueryKey": "1",
    }

    mock_fetch_handle = MagicMock()
    mock_fetch_handle.read.return_value = MOCK_PMC_XML
    mock_entrez.efetch.return_value = mock_fetch_handle

    scraper = PMCScraper(_make_settings())
    docs = scraper.search("aspirin", max_results=5)

    assert len(docs) == 1
    doc = docs[0]
    assert doc.source == "pmc"
    assert doc.title == "Aspirin mechanism of action"
    assert doc.abstract == "Aspirin inhibits COX-1 irreversibly."
    assert doc.full_text is not None
    assert "widely used drug" in doc.full_text
    assert doc.journal == "Journal of Pharmacology"
    assert doc.publication_date == date(2024, 3, 15)

    id_types = {i.type for i in doc.identifiers}
    assert "pmid" in id_types
    assert "pmcid" in id_types
    assert "doi" in id_types

    assert len(doc.authors) == 1
    assert doc.authors[0].name == "John Smith"

    assert "Introduction" in doc.sections
    assert "Methods" in doc.sections

    assert "aspirin" in doc.keywords


@patch("chem2textqa.scrapers.pmc.Entrez")
def test_pmc_fetch_by_pmcids(mock_entrez):
    mock_entrez.epost.return_value = MagicMock()
    mock_entrez.read.return_value = {
        "WebEnv": "fake_webenv",
        "QueryKey": "1",
    }

    mock_fetch_handle = MagicMock()
    mock_fetch_handle.read.return_value = MOCK_PMC_XML
    mock_entrez.efetch.return_value = mock_fetch_handle

    scraper = PMCScraper(_make_settings())
    docs = scraper.fetch_by_pmcids(["PMC9876543"])

    assert len(docs) == 1
    assert docs[0].title == "Aspirin mechanism of action"


def test_parse_pub_date():
    from lxml import etree
    xml = b"""<article>
      <front><article-meta>
        <pub-date pub-type="epub">
          <year>2024</year><month>6</month><day>20</day>
        </pub-date>
      </article-meta></front>
    </article>"""
    article = etree.fromstring(xml)
    result = PMCScraper._parse_pub_date(article)
    assert result == date(2024, 6, 20)
