from chem2textqa.scrapers.google_patents import GooglePatentsScraper


def test_serpapi_result_to_document():
    """SerpAPI result dict maps correctly to ScientificDocument."""
    result = {
        "title": "Drug compound with improved bioavailability",
        "patent_id": "US20240001234A1",
        "snippet": "A pharmaceutical composition...",
        "inventor": "John Smith",
        "assignee": "BioPharm Corp",
        "priority_date": "2023-06-15",
        "link": "https://patents.google.com/patent/US20240001234A1",
        "pdf": "https://patentimages.storage.googleapis.com/...",
        "grant_date": "2024-01-10",
    }

    doc = GooglePatentsScraper._serpapi_result_to_document(result)

    assert doc is not None
    assert doc.source == "google_patents"
    assert "bioavailability" in doc.title.lower()
    assert doc.identifiers[0].value == "US20240001234A1"
    assert doc.authors[0].name == "John Smith"
    assert doc.abstract == "A pharmaceutical composition..."
    assert doc.publication_date is not None


def test_serpapi_result_no_title():
    """Results without titles are skipped."""
    doc = GooglePatentsScraper._serpapi_result_to_document({"title": ""})
    assert doc is None


def test_html_result_to_document_empty():
    """Empty HTML elements return None."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup("<div></div>", "html.parser")
    elem = soup.find("div")
    doc = GooglePatentsScraper._html_result_to_document(elem)
    assert doc is None
