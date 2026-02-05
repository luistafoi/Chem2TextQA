from datetime import date, datetime, timezone

import pytest
from pydantic import ValidationError

from chem2textqa.models.document import (
    Author,
    Identifier,
    ScientificDocument,
    SourceType,
)


def test_minimal_document():
    """A document with only required fields is valid."""
    doc = ScientificDocument(source=SourceType.PUBMED, title="Test Paper")
    assert doc.source == SourceType.PUBMED
    assert doc.title == "Test Paper"
    assert doc.abstract is None
    assert doc.authors == []
    assert doc.identifiers == []
    assert doc.chemical_entities == []
    assert isinstance(doc.scraped_at, datetime)


def test_full_document():
    """A document with all fields round-trips correctly."""
    doc = ScientificDocument(
        source=SourceType.USPTO,
        title="Novel Drug Compound",
        abstract="A new compound for treating...",
        authors=[Author(name="Jane Doe", affiliation="MIT")],
        publication_date=date(2024, 6, 15),
        identifiers=[Identifier(type="patent_number", value="US12345678")],
        chemical_entities=["aspirin", "ibuprofen"],
        full_text_url="https://example.com/patent/123",
        keywords=["drug", "compound"],
        journal_or_office="USPTO",
        metadata={"cpc_codes": ["A61K"]},
    )
    # Round-trip through JSON
    json_str = doc.model_dump_json()
    restored = ScientificDocument.model_validate_json(json_str)
    assert restored.title == doc.title
    assert restored.authors[0].name == "Jane Doe"
    assert restored.identifiers[0].value == "US12345678"
    assert restored.publication_date == date(2024, 6, 15)


def test_missing_required_fields():
    """Missing source or title raises ValidationError."""
    with pytest.raises(ValidationError):
        ScientificDocument(title="No source")

    with pytest.raises(ValidationError):
        ScientificDocument(source=SourceType.PUBMED)


def test_source_type_enum():
    """All expected source types exist."""
    assert SourceType.PUBMED == "pubmed"
    assert SourceType.USPTO == "uspto"
    assert SourceType.EPO == "epo"
    assert SourceType.GOOGLE_PATENTS == "google_patents"


def test_scraped_at_default_is_utc():
    doc = ScientificDocument(source=SourceType.PUBMED, title="Test")
    assert doc.scraped_at.tzinfo == timezone.utc
