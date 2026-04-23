from datetime import date, datetime, timezone

import pytest
from pydantic import ValidationError

from chem2textqa.models.document import (
    Author,
    CompoundRecord,
    Identifier,
    ScientificDocument,
    SourceType,
)


def test_minimal_document():
    doc = ScientificDocument(source=SourceType.PUBMED, title="Test Paper")
    assert doc.source == SourceType.PUBMED
    assert doc.title == "Test Paper"
    assert doc.abstract is None
    assert doc.full_text is None
    assert doc.authors == []
    assert doc.identifiers == []
    assert doc.chemical_entities == []
    assert isinstance(doc.scraped_at, datetime)


def test_full_document():
    doc = ScientificDocument(
        source=SourceType.PUBMED,
        title="Novel Drug Compound",
        abstract="A new compound for treating...",
        full_text="Introduction\nThis paper describes...",
        authors=[Author(name="Jane Doe", affiliation="MIT")],
        publication_date=date(2024, 6, 15),
        identifiers=[Identifier(type="pmid", value="12345678")],
        chemical_entities=["aspirin", "ibuprofen"],
        full_text_url="https://pubmed.ncbi.nlm.nih.gov/12345678/",
        keywords=["drug", "compound"],
        journal="Journal of Pharmacology",
        sections={"Introduction": "This paper..."},
        metadata={"mesh_headings": ["Aspirin/pharmacology"]},
    )
    json_str = doc.model_dump_json()
    restored = ScientificDocument.model_validate_json(json_str)
    assert restored.title == doc.title
    assert restored.authors[0].name == "Jane Doe"
    assert restored.identifiers[0].value == "12345678"
    assert restored.publication_date == date(2024, 6, 15)
    assert restored.full_text is not None
    assert restored.sections["Introduction"] == "This paper..."


def test_missing_required_fields():
    with pytest.raises(ValidationError):
        ScientificDocument(title="No source")

    with pytest.raises(ValidationError):
        ScientificDocument(source=SourceType.PUBMED)


def test_source_type_enum():
    assert SourceType.PUBMED == "pubmed"
    assert SourceType.PUBCHEM == "pubchem"
    assert SourceType.PMC == "pmc"


def test_scraped_at_default_is_utc():
    doc = ScientificDocument(source=SourceType.PUBMED, title="Test")
    assert doc.scraped_at.tzinfo == timezone.utc


def test_compound_record():
    rec = CompoundRecord(
        cid=2244,
        name="Aspirin",
        molecular_formula="C9H8O4",
        molecular_weight=180.16,
        canonical_smiles="CC(=O)OC1=CC=CC=C1C(O)=O",
        linked_pmids=[12345, 67890],
        pharmacological_actions=["Anti-Inflammatory Agents"],
        categories=["drug"],
    )
    json_str = rec.model_dump_json()
    restored = CompoundRecord.model_validate_json(json_str)
    assert restored.cid == 2244
    assert restored.name == "Aspirin"
    assert len(restored.linked_pmids) == 2
    assert restored.pharmacological_actions == ["Anti-Inflammatory Agents"]


def test_compound_record_minimal():
    rec = CompoundRecord(cid=1, name="Methane")
    assert rec.linked_pmids == []
    assert rec.pharmacological_actions == []
    assert isinstance(rec.scraped_at, datetime)
