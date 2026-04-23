from __future__ import annotations

from datetime import date, datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class SourceType(str, Enum):
    PUBCHEM = "pubchem"
    PUBMED = "pubmed"
    PMC = "pmc"


class Identifier(BaseModel):
    """A typed identifier (DOI, PMID, CID, etc.)."""

    type: str  # "doi", "pmid", "cid", "pmcid"
    value: str


class Author(BaseModel):
    name: str
    affiliation: Optional[str] = None


class CompoundRecord(BaseModel):
    """A PubChem compound with its metadata and linked article IDs."""

    cid: int
    name: str
    iupac_name: Optional[str] = None
    molecular_formula: Optional[str] = None
    molecular_weight: Optional[float] = None
    canonical_smiles: Optional[str] = None
    inchi: Optional[str] = None
    inchi_key: Optional[str] = None
    description: Optional[str] = None
    pharmacological_actions: list[str] = Field(default_factory=list)
    mesh_headings: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    linked_pmids: list[int] = Field(default_factory=list)
    cross_references: dict[str, list[str]] = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)
    scraped_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ScientificDocument(BaseModel):
    """Unified schema for articles scraped from PubMed or PMC."""

    source: SourceType
    title: str
    abstract: Optional[str] = None
    full_text: Optional[str] = None
    authors: list[Author] = Field(default_factory=list)
    publication_date: Optional[date] = None
    identifiers: list[Identifier] = Field(default_factory=list)
    chemical_entities: list[str] = Field(default_factory=list)
    qa_categories: list[str] = Field(default_factory=list)
    full_text_url: Optional[str] = None
    keywords: list[str] = Field(default_factory=list)
    journal: Optional[str] = None
    sections: dict[str, str] = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)
    scraped_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
