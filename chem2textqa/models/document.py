from __future__ import annotations

from datetime import date, datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class SourceType(str, Enum):
    PUBMED = "pubmed"
    GOOGLE_PATENTS = "google_patents"
    USPTO = "uspto"
    EPO = "epo"


class Identifier(BaseModel):
    """A typed identifier (DOI, PMID, patent number, etc.)."""

    type: str  # "doi", "pmid", "patent_number", "application_number"
    value: str


class Author(BaseModel):
    name: str
    affiliation: Optional[str] = None


class ScientificDocument(BaseModel):
    """Unified schema for papers and patents scraped from any source."""

    source: SourceType
    title: str
    abstract: Optional[str] = None
    authors: list[Author] = Field(default_factory=list)
    publication_date: Optional[date] = None
    identifiers: list[Identifier] = Field(default_factory=list)
    chemical_entities: list[str] = Field(default_factory=list)
    full_text_url: Optional[str] = None
    keywords: list[str] = Field(default_factory=list)
    journal_or_office: Optional[str] = None
    metadata: dict = Field(default_factory=dict)
    scraped_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
