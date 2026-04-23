"""Streaming parser for PubMed baseline XML files.

Each baseline file (`pubmed26nNNNN.xml.gz`) contains a `<PubmedArticleSet>`
with thousands of `<PubmedArticle>` elements. We use lxml's iterparse to
process them one at a time without loading the whole file into memory.
"""
from __future__ import annotations

import gzip
import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterator

from lxml import etree

logger = logging.getLogger(__name__)


@dataclass
class PubMedArticle:
    """A parsed PubMed article (abstract record)."""

    pmid: int
    title: str = ""
    abstract: str = ""
    authors: list[str] = field(default_factory=list)
    journal: str = ""
    publication_date: str | None = None  # YYYY-MM-DD
    mesh_headings: list[str] = field(default_factory=list)
    chemicals: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    doi: str = ""
    pmcid: str = ""
    publication_types: list[str] = field(default_factory=list)
    language: str = ""

    def to_dict(self) -> dict:
        return {
            "pmid": self.pmid,
            "title": self.title,
            "abstract": self.abstract,
            "authors": self.authors,
            "journal": self.journal,
            "publication_date": self.publication_date,
            "mesh_headings": self.mesh_headings,
            "chemicals": self.chemicals,
            "keywords": self.keywords,
            "doi": self.doi,
            "pmcid": self.pmcid,
            "publication_types": self.publication_types,
            "language": self.language,
        }


def iter_pubmed_articles(
    xml_gz_path: Path,
    target_pmids: set[int] | None = None,
) -> Iterator[PubMedArticle]:
    """Stream articles from a single PubMed baseline XML.gz file.

    Args:
        xml_gz_path: path to a single .xml.gz file
        target_pmids: optional set of PMIDs to keep. If provided, only
            yield articles whose PMID is in the set.
    """
    xml_gz_path = Path(xml_gz_path)
    with gzip.open(xml_gz_path, "rb") as f:
        # iterparse with end events on PubmedArticle elements
        context = etree.iterparse(f, events=("end",), tag="PubmedArticle")
        for _event, elem in context:
            try:
                article = _parse_article(elem)
                if article is None:
                    continue
                if target_pmids is None or article.pmid in target_pmids:
                    yield article
            finally:
                # Free memory: clear element and its previous siblings
                elem.clear()
                while elem.getprevious() is not None:
                    del elem.getparent()[0]


def _parse_article(elem) -> PubMedArticle | None:
    """Convert a <PubmedArticle> element to a PubMedArticle dataclass."""
    pmid_elem = elem.find(".//MedlineCitation/PMID")
    if pmid_elem is None or not pmid_elem.text:
        return None
    try:
        pmid = int(pmid_elem.text.strip())
    except ValueError:
        return None

    article = PubMedArticle(pmid=pmid)

    # Title
    title_el = elem.find(".//Article/ArticleTitle")
    if title_el is not None:
        article.title = "".join(title_el.itertext()).strip()

    # Abstract (may have multiple AbstractText sections, e.g. Background, Methods)
    abstract_parts: list[str] = []
    for abs_el in elem.findall(".//Article/Abstract/AbstractText"):
        label = abs_el.get("Label", "")
        text = "".join(abs_el.itertext()).strip()
        if not text:
            continue
        if label:
            abstract_parts.append(f"{label}: {text}")
        else:
            abstract_parts.append(text)
    article.abstract = " ".join(abstract_parts)

    # Authors
    for author_el in elem.findall(".//Article/AuthorList/Author"):
        last = author_el.findtext("LastName", "")
        fore = author_el.findtext("ForeName", "")
        initials = author_el.findtext("Initials", "")
        if last:
            name = f"{fore or initials} {last}".strip()
            article.authors.append(name)
        else:
            collective = author_el.findtext("CollectiveName", "")
            if collective:
                article.authors.append(collective)

    # Journal
    journal_el = elem.find(".//Article/Journal/Title")
    if journal_el is not None and journal_el.text:
        article.journal = journal_el.text.strip()

    # Publication date — try ArticleDate first, then PubDate
    pub_date = _parse_pub_date(elem)
    if pub_date:
        article.publication_date = pub_date

    # MeSH headings (with subheadings as "Heading/Subheading")
    for mesh_el in elem.findall(".//MedlineCitation/MeshHeadingList/MeshHeading"):
        desc_el = mesh_el.find("DescriptorName")
        if desc_el is None or not desc_el.text:
            continue
        heading_name = desc_el.text.strip()
        # Mark major-topic with asterisk to match standard MEDLINE format
        if desc_el.get("MajorTopicYN") == "Y":
            heading_name = "*" + heading_name

        # Subheadings (Qualifiers)
        qualifiers = mesh_el.findall("QualifierName")
        if qualifiers:
            for q_el in qualifiers:
                if q_el.text:
                    qual = q_el.text.strip()
                    if q_el.get("MajorTopicYN") == "Y":
                        qual = "*" + qual
                    article.mesh_headings.append(f"{heading_name}/{qual}")
        else:
            article.mesh_headings.append(heading_name)

    # Chemicals (substances mentioned in the article)
    for chem_el in elem.findall(".//MedlineCitation/ChemicalList/Chemical/NameOfSubstance"):
        if chem_el.text:
            article.chemicals.append(chem_el.text.strip())

    # Keywords
    for kw_el in elem.findall(".//MedlineCitation/KeywordList/Keyword"):
        if kw_el.text:
            article.keywords.append(kw_el.text.strip())

    # DOI and PMCID from ArticleIdList (PubmedData)
    for id_el in elem.findall(".//PubmedData/ArticleIdList/ArticleId"):
        id_type = id_el.get("IdType", "")
        if not id_el.text:
            continue
        value = id_el.text.strip()
        if id_type == "doi":
            article.doi = value
        elif id_type == "pmc":
            article.pmcid = value

    # Publication types
    for pt_el in elem.findall(".//Article/PublicationTypeList/PublicationType"):
        if pt_el.text:
            article.publication_types.append(pt_el.text.strip())

    # Language
    lang_el = elem.find(".//Article/Language")
    if lang_el is not None and lang_el.text:
        article.language = lang_el.text.strip()

    return article


def _parse_pub_date(elem) -> str | None:
    """Best-effort publication date extraction. Returns YYYY-MM-DD or None."""
    # Prefer ArticleDate (most precise)
    for ad in elem.findall(".//Article/ArticleDate"):
        year = ad.findtext("Year")
        month = ad.findtext("Month") or "1"
        day = ad.findtext("Day") or "1"
        if year:
            try:
                return date(int(year), int(month), int(day)).isoformat()
            except ValueError:
                pass

    # Fall back to PubDate
    pd = elem.find(".//Article/Journal/JournalIssue/PubDate")
    if pd is not None:
        year = pd.findtext("Year")
        month = pd.findtext("Month") or "1"
        day = pd.findtext("Day") or "1"
        if year:
            month_map = {
                "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
                "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
            }
            try:
                m = int(month) if month.isdigit() else month_map.get(month, 1)
                return date(int(year), m, int(day)).isoformat()
            except ValueError:
                pass

    return None
