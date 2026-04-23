"""Local MeSH-subheading-based QA categorization.

The MeSHCategoryFilter in chem2textqa/filters/ uses NCBI Entrez to filter
PMIDs server-side. For offline bulk processing we replicate the same logic
locally — given an article's MeSH headings list, return the QA categories
that match.

Each MeSH heading in PubMed XML looks like:
    Aspirin/*pharmacology
    Anti-Inflammatory Agents/therapeutic use
    Drug Interactions
The text after the slash is the "subheading"; the asterisk marks a major
topic. We match on the subheading (or the heading itself for things like
"Drug Interactions").
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LocalQACategory:
    name: str
    description: str
    # Subheadings to match (lowercase, without leading slash)
    subheadings: tuple[str, ...] = ()
    # Full MeSH headings to match (lowercase)
    headings: tuple[str, ...] = ()


LOCAL_QA_CATEGORIES: list[LocalQACategory] = [
    LocalQACategory(
        name="mechanism_of_action",
        description="How the compound acts at molecular/cellular level",
        subheadings=("pharmacology",),
    ),
    LocalQACategory(
        name="therapeutic_use",
        description="Diseases and conditions the compound treats",
        subheadings=("therapeutic use",),
    ),
    LocalQACategory(
        name="toxicity",
        description="Side effects, adverse reactions, LD50",
        subheadings=("toxicity", "adverse effects", "poisoning"),
    ),
    LocalQACategory(
        name="metabolism",
        description="Metabolic pathways, CYP enzymes, metabolites",
        subheadings=("metabolism", "pharmacokinetics"),
    ),
    LocalQACategory(
        name="drug_interactions",
        description="Interactions with other drugs or substances",
        headings=("drug interactions",),
    ),
    LocalQACategory(
        name="chemistry",
        description="Chemical synthesis, structural analysis",
        subheadings=("chemistry", "chemical synthesis"),
    ),
]


def categorize_mesh_headings(mesh_headings: list[str]) -> list[str]:
    """Return QA category names that match this article's MeSH headings.

    A heading like "Aspirin/*pharmacology" will match the "mechanism_of_action"
    category (subheading == pharmacology). An article can match multiple
    categories.
    """
    matched: set[str] = set()

    for raw in mesh_headings:
        if not raw:
            continue
        # Strip the major-topic asterisk and lowercase
        clean = raw.replace("*", "").lower()

        # Split heading from subheading (first slash)
        if "/" in clean:
            heading, subheading = clean.split("/", 1)
            heading = heading.strip()
            subheading = subheading.strip()
        else:
            heading = clean.strip()
            subheading = ""

        # Check each category
        for cat in LOCAL_QA_CATEGORIES:
            if cat.name in matched:
                continue
            if subheading and subheading in cat.subheadings:
                matched.add(cat.name)
            elif heading in cat.headings:
                matched.add(cat.name)

    return sorted(matched)
