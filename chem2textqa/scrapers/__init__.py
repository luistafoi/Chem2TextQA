from chem2textqa.scrapers.base import BaseScraper
from chem2textqa.scrapers.pubchem import PubChemScraper
from chem2textqa.scrapers.pubmed import PubMedScraper
from chem2textqa.scrapers.pmc import PMCScraper

SCRAPER_REGISTRY: dict[str, type[BaseScraper]] = {
    "pubchem": PubChemScraper,
    "pubmed": PubMedScraper,
    "pmc": PMCScraper,
}

__all__ = [
    "BaseScraper",
    "SCRAPER_REGISTRY",
    "PubChemScraper",
    "PubMedScraper",
    "PMCScraper",
]
