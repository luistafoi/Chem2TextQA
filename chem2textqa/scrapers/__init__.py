from chem2textqa.scrapers.base import BaseScraper
from chem2textqa.scrapers.epo import EPOScraper
from chem2textqa.scrapers.google_patents import GooglePatentsScraper
from chem2textqa.scrapers.pubmed import PubMedScraper
from chem2textqa.scrapers.uspto import USPTOScraper

SCRAPER_REGISTRY: dict[str, type[BaseScraper]] = {
    "pubmed": PubMedScraper,
    "google_patents": GooglePatentsScraper,
    "uspto": USPTOScraper,
    "epo": EPOScraper,
}

__all__ = [
    "BaseScraper",
    "SCRAPER_REGISTRY",
    "PubMedScraper",
    "GooglePatentsScraper",
    "USPTOScraper",
    "EPOScraper",
]
