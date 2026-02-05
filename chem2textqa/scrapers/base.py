from abc import ABC, abstractmethod
from pathlib import Path
import logging

from chem2textqa.config.settings import Settings
from chem2textqa.models.document import ScientificDocument


class BaseScraper(ABC):
    """Abstract base for all data-source scrapers."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = logging.getLogger(f"chem2textqa.scrapers.{self.name}")

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in logging and file names (e.g. 'pubmed')."""
        ...

    @abstractmethod
    def search(
        self,
        query: str,
        max_results: int = 100,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[ScientificDocument]:
        """Execute a search and return normalized documents."""
        ...

    def default_output_path(self) -> Path:
        return self.settings.output_dir / f"{self.name}.jsonl"
