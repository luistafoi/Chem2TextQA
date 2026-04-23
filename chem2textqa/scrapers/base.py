from abc import ABC, abstractmethod
from pathlib import Path
import logging

from chem2textqa.config.settings import Settings


class BaseScraper(ABC):
    """Abstract base for all data-source scrapers."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = logging.getLogger(f"chem2textqa.scrapers.{self.name}")

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in logging and file names."""
        ...

    def default_output_path(self) -> Path:
        return self.settings.output_dir / f"{self.name}.jsonl"
