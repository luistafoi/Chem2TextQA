from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # NCBI / PubMed
    ncbi_api_key: Optional[str] = None
    ncbi_email: str = "chem2textqa@example.com"

    # USPTO PatentsView
    uspto_api_key: Optional[str] = None

    # EPO Open Patent Services
    epo_key: Optional[str] = None
    epo_secret: Optional[str] = None

    # SerpAPI (optional, for Google Patents)
    serpapi_key: Optional[str] = None

    # Storage
    output_dir: Path = Path("./data")

    # Logging
    log_level: str = "INFO"

    @property
    def ncbi_rate_limit(self) -> int:
        """Requests per second for NCBI: 10 with API key, 3 without."""
        return 10 if self.ncbi_api_key else 3


def get_settings() -> Settings:
    """Factory that creates a Settings instance. Can be overridden in tests."""
    return Settings()
