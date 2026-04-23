from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # NCBI / PubMed / PMC
    ncbi_api_key: Optional[str] = None
    ncbi_email: str = "chem2textqa@example.com"

    # Storage
    output_dir: Path = Path("./data")

    # Logging
    log_level: str = "INFO"

    # PubChem rate limit (5 req/sec without key)
    pubchem_rate_limit: float = 5.0

    @property
    def ncbi_rate_limit(self) -> float:
        """Requests per second for NCBI: 10 with API key, 3 without."""
        return 10.0 if self.ncbi_api_key else 3.0


def get_settings() -> Settings:
    """Factory that creates a Settings instance. Can be overridden in tests."""
    return Settings()
