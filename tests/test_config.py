from pathlib import Path

from chem2textqa.config.settings import Settings


def test_default_settings():
    # _env_file=None ensures any real .env in cwd is ignored.
    settings = Settings(_env_file=None, ncbi_email="test@example.com")
    assert not settings.ncbi_api_key  # None or empty string
    assert settings.ncbi_email == "test@example.com"
    assert settings.output_dir == Path("./data")
    assert settings.log_level == "INFO"
    assert settings.pubchem_rate_limit == 5.0


def test_ncbi_rate_limit_without_key():
    settings = Settings(_env_file=None, ncbi_email="test@example.com")
    assert settings.ncbi_rate_limit == 3.0


def test_ncbi_rate_limit_with_key():
    settings = Settings(ncbi_email="test@example.com", ncbi_api_key="fake-key")
    assert settings.ncbi_rate_limit == 10.0


def test_output_dir_custom(tmp_path):
    settings = Settings(ncbi_email="test@example.com", output_dir=tmp_path)
    assert settings.output_dir == tmp_path
