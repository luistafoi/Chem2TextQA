import pytest

from chem2textqa.config.settings import Settings


@pytest.fixture
def tmp_output_dir(tmp_path):
    return tmp_path / "output"


@pytest.fixture
def test_settings(tmp_output_dir):
    """Settings with all optional keys unset and a temp output dir."""
    return Settings(
        ncbi_email="test@test.com",
        output_dir=tmp_output_dir,
    )
