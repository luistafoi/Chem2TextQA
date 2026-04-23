from unittest.mock import MagicMock, patch

from chem2textqa.config.settings import Settings
from chem2textqa.filters.mesh_categories import (
    MeSHCategoryFilter,
    QACategory,
    QA_CATEGORIES,
)


def _make_settings() -> Settings:
    return Settings(ncbi_email="test@test.com")


SIMPLE_CATEGORIES = [
    QACategory(
        name="mechanism",
        description="How it works",
        pubmed_filter='"pharmacology"[sh]',
    ),
    QACategory(
        name="toxicity",
        description="Side effects",
        pubmed_filter='"toxicity"[sh]',
    ),
]


@patch("chem2textqa.filters.mesh_categories.Entrez")
def test_categorize_basic(mock_entrez):
    """Posts PMIDs, runs category queries, returns correct mapping."""
    # Mock epost
    mock_entrez.epost.return_value = MagicMock()
    mock_entrez.read.side_effect = [
        # epost result
        {"WebEnv": "fake_webenv", "QueryKey": "1"},
        # esearch for "mechanism" category
        {"IdList": ["111", "222"]},
        # esearch for "toxicity" category
        {"IdList": ["222", "333"]},
    ]
    mock_entrez.esearch.return_value = MagicMock()

    filt = MeSHCategoryFilter(_make_settings(), categories=SIMPLE_CATEGORIES)
    result = filt.categorize([111, 222, 333, 444])

    # 111 → mechanism only
    assert result[111] == ["mechanism"]
    # 222 → both
    assert sorted(result[222]) == ["mechanism", "toxicity"]
    # 333 → toxicity only
    assert result[333] == ["toxicity"]
    # 444 → not in result (no categories matched)
    assert 444 not in result


@patch("chem2textqa.filters.mesh_categories.Entrez")
def test_categorize_with_counts(mock_entrez):
    mock_entrez.epost.return_value = MagicMock()
    mock_entrez.read.side_effect = [
        {"WebEnv": "fake_webenv", "QueryKey": "1"},
        {"IdList": ["111", "222", "333"]},  # mechanism: 3
        {"IdList": ["222"]},                 # toxicity: 1
    ]
    mock_entrez.esearch.return_value = MagicMock()

    filt = MeSHCategoryFilter(_make_settings(), categories=SIMPLE_CATEGORIES)
    mapping, counts = filt.categorize_with_counts([111, 222, 333])

    assert counts["mechanism"] == 3
    assert counts["toxicity"] == 1
    assert len(mapping) == 3  # all 3 PMIDs have at least one category


@patch("chem2textqa.filters.mesh_categories.Entrez")
def test_categorize_empty_input(mock_entrez):
    filt = MeSHCategoryFilter(_make_settings(), categories=SIMPLE_CATEGORIES)
    result = filt.categorize([])
    assert result == {}
    mock_entrez.epost.assert_not_called()


@patch("chem2textqa.filters.mesh_categories.Entrez")
def test_filter_uses_history_server(mock_entrez):
    """Verify the #QueryKey syntax is used for server-side intersection."""
    mock_entrez.epost.return_value = MagicMock()
    mock_entrez.read.side_effect = [
        {"WebEnv": "test_env", "QueryKey": "42"},
        {"IdList": []},
        {"IdList": []},
    ]
    mock_entrez.esearch.return_value = MagicMock()

    filt = MeSHCategoryFilter(_make_settings(), categories=SIMPLE_CATEGORIES)
    filt.categorize([100])

    # Check that esearch was called with #QueryKey AND filter
    calls = mock_entrez.esearch.call_args_list
    assert len(calls) == 2

    first_call_kwargs = calls[0][1]
    assert "#42" in first_call_kwargs["term"]
    assert '"pharmacology"[sh]' in first_call_kwargs["term"]
    assert first_call_kwargs["webenv"] == "test_env"

    second_call_kwargs = calls[1][1]
    assert "#42" in second_call_kwargs["term"]
    assert '"toxicity"[sh]' in second_call_kwargs["term"]


def test_qa_categories_defined():
    """All expected QA categories exist."""
    names = {c.name for c in QA_CATEGORIES}
    assert "mechanism_of_action" in names
    assert "therapeutic_use" in names
    assert "toxicity" in names
    assert "metabolism" in names
    assert "drug_interactions" in names
    assert "chemistry" in names
    assert "structure_activity" in names
