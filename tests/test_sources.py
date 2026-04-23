from unittest.mock import MagicMock, patch

from chem2textqa.processing.sources import (
    CURATED_SOURCES,
    download_all_sources,
    fetch_cids_from_source,
    load_cid_list,
    load_source_cids,
    save_cid_list,
)


def test_curated_sources_defined():
    assert "DrugBank" in CURATED_SOURCES
    assert "HMDB" in CURATED_SOURCES
    assert "KEGG" in CURATED_SOURCES
    assert "ChEBI" in CURATED_SOURCES
    assert "BindingDB" in CURATED_SOURCES
    assert "ChEMBL" in CURATED_SOURCES


def test_save_and_load_cid_list(tmp_path):
    path = tmp_path / "DrugBank.txt"
    save_cid_list([2244, 4091, 5743], path)

    loaded = load_cid_list(path)
    assert loaded == [2244, 4091, 5743]


def test_load_cid_list_skips_non_digits(tmp_path):
    path = tmp_path / "DrugBank.txt"
    path.write_text("2244\n\n4091\nnot_a_cid\n5743\n")
    assert load_cid_list(path) == [2244, 4091, 5743]


@patch("chem2textqa.processing.sources.requests.get")
def test_fetch_cids_from_source(mock_get):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "InformationList": {
            "Information": [
                {"SID": 1, "CID": [2244]},
                {"SID": 2, "CID": [4091]},
                {"SID": 3, "CID": [5743, 2244]},  # duplicate CID
                {"SID": 4},  # no CID
            ]
        }
    }
    mock_resp.raise_for_status = MagicMock()
    mock_get.return_value = mock_resp

    cids = fetch_cids_from_source("DrugBank")
    assert cids == [2244, 4091, 5743]  # sorted + deduplicated


def test_fetch_cids_from_source_unknown():
    import pytest
    with pytest.raises(ValueError, match="Unknown source"):
        fetch_cids_from_source("NotARealSource")


def test_load_source_cids_union(tmp_path):
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "DrugBank.txt").write_text("1\n2\n3\n")
    (sources_dir / "HMDB.txt").write_text("3\n4\n5\n")

    union = load_source_cids(tmp_path, ["DrugBank", "HMDB"])
    assert union == {1, 2, 3, 4, 5}


def test_load_source_cids_missing_file(tmp_path):
    (tmp_path / "sources").mkdir()
    (tmp_path / "sources" / "DrugBank.txt").write_text("1\n2\n")

    # Missing file for HMDB should warn but not crash
    union = load_source_cids(tmp_path, ["DrugBank", "HMDB"])
    assert union == {1, 2}


@patch("chem2textqa.processing.sources.fetch_cids_from_source")
def test_download_all_sources_caches(mock_fetch, tmp_path):
    mock_fetch.side_effect = lambda src: [1, 2, 3] if src == "DrugBank" else [4, 5]

    # First call fetches
    counts = download_all_sources(tmp_path, sources=["DrugBank", "HMDB"])
    assert counts == {"DrugBank": 3, "HMDB": 2}
    assert mock_fetch.call_count == 2

    # Second call uses cache
    mock_fetch.reset_mock()
    counts = download_all_sources(tmp_path, sources=["DrugBank", "HMDB"])
    assert counts == {"DrugBank": 3, "HMDB": 2}
    assert mock_fetch.call_count == 0


@patch("chem2textqa.processing.sources.fetch_cids_from_source")
def test_download_all_sources_force(mock_fetch, tmp_path):
    mock_fetch.return_value = [1, 2, 3]
    download_all_sources(tmp_path, sources=["DrugBank"])
    assert mock_fetch.call_count == 1

    # Force should re-download
    download_all_sources(tmp_path, sources=["DrugBank"], force=True)
    assert mock_fetch.call_count == 2


# ------------------------------------------------------------------
# Integration: load_compound_index with extra_cids
# ------------------------------------------------------------------


def test_load_compound_index_with_extra_cids(tmp_path):
    import gzip

    from chem2textqa.processing.compounds import load_compound_index

    bulk = tmp_path / "bulk"
    bulk.mkdir()

    # CID-MeSH: only CID 2244 (Aspirin)
    (bulk / "CID-MeSH").write_text("2244\tAspirin\n")

    def _gz(name, content):
        with gzip.open(bulk / name, "wt") as f:
            f.write(content)

    _gz("CID-Title.gz",
        "2244\tAspirin\n"
        "4091\tMetformin\n"
        "5743\tMorphine\n")
    _gz("CID-SMILES.gz",
        "2244\tCC(=O)OC1=CC=CC=C1C(=O)O\n"
        "4091\tCN(C)C(=N)N=C(N)N\n"
        "5743\tCN1CCC23C4OC5=CC=CC=C5C2C=CC4\n")
    _gz("CID-IUPAC.gz", "")
    _gz("CID-InChI-Key.gz", "")
    _gz("CID-Mass.gz",
        "2244\tC9H8O4\t180.04225873\n"
        "4091\tC4H11N5\t129.098745\n"
        "5743\tC17H19NO3\t285.136493\n")

    # Without extra_cids: only Aspirin from MeSH
    compounds = load_compound_index(bulk, require_mesh=True)
    assert set(compounds.keys()) == {2244}

    # With extra_cids: Aspirin + Metformin + Morphine
    compounds = load_compound_index(
        bulk, require_mesh=True, extra_cids={4091, 5743},
    )
    assert set(compounds.keys()) == {2244, 4091, 5743}

    # Verify fields populated for the new compounds (not in MeSH)
    metformin = compounds[4091]
    assert metformin.name == "Metformin"
    assert metformin.smiles == "CN(C)C(=N)N=C(N)N"
    assert metformin.molecular_formula == "C4H11N5"
    assert metformin.molecular_weight == 129.098745
    assert metformin.mesh_terms == []  # no MeSH for this one


def test_load_compound_index_requires_some_source(tmp_path):
    import pytest

    from chem2textqa.processing.compounds import load_compound_index

    bulk = tmp_path / "bulk"
    bulk.mkdir()

    # No MeSH, no extra_cids → should raise
    with pytest.raises(ValueError, match="No compounds loaded"):
        load_compound_index(bulk, require_mesh=False, extra_cids=None)
