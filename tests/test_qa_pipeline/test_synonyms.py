import gzip

from chem2textqa.qa_pipeline.phase_0_evidence.synonyms import (
    collect_compound_synonyms,
    filter_synonyms,
    is_usable_synonym,
    load_pubchem_synonyms,
)


def test_is_usable_synonym_basic():
    assert is_usable_synonym("Aspirin")
    assert is_usable_synonym("Acetylsalicylic Acid")
    assert is_usable_synonym("acetylsalicylic-acid")


def test_is_usable_synonym_rejects_too_short():
    assert not is_usable_synonym("AB")
    assert not is_usable_synonym("")
    assert not is_usable_synonym("   ")


def test_is_usable_synonym_rejects_no_letters():
    assert not is_usable_synonym("12345")
    assert not is_usable_synonym("50-78-2")  # CAS number — pure digits + dashes


def test_is_usable_synonym_rejects_common_words():
    assert not is_usable_synonym("acid")
    assert not is_usable_synonym("Acid")
    assert not is_usable_synonym("drug")
    assert not is_usable_synonym("alpha")


def test_is_usable_synonym_rejects_smiles_like():
    assert not is_usable_synonym("C1=CC(=O)OC(=O)C1")  # plain SMILES
    assert not is_usable_synonym("InChI=1S/C9H8O4/c10")


def test_collect_compound_synonyms():
    compound = {
        "name": "Aspirin",
        "iupac_name": "2-acetyloxybenzoic acid",
        "mesh_terms": ["Aspirin", "Acetylsalicylic Acid"],
    }
    syns = collect_compound_synonyms(compound, extra={"ASA", "Bayer Aspirin"})
    assert "Aspirin" in syns
    assert "2-acetyloxybenzoic acid" in syns
    assert "Acetylsalicylic Acid" in syns
    assert "ASA" in syns
    assert "Bayer Aspirin" in syns


def test_filter_synonyms_sort_longest_first():
    raw = {"Aspirin", "Acetylsalicylic Acid", "ASA"}
    result = filter_synonyms(raw)
    # "ASA" filtered out (too short, <4 chars)
    assert "ASA" not in result
    # Longer synonym should come first
    assert result[0] == "Acetylsalicylic Acid"
    assert "Aspirin" in result


def test_filter_synonyms_respects_max_count():
    raw = {f"compound{i}" for i in range(200)}
    result = filter_synonyms(raw, max_count=50)
    assert len(result) == 50


def test_load_pubchem_synonyms(tmp_path):
    path = tmp_path / "CID-Synonym-filtered.gz"
    with gzip.open(path, "wt") as f:
        f.write("2244\tAspirin\n")
        f.write("2244\tAcetylsalicylic Acid\n")
        f.write("2244\tASA\n")
        f.write("4091\tMetformin\n")
        f.write("9999\tSomeotherCompound\n")  # not in target set

    result = load_pubchem_synonyms(path, target_cids={2244, 4091})
    assert 2244 in result
    assert 4091 in result
    assert 9999 not in result
    assert "Aspirin" in result[2244]
    assert "Acetylsalicylic Acid" in result[2244]
    assert result[4091] == {"Metformin"}
