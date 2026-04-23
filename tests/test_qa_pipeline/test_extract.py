import json

from chem2textqa.qa_pipeline.phase_0_evidence.extract import (
    build_compound_index,
    build_evidence_record,
    iter_compound_evidence,
    run_phase_0,
)
from chem2textqa.qa_pipeline.phase_0_evidence.redact import compile_redaction_regex


def _make_dataset(tmp_path, records):
    path = tmp_path / "dataset.jsonl"
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return path


def test_build_compound_index(tmp_path):
    records = [
        {
            "pmid": 100,
            "abstract": "Aspirin reduces pain.",
            "linked_compounds": [
                {"cid": 2244, "name": "Aspirin", "smiles": "CC(=O)OC1...",
                 "molecular_formula": "C9H8O4", "mesh_terms": ["Aspirin"]},
            ],
        },
        {
            "pmid": 200,
            "abstract": "Metformin treats diabetes. Aspirin may help.",
            "linked_compounds": [
                {"cid": 2244, "name": "Aspirin", "smiles": "CC(=O)OC1...",
                 "molecular_formula": "C9H8O4", "mesh_terms": []},
                {"cid": 4091, "name": "Metformin", "smiles": "CN(C)C...",
                 "molecular_formula": "C4H11N5", "mesh_terms": []},
            ],
        },
    ]
    dataset = _make_dataset(tmp_path, records)
    index = build_compound_index(dataset)

    assert set(index.keys()) == {2244, 4091}
    assert len(index[2244].article_positions) == 2  # aspirin appears in both
    assert len(index[4091].article_positions) == 1
    assert index[2244].name == "Aspirin"
    assert index[2244].molecular_formula == "C9H8O4"


def test_iter_compound_evidence_redacts_and_matches(tmp_path):
    records = [
        {
            "pmid": 42,
            "abstract": (
                "Aspirin is a nonsteroidal anti-inflammatory drug and was a "
                "subject of study in this paper. The compound reduced platelet "
                "aggregation significantly. Unrelated sentence about weather."
            ),
            "linked_compounds": [
                {"cid": 2244, "name": "Aspirin", "mesh_terms": ["Aspirin"]},
            ],
        },
    ]
    dataset = _make_dataset(tmp_path, records)
    index = build_compound_index(dataset)
    entry = index[2244]

    pattern = compile_redaction_regex(["Aspirin"])
    evidence = list(iter_compound_evidence(dataset, entry, pattern))

    # Only the sentence mentioning aspirin should be kept, redacted
    assert len(evidence) == 1
    assert "[COMPOUND]" in evidence[0]["text"]
    assert "Aspirin" not in evidence[0]["text"]
    assert evidence[0]["pmid"] == "42"
    assert evidence[0]["source"] == "abstract"


def test_iter_compound_evidence_scans_abstract_and_fulltext(tmp_path):
    """When both abstract and full_text are present, both are scanned and
    deduplicated."""
    records = [
        {
            "pmid": 42,
            "abstract": "Aspirin irreversibly acetylates the COX-1 enzyme on Ser530.",
            "full_text": (
                "Aspirin irreversibly acetylates the COX-1 enzyme on Ser530. "  # shared
                "Aspirin's half life in humans is about twenty minutes only. "
                "In the liver Aspirin undergoes hepatic glucuronidation rapidly."
            ),
            "linked_compounds": [
                {"cid": 2244, "name": "Aspirin", "mesh_terms": ["Aspirin"]},
            ],
        },
    ]
    dataset = _make_dataset(tmp_path, records)
    index = build_compound_index(dataset)
    entry = index[2244]
    pattern = compile_redaction_regex(["Aspirin"])
    evidence = list(iter_compound_evidence(dataset, entry, pattern))

    # 3 unique sentences: 1 shared + 2 unique to full_text
    assert len(evidence) == 3
    sources = [e["source"] for e in evidence]
    assert "abstract" in sources
    assert "full_text" in sources


def test_iter_compound_evidence_dedupes(tmp_path):
    """Duplicate sentences across articles should be kept only once."""
    rec1 = {
        "pmid": 1, "abstract": "Aspirin inhibits COX-1 at therapeutic doses.",
        "linked_compounds": [{"cid": 2244, "name": "Aspirin", "mesh_terms": ["Aspirin"]}],
    }
    rec2 = {
        "pmid": 2, "abstract": "Aspirin inhibits COX-1 at therapeutic doses.",
        "linked_compounds": [{"cid": 2244, "name": "Aspirin", "mesh_terms": ["Aspirin"]}],
    }
    dataset = _make_dataset(tmp_path, [rec1, rec2])
    index = build_compound_index(dataset)
    pattern = compile_redaction_regex(["Aspirin"])
    evidence = list(iter_compound_evidence(dataset, index[2244], pattern))
    assert len(evidence) == 1


def test_build_evidence_record():
    from chem2textqa.qa_pipeline.phase_0_evidence.extract import CompoundIndexEntry

    entry = CompoundIndexEntry(
        cid=2244, name="Aspirin", iupac_name="2-acetyloxybenzoic acid",
        smiles="CC(=O)OC1=CC=CC=C1C(=O)O",
        molecular_formula="C9H8O4", molecular_weight=180.16,
        inchi_key="BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
        mesh_terms=["Aspirin"],
    )
    synonyms = ["Aspirin", "Acetylsalicylic Acid"]
    evidence = [
        {"id": 1, "pmid": "42", "text": "[COMPOUND] inhibits COX-1."},
        {"id": 2, "pmid": "43", "text": "[COMPOUND] reduces fever."},
    ]
    record = build_evidence_record(entry, synonyms, evidence)

    assert record["cid"] == 2244
    assert record["name"] == "Aspirin"
    assert record["smiles"] == "CC(=O)OC1=CC=CC=C1C(=O)O"
    assert record["num_synonyms"] == 2
    assert record["num_pmids"] == 2
    assert record["pmids"] == ["42", "43"]
    assert len(record["evidence_sentences"]) == 2


def test_run_phase_0_end_to_end(tmp_path):
    """Full pilot: one dataset, one compound, verify output shape."""
    records = [
        {
            "pmid": 101,
            "abstract": (
                "Aspirin is widely used as an antithrombotic agent. Its mechanism "
                "involves acetylation of COX-1. The drug shows dose-dependent "
                "effects on platelet aggregation."
            ),
            "full_text": "",
            "linked_compounds": [
                {"cid": 2244, "name": "Aspirin",
                 "iupac_name": "2-acetyloxybenzoic acid",
                 "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
                 "molecular_formula": "C9H8O4",
                 "molecular_weight": 180.16,
                 "inchi_key": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
                 "mesh_terms": ["Aspirin", "Acetylsalicylic Acid"]},
            ],
        },
    ]
    dataset = _make_dataset(tmp_path, records)
    output = tmp_path / "evidence.jsonl"
    empty_synfile = tmp_path / "synfile_does_not_exist.gz"

    stats = run_phase_0(dataset, output, cid_synonym_file=empty_synfile)

    assert stats.compounds_scanned == 1
    assert stats.compounds_with_evidence == 1

    lines = output.read_text().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["cid"] == 2244
    assert rec["smiles"] == "CC(=O)OC1=CC=CC=C1C(=O)O"
    assert "[COMPOUND]" in rec["evidence_sentences"][0]["text"]
