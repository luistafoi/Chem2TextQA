import json

from chem2textqa.qa_pipeline.assemble import assemble_dataset


def _write_jsonl(path, records):
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _make_phase_files(tmp_path):
    """Create minimal phase 0-3 files for 2 compounds with 2 QA each."""
    p0 = tmp_path / "p0.jsonl"
    _write_jsonl(p0, [
        {
            "cid": 2244, "name": "Aspirin",
            "iupac_name": "2-acetyloxybenzoic acid",
            "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
            "molecular_formula": "C9H8O4", "molecular_weight": 180.16,
            "inchi_key": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
            "num_pmids": 5, "num_synonyms": 42,
            "evidence_sentences": [
                {"id": 1, "pmid": "100", "source": "abstract",
                 "text": "[COMPOUND] inhibits COX-1."},
            ],
        },
        {
            "cid": 4091, "name": "Metformin", "iupac_name": "...",
            "smiles": "CN(C)C(=N)N=C(N)N", "molecular_formula": "C4H11N5",
            "molecular_weight": 129.16, "inchi_key": "XZWYZXLIPXDOLR-UHFFFAOYSA-N",
            "num_pmids": 3, "num_synonyms": 20,
            "evidence_sentences": [
                {"id": 1, "pmid": "200", "source": "abstract",
                 "text": "[COMPOUND] activates AMPK."},
            ],
        },
    ])

    p1 = tmp_path / "p1.jsonl"
    _write_jsonl(p1, [
        {
            "cid": 2244,
            "qa_pairs": [
                {"topic": "functional_groups", "question": "Q1a", "answer": "A1a"},
                {"topic": "scaffold", "question": "Q1b", "answer": "A1b"},
            ],
        },
        {
            "cid": 4091,
            "qa_pairs": [
                {"topic": "adme", "question": "Q2a", "answer": "A2a"},
                {"topic": "toxicity", "question": "Q2b", "answer": "A2b"},
            ],
        },
    ])

    p2 = tmp_path / "p2.jsonl"
    _write_jsonl(p2, [
        {"cid": 2244, "qa_index": 1, "phase2_answer": "P2-1a"},
        {"cid": 2244, "qa_index": 2, "phase2_answer": "P2-1b"},
        {"cid": 4091, "qa_index": 1, "phase2_answer": "P2-2a"},
        {"cid": 4091, "qa_index": 2, "phase2_answer": "P2-2b"},
    ])

    p3 = tmp_path / "p3.jsonl"
    _write_jsonl(p3, [
        {"cid": 2244, "qa_index": 1, "verdict": "agree", "judge_reasoning": "r1"},
        {"cid": 2244, "qa_index": 2, "verdict": "disagree", "judge_reasoning": "r2"},
        {"cid": 4091, "qa_index": 1, "verdict": "agree", "judge_reasoning": "r3"},
        {"cid": 4091, "qa_index": 2, "verdict": "unclear", "judge_reasoning": "r4"},
    ])

    return p0, p1, p2, p3


def test_assemble_end_to_end(tmp_path):
    p0, p1, p2, p3 = _make_phase_files(tmp_path)
    out_jsonl = tmp_path / "final.jsonl"
    out_json = tmp_path / "final.json"
    summary = tmp_path / "summary.json"

    stats = assemble_dataset(
        phase0_path=p0, phase1_path=p1, phase2_path=p2, phase3_path=p3,
        output_jsonl=out_jsonl, output_json=out_json, summary_path=summary,
    )

    assert stats.compounds == 2
    assert stats.total_qa == 4
    assert stats.agree == 2
    assert stats.disagree == 1
    assert stats.unclear == 1

    records = [json.loads(line) for line in out_jsonl.read_text().splitlines()]
    assert len(records) == 2

    aspirin = next(r for r in records if r["cid"] == 2244)
    assert aspirin["name"] == "Aspirin"
    assert aspirin["smiles"] == "CC(=O)OC1=CC=CC=C1C(=O)O"
    assert aspirin["molecular_formula"] == "C9H8O4"
    assert aspirin["num_pmids"] == 5
    assert len(aspirin["qa_pairs"]) == 2

    qa1 = aspirin["qa_pairs"][0]
    assert qa1["qa_index"] == 1
    assert qa1["topic"] == "functional_groups"
    assert qa1["phase1_answer"] == "A1a"
    assert qa1["phase2_answer"] == "P2-1a"
    assert qa1["verdict"] == "agree"

    # Pretty JSON should also exist for small datasets
    assert out_json.exists()
    data = json.loads(out_json.read_text())
    assert len(data) == 2

    # Summary
    s = json.loads(summary.read_text())
    assert s["compounds"] == 2
    assert s["verdicts"]["agree"] == 2
    assert s["verdicts"]["disagree"] == 1
    assert s["agree_rate"] == 0.5  # 2/4


def test_assemble_agree_only(tmp_path):
    p0, p1, p2, p3 = _make_phase_files(tmp_path)
    out_jsonl = tmp_path / "gold.jsonl"
    out_json = tmp_path / "gold.json"
    summary = tmp_path / "gold_summary.json"

    stats = assemble_dataset(
        phase0_path=p0, phase1_path=p1, phase2_path=p2, phase3_path=p3,
        output_jsonl=out_jsonl, output_json=out_json, summary_path=summary,
        agree_only=True,
    )

    # Only the 2 "agree" QAs should survive — one per compound
    assert stats.compounds == 2
    assert stats.total_qa == 2
    assert stats.agree == 2
    assert stats.disagree == 0

    records = [json.loads(line) for line in out_jsonl.read_text().splitlines()]
    for rec in records:
        for qa in rec["qa_pairs"]:
            assert qa["verdict"] == "agree"


def test_assemble_skips_compounds_without_qa(tmp_path):
    """A compound with only disagree QAs should be dropped in agree_only mode."""
    p0 = tmp_path / "p0.jsonl"
    _write_jsonl(p0, [{"cid": 999, "name": "X", "smiles": "C",
                        "evidence_sentences": []}])
    p1 = tmp_path / "p1.jsonl"
    _write_jsonl(p1, [{"cid": 999, "qa_pairs":
                        [{"topic": "x", "question": "q", "answer": "a"}]}])
    p2 = tmp_path / "p2.jsonl"
    _write_jsonl(p2, [{"cid": 999, "qa_index": 1, "phase2_answer": "p2"}])
    p3 = tmp_path / "p3.jsonl"
    _write_jsonl(p3, [{"cid": 999, "qa_index": 1, "verdict": "disagree"}])

    stats = assemble_dataset(
        phase0_path=p0, phase1_path=p1, phase2_path=p2, phase3_path=p3,
        output_jsonl=tmp_path / "out.jsonl",
        output_json=tmp_path / "out.json",
        summary_path=tmp_path / "sum.json",
        agree_only=True,
    )
    assert stats.compounds == 0
    assert stats.total_qa == 0
