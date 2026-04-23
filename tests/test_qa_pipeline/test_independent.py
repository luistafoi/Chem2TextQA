import json

from chem2textqa.qa_pipeline.phase_2_independent.independent import (
    _extract_json,
    _iter_questions,
    build_user_prompt,
    load_processed_keys,
)


def test_build_user_prompt_is_blind_and_topic_hinted():
    prompt = build_user_prompt(
        question="What is the scaffold?",
        smiles="CC(=O)OC1=CC=CC=C1C(=O)O",
        molecular_formula="C9H8O4",
        molecular_weight=180.16,
        evidence_sentences=[{"id": 1, "text": "[COMPOUND] contains a phenyl ring."}],
    )
    # Structure + question present
    assert "C9H8O4" in prompt
    assert "What is the scaffold?" in prompt
    # Uses topic-hint framing
    assert "PRIVATE TOPIC HINT" in prompt
    # No citation instructions
    assert "[E1]" not in prompt
    # No leakage of "phase1" state
    assert "phase1" not in prompt.lower()


def test_load_processed_keys(tmp_path):
    path = tmp_path / "p2.jsonl"
    path.write_text(
        json.dumps({"cid": 2244, "qa_index": 1}) + "\n"
        + json.dumps({"cid": 2244, "qa_index": 2}) + "\n"
        + json.dumps({"cid": 4091, "qa_index": 1}) + "\n"
    )
    result = load_processed_keys(path)
    assert result == {"2244:1", "2244:2", "4091:1"}


def test_iter_questions_yields_index(tmp_path):
    """Each question gets a 1-based index within its compound."""
    path = tmp_path / "p1.jsonl"
    records = [
        {
            "cid": 2244,
            "qa_pairs": [
                {"topic": "scaffold", "question": "q1", "answer": "a1"},
                {"topic": "toxicity", "question": "q2", "answer": "a2"},
            ],
        },
        {
            "cid": 4091,
            "qa_pairs": [
                {"topic": "adme", "question": "q3", "answer": "a3"},
            ],
        },
    ]
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    results = list(_iter_questions(path))
    assert len(results) == 3
    assert results[0][1] == 1 and results[0][2]["question"] == "q1"
    assert results[1][1] == 2 and results[1][2]["question"] == "q2"
    assert results[2][1] == 1 and results[2][2]["question"] == "q3"


def test_extract_json_handles_fences():
    r = '```json\n{"answer": "x"}\n```'
    assert _extract_json(r) == {"answer": "x"}
