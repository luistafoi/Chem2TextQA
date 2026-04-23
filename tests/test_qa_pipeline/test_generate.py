import json

from chem2textqa.qa_pipeline.phase_1_qa.generate import (
    MAX_QA_PAIRS,
    MIN_QA_PAIRS,
    _extract_json,
    _validate_qa_output,
    load_processed_cids,
)


def _pair(topic="composition", question="What is it?", answer="An answer."):
    return {"topic": topic, "question": question, "answer": answer}


def test_extract_json_fenced():
    raw = "Sure thing!\n```json\n{\"qa_pairs\": []}\n```\n"
    assert _extract_json(raw) == {"qa_pairs": []}


def test_extract_json_bare():
    raw = '{"qa_pairs": [{"topic": "x"}]}'
    assert _extract_json(raw) == {"qa_pairs": [{"topic": "x"}]}


def test_extract_json_malformed():
    assert _extract_json("this is not json") is None
    assert _extract_json("") is None
    assert _extract_json(None) is None


def test_validate_happy_path():
    topics = ["composition", "mechanism", "engineering", "metabolism", "reactivity", "toxicity"]
    parsed = {"qa_pairs": [_pair(topic=t) for t in topics[:MIN_QA_PAIRS + 2]]}
    result = _validate_qa_output(parsed)
    assert result is not None
    assert len(result) == MIN_QA_PAIRS + 2
    for p in result:
        assert "topic" in p
        assert "question" in p
        assert "answer" in p


def test_validate_rejects_too_few():
    parsed = {"qa_pairs": [_pair() for _ in range(MIN_QA_PAIRS - 1)]}
    assert _validate_qa_output(parsed) is None


def test_validate_rejects_too_many():
    parsed = {"qa_pairs": [_pair() for _ in range(MAX_QA_PAIRS + 1)]}
    assert _validate_qa_output(parsed) is None


def test_validate_rejects_empty_question_or_answer():
    pairs = [_pair() for _ in range(MIN_QA_PAIRS)]
    pairs[0]["question"] = ""
    assert _validate_qa_output({"qa_pairs": pairs}) is None

    pairs = [_pair() for _ in range(MIN_QA_PAIRS)]
    pairs[0]["answer"] = "   "
    assert _validate_qa_output({"qa_pairs": pairs}) is None


def test_validate_freeform_topic_accepted():
    """Any non-empty string is a valid topic tag under the soft-rule design."""
    pairs = [_pair() for _ in range(MIN_QA_PAIRS)]
    pairs[0]["topic"] = "prodrug_activation"
    result = _validate_qa_output({"qa_pairs": pairs})
    assert result is not None
    assert result[0]["topic"] == "prodrug_activation"


def test_validate_empty_topic_falls_back_to_other():
    pairs = [_pair() for _ in range(MIN_QA_PAIRS)]
    pairs[0]["topic"] = ""
    result = _validate_qa_output({"qa_pairs": pairs})
    assert result is not None
    assert result[0]["topic"] == "other"


def test_validate_rejects_non_string_fields():
    pairs = [_pair() for _ in range(MIN_QA_PAIRS)]
    pairs[0]["answer"] = 42
    assert _validate_qa_output({"qa_pairs": pairs}) is None


def test_validate_rejects_non_list():
    assert _validate_qa_output({"qa_pairs": "not a list"}) is None
    assert _validate_qa_output({}) is None


def test_load_processed_cids(tmp_path):
    path = tmp_path / "out.jsonl"
    path.write_text(
        json.dumps({"cid": 2244}) + "\n"
        + json.dumps({"cid": 4091}) + "\n"
        + "\n"
        + "garbage line without json\n"
        + json.dumps({"cid": 5743}) + "\n"
    )
    result = load_processed_cids(path)
    assert result == {2244, 4091, 5743}


def test_load_processed_cids_missing_file(tmp_path):
    assert load_processed_cids(tmp_path / "nope.jsonl") == set()
