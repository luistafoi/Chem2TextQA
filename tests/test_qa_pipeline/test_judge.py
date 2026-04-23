import json

from chem2textqa.qa_pipeline.phase_3_validate.judge import (
    _extract_json,
    _judge_user_prompt,
    load_processed_keys,
)


def test_judge_user_prompt():
    prompt = _judge_user_prompt(
        question="What is the scaffold?",
        a1="A benzene ring with a carboxylic acid.",
        a2="An aromatic ring bearing -COOH.",
    )
    assert "ANSWER 1" in prompt
    assert "ANSWER 2" in prompt
    assert "What is the scaffold?" in prompt


def test_extract_json_judge_output():
    r = '{"verdict": "agree", "reasoning": "Both say benzene + carboxylic acid."}'
    out = _extract_json(r)
    assert out["verdict"] == "agree"


def test_load_processed_keys(tmp_path):
    path = tmp_path / "p3.jsonl"
    path.write_text(
        json.dumps({"cid": 2244, "qa_index": 1, "verdict": "agree"}) + "\n"
        + json.dumps({"cid": 2244, "qa_index": 2, "verdict": "disagree"}) + "\n"
    )
    result = load_processed_keys(path)
    assert result == {"2244:1", "2244:2"}
