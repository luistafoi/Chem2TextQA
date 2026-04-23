import json

from chem2textqa.qa_pipeline.compare import compare, load_run


def _write_dataset(path, records):
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _qa(topic, verdict):
    return {
        "topic": topic,
        "question": "q",
        "phase1_answer": "a1",
        "phase2_answer": "a2",
        "verdict": verdict,
    }


def test_load_run_counts(tmp_path):
    path = tmp_path / "a.jsonl"
    _write_dataset(path, [
        {"cid": 1, "qa_pairs": [
            _qa("scaffold", "agree"), _qa("adme", "disagree"),
        ]},
        {"cid": 2, "qa_pairs": [
            _qa("scaffold", "agree"), _qa("toxicity", "unclear"),
        ]},
    ])
    stats = load_run(path, "claude")
    assert stats.compounds == 2
    assert stats.total_qa == 4
    assert stats.agree == 2
    assert stats.disagree == 1
    assert stats.unclear == 1
    assert stats.per_topic_agree["scaffold"] == 2
    assert stats.per_topic_total["adme"] == 1
    assert stats.agree_rate() == 0.5


def test_compare_reports_delta(tmp_path):
    a_path = tmp_path / "a.jsonl"
    _write_dataset(a_path, [
        {"cid": 1, "qa_pairs": [
            _qa("scaffold", "agree"), _qa("adme", "agree"),
        ]},
    ])
    b_path = tmp_path / "b.jsonl"
    _write_dataset(b_path, [
        {"cid": 1, "qa_pairs": [
            _qa("scaffold", "agree"), _qa("adme", "disagree"),
        ]},
    ])

    a = load_run(a_path, "claude")
    b = load_run(b_path, "deepseek")
    report = compare(a, b)

    assert report["overall"]["claude"]["agree_rate"] == 1.0
    assert report["overall"]["deepseek"]["agree_rate"] == 0.5
    assert report["overall"]["delta_agree_rate"] == -0.5

    # Per-compound presence
    by_cid = {r["cid"]: r for r in report["per_compound"]}
    assert 1 in by_cid
    assert by_cid[1]["claude_agree"] == 2
    assert by_cid[1]["deepseek_agree"] == 1

    # Per-topic
    by_topic = {r["topic"]: r for r in report["per_topic"]}
    assert by_topic["scaffold"]["claude_agree_rate"] == 1.0
    assert by_topic["scaffold"]["deepseek_agree_rate"] == 1.0
    assert by_topic["adme"]["deepseek_agree_rate"] == 0.0
