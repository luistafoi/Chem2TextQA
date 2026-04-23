from chem2textqa.qa_pipeline.phase_1_qa.prompts import (
    QA_TEMPLATES,
    STRUCTURAL_TOPICS,
    SYSTEM_PROMPT,
    build_user_prompt,
)


def test_structural_topics_have_unique_keys():
    keys = [t.key for t in STRUCTURAL_TOPICS]
    assert len(set(keys)) == len(keys)
    assert len(keys) >= 10


def test_qa_templates_aliases_structural_topics():
    """QA_TEMPLATES is kept as an alias for legacy imports."""
    assert QA_TEMPLATES is STRUCTURAL_TOPICS


def test_system_prompt_requires_blind():
    lower = SYSTEM_PROMPT.lower()
    assert "do not" in lower or "must not" in lower
    assert "identify" in lower or "name" in lower


def test_system_prompt_forbids_article_references():
    lower = SYSTEM_PROMPT.lower()
    assert "article" in lower
    assert "study" in lower
    assert "do not" in lower or "must not" in lower


def test_system_prompt_mandates_json():
    assert "JSON" in SYSTEM_PROMPT


def test_system_prompt_mentions_topic_hint_privacy():
    """The 'private topic hint' framing is load-bearing — don't drop it."""
    lower = SYSTEM_PROMPT.lower()
    assert "topic hint" in lower or "topic hints" in lower


def test_build_user_prompt_includes_structure():
    prompt = build_user_prompt(
        smiles="CC(=O)OC1=CC=CC=C1C(=O)O",
        molecular_formula="C9H8O4",
        molecular_weight=180.16,
        evidence_sentences=[
            {"id": 1, "text": "[COMPOUND] inhibits COX-1."},
            {"id": 2, "text": "[COMPOUND] reduces platelet aggregation."},
        ],
    )
    assert "CC(=O)OC1=CC=CC=C1C(=O)O" in prompt
    assert "C9H8O4" in prompt
    assert "180.16" in prompt


def test_build_user_prompt_wraps_hints_as_private():
    prompt = build_user_prompt(
        smiles="C", molecular_formula="CH4", molecular_weight=16.0,
        evidence_sentences=[{"id": 1, "text": "[COMPOUND] reduces pain."}],
    )
    # Must be framed as private topic hint, not citable evidence
    assert "PRIVATE TOPIC HINT" in prompt
    assert "DO NOT MENTION" in prompt or "DO NOT QUOTE" in prompt.upper()
    # Must NOT instruct LLM to cite [E<id>]
    assert "[E1]" not in prompt
    assert "[E<" not in prompt


def test_build_user_prompt_without_mw():
    prompt = build_user_prompt("C", "CH4", None, [])
    assert "?" in prompt  # placeholder for missing MW


def test_build_user_prompt_includes_topic_taxonomy():
    prompt = build_user_prompt("C", "CH4", 16.0, [])
    for t in STRUCTURAL_TOPICS:
        assert t.key in prompt
