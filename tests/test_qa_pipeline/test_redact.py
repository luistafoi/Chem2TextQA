from chem2textqa.qa_pipeline.phase_0_evidence.redact import (
    compile_redaction_regex,
    count_hits,
    redact,
)


def test_compile_none_for_empty():
    assert compile_redaction_regex([]) is None


def test_redact_simple():
    pattern = compile_redaction_regex(["Aspirin"])
    assert redact("Aspirin reduces pain.", pattern) == "[COMPOUND] reduces pain."


def test_redact_case_insensitive():
    pattern = compile_redaction_regex(["aspirin"])
    assert redact("ASPIRIN is an NSAID.", pattern) == "[COMPOUND] is an NSAID."
    assert redact("Aspirin is an NSAID.", pattern) == "[COMPOUND] is an NSAID."


def test_redact_longest_match_first():
    """'Acetylsalicylic Acid' should match as a whole unit, not leave 'Acetyl'."""
    pattern = compile_redaction_regex(["Acetylsalicylic Acid", "Acetyl"])
    result = redact("Acetylsalicylic Acid inhibits COX-1.", pattern)
    assert result == "[COMPOUND] inhibits COX-1."


def test_redact_whole_word_only():
    """'Aspirin' should not match inside 'Aspirins' — but should with hyphen break."""
    pattern = compile_redaction_regex(["Aspirin"])
    # Should match with space/punctuation boundary
    assert redact("Aspirin-induced effects", pattern) == "[COMPOUND]-induced effects"
    # Should NOT consume part of a longer word
    assert redact("Aspirins are used", pattern) == "Aspirins are used"


def test_redact_handles_regex_metacharacters():
    """Synonyms with parentheses / dots etc. must be escaped."""
    pattern = compile_redaction_regex(["(S)-(+)-Ibuprofen", "alpha-methylbenzyl alcohol"])
    text = "The (S)-(+)-Ibuprofen enantiomer is more active."
    assert "[COMPOUND]" in redact(text, pattern)


def test_redact_multiple_synonyms_in_sentence():
    pattern = compile_redaction_regex(["Aspirin", "Acetylsalicylic Acid"])
    text = "Aspirin (also known as Acetylsalicylic Acid) inhibits COX-1."
    result = redact(text, pattern)
    assert result.count("[COMPOUND]") == 2


def test_count_hits():
    pattern = compile_redaction_regex(["Aspirin"])
    assert count_hits("Aspirin reduces inflammation. Aspirin is cheap.", pattern) == 2
    assert count_hits("Nothing relevant here.", pattern) == 0


def test_redact_no_pattern_returns_original():
    assert redact("Text", None) == "Text"
    assert redact("", None) == ""
