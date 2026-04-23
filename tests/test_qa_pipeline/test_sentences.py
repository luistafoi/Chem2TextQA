from chem2textqa.qa_pipeline.phase_0_evidence.sentences import split_into_sentences


def test_basic_split():
    text = "First sentence. Second sentence! Third one?"
    result = split_into_sentences(text)
    assert result == ["First sentence.", "Second sentence!", "Third one?"]


def test_handles_initials():
    text = "Dr. Smith studied the reaction. J. Doe confirmed the results."
    result = split_into_sentences(text)
    assert len(result) == 2


def test_handles_et_al():
    text = "As shown by Vane et al. the mechanism involves COX-1. Later studies confirmed."
    result = split_into_sentences(text)
    assert len(result) == 2


def test_handles_figure_references():
    text = "See Fig. 3 for details. The results support the hypothesis."
    result = split_into_sentences(text)
    assert len(result) == 2


def test_handles_decimal_numbers():
    text = "The IC50 was 3.14 uM. The pKa was 4.5."
    result = split_into_sentences(text)
    assert len(result) == 2


def test_empty_input():
    assert split_into_sentences("") == []
    assert split_into_sentences("   ") == []


def test_strips_whitespace():
    result = split_into_sentences("  First.   Second.  ")
    assert result == ["First.", "Second."]
