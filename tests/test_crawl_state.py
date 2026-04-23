from chem2textqa.crawl_state import CrawlState


def test_save_and_load(tmp_path):
    path = tmp_path / "state.json"

    state = CrawlState()
    state.seen_pmids = {111, 222, 333}
    state.seen_cids = {2244, 5988}
    state.discovered_terms = {"Aspirin", "Metformin"}
    state.queried_terms = {"Aspirin"}
    state.total_abstracts = 42
    state.rounds_completed = 2

    state.save(path)

    loaded = CrawlState.load(path)
    assert loaded.seen_pmids == {111, 222, 333}
    assert loaded.seen_cids == {2244, 5988}
    assert loaded.discovered_terms == {"Aspirin", "Metformin"}
    assert loaded.queried_terms == {"Aspirin"}
    assert loaded.total_abstracts == 42
    assert loaded.rounds_completed == 2


def test_load_missing_file(tmp_path):
    path = tmp_path / "nonexistent.json"
    state = CrawlState.load(path)
    assert len(state.seen_pmids) == 0
    assert state.rounds_completed == 0


def test_filter_new_pmids():
    state = CrawlState(seen_pmids={100, 200, 300})
    new = state.filter_new_pmids([200, 300, 400, 500])
    assert new == [400, 500]


def test_filter_new_cids():
    state = CrawlState(seen_cids={1, 2, 3})
    new = state.filter_new_cids([2, 3, 4, 5])
    assert new == [4, 5]


def test_new_terms():
    state = CrawlState(
        discovered_terms={"Aspirin", "Metformin", "Ibuprofen"},
        queried_terms={"Aspirin"},
    )
    unused = state.new_terms()
    assert unused == {"Metformin", "Ibuprofen"}


def test_extract_terms_from_document():
    state = CrawlState()
    metadata = {
        "mesh_headings": [
            "Aspirin/*pharmacology",
            "Humans",
            "Cyclooxygenase Inhibitors/therapeutic use",
            "Anti-Inflammatory Agents, Non-Steroidal/toxicity",
            "Male",
        ],
    }
    terms = state.extract_terms_from_document(metadata)
    assert "Aspirin" in terms
    assert "Cyclooxygenase Inhibitors" in terms
    assert "Anti-Inflammatory Agents, Non-Steroidal" in terms
    # "Humans" and "Male" should NOT be extracted (no drug subheading)
    assert "Humans" not in terms
    assert "Male" not in terms


def test_mark_and_filter():
    state = CrawlState()
    state.mark_pmids([100, 200])
    state.mark_cids([1, 2])

    assert state.filter_new_pmids([100, 300]) == [300]
    assert state.filter_new_cids([1, 3]) == [3]
