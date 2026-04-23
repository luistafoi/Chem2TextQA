import json

from chem2textqa.processing.merge_fulltext import (
    build_pmid_offset_index,
    merge_pmc_fulltext,
)


def _write_jsonl(path, records):
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_build_pmid_offset_index(tmp_path):
    pmc = tmp_path / "pmc.jsonl"
    records = [
        {"pmid": "100", "full_text": "abc"},
        {"pmid": "200", "full_text": "defghij"},
        {"pmid": "300", "full_text": "x"},
    ]
    _write_jsonl(pmc, records)

    offsets = build_pmid_offset_index(pmc)
    assert set(offsets) == {100, 200, 300}

    # Verify each offset actually points at its record
    with pmc.open("rb") as f:
        for pmid, offset in offsets.items():
            f.seek(offset)
            rec = json.loads(f.readline())
            assert int(rec["pmid"]) == pmid


def test_build_index_keeps_longest_for_duplicates(tmp_path):
    """If the same PMID appears twice (e.g. oa_comm + oa_noncomm), keep the
    one with longer full_text."""
    pmc = tmp_path / "pmc.jsonl"
    records = [
        {"pmid": "100", "full_text": "short"},
        {"pmid": "100", "full_text": "this one is much longer than the first"},
        {"pmid": "200", "full_text": "unique"},
    ]
    _write_jsonl(pmc, records)

    offsets = build_pmid_offset_index(pmc)

    # The offset for 100 should point at the second (longer) record
    with pmc.open("rb") as f:
        f.seek(offsets[100])
        rec = json.loads(f.readline())
    assert "much longer" in rec["full_text"]


def test_merge_fills_missing_fulltext(tmp_path):
    pmc = tmp_path / "pmc.jsonl"
    _write_jsonl(pmc, [
        {"pmid": "100", "full_text": "body of 100", "sections": {"Intro": "i"},
         "pmcid": "PMC100", "doi": "10.1/100"},
        {"pmid": "200", "full_text": "body of 200", "sections": {},
         "pmcid": "PMC200", "doi": "10.1/200"},
    ])

    input_path = tmp_path / "in.jsonl"
    _write_jsonl(input_path, [
        {"pmid": 100, "title": "A", "full_text": ""},
        {"pmid": 200, "title": "B", "full_text": "", "doi": "existing"},
        {"pmid": 300, "title": "C", "full_text": ""},  # no PMC match
        {"pmid": 400, "title": "D", "full_text": "already has"},  # preserved
    ])

    output_path = tmp_path / "out.jsonl"
    stats = merge_pmc_fulltext(input_path, output_path, pmc)

    assert stats.total == 4
    assert stats.newly_merged == 2
    assert stats.already_had_fulltext == 1
    assert stats.no_pmc_match == 1

    out = [json.loads(line) for line in output_path.read_text().splitlines()]

    # PMID 100: filled in
    assert out[0]["full_text"] == "body of 100"
    assert out[0]["sections"] == {"Intro": "i"}
    assert out[0]["pmcid"] == "PMC100"
    assert out[0]["doi"] == "10.1/100"

    # PMID 200: filled in, but existing doi preserved
    assert out[1]["full_text"] == "body of 200"
    assert out[1]["doi"] == "existing"

    # PMID 300: left alone
    assert out[2]["full_text"] == ""
    assert "sections" not in out[2]

    # PMID 400: pre-existing full_text preserved
    assert out[3]["full_text"] == "already has"


def test_merge_skips_pmc_records_with_empty_fulltext(tmp_path):
    """PMC records with empty full_text should not trigger a merge."""
    pmc = tmp_path / "pmc.jsonl"
    _write_jsonl(pmc, [
        {"pmid": "100", "full_text": "", "sections": {}},
    ])

    input_path = tmp_path / "in.jsonl"
    _write_jsonl(input_path, [
        {"pmid": 100, "full_text": ""},
    ])

    output_path = tmp_path / "out.jsonl"
    stats = merge_pmc_fulltext(input_path, output_path, pmc)

    assert stats.newly_merged == 0
    assert stats.no_pmc_match == 1
