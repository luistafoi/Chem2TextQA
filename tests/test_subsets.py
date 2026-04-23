import json

from chem2textqa.processing.subsets import classify_record, make_subsets


def test_classify_major_topic():
    rec = {
        "title": "Cardiovascular outcomes",
        "mesh_headings": ["*Aspirin/pharmacology", "Humans"],
        "linked_compounds": [
            {"name": "Aspirin", "mesh_terms": ["Aspirin"]},
        ],
    }
    is_major, is_title = classify_record(rec)
    assert is_major is True
    assert is_title is False


def test_classify_in_title():
    rec = {
        "title": "The effect of Aspirin on platelet aggregation",
        "mesh_headings": ["Humans"],  # no major topic
        "linked_compounds": [
            {"name": "Aspirin", "mesh_terms": ["Aspirin"]},
        ],
    }
    is_major, is_title = classify_record(rec)
    assert is_major is False
    assert is_title is True


def test_classify_matches_mesh_synonym():
    rec = {
        "title": "The effect of acetylsalicylic acid on platelets",
        "mesh_headings": ["Humans"],
        "linked_compounds": [
            {"name": "Aspirin",
             "mesh_terms": ["Aspirin", "Acetylsalicylic Acid"]},
        ],
    }
    is_major, is_title = classify_record(rec)
    assert is_title is True


def test_classify_neither():
    rec = {
        "title": "General cardiovascular study",
        "mesh_headings": ["Humans", "Cardiovascular Diseases"],
        "linked_compounds": [
            {"name": "Aspirin", "mesh_terms": ["Aspirin"]},
        ],
    }
    is_major, is_title = classify_record(rec)
    assert is_major is False
    assert is_title is False


def test_classify_no_compounds():
    rec = {"title": "no compounds", "mesh_headings": [], "linked_compounds": []}
    assert classify_record(rec) == (False, False)


def test_classify_ignores_short_compound_names():
    """Very short names (<3 chars) shouldn't match in title (too many false positives)."""
    rec = {
        "title": "This is a short study",
        "mesh_headings": [],
        "linked_compounds": [
            {"name": "AB", "mesh_terms": []},  # 2 chars — ignored
        ],
    }
    _, is_title = classify_record(rec)
    assert is_title is False


def test_classify_major_topic_with_subheading():
    """Major-topic marker on the subheading should still count the descriptor."""
    rec = {
        "title": "Study",
        "mesh_headings": ["*Aspirin/*pharmacology"],
        "linked_compounds": [{"name": "Aspirin", "mesh_terms": []}],
    }
    is_major, _ = classify_record(rec)
    assert is_major is True


def test_make_subsets_end_to_end(tmp_path):
    input_path = tmp_path / "in.jsonl"
    records = [
        # Major topic → premium + standard
        {"pmid": 1, "title": "Cardio outcomes",
         "mesh_headings": ["*Aspirin/pharmacology"],
         "linked_compounds": [{"name": "Aspirin", "mesh_terms": []}]},
        # In title only → standard
        {"pmid": 2, "title": "Aspirin in patients",
         "mesh_headings": ["Humans"],
         "linked_compounds": [{"name": "Aspirin", "mesh_terms": []}]},
        # Neither → broad only
        {"pmid": 3, "title": "Cardiovascular disease",
         "mesh_headings": ["Humans"],
         "linked_compounds": [{"name": "Aspirin", "mesh_terms": []}]},
        # Both → premium + standard
        {"pmid": 4, "title": "Metformin in PCOS",
         "mesh_headings": ["*Metformin/therapeutic use"],
         "linked_compounds": [{"name": "Metformin", "mesh_terms": []}]},
    ]
    with input_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    premium_path = tmp_path / "premium.jsonl"
    standard_path = tmp_path / "standard.jsonl"
    stats = make_subsets(input_path, premium_path, standard_path)

    assert stats.total == 4
    assert stats.broad == 4
    assert stats.standard == 3
    assert stats.premium == 2

    premium_pmids = {json.loads(line)["pmid"]
                     for line in premium_path.read_text().splitlines()}
    standard_pmids = {json.loads(line)["pmid"]
                      for line in standard_path.read_text().splitlines()}

    assert premium_pmids == {1, 4}
    assert standard_pmids == {1, 2, 4}
