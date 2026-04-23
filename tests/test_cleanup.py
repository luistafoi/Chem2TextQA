import json

from chem2textqa.processing.cleanup import (
    FilterConfig,
    FilterStats,
    GENERIC_COMPOUNDS,
    cleanup_dataset,
    passes_filters,
)


def _base_record(**overrides) -> dict:
    """A minimal valid record that passes all default filters."""
    rec = {
        "pmid": 1,
        "title": "Aspirin inhibits platelet aggregation",
        "abstract": (
            "Aspirin has been shown to inhibit platelet aggregation through "
            "irreversible acetylation of cyclooxygenase-1 (COX-1). This mechanism "
            "underlies its antithrombotic effect at low doses and is clinically "
            "relevant for preventing cardiovascular events in at-risk populations. "
            "The present review summarizes recent findings on aspirin's pharmacology, "
            "including dose-response relationships, metabolic fate in liver, and "
            "long-term outcomes in primary and secondary prevention trials. "
            "Aspirin remains a cornerstone of cardiovascular prophylaxis worldwide, "
            "with ongoing research into novel formulations and combination regimens."
        ),
        "authors": ["Smith J"],
        "journal": "Test Journal",
        "publication_date": "2020-01-01",
        "mesh_headings": ["*Aspirin/pharmacology"],
        "chemicals": ["Aspirin"],
        "keywords": [],
        "doi": "",
        "pmcid": "",
        "publication_types": ["Journal Article"],
        "language": "eng",
        "qa_categories": ["mechanism_of_action"],
        "linked_compounds": [
            {"cid": 2244, "name": "Aspirin", "smiles": "CC(=O)O...",
             "mesh_terms": ["Aspirin"]},
        ],
    }
    rec.update(overrides)
    return rec


def test_passes_default():
    stats = FilterStats()
    assert passes_filters(_base_record(), FilterConfig(), stats) is True
    assert stats.kept == 1


def test_drops_non_english():
    stats = FilterStats()
    rec = _base_record(language="fre")
    assert passes_filters(rec, FilterConfig(), stats) is False
    assert stats.dropped_language == 1


def test_drops_short_abstract():
    stats = FilterStats()
    rec = _base_record(abstract="Too short.")
    assert passes_filters(rec, FilterConfig(), stats) is False
    assert stats.dropped_short_abstract == 1


def test_drops_retracted():
    stats = FilterStats()
    rec = _base_record(publication_types=["Journal Article", "Retracted Publication"])
    assert passes_filters(rec, FilterConfig(), stats) is False
    assert stats.dropped_retracted == 1


def test_drops_editorial():
    stats = FilterStats()
    rec = _base_record(publication_types=["Editorial"])
    assert passes_filters(rec, FilterConfig(), stats) is False
    assert stats.dropped_non_primary == 1


def test_drops_no_compounds():
    stats = FilterStats()
    rec = _base_record(linked_compounds=[])
    assert passes_filters(rec, FilterConfig(), stats) is False
    assert stats.dropped_no_compounds == 1


def test_drops_only_generic_compounds():
    stats = FilterStats()
    rec = _base_record(
        linked_compounds=[
            {"cid": 5793, "name": "D-Glucose", "mesh_terms": ["D-Glucose", "Glucose"]},
            {"cid": 271, "name": "Calcium", "mesh_terms": ["Calcium"]},
        ],
        abstract="Glucose and calcium homeostasis. " * 20,
    )
    assert passes_filters(rec, FilterConfig(), stats) is False
    assert stats.dropped_generic_only == 1


def test_keeps_mixed_generic_and_specific():
    """Article with one generic + one specific compound should be kept."""
    stats = FilterStats()
    rec = _base_record(
        linked_compounds=[
            {"cid": 5793, "name": "D-Glucose", "mesh_terms": ["Glucose"]},
            {"cid": 2244, "name": "Aspirin", "mesh_terms": ["Aspirin"]},
        ],
    )
    assert passes_filters(rec, FilterConfig(), stats) is True


def test_drops_compound_not_mentioned():
    """Article links a compound that doesn't appear anywhere in the text."""
    stats = FilterStats()
    rec = _base_record(
        title="Cardiovascular disease in at-risk populations",
        abstract=(
            "This is a long abstract about cardiovascular disease that does "
            "not mention the linked compound by any of its names. It focuses "
            "on epidemiology and general clinical findings without touching on "
            "any specific pharmacological agent in detail, making it a poor fit. "
            "More filler text to meet the minimum abstract length requirement. "
            "We further discuss population characteristics, study design choices, "
            "follow-up durations, and the overall impact on patient outcomes "
            "across geographic regions and subgroups to ensure adequate length."
        ),
        mesh_headings=["Humans"],  # no major topic
        linked_compounds=[
            {"cid": 12345, "name": "Fosmidomycin", "mesh_terms": ["Fosmidomycin"]},
        ],
    )
    assert passes_filters(rec, FilterConfig(), stats) is False
    assert stats.dropped_no_mention == 1


def test_keeps_compound_mentioned_in_abstract():
    stats = FilterStats()
    rec = _base_record(
        title="Study of cardiovascular outcomes",
        abstract=(
            "We evaluated the effect of Fosmidomycin in patients with "
            "cardiovascular disease. Fosmidomycin demonstrated significant "
            "improvements in multiple endpoints over the 12 week follow-up "
            "period compared to standard of care. These results are promising "
            "and warrant further investigation in larger phase III trials across "
            "multiple countries. Additional analyses examined dose response, "
            "safety profile, and interactions with commonly co-administered agents. "
            "Fosmidomycin was well tolerated in most patients enrolled in this study "
            "with minimal serious adverse events observed throughout the entire "
            "follow-up period, supporting its potential for clinical adoption."
        ),
        mesh_headings=["Humans"],
        linked_compounds=[
            {"cid": 12345, "name": "Fosmidomycin", "mesh_terms": ["Fosmidomycin"]},
        ],
    )
    assert passes_filters(rec, FilterConfig(), stats) is True


def test_keeps_compound_mentioned_via_synonym():
    """Match via MeSH synonym in linked_compounds.mesh_terms."""
    stats = FilterStats()
    rec = _base_record(
        title="Antithrombotic outcomes study",
        abstract=(
            "This comprehensive study examined acetylsalicylic acid in patients "
            "with acute coronary syndrome over a 24 month follow-up period. "
            "Multiple clinical endpoints were assessed to establish efficacy "
            "and safety, with detailed statistical analysis. Results were "
            "stratified across several important subgroups including age, sex, "
            "and baseline risk factors. The compound showed a consistent benefit "
            "across pre-specified strata, with no major safety signals detected. "
            "These findings extend prior work on antithrombotic therapy."
        ),
        mesh_headings=["Humans"],
        linked_compounds=[
            {"cid": 2244, "name": "Aspirin",
             "mesh_terms": ["Aspirin", "Acetylsalicylic Acid"]},
        ],
    )
    assert passes_filters(rec, FilterConfig(), stats) is True


def test_keeps_major_topic_even_if_not_in_abstract():
    stats = FilterStats()
    rec = _base_record(
        title="Cardiovascular outcomes",
        abstract=(
            "Long abstract without the compound name in text but the compound "
            "is marked as a major topic in the MeSH headings which should be "
            "sufficient to consider the article as being about the compound. "
            "Just adding more text to make sure the abstract is long enough. "
            "We include population characteristics, endpoints, statistical "
            "methods, and overall interpretation to reach the required length. "
            "The results support further pharmacological investigation in "
            "randomized controlled trials with larger patient cohorts across "
            "multiple centers, which will help establish generalizability and "
            "long-term safety of this class of pharmacological agents."
        ),
        mesh_headings=["*Fosmidomycin/pharmacology", "Humans"],
        linked_compounds=[
            {"cid": 12345, "name": "Fosmidomycin", "mesh_terms": ["Fosmidomycin"]},
        ],
    )
    assert passes_filters(rec, FilterConfig(), stats) is True


def test_config_disable_mention_check():
    stats = FilterStats()
    rec = _base_record(
        title="Cardiovascular disease",
        abstract="Long abstract that does not mention the linked compound. " * 10,
        mesh_headings=["Humans"],
        linked_compounds=[
            {"cid": 99999, "name": "Rarecompound", "mesh_terms": ["Rarecompound"]},
        ],
    )
    config = FilterConfig(require_compound_mention=False)
    assert passes_filters(rec, config, stats) is True


def test_generic_compounds_set_includes_expected():
    assert "Calcium" in GENERIC_COMPOUNDS
    assert "D-Glucose" in GENERIC_COMPOUNDS
    assert "Water" in GENERIC_COMPOUNDS
    assert "Aspirin" not in GENERIC_COMPOUNDS
    assert "Metformin" not in GENERIC_COMPOUNDS


def test_end_to_end_jsonl(tmp_path):
    input_path = tmp_path / "in.jsonl"
    output_path = tmp_path / "out.jsonl"
    stats_path = tmp_path / "stats.json"

    records = [
        _base_record(pmid=1),  # keep
        _base_record(pmid=2, language="ger"),  # drop (non-English)
        _base_record(pmid=3, abstract="short"),  # drop (short)
        _base_record(pmid=4, publication_types=["Editorial"]),  # drop
        _base_record(pmid=5),  # keep
    ]

    with input_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    stats = cleanup_dataset(input_path, output_path, stats_path=stats_path)

    assert stats.total == 5
    assert stats.kept == 2
    assert stats.dropped_language == 1
    assert stats.dropped_short_abstract == 1
    assert stats.dropped_non_primary == 1

    # Verify output file
    kept_pmids = set()
    with output_path.open() as f:
        for line in f:
            kept_pmids.add(json.loads(line)["pmid"])
    assert kept_pmids == {1, 5}

    # Verify stats file
    stats_data = json.loads(stats_path.read_text())
    assert stats_data["kept"] == 2
    assert stats_data["total"] == 5
