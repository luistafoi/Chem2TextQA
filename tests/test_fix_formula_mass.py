import gzip
import json

from chem2textqa.processing.fix_formula_mass import (
    load_cid_mass_map,
    patch_jsonl,
)


def _write_cid_mass(path, rows):
    with gzip.open(path, "wt") as f:
        for row in rows:
            f.write("\t".join(str(x) for x in row) + "\n")


def test_load_cid_mass_map(tmp_path):
    path = tmp_path / "CID-Mass.gz"
    _write_cid_mass(path, [
        (2244, "C9H8O4", 180.04225873, 180.04225873),
        (4091, "C4H11N5", 129.098745, 129.098745),
        (5743, "C17H19NO3", 285.136493, 285.136493),
    ])

    mapping = load_cid_mass_map(path)
    assert mapping[2244] == ("C9H8O4", 180.04225873)
    assert mapping[4091] == ("C4H11N5", 129.098745)
    assert mapping[5743] == ("C17H19NO3", 285.136493)


def test_load_cid_mass_skips_malformed(tmp_path):
    path = tmp_path / "CID-Mass.gz"
    _write_cid_mass(path, [
        (2244, "C9H8O4", 180.0),
        ("not-a-cid", "X", "Y"),
    ])
    # Use open to write a malformed line
    with gzip.open(path, "at") as f:
        f.write("badline\n")

    mapping = load_cid_mass_map(path)
    assert 2244 in mapping


def test_patch_jsonl_fills_missing_fields(tmp_path):
    # Existing record has compounds with empty formula/mw
    input_path = tmp_path / "in.jsonl"
    records = [
        {
            "pmid": 1,
            "linked_compounds": [
                {"cid": 2244, "name": "Aspirin",
                 "molecular_formula": "", "molecular_weight": None},
                {"cid": 4091, "name": "Metformin",
                 "molecular_formula": "", "molecular_weight": None},
            ],
        },
        {
            "pmid": 2,
            "linked_compounds": [
                {"cid": 9999, "name": "Unknown",
                 "molecular_formula": "", "molecular_weight": None},
            ],
        },
    ]
    with input_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    cid_mass_map = {
        2244: ("C9H8O4", 180.04225873),
        4091: ("C4H11N5", 129.098745),
    }

    output_path = tmp_path / "out.jsonl"
    count, patched = patch_jsonl(input_path, output_path, cid_mass_map)

    assert count == 2
    assert patched == 2  # 2244 and 4091 got patched; 9999 had no lookup

    out_records = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert out_records[0]["linked_compounds"][0]["molecular_formula"] == "C9H8O4"
    assert out_records[0]["linked_compounds"][0]["molecular_weight"] == 180.04225873
    assert out_records[0]["linked_compounds"][1]["molecular_formula"] == "C4H11N5"
    # Unknown CID left alone
    assert out_records[1]["linked_compounds"][0]["molecular_formula"] == ""


def test_patch_jsonl_preserves_other_fields(tmp_path):
    input_path = tmp_path / "in.jsonl"
    record = {
        "pmid": 42,
        "title": "Some title",
        "abstract": "Some abstract",
        "qa_categories": ["mechanism_of_action"],
        "linked_compounds": [
            {"cid": 2244, "name": "Aspirin",
             "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
             "molecular_formula": "", "molecular_weight": None,
             "mesh_terms": ["Aspirin"]},
        ],
    }
    input_path.write_text(json.dumps(record) + "\n")

    cid_mass_map = {2244: ("C9H8O4", 180.04)}
    output_path = tmp_path / "out.jsonl"
    patch_jsonl(input_path, output_path, cid_mass_map)

    out = json.loads(output_path.read_text().strip())
    assert out["pmid"] == 42
    assert out["title"] == "Some title"
    assert out["qa_categories"] == ["mechanism_of_action"]
    c = out["linked_compounds"][0]
    assert c["molecular_formula"] == "C9H8O4"
    assert c["molecular_weight"] == 180.04
    assert c["smiles"] == "CC(=O)OC1=CC=CC=C1C(=O)O"
    assert c["mesh_terms"] == ["Aspirin"]
