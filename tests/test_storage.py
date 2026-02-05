from pathlib import Path

from chem2textqa.models.document import ScientificDocument, SourceType
from chem2textqa.storage.jsonl import append_documents, count_documents, read_documents


def _make_doc(title: str = "Test Doc") -> ScientificDocument:
    return ScientificDocument(source=SourceType.PUBMED, title=title)


def test_write_and_read_roundtrip(tmp_path):
    path = tmp_path / "test.jsonl"
    docs = [_make_doc(f"Doc {i}") for i in range(3)]

    written = append_documents(path, docs)
    assert written == 3

    read_back = list(read_documents(path))
    assert len(read_back) == 3
    assert read_back[0].title == "Doc 0"
    assert read_back[2].title == "Doc 2"


def test_append_mode(tmp_path):
    path = tmp_path / "test.jsonl"

    append_documents(path, [_make_doc("First")])
    append_documents(path, [_make_doc("Second"), _make_doc("Third")])

    read_back = list(read_documents(path))
    assert len(read_back) == 3
    assert read_back[0].title == "First"
    assert read_back[2].title == "Third"


def test_count_documents(tmp_path):
    path = tmp_path / "test.jsonl"

    assert count_documents(path) == 0  # file doesn't exist yet

    append_documents(path, [_make_doc() for _ in range(5)])
    assert count_documents(path) == 5


def test_creates_parent_dirs(tmp_path):
    path = tmp_path / "nested" / "dir" / "test.jsonl"
    append_documents(path, [_make_doc()])
    assert path.exists()
    assert count_documents(path) == 1


def test_empty_list_writes_nothing(tmp_path):
    path = tmp_path / "test.jsonl"
    written = append_documents(path, [])
    assert written == 0
