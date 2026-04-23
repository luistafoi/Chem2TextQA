from pathlib import Path
from typing import Iterator, Union

from chem2textqa.models.document import CompoundRecord, ScientificDocument

Record = Union[CompoundRecord, ScientificDocument]


def append_documents(path: Path, documents: list[Record]) -> int:
    """Append documents or compound records to a JSONL file. Returns count written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "a", encoding="utf-8") as f:
        for doc in documents:
            f.write(doc.model_dump_json() + "\n")
            count += 1
    return count


def read_documents(path: Path) -> Iterator[ScientificDocument]:
    """Lazily read ScientificDocuments from a JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield ScientificDocument.model_validate_json(line)


def read_compounds(path: Path) -> Iterator[CompoundRecord]:
    """Lazily read CompoundRecords from a JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield CompoundRecord.model_validate_json(line)


def count_documents(path: Path) -> int:
    """Count records in a JSONL file without loading all into memory."""
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())
