"""Merge all 4 phases into a single human-readable dataset.

For each compound:
  - structure (SMILES, formula, MW, InChIKey — withheld names)
  - evidence sentences used as input
  - list of QA pairs, each with:
      * question, topic
      * phase1_answer (LLM1 — generator)
      * phase2_answer (LLM2 — blind independent answer)
      * verdict (agree / disagree / unclear) + judge reasoning

Output is one JSON per compound, plus a combined JSONL. Also produces
a summary.json with dataset-wide statistics.
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from chem2textqa.qa_pipeline.config import (
    PHASE_0_EVIDENCE,
    PHASE_1_QA,
    PHASE_2_QA,
    PHASE_3_VALIDATED,
    QA_DIR,
)

logger = logging.getLogger(__name__)


@dataclass
class AssembleStats:
    compounds: int = 0
    total_qa: int = 0
    agree: int = 0
    disagree: int = 0
    unclear: int = 0
    missing_verdict: int = 0
    topics: Counter = field(default_factory=Counter)


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def assemble_dataset(
    phase0_path: Path = PHASE_0_EVIDENCE,
    phase1_path: Path = PHASE_1_QA,
    phase2_path: Path = PHASE_2_QA,
    phase3_path: Path = PHASE_3_VALIDATED,
    output_jsonl: Path = QA_DIR / "dataset_final.jsonl",
    output_json: Path = QA_DIR / "dataset_final.json",
    summary_path: Path = QA_DIR / "dataset_summary.json",
    agree_only: bool = False,
) -> AssembleStats:
    """Merge all phase outputs into a per-compound record.

    Args:
        agree_only: if True, only include QA pairs with judge verdict "agree"
            (the gold subset).
    """
    output_jsonl = Path(output_jsonl)
    output_json = Path(output_json)
    summary_path = Path(summary_path)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    # Phase 0: evidence per compound (the "spine")
    logger.info("Loading Phase 0 evidence...")
    p0_records = _load_jsonl(Path(phase0_path))
    by_cid: dict[int, dict] = {int(r["cid"]): r for r in p0_records if "cid" in r}

    # Phase 1: QA pairs per compound (qa_index maps by position 1..N)
    logger.info("Loading Phase 1 QA pairs...")
    p1_records = _load_jsonl(Path(phase1_path))
    p1_by_cid: dict[int, dict] = {int(r["cid"]): r for r in p1_records if "cid" in r}

    # Phase 2 + 3 are per-question rows keyed by (cid, qa_index)
    logger.info("Loading Phase 2 independent answers...")
    p2_records = _load_jsonl(Path(phase2_path))
    p2_by_key: dict[tuple[int, int], dict] = {
        (int(r["cid"]), int(r["qa_index"])): r
        for r in p2_records if "cid" in r and "qa_index" in r
    }

    logger.info("Loading Phase 3 verdicts...")
    p3_records = _load_jsonl(Path(phase3_path))
    p3_by_key: dict[tuple[int, int], dict] = {
        (int(r["cid"]), int(r["qa_index"])): r
        for r in p3_records if "cid" in r and "qa_index" in r
    }

    stats = AssembleStats()

    # Merge per compound
    all_records: list[dict] = []
    for cid, p0 in by_cid.items():
        p1 = p1_by_cid.get(cid)
        if not p1:
            continue

        qa_pairs_out = []
        for i, qa in enumerate(p1.get("qa_pairs", []) or [], start=1):
            key = (cid, i)
            p2 = p2_by_key.get(key, {})
            p3 = p3_by_key.get(key, {})

            verdict = p3.get("verdict")
            if agree_only and verdict != "agree":
                continue

            if verdict == "agree":
                stats.agree += 1
            elif verdict == "disagree":
                stats.disagree += 1
            elif verdict == "unclear":
                stats.unclear += 1
            else:
                stats.missing_verdict += 1

            topic = qa.get("topic", "other")
            stats.topics[topic] += 1
            stats.total_qa += 1

            qa_pairs_out.append({
                "qa_index": i,
                "topic": topic,
                "question": qa.get("question", ""),
                "phase1_answer": qa.get("answer", ""),
                "phase2_answer": p2.get("phase2_answer", ""),
                "verdict": verdict,
                "judge_reasoning": p3.get("judge_reasoning", ""),
            })

        if not qa_pairs_out:
            continue

        stats.compounds += 1
        record = {
            "cid": cid,
            "name": p0.get("name", ""),
            "iupac_name": p0.get("iupac_name", ""),
            "smiles": p0.get("smiles", ""),
            "molecular_formula": p0.get("molecular_formula", ""),
            "molecular_weight": p0.get("molecular_weight"),
            "inchi_key": p0.get("inchi_key", ""),
            "num_pmids": p0.get("num_pmids", 0),
            "num_synonyms": p0.get("num_synonyms", 0),
            "num_evidence_sentences": len(p0.get("evidence_sentences", []) or []),
            "evidence_sentences": p0.get("evidence_sentences", []) or [],
            "qa_pairs": qa_pairs_out,
        }
        all_records.append(record)

    # Sort by CID for deterministic output
    all_records.sort(key=lambda r: r["cid"])

    # Write JSONL (streaming-friendly)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Write pretty JSON (human-readable array) — small datasets only
    if len(all_records) <= 5000:
        with output_json.open("w", encoding="utf-8") as f:
            json.dump(all_records, f, indent=2, ensure_ascii=False)
    else:
        logger.info("Skipping pretty JSON (%d records > 5000); use JSONL",
                     len(all_records))
        # Remove the stale pretty JSON if it exists
        if output_json.exists():
            output_json.unlink()

    # Summary
    summary = {
        "compounds": stats.compounds,
        "total_qa_pairs": stats.total_qa,
        "verdicts": {
            "agree": stats.agree,
            "disagree": stats.disagree,
            "unclear": stats.unclear,
            "missing": stats.missing_verdict,
        },
        "topics": dict(stats.topics.most_common()),
        "agree_rate": (stats.agree / stats.total_qa) if stats.total_qa else 0.0,
        "outputs": {
            "jsonl": str(output_jsonl),
            "json": str(output_json) if len(all_records) <= 5000 else None,
        },
        "filtered_to_agree_only": agree_only,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return stats
