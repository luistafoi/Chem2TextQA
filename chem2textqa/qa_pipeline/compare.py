"""Compare two QA datasets (typically: baseline vs experimental Phase 1 model).

Reads the `dataset_summary.json` + `dataset_final.jsonl` from two runs and
reports: agree-rate delta, per-compound disagreement deltas, per-topic
disagreement rates.
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class RunStats:
    name: str
    compounds: int = 0
    total_qa: int = 0
    agree: int = 0
    disagree: int = 0
    unclear: int = 0
    per_compound_agree: dict[int, int] = None
    per_compound_total: dict[int, int] = None
    per_topic_agree: Counter = None
    per_topic_total: Counter = None

    def agree_rate(self) -> float:
        return self.agree / self.total_qa if self.total_qa else 0.0


def load_run(dataset_jsonl: Path, label: str) -> RunStats:
    stats = RunStats(
        name=label,
        per_compound_agree={},
        per_compound_total={},
        per_topic_agree=Counter(),
        per_topic_total=Counter(),
    )
    with dataset_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cid = rec.get("cid")
            if cid is None:
                continue
            stats.compounds += 1
            stats.per_compound_agree.setdefault(cid, 0)
            stats.per_compound_total.setdefault(cid, 0)
            for qa in rec.get("qa_pairs", []) or []:
                stats.total_qa += 1
                stats.per_compound_total[cid] += 1
                topic = qa.get("topic", "other")
                stats.per_topic_total[topic] += 1
                verdict = qa.get("verdict")
                if verdict == "agree":
                    stats.agree += 1
                    stats.per_compound_agree[cid] += 1
                    stats.per_topic_agree[topic] += 1
                elif verdict == "disagree":
                    stats.disagree += 1
                elif verdict == "unclear":
                    stats.unclear += 1
    return stats


def compare(a: RunStats, b: RunStats) -> dict:
    """Return a dict summarizing differences between two runs."""
    all_cids = set(a.per_compound_total) | set(b.per_compound_total)
    per_compound = []
    for cid in sorted(all_cids):
        a_agree = a.per_compound_agree.get(cid, 0)
        a_total = a.per_compound_total.get(cid, 0)
        b_agree = b.per_compound_agree.get(cid, 0)
        b_total = b.per_compound_total.get(cid, 0)
        per_compound.append({
            "cid": cid,
            f"{a.name}_agree": a_agree,
            f"{a.name}_total": a_total,
            f"{b.name}_agree": b_agree,
            f"{b.name}_total": b_total,
        })

    per_topic = []
    all_topics = set(a.per_topic_total) | set(b.per_topic_total)
    for topic in sorted(all_topics):
        a_agree = a.per_topic_agree[topic]
        a_total = a.per_topic_total[topic]
        b_agree = b.per_topic_agree[topic]
        b_total = b.per_topic_total[topic]
        per_topic.append({
            "topic": topic,
            f"{a.name}_agree_rate": a_agree / a_total if a_total else None,
            f"{b.name}_agree_rate": b_agree / b_total if b_total else None,
            f"{a.name}_total": a_total,
            f"{b.name}_total": b_total,
        })

    return {
        "overall": {
            a.name: {
                "compounds": a.compounds,
                "total_qa": a.total_qa,
                "agree": a.agree,
                "disagree": a.disagree,
                "unclear": a.unclear,
                "agree_rate": a.agree_rate(),
            },
            b.name: {
                "compounds": b.compounds,
                "total_qa": b.total_qa,
                "agree": b.agree,
                "disagree": b.disagree,
                "unclear": b.unclear,
                "agree_rate": b.agree_rate(),
            },
            "delta_agree_rate": b.agree_rate() - a.agree_rate(),
        },
        "per_compound": per_compound,
        "per_topic": per_topic,
    }
