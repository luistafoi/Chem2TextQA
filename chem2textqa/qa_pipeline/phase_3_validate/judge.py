"""Phase 3: cross-validation judge.

For each (cid, question) from Phase 2, ask LLM3 (the judge) two questions:
  1. Does LLM1's answer agree with LLM2's answer?
  2. Are both answers consistent with the cited evidence?

The judge gives a verdict in {agree, disagree, unclear} plus a brief
reason. Records that pass both checks are the "gold" subset.

Input:  phase_2_independent/qa_independent.jsonl
Output: phase_3_validate/validated.jsonl
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import aiohttp
from tqdm.asyncio import tqdm_asyncio

from chem2textqa.qa_pipeline.config import (
    DEFAULT_PHASE3_MODEL,
    PHASE_2_QA,
    PHASE_3_VALIDATED,
)
from chem2textqa.qa_pipeline.openrouter import OpenRouterClient
from chem2textqa.qa_pipeline.phase_3_validate.heuristic import classify as heuristic_classify

logger = logging.getLogger(__name__)


JUDGE_SYSTEM_PROMPT = """You are a judge evaluating whether two independent \
answers to the same question about a chemical compound agree with each other.

Both answers were produced blind to each other — they never saw each other's \
reply. Your job is to classify the agreement level as agree, disagree, or \
unclear.

Use:
  "agree"    — both answers reach the same substantive conclusion on the main \
claim, even if phrased differently or with different levels of detail. Minor \
wording differences, partial overlap on supplementary details, or one answer \
being more concise than the other still counts as agree.
  "disagree" — the answers contradict each other on the main claim (e.g., one \
says yes and the other says no; one says 3 rings and the other says 1 ring).
  "unclear"  — the two answers address different aspects of the question, one \
says "N/A" / "cannot determine", or both are so vague that comparing them is \
not meaningful.

WORKED EXAMPLES:

Example A (agree):
  QUESTION: How many rotatable bonds does the compound have?
  ANSWER 1: Approximately 5 rotatable bonds, located in the aliphatic chain \
and the ester linkage.
  ANSWER 2: The compound has about 5 rotatable bonds, primarily in the side chain.
  → {"verdict": "agree", "reasoning": "Both report the same count (~5) and \
attribute it to the same flexible region; wording differs but substance matches."}

Example B (disagree):
  QUESTION: Does the compound contain a nitro group?
  ANSWER 1: Yes, there is a nitro group attached to the aromatic ring.
  ANSWER 2: No, the compound has no nitro functionality; the nitrogen is part \
of an amine.
  → {"verdict": "disagree", "reasoning": "Direct contradiction on whether a \
nitro group is present."}

Example C (unclear):
  QUESTION: What is the aromaticity pattern of the compound?
  ANSWER 1: The molecule contains one aromatic benzene ring.
  ANSWER 2: N/A — the question cannot be answered without 3D coordinates.
  → {"verdict": "unclear", "reasoning": "One answer addresses the question, \
the other declines to answer, so they cannot be compared."}

Output JSON only, matching exactly this shape:
{
  "verdict": "agree" | "disagree" | "unclear",
  "reasoning": "<one or two sentences>"
}"""


def _judge_user_prompt(question: str, a1: str, a2: str) -> str:
    return (
        f"QUESTION: {question}\n\n"
        f"ANSWER 1: {a1}\n\n"
        f"ANSWER 2: {a2}\n\n"
        f"Do the two answers agree? Output JSON only."
    )


@dataclass
class Phase3Stats:
    total: int = 0
    already_done: int = 0
    agree: int = 0
    disagree: int = 0
    unclear: int = 0
    failed: int = 0
    heuristic_agree: int = 0
    heuristic_unclear: int = 0
    llm_calls: int = 0


def _extract_json(response: str) -> dict | None:
    if not response:
        return None
    text = response.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0]
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return None


def load_processed_keys(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    done: set[str] = set()
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                cid = rec.get("cid")
                idx = rec.get("qa_index")
                if cid is not None and idx is not None:
                    done.add(f"{cid}:{idx}")
            except json.JSONDecodeError:
                continue
    return done


async def judge_one_pair(
    client: OpenRouterClient,
    session: aiohttp.ClientSession,
    record: dict,
    use_heuristic: bool = True,
) -> tuple[dict | None, str | None, str]:
    """Judge a single (question, a1, a2) record.

    Returns (augmented_record_or_None, error_or_None, source) where source is
    "heuristic" if the verdict came from the token-overlap pre-filter, or
    "llm" if the judge model was called.
    """
    a1 = record.get("phase1_answer", "") or ""
    a2 = record.get("phase2_answer", "") or ""
    question = record.get("question", "") or ""

    if not (question and a1 and a2):
        return None, "missing question or answers", "heuristic"

    if use_heuristic:
        h = heuristic_classify(a1, a2)
        if h.verdict is not None:
            out = dict(record)
            out["verdict"] = h.verdict
            out["judge_reasoning"] = h.reason
            out["judge_source"] = "heuristic"
            return out, None, "heuristic"

    response, error = await client.complete(
        session, JUDGE_SYSTEM_PROMPT,
        _judge_user_prompt(question, a1, a2),
        max_tokens=400, temperature=0.0,
        reasoning={"enabled": False},
    )
    if error:
        return None, f"api error: {error}", "llm"
    if not response:
        return None, "empty response", "llm"

    parsed = _extract_json(response)
    if parsed is None:
        return None, "could not parse JSON", "llm"

    verdict = (parsed.get("verdict") or "").strip().lower()
    if verdict not in ("agree", "disagree", "unclear"):
        return None, f"invalid verdict: {verdict!r}", "llm"

    out = dict(record)
    out["verdict"] = verdict
    out["judge_reasoning"] = parsed.get("reasoning", "")
    out["judge_source"] = "llm"
    return out, None, "llm"


async def run_phase_3(
    input_path: Path = PHASE_2_QA,
    output_path: Path = PHASE_3_VALIDATED,
    model: str = DEFAULT_PHASE3_MODEL,
    workers: int = 20,
    api_key: str | None = None,
    use_heuristic: bool = True,
) -> Phase3Stats:
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    client = OpenRouterClient(api_key=api_key, model=model)
    client.semaphore = asyncio.Semaphore(workers)

    done_keys = load_processed_keys(output_path)
    stats = Phase3Stats(already_done=len(done_keys))

    todo = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = rec.get("cid")
            idx = rec.get("qa_index")
            if cid is None or idx is None:
                continue
            if f"{cid}:{idx}" in done_keys:
                continue
            todo.append(rec)

    stats.total = len(todo) + stats.already_done

    async with aiohttp.ClientSession() as session:
        out_f = output_path.open("a", encoding="utf-8")
        out_lock = asyncio.Lock()

        async def _process(rec):
            result, err, source = await judge_one_pair(
                client, session, rec, use_heuristic=use_heuristic,
            )
            async with out_lock:
                if result is None:
                    stats.failed += 1
                    return
                verdict = result["verdict"]
                if verdict == "agree":
                    stats.agree += 1
                elif verdict == "disagree":
                    stats.disagree += 1
                else:
                    stats.unclear += 1
                if source == "heuristic":
                    if verdict == "agree":
                        stats.heuristic_agree += 1
                    else:
                        stats.heuristic_unclear += 1
                else:
                    stats.llm_calls += 1
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()

        tasks = [_process(rec) for rec in todo]
        await tqdm_asyncio.gather(*tasks, desc="Phase 3")

        out_f.close()

    return stats
