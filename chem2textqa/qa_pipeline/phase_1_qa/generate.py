"""Phase 1: generate structured Q&A for each compound using LLM1.

Input:  phase_0_evidence/evidence_per_cid.jsonl
Output: phase_1_qa/qa_pairs.jsonl

Each input record is one compound; output is the same compound with a
populated `qa_pairs` list.

Async batching mirrors Robert's pattern: a semaphore caps concurrent
OpenRouter calls, checkpoints are written every N records, and the output
file is append-only so the job is resumable.
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
    DEFAULT_PHASE1_MODEL,
    PHASE_0_EVIDENCE,
    PHASE_1_ERRORS,
    PHASE_1_QA,
)
from chem2textqa.qa_pipeline.openrouter import OpenRouterClient
from chem2textqa.qa_pipeline.phase_1_qa.prompts import (
    QA_TEMPLATES,
    SYSTEM_PROMPT,
    build_user_prompt,
)

logger = logging.getLogger(__name__)


@dataclass
class Phase1Stats:
    total: int = 0
    already_done: int = 0
    succeeded: int = 0
    failed: int = 0
    total_qa_pairs: int = 0


def _extract_json(response: str) -> dict | None:
    """Pull a JSON object out of an LLM response. Handles fenced code blocks."""
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


MIN_QA_PAIRS = 4
MAX_QA_PAIRS = 60


def _validate_qa_output(parsed: dict) -> list[dict] | None:
    """Validate a flexible qa_pairs list.

    Each pair must have a non-empty `question` and non-empty `answer`.
    `topic` is a free-form string tag — any non-empty value is accepted
    (the taxonomy in the prompt is inspiration, not a closed enum).
    Count must fall within [MIN_QA_PAIRS, MAX_QA_PAIRS].
    """
    pairs = parsed.get("qa_pairs") if isinstance(parsed, dict) else None
    if not isinstance(pairs, list):
        return None
    if not (MIN_QA_PAIRS <= len(pairs) <= MAX_QA_PAIRS):
        return None

    cleaned = []
    for p in pairs:
        if not isinstance(p, dict):
            return None
        question = p.get("question", "")
        answer = p.get("answer", "")
        if not isinstance(question, str) or not isinstance(answer, str):
            return None
        if not question.strip() or not answer.strip():
            return None
        topic = (p.get("topic") or "").strip() or "other"
        cleaned.append({
            "topic": topic,
            "question": question.strip(),
            "answer": answer.strip(),
        })
    return cleaned


def load_processed_cids(output_path: Path) -> set[int]:
    """Scan the output JSONL for already-processed CIDs so we can resume."""
    if not output_path.exists():
        return set()
    done: set[int] = set()
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                cid = rec.get("cid")
                if cid is not None:
                    done.add(int(cid))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
    return done


async def generate_for_compound(
    client: OpenRouterClient,
    session: aiohttp.ClientSession,
    record: dict,
) -> tuple[dict | None, str | None]:
    """Call the LLM once for a compound record and return the augmented record."""
    smiles = record.get("smiles") or ""
    mw = record.get("molecular_weight")
    formula = record.get("molecular_formula") or ""
    evidence = record.get("evidence_sentences", []) or []

    if not smiles:
        return None, "missing smiles"
    if not evidence:
        return None, "no evidence sentences"

    user_prompt = build_user_prompt(
        smiles=smiles,
        molecular_formula=formula,
        molecular_weight=mw,
        evidence_sentences=evidence,
    )

    response, error = await client.complete(
        session, SYSTEM_PROMPT, user_prompt,
        max_tokens=10000, temperature=0.3,
    )
    if error:
        return None, f"api error: {error}"
    if not response:
        return None, "empty response"

    parsed = _extract_json(response)
    if parsed is None:
        return None, "could not parse JSON"

    qa_pairs = _validate_qa_output(parsed)
    if qa_pairs is None:
        return None, "qa_pairs failed validation"

    return {
        "cid": record["cid"],
        "smiles": smiles,
        "molecular_formula": formula,
        "molecular_weight": mw,
        "num_pmids": record.get("num_pmids", 0),
        "num_evidence_sentences": len(evidence),
        "evidence_sentences": evidence,
        "qa_pairs": qa_pairs,
    }, None


async def run_phase_1(
    input_path: Path = PHASE_0_EVIDENCE,
    output_path: Path = PHASE_1_QA,
    error_path: Path = PHASE_1_ERRORS,
    model: str = DEFAULT_PHASE1_MODEL,
    workers: int = 20,
    api_key: str | None = None,
) -> Phase1Stats:
    """Drive Phase 1 end-to-end."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    error_path = Path(error_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    error_path.parent.mkdir(parents=True, exist_ok=True)

    api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    client = OpenRouterClient(api_key=api_key, model=model)
    client.semaphore = asyncio.Semaphore(workers)

    done_cids = load_processed_cids(output_path)
    stats = Phase1Stats()
    stats.already_done = len(done_cids)

    # Read all input records up-front — phase 0 output is small (22K records)
    todo: list[dict] = []
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
            if cid is None or int(cid) in done_cids:
                continue
            todo.append(rec)

    stats.total = len(todo) + stats.already_done
    logger.info("Phase 1: %d compounds to process (%d already done)",
                 len(todo), stats.already_done)

    async with aiohttp.ClientSession() as session:
        out_f = output_path.open("a", encoding="utf-8")
        err_f = error_path.open("a", encoding="utf-8")
        out_lock = asyncio.Lock()

        async def _process(rec):
            result, err = await generate_for_compound(client, session, rec)
            async with out_lock:
                if result is not None:
                    out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    out_f.flush()
                    stats.succeeded += 1
                    stats.total_qa_pairs += len(result.get("qa_pairs", []))
                else:
                    err_f.write(json.dumps({
                        "cid": rec.get("cid"),
                        "error": err,
                    }) + "\n")
                    err_f.flush()
                    stats.failed += 1

        tasks = [_process(rec) for rec in todo]
        await tqdm_asyncio.gather(*tasks, desc="Phase 1")

        out_f.close()
        err_f.close()

    return stats
