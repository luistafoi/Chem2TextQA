"""Phase 2: independent answers from LLM2.

For each (cid, question) from Phase 1, ask a DIFFERENT model to answer
the same question using the same redacted evidence — without ever seeing
Phase 1's answer. This enables Phase 3 cross-validation of whether the
two models independently reached the same conclusion.

Input:  phase_1_qa/qa_pairs.jsonl
Output: phase_2_independent/qa_independent.jsonl  (same shape + phase2_answer fields)
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
    DEFAULT_PHASE2_MODEL,
    PHASE_1_QA,
    PHASE_2_ERRORS,
    PHASE_2_QA,
)
from chem2textqa.qa_pipeline.openrouter import OpenRouterClient

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are an expert medicinal chemist answering a single \
question about a compound.

You receive the SMILES, molecular formula, and molecular weight, plus a \
list of redacted literature sentences about the compound (PRIVATE topic \
hints — absorbed silently, never quoted).

RULES:
- Refer to the molecule only as "the compound". Never name it or guess \
its identity.
- Structural claims (scaffold, functional groups, descriptors, \
stereochemistry, reactivity, physicochemical properties) must be \
derivable from the SMILES / formula / MW.
- Functional claims (mechanism, metabolism, therapeutic use, toxicity, \
drug interactions, ADME, design/engineering) may be supported by the \
evidence sentences, used silently as background knowledge — you answer as \
an expert who happens to know these facts.
- Do NOT quote, paraphrase, or cite the evidence. Do NOT use phrases like \
"as reported", "according to the article", "in this study", "[E1]", etc. \
No markers, no pointers.
- The reader should not see the evidence as literature. Write as a \
knowledgeable chemist, not a literature reviewer.
- Answer should be 1–4 sentences, specific, not encyclopedic.

OUTPUT FORMAT (JSON only):
{
  "answer": "<answer grounded in SMILES (structural) and/or evidence (functional)>"
}"""


def build_user_prompt(
    question: str,
    smiles: str,
    molecular_formula: str,
    molecular_weight,
    evidence_sentences: list[dict],
) -> str:
    mw_str = f"{molecular_weight:.2f}" if isinstance(molecular_weight, (int, float)) else "?"
    hints_text = "\n".join(f"- {e['text']}" for e in evidence_sentences)
    return (
        f"SMILES: {smiles}\n"
        f"Molecular formula: {molecular_formula}\n"
        f"Molecular weight: {mw_str}\n\n"
        f"[PRIVATE TOPIC HINT — DO NOT MENTION OR QUOTE]\n"
        f"{hints_text}\n"
        f"[END PRIVATE TOPIC HINT]\n\n"
        f"QUESTION: {question}\n\n"
        f"Answer as an expert medicinal chemist. Structural claims from the "
        f"SMILES; functional claims (mechanism, metabolism, therapeutic use, "
        f"toxicity, drug interactions, design) may be grounded silently in the "
        f"evidence — never quote, paraphrase, or cite it. Output JSON only."
    )


@dataclass
class Phase2Stats:
    total_questions: int = 0
    already_done: int = 0
    succeeded: int = 0
    failed: int = 0


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


def _iter_questions(phase1_path: Path):
    """Yield (compound_record, qa_index, qa_dict) triples.

    qa_index is a 1-based position within a compound; it's our stable key
    since the new prompt produces a flexible number of Q&A per compound.
    """
    with phase1_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            for i, qa in enumerate(rec.get("qa_pairs", []) or [], start=1):
                yield rec, i, qa


def load_processed_keys(output_path: Path) -> set[str]:
    """Resume support: which (cid, qa_index) pairs have already been answered."""
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


async def answer_one_question(
    client: OpenRouterClient,
    session: aiohttp.ClientSession,
    compound_record: dict,
    qa_index: int,
    qa: dict,
) -> tuple[dict | None, str | None]:
    """Generate LLM2's independent answer for one (compound, question)."""
    question = qa.get("question", "")
    if not question:
        return None, "missing question"

    user_prompt = build_user_prompt(
        question=question,
        smiles=compound_record.get("smiles") or "",
        molecular_formula=compound_record.get("molecular_formula") or "",
        molecular_weight=compound_record.get("molecular_weight"),
        evidence_sentences=compound_record.get("evidence_sentences", []) or [],
    )

    response, error = await client.complete(
        session, SYSTEM_PROMPT, user_prompt,
        max_tokens=1000, temperature=0.3,
        # Disable reasoning — hybrid models like Kimi K2.5 otherwise burn
        # the whole token budget on internal reasoning and emit empty content.
        reasoning={"enabled": False},
    )
    if error:
        return None, f"api error: {error}"
    if not response:
        return None, "empty response"

    parsed = _extract_json(response)
    if parsed is None:
        return None, "could not parse JSON"

    phase2_answer = parsed.get("answer", "") or ""
    if not isinstance(phase2_answer, str) or not phase2_answer.strip():
        return None, "empty answer"

    return {
        "cid": compound_record["cid"],
        "smiles": compound_record.get("smiles", ""),
        "qa_index": qa_index,
        "topic": qa.get("topic", "other"),
        "question": question,
        "phase1_answer": qa.get("answer", ""),
        "phase2_answer": phase2_answer,
    }, None


async def run_phase_2(
    input_path: Path = PHASE_1_QA,
    output_path: Path = PHASE_2_QA,
    error_path: Path = PHASE_2_ERRORS,
    model: str = DEFAULT_PHASE2_MODEL,
    workers: int = 20,
    api_key: str | None = None,
) -> Phase2Stats:
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

    done_keys = load_processed_keys(output_path)
    stats = Phase2Stats(already_done=len(done_keys))

    todo = []
    for compound_rec, qa_index, qa in _iter_questions(input_path):
        cid = compound_rec.get("cid")
        if cid is None:
            continue
        if f"{cid}:{qa_index}" in done_keys:
            continue
        todo.append((compound_rec, qa_index, qa))

    stats.total_questions = len(todo) + stats.already_done
    logger.info("Phase 2: %d questions to process (%d done)",
                 len(todo), stats.already_done)

    async with aiohttp.ClientSession() as session:
        out_f = output_path.open("a", encoding="utf-8")
        err_f = error_path.open("a", encoding="utf-8")
        out_lock = asyncio.Lock()

        async def _process(rec, qa_index, qa):
            result, err = await answer_one_question(
                client, session, rec, qa_index, qa
            )
            async with out_lock:
                if result is not None:
                    out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    out_f.flush()
                    stats.succeeded += 1
                else:
                    err_f.write(json.dumps({
                        "cid": rec.get("cid"),
                        "qa_index": qa_index,
                        "error": err,
                    }) + "\n")
                    err_f.flush()
                    stats.failed += 1

        tasks = [_process(rec, qa_index, qa) for rec, qa_index, qa in todo]
        await tqdm_asyncio.gather(*tasks, desc="Phase 2")

        out_f.close()
        err_f.close()

    return stats
