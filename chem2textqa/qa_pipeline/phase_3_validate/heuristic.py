"""Cheap token-overlap pre-filter for Phase 3.

Most QA pairs where LLM1 and LLM2 agree are obvious — the answers share
most content words and neither negates the other. Sending those to the
LLM judge is wasted cost. This module classifies the easy cases locally
and escalates the ambiguous ones to the judge.

Conservative by design: we only auto-emit `agree` verdicts. Anything
that could plausibly be disagree/unclear escalates. In the 500-compound
pilot ~83% of pairs were `agree`; even auto-classifying half of them
cuts Phase 3 cost by ~40%.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Small English stopword set — we want to compare content words.
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but",
    "by", "can", "could", "did", "do", "does", "doing", "done", "for",
    "from", "had", "has", "have", "having", "he", "her", "here", "hers",
    "him", "his", "how", "i", "if", "in", "into", "is", "it", "its",
    "itself", "me", "my", "of", "on", "or", "our", "ours", "she",
    "so", "some", "such", "than", "that", "the", "their", "theirs",
    "them", "then", "there", "these", "they", "this", "those", "to",
    "too", "us", "very", "was", "we", "were", "what", "when", "where",
    "which", "while", "who", "whom", "why", "will", "with", "would",
    "you", "your", "yours", "also", "about", "around", "between",
    "both", "each", "other", "than", "within", "without",
    # Chem-generic hedge words that show up in almost every answer
    "compound", "molecule", "structure", "structural", "chemical",
    "approximately", "roughly", "around", "about",
})

# Explicit negation tokens — a mismatch here is a strong disagree signal.
_NEGATIONS = frozenset({
    "no", "not", "none", "never", "cannot", "isn't", "aren't", "doesn't",
    "don't", "didn't", "won't", "absent", "lacks", "without", "neither",
    "nor",
})

_NA_PATTERNS = (
    re.compile(r"^\s*n\s*/\s*a\s*[.!]?\s*$", re.IGNORECASE),
    re.compile(r"^\s*not\s+applicable\b", re.IGNORECASE),
    re.compile(r"^\s*(i\s+)?cannot\s+(tell|determine|answer)\b", re.IGNORECASE),
    re.compile(r"^\s*unknown\b", re.IGNORECASE),
    re.compile(r"^\s*insufficient\b", re.IGNORECASE),
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _is_na(text: str) -> bool:
    t = text.strip()
    if not t or len(t) < 4:
        return True
    return any(p.match(t) for p in _NA_PATTERNS)


def _content_tokens(text: str) -> set[str]:
    lowered = text.lower()
    toks = _TOKEN_RE.findall(lowered)
    return {t for t in toks if t not in _STOPWORDS and len(t) > 2}


def _negation_count(text: str) -> int:
    lowered = text.lower()
    toks = _TOKEN_RE.findall(lowered)
    return sum(1 for t in toks if t in _NEGATIONS)


@dataclass(frozen=True)
class HeuristicResult:
    verdict: str | None  # "agree", "unclear", or None (escalate)
    jaccard: float
    reason: str


def classify(
    answer1: str,
    answer2: str,
    agree_threshold: float = 0.5,
) -> HeuristicResult:
    """Return a cheap verdict without calling the LLM, or None to escalate.

    Rules (conservative — prefer to escalate):
      - Either side is N/A-like  → unclear
      - Both sides very short (<3 content tokens) → escalate
      - Large mismatch in negation counts → escalate (likely disagree)
      - Jaccard >= agree_threshold AND matching negation parity → agree
      - Otherwise → escalate
    """
    a1 = (answer1 or "").strip()
    a2 = (answer2 or "").strip()

    if _is_na(a1) or _is_na(a2):
        return HeuristicResult(
            verdict="unclear", jaccard=0.0,
            reason="one or both answers are N/A-like",
        )

    t1 = _content_tokens(a1)
    t2 = _content_tokens(a2)
    if len(t1) < 3 or len(t2) < 3:
        return HeuristicResult(
            verdict=None, jaccard=0.0,
            reason="too few content tokens — escalate",
        )

    union = t1 | t2
    inter = t1 & t2
    jaccard = len(inter) / len(union) if union else 0.0

    n1 = _negation_count(a1)
    n2 = _negation_count(a2)
    negation_mismatch = abs(n1 - n2) >= 2 or (n1 == 0) != (n2 == 0)

    if jaccard >= agree_threshold and not negation_mismatch:
        return HeuristicResult(
            verdict="agree", jaccard=jaccard,
            reason=f"jaccard={jaccard:.2f} and matching negation parity",
        )

    return HeuristicResult(
        verdict=None, jaccard=jaccard,
        reason=f"jaccard={jaccard:.2f}, needs LLM judge",
    )
