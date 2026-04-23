"""Sentence splitter, ported from Robert's pipeline.

Handles common abbreviations that would otherwise cause split errors
(initials, "et al.", "Fig.", decimal numbers, etc.).
"""
from __future__ import annotations

import re

_ABBR_RE = re.compile(
    r"(Fig|Tab|Ref|No|vs|Dr|Mr|Mrs|Ms|approx|ca|cf|eg|ie|etc)\.",
    re.IGNORECASE,
)


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences using regex-based heuristics.

    Replaces dots in common abbreviations with a placeholder, splits on
    remaining terminal punctuation + whitespace, then restores the dots.
    """
    if not text:
        return []

    # Initials like "A. Smith"
    t = re.sub(r"(\b[A-Z])\.", r"\1<DOT>", text)
    # "et al."
    t = re.sub(r"(et al)\.", r"\1<DOT>", t)
    # Common abbreviations
    t = _ABBR_RE.sub(lambda m: m.group(1) + "<DOT>", t)
    # Decimal numbers like "3.14"
    t = re.sub(r"(\d)\.(\d)", r"\1<DOT>\2", t)

    # Split on sentence-terminal punctuation followed by whitespace.
    parts = re.split(r"(?<=[.!?])\s+", t)

    sentences = []
    for s in parts:
        s = s.replace("<DOT>", ".").strip()
        if s:
            sentences.append(s)
    return sentences
