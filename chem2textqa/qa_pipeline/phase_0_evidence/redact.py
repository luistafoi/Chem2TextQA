"""Compile synonym lists into a redaction regex and apply it to text.

Synonyms can contain regex metacharacters (parentheses, dots, brackets),
so we `re.escape` each. We sort longest-first so overlapping matches
prefer the most specific synonym; that way "acetylsalicylic acid" is
redacted as a whole before "salicylic acid" inside it.
"""
from __future__ import annotations

import re
from typing import Pattern

from chem2textqa.qa_pipeline.config import REDACTION_TOKEN


def compile_redaction_regex(synonyms: list[str]) -> Pattern[str] | None:
    """Return a compiled, case-insensitive regex that matches any of the
    given synonyms as whole words. Returns None if the synonym list is
    empty (nothing to redact).
    """
    if not synonyms:
        return None

    # Sort longest-first so alternation prefers specific matches.
    ordered = sorted(synonyms, key=lambda s: (-len(s), s))
    escaped = [re.escape(s) for s in ordered]
    # Whole-word-ish boundaries. We use lookbehind/lookahead on word chars so
    # we don't split hyphenated or parenthesized synonyms inside themselves.
    pattern = r"(?<!\w)(?:" + "|".join(escaped) + r")(?!\w)"
    return re.compile(pattern, re.IGNORECASE)


def redact(text: str, pattern: Pattern[str] | None,
           token: str = REDACTION_TOKEN) -> str:
    """Replace every match of `pattern` in `text` with `token`."""
    if not pattern or not text:
        return text
    return pattern.sub(token, text)


def count_hits(text: str, pattern: Pattern[str] | None) -> int:
    """Count how many synonym hits a piece of text contains."""
    if not pattern or not text:
        return 0
    return len(pattern.findall(text))
