"""Config for the QA generation pipeline.

Paths for intermediate outputs, default model names, and constants.
All paths are relative to the repo root by default — callers can override.
"""
from __future__ import annotations

from pathlib import Path

# ------------------------------------------------------------------
# Inputs
# ------------------------------------------------------------------

FILTERED_DIR = Path("data/filtered")

# Default tier to run the QA pipeline on. Premium is recommended.
PREMIUM_DATASET = FILTERED_DIR / "drug_articles_v2_premium.jsonl"
STANDARD_DATASET = FILTERED_DIR / "drug_articles_v2_standard.jsonl"
BROAD_DATASET = FILTERED_DIR / "drug_articles_v2_final.jsonl"

# Synonym source — gigabytes of PubChem synonyms (we stream, never load).
CID_SYNONYM_FILE = Path("data/bulk/CID-Synonym-filtered.gz")

# ------------------------------------------------------------------
# Outputs (one directory per phase)
# ------------------------------------------------------------------

QA_DIR = Path("data/qa_pipeline")
PHASE_0_DIR = QA_DIR / "phase_0_evidence"
PHASE_1_DIR = QA_DIR / "phase_1_qa"
PHASE_2_DIR = QA_DIR / "phase_2_independent"
PHASE_3_DIR = QA_DIR / "phase_3_validate"

# Phase 0 artifacts
PHASE_0_SYNONYMS = PHASE_0_DIR / "synonyms_per_cid.jsonl"
PHASE_0_EVIDENCE = PHASE_0_DIR / "evidence_per_cid.jsonl"

# Canonical full-premium Phase 0 output (15,667 compounds with evidence).
# Produced once by `run_phase0_full_premium.sh`. All downstream Phase 1 runs
# should point here via --input so we don't recompute evidence per experiment.
PHASE_0_FULL_PREMIUM = QA_DIR / "phase0_full_premium" / "evidence_per_cid.jsonl"

# Phase 1 artifacts
PHASE_1_QA = PHASE_1_DIR / "qa_pairs.jsonl"
PHASE_1_ERRORS = PHASE_1_DIR / "errors.jsonl"
PHASE_1_CHECKPOINT = PHASE_1_DIR / "checkpoint.json"

# Phase 2 artifacts
PHASE_2_QA = PHASE_2_DIR / "qa_independent.jsonl"
PHASE_2_ERRORS = PHASE_2_DIR / "errors.jsonl"
PHASE_2_CHECKPOINT = PHASE_2_DIR / "checkpoint.json"

# Phase 3 artifacts
PHASE_3_VALIDATED = PHASE_3_DIR / "validated.jsonl"
PHASE_3_CHECKPOINT = PHASE_3_DIR / "checkpoint.json"

# ------------------------------------------------------------------
# Models (via OpenRouter)
# ------------------------------------------------------------------

# Cross-validation only works if Phase 1 and Phase 2 come from different
# model families (different training data → different biases). Don't set
# both to the same provider.
#
# The shell scripts (run_qa_full_premium.sh, run_1000_pilot.sh) pass
# --model explicitly, so these defaults are only used when callers omit
# --model. Prices below are per 1M tokens, input / output (April 2026):
#   Phase 1 — Gemini 3 Flash preview: inexpensive, fast, solid Q&A gen
#   Phase 2 — Kimi K2.5:              $0.38 / $1.72    (diverse peer)
#   Phase 3 — Gemma 4 31B (paid):     $0.13 / $0.38    (cheap classifier)
DEFAULT_PHASE1_MODEL = "google/gemini-3-flash-preview"
DEFAULT_PHASE2_MODEL = "moonshotai/kimi-k2.5"
DEFAULT_PHASE3_MODEL = "google/gemma-4-31b-it"

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

REDACTION_TOKEN = "[COMPOUND]"

# Synonym filtering
MIN_SYNONYM_LENGTH = 4
MAX_SYNONYMS_PER_COMPOUND = 1000  # hard cap — keep regex sane

# Evidence extraction
MAX_EVIDENCE_SENTENCES_PER_COMPOUND = 500
MAX_SENTENCE_LENGTH = 800   # drop absurdly long "sentences" that indicate broken splitting

# Common English / chemistry words we never redact even if they're in synonym lists.
# (Keeps the regex from destroying sentence grammar.)
COMMON_WORD_BLOCKLIST = frozenset({
    "acid", "base", "salt", "ester", "alcohol", "amine", "amide",
    "ether", "water", "oxygen", "hydrogen", "carbon", "nitrogen",
    "methyl", "ethyl", "phenyl", "alkyl", "aryl",
    "drug", "compound", "molecule", "agent", "substance",
    "peptide", "protein", "enzyme", "inhibitor", "activator",
    "factor", "receptor", "vitamin", "ion", "hormone",
    "sugar", "lipid", "gene", "dna", "rna",
    "crystal", "powder", "solid", "liquid", "gas",
    "isomer", "racemate", "mixture", "solution",
    # Common chemistry prefixes that shouldn't be redacted standalone
    "alpha", "beta", "gamma", "delta", "omega",
    "cis", "trans", "meta", "para", "ortho",
})
