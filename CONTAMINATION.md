# Contamination analysis plan

## Why contamination matters for this dataset

Chem2TextQA draws evidence sentences from PubMed abstracts and PMC
open-access full text. These corpora are present in the pretraining data
of every frontier LLM called by this pipeline. The dataset therefore
risks being a **sophisticated paraphrase** of what the Q&A-generating
models already know. A model fine-tuned on it may show downstream
accuracy gains that reflect format exposure rather than knowledge
transfer.

Contamination must be surfaced explicitly and quantified.

## Canary methodology

We will construct a **post-cutoff canary set** and report accuracy on it
separately from the main dataset.

### Step 1 — Fix the reference cutoff

The pretraining cutoffs of the three models:
- Gemini 3 Flash preview — approximately early 2025 per model card
- Kimi K2.5 — approximately mid-2025
- Gemma 4 31B — approximately mid-2025

We adopt **2025-09-01** as a conservative post-cutoff boundary. Anything
first reported in literature after this date is treated as not available
to any of the three models during pretraining.

### Step 2 — Identify post-cutoff compounds

A compound is **post-cutoff** if:
1. Its earliest PubMed indexing date for any article in the premium tier
   is on or after 2025-09-01, AND
2. At least 3 articles meeting the Phase 0 evidence criteria exist for
   it, AND
3. Its PubChem CID was created on or after 2025-06-01 (proxy for
   compound-discovery recency).

This produces a set of compounds whose evidence is temporally disjoint
from any of the pretraining corpora.

### Step 3 — Run the pipeline on the canary set separately

Same prompts, same models, same Phase 0-3 flow; emit
`data/qa_pipeline/canary/evidence_per_cid.jsonl` and downstream.

### Step 4 — Report per-bucket accuracy on the canary vs the main set

For the human-evaluated sample (scheduled for Round 2):
- Sample 50 Q&A pairs from the main (pre-cutoff) dataset.
- Sample 50 Q&A pairs from the canary (post-cutoff) dataset.
- Blind-annotate correctness (two annotators, kappa reported).
- Report bucketed accuracy by {structural, functional} × {main, canary}.

If **structural** accuracy is similar on main and canary, structural
claims are genuinely SMILES-derivable and transfer.

If **functional** accuracy drops sharply on the canary, we have direct
evidence that the pipeline's functional claims are memorization-heavy
and the dataset's evaluative claim must be narrowed to structural-only
instruction tuning.

## Implementation status

- `scripts/build_contamination_canary.py` — identifies post-cutoff
  compounds from PubMed indexing dates. **Implemented and run.**
- Canary evidence split: `phase0_full_premium_v3/evidence_canary.jsonl`
  (120 compounds) and `evidence_main.jsonl` (15,547 compounds).
  **Produced.**
- Canary pass in the master script: `run_qa_full_premium.sh` now runs
  Phase 1-3 separately on the canary evidence, producing
  `full_premium_kimi/canary/dataset_final.jsonl`. **Implemented.**
- `scripts/analyze_canary_results.py` — compares accuracy on canary vs
  main after human evaluation. **Deferred (requires human annotators).**

## Realized canary size and its limitation

Using cutoff 2024-01-01 (minimum feasible given PubMed baseline date
range in the premium tier), we retain:
- 154 compounds whose earliest premium-tier article is ≥2024-01-01
  and which have ≥3 articles.
- 120 of those retain usable evidence after Phase 0 extraction.

**This is a "late-training" canary, not a true post-cutoff canary.**
Gemini 3 Flash preview, Kimi K2.5, and Gemma 4 31B have training
cutoffs in mid-2025, so compounds first reported in 2024 may be in
their training data. A stricter canary (≥2025-09-01) yields zero
compounds with sufficient evidence, so we accept the weaker test as
the best available.

**Interpretation.** If structural accuracy on the canary is comparable
to the main set, the pipeline's structural reasoning genuinely
transfers. If functional accuracy on the canary is meaningfully lower
than on the main set, that is direct evidence of memorization-heavy
functional claims, and the dataset's evaluative claim should be
narrowed accordingly. This comparison will be published after the
human-evaluation round.
