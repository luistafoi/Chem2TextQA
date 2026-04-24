# Limitations of Chem2TextQA

A complete, uncomfortable list. Anyone considering using this dataset as
training data must read this in full.

## 1. Labels are LLM-consensus, not correctness

The "gold" subset is defined as Q&A pairs where Phase 1 (Gemini 3 Flash
preview) and Phase 2 (Kimi K2.5) produce answers that Phase 3 (Gemma 4
31B) classifies as "agree". This measures inter-model consistency, not
factual correctness. Two LLMs with overlapping pretraining can agree on
the same wrong claim with high confidence.

Until a human-evaluated sample is reported (scheduled for Round 2), the
agree rate (~88%) should be treated as an upper bound on correctness,
not a direct proxy.

## 2. Training-data contamination is pervasive

Every evidence sentence comes from PubMed or PMC. These corpora are
near-certainly present in the pretraining data of every frontier LLM,
including the three models in this pipeline. Consequences:

- A model fine-tuned on Chem2TextQA may exhibit stronger apparent
  performance on chemistry tasks without gaining transferable knowledge
  — the improvements may reflect exposure to a specific Q&A format
  applied to knowledge the model already has.
- A functional claim marked "grounded in evidence" may in fact be the
  model recalling a training-time fact, with the evidence sentence
  merely labeling the topic.
- The empty-evidence probe shows the model produces ~27% of its normal
  functional-Q&A volume with zero evidence present, confirming training
  recall is at play.

See `CONTAMINATION.md` for the proposed canary-based validation
methodology.

## 3. The "soft rule" permits training-recall

Phase 1 and Phase 2 system prompts explicitly allow functional claims
"supported by the evidence ... used silently as background knowledge".
This phrasing admits recall from pretraining. There is no mechanism at
generation time to distinguish a claim supported by a specific evidence
sentence from one the model would have made anyway. A Phase 4 grounding
check (LLM-based claim-to-evidence alignment scoring) is scheduled for
Round 2 but is not yet implemented.

## 4. Compound coverage is biased toward well-studied drugs

The premium tier draws from MeSH drug headings ∪ DrugBank ∪ HMDB ∪ KEGG
∪ ChEBI ∪ BindingDB ∪ ChEMBL.

**Measured distribution on v3 (15,667 compounds):**

Molecular weight:
| Bucket | Share |
|---|---|
| 300–499 Da (drug-like) | 33.9% |
| 150–299 Da (Lipinski small) | 32.1% |
| <150 Da (fragments, ions) | 13.5% |
| 500–799 Da (large drug-like) | 12.6% |
| 800–1499 Da (macrocyclic / peptide) | 5.8% |
| ≥1500 Da (biologic / large peptide) | 2.2% |

Evidence concentration:
- Mean sentences/compound: **71.0**
- Median: **7** (heavily right-skewed)
- p90: **281**, p99: **500 (cap)**
- **Top 10% of compounds hold 65% of total evidence.**

This means the training signal is dominated by well-studied drugs. A
model fine-tuned on Chem2TextQA will be disproportionately shaped by
high-citation-count compounds and should be expected to fail silently
on out-of-distribution chemistry (orphan drugs, research chemicals,
natural-product intermediates). Users wanting balanced coverage should
consider downsampling compounds with ≥p90 sentence counts before
fine-tuning.

## 5. Redaction coverage (measured)

Synonyms used to build the compound-mention regex are drawn from
PubChem's `CID-Synonym-filtered.gz`, MeSH headings, primary PubChem
name, and IUPAC name.

**Measured leak rate on v3 evidence (~1.1M redacted sentences):**

| Held-out synonym source | Sentences with leak | Sentence-level leak rate |
|---|---|---|
| IUPAC name | 315 / 1,105,522 | **0.03%** |
| Primary name | 768 / 1,112,426 | **0.07%** |

Both are well under the 5% threshold that would compromise Phase 2's
blind-re-answer premise. Leakage via trade names, non-English INNs, and
target-derived descriptors is not directly measured here; users with an
external lexicon of interest can run `scripts/audit_redaction.py
external --lexicon <file>` for a per-compound leak-rate report.

## 6. Pipeline output is stochastic (measured)

- Phase 1 temperature = 0.3.
- Evidence sentence sampling uses a CID-seeded RNG (deterministic if
  re-run with the same cap/pool, but resampling across different
  random-sampling caps will yield different evidence).
- Phase 2 and Phase 3 temperatures are 0.3 and 0.0 respectively.

**Measured seed-to-seed variance on Phase 1 (30 compounds × 3 independent
runs, temperature 0.3, identical evidence):**

| Metric | Value |
|---|---|
| Mean (max−min) Q&A count per compound across seeds | **0.36** |
| Mean pairwise answer token-Jaccard across seeds | **0.344** |
| Per-run API failure rate | ~3% (1–2 of 30 compounds per seed) |
| Compounds producing Q&A in all three runs | 28 / 30 |

Interpretation. Q&A **count** per compound is stable across seeds (a
given compound yields within 1 Q&A of its median across three independent
runs). Q&A **content** shares ~34% tokens across seeds — most variation
is paraphrasing; structural facts and topic selection are stable. This
is consistent with the ablation probe's finding that the taxonomy +
SMILES dominate content selection, not the temperature-level noise.

The published dataset will be a **single seed** run. Users wishing to
average across seeds would need to re-invoke the pipeline with different
seeds (per-call RNG seed is controllable; see
`scripts/measure_seed_variance.py`). Seed-level provenance of the
shipped release is documented in the release notes.

## 7. Model dependency risks archival reproducibility

The pipeline pins to specific OpenRouter model IDs:
- `google/gemini-3-flash-preview`
- `moonshotai/kimi-k2.5`
- `google/gemma-4-31b-it`

If any of these are deprecated, re-generating the dataset from scratch
will yield different outputs. The mitigation (not yet implemented) is
Phase-1/2/3 response caching with content-addressable storage, so the
exact prompts and responses can be replayed independent of model
availability.

## 8. Scope is premium tier only

The standard and broad tiers (74K and 105K compounds respectively)
exist and are lower-quality filter tiers of the same underlying
PubMed/PMC data. They are not processed by the QA pipeline in this
release. Expanding to standard/broad is straightforward operationally
but would require reassessment of all the biases above for the lower
tiers.

## 9. Anonymization / double-blind caveat

This repository contains author-identifying information in git history.
Submitting to a double-blind venue requires a fresh clone into a new
anonymized repository with the git history squashed. This is a manual
step not handled by the codebase.

## 10. Public-health implications of chemistry misinformation

The dataset contains claims about drug mechanism, metabolism, toxicity,
drug interactions, and therapeutic use. A model fine-tuned on inaccurate
or hallucinated claims of this kind could produce confidently wrong
clinical-relevant outputs. See `RESPONSIBLE_AI.md` for the full risk
discussion and intended-use restriction.
