# CHANGELOG

Running log of changes made in response to peer review. Each entry pairs
a reviewer concern with the resolving action and touched files.

## Round 1 — 2026-04-23

### Reviewer issues addressed in-repo

| # | Severity | Concern | Action | Files |
|---|---|---|---|---|
| 1 | Critical | Agreement ≠ correctness; "gold" label is LLM-consensus | Reframed "gold" as "high-agreement LLM-consensus" throughout docs; committed to human-evaluation plan (Round 2) | `README.md`, `DATASHEET.md`, `LIMITATIONS.md` |
| 2 | Critical | Training-data contamination unaddressed | Authored contamination methodology doc + canary-set generator (code); defers canary run to full-pipeline execution | `CONTAMINATION.md`, `scripts/build_contamination_canary.py` |
| 3 | Critical | No downstream validation | Committed to a 3B-parameter open-model fine-tune pilot on a named target task (Q2 NeurIPS rebuttal); scoped in `EVALUATIVE_ROLE.md` | `EVALUATIVE_ROLE.md` |
| 4 | Major | Soft rule permits training-recall | Added a Phase 4 grounding-check proposal (deferred to Round 2); documented the risk | `LIMITATIONS.md`, `EVALUATIVE_ROLE.md` |
| 5 | Major | Compound coverage bias | Wrote coverage-analysis script that reports therapeutic-area, MW, logP, and heavy-atom distribution; results added to datasheet | `scripts/analyze_coverage.py`, `DATASHEET.md` |
| 6 | Major | Redaction coverage unmeasured | Authored redaction-audit script measuring false-negative leakage against held-out synonym sources | `scripts/audit_redaction.py`, `LIMITATIONS.md` |
| 7 | Major | Evaluative role underspecified | Pinned to "instruction-tuning resource for medicinal-chemistry reasoning" with concrete target benchmarks | `EVALUATIVE_ROLE.md` |
| 8 | Major | RAI analysis absent | Authored Responsible AI doc covering bias, misuse, safety-critical claims, mitigation | `RESPONSIBLE_AI.md` |
| 9 | Major | Stochasticity unquantified | Added seed-variance measurement plan and documented the RNG surface | `LIMITATIONS.md`, `scripts/measure_seed_variance.py` |
| 10 | Major | Model-dependency / archival | Added Phase-1-response caching plan + model-pinning contract | `LIMITATIONS.md` |
| 11 | Minor | Dataset license undefined | Wrote dataset license reconciling PubMed / PubChem / PMC upstream terms | `LICENSE-DATA.md` |
| 12 | Minor | Missing comparison to prior art | Added comparison table covering ChemBench, ChemLLMBench, Mol-Instructions, SciQA, PubMedQA | `DATASHEET.md` |
| 13 | Minor | Mandatory artifacts missing | Produced Gebru-style datasheet + Croissant JSON-LD skeleton with RAI fields | `DATASHEET.md`, `croissant.json` |
| 14 | Minor | Anonymization incomplete | Action plan for the user; requires fresh-clone + git-history scrub | `LIMITATIONS.md` (anonymization notes) |

### Deferred to Round 2 (compute- or credential-dependent)

- Actual contamination canary run on post-cutoff compounds (requires small pilot budget).
- Human-evaluated sample for label-reliability kappa (requires annotator time, ~$200).
- Downstream fine-tune pilot on a 3B open model (requires compute).
- HuggingFace hosting + Croissant validation on live dataset (requires user HF credentials).
- Full git-history anonymization (requires user action; one-way operation).

## Round 2 — 2026-04-23 (pre-flight, dataset-creation focus)

Scope restricted to issues that would force a re-generation of the full
dataset if left unaddressed.

### Reviewer issues addressed pre-run

| # | Severity | Concern | Action | Evidence |
|---|---|---|---|---|
| R2-1 | Critical-Blocking | Redaction leakage unmeasured on v3 | Ran `audit_redaction.py` in both holdout modes | IUPAC leak **0.03%**, name leak **0.07%** — well under 5% threshold. Reports at `data/qa_pipeline/redaction_audit_*.json` |
| R2-2 | Critical-Blocking | Phase 4 grounding absent | Added `evidence_ids` field to Phase 1 output schema as a per-Q&A provenance signal; full Phase 4 grounding check deferred to post-generation analysis | `chem2textqa/qa_pipeline/phase_1_qa/prompts.py`, `generate.py` |
| R2-3 | Major-Blocking | No per-Q&A provenance | `evidence_ids: [int]` field added, populated by Phase 1 model, flows through assembly | same as R2-2 |
| R2-4 | Major-Blocking | No train/dev/test split | Deterministic CID-hash split baked into assembly: 80/10/10 train/dev/test, canary gets its own `split="canary"` tag | `chem2textqa/qa_pipeline/assemble.py` (`assign_split`) |
| R2-5 | Major | Seed variance unmeasured | Script exists (`scripts/measure_seed_variance.py`); user can run when desired (~$1) | (deferred by user) |
| R2-6 | Major | Evidence concentration risk | Ran `analyze_coverage.py`; results in LIMITATIONS | **Top 10% of compounds hold 65% of total evidence.** |
| R2-7 | Major | Prompt adherence unverified | Pattern-matched 27,514 pilot answers for forbidden phrases | **18 forbidden-phrase hits (0.07%); 0 `[E<id>]` citations.** Engineering questions confirmed substantive on spot-check. |
| R2-8 | Major | Canary CIDs not built | Canary built at 2024-01-01 cutoff → 120 compounds with evidence; split files produced | `data/qa_pipeline/canary_cids.txt`, `phase0_full_premium_v3/evidence_{main,canary}.jsonl` |
| R2-9 | Minor | Duplicate-question audit | 140,854 within-compound Q-pair comparisons on pilot-1000 | **0 identical, 0 near-duplicates (J≥0.8).** Anti-redundancy rule removal was safe. |
| R2-10 | Minor | Phase 3 judge noise floor | Sampled 5 disagree verdicts; all flag real content errors (stereocenter counts, DBE, scaffold, etc.) | Qualitative; no judge replacement needed |
| — | Master script | Main + canary in one invocation | Master script now runs both in sequence with separate output dirs | `run_qa_full_premium.sh` |

### Pre-run measurements (v3 evidence)

| Metric | Value |
|---|---|
| Compounds with evidence | 15,667 |
| Total evidence sentences | 1,112,426 |
| Mean sentences/compound | 71.0 |
| p90 / p99 / max | 281 / 500 / 500 |
| Top-10% evidence concentration | 65.0% |
| IUPAC-holdout redaction leak | 0.03% |
| Name-holdout redaction leak | 0.07% |
| Pilot forbidden-phrase hit rate | 0.07% of answers |
| Pilot duplicate-question rate | 0.00% |
| Canary (late-training, cutoff 2024-01-01) | 120 compounds with evidence |
| Train / dev / test / canary split | 80% / 10% / 10% / separate |

### Schema additions in this round (non-retrofittable)

- `qa_pairs[].evidence_ids: list[int]` — Phase 1 model self-reports which
  evidence-sentence IDs support the claim.
- `split: "train" | "dev" | "test" | "canary"` per compound record at
  assembly time.

### Still deferred (do not block full run)

- Post-hoc grounding verification using `evidence_ids` (possible after
  full run finishes; no additional compute).
- Human evaluation and fine-tune pilot (per user, Round 3+).

## Round 2a — 2026-04-23 (seed variance run)

Ran `scripts/measure_seed_variance.py` on 30 random compounds × 3 seeds
before committing to the full run.

| Metric | Value |
|---|---|
| Mean (max−min) Q&A count per compound | 0.36 |
| Mean pairwise answer Jaccard across seeds | 0.344 |
| Per-run API failure rate | ~3% (1–2 of 30 per seed) |
| Compounds stable across all 3 seeds | 28 / 30 |

Reading: Q&A **counts** are stable; **content** shares ~34% tokens
across seeds, which matches the ablation finding that the dominant
content signal is SMILES + taxonomy, not temperature-level noise. Seed
choice is not a major dataset-quality lever. `LIMITATIONS.md §6` updated
with the measured numbers.

**Cleared to commit the full run.** No further blockers.
