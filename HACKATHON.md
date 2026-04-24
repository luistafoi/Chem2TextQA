# Chem2TextQA — Lab Hackathon Agenda

A one-day push to stress-test the dataset from every angle a NeurIPS
reviewer would, and to produce the figures/tables we need for the paper.

**Before you start:** read sections 1–4. Pick a task in section 5 that
matches your skills and time budget. Ping Luis if the inputs you need
don't match what's in this document — things may have moved.

---

## 1. What Chem2TextQA is (2-minute read)

Chem2TextQA is a training dataset that pairs drug-like compounds
(SMILES) with natural-language Q&A pairs about their structure and
function. Each Q&A is cross-validated by two different LLMs, plus a
third as a judge.

- **Source:** 22,438 premium-tier compounds from a local PubChem ×
  PubMed × PMC join. After evidence extraction, 15,667 compounds retain
  ≥1 usable literature sentence about them.
- **Output scale:** 15,509 main compounds + 119 canary compounds,
  ~211K raw Q&A, ~189K in the agree-only (gold) subset.
- **Target role:** instruction-tuning resource for medicinal-chemistry
  reasoning. **Not** a benchmark. **Not** a clinical resource.

### How each Q&A was produced (4-phase pipeline)

| Phase | What | Model | Input | Output |
|---|---|---|---|---|
| 0 | Extract redacted evidence sentences per compound | none (local RDKit + regex) | Premium tier JSONL | per-compound evidence bundles; compound name replaced with `[COMPOUND]` |
| 1 | Generate Q&A | Gemini 3 Flash preview | SMILES + evidence | ~14 Q&A per compound with `{topic, question, answer, evidence_ids}` |
| 2 | Blind independent re-answer | Kimi K2.5 (reasoning disabled) | Same SMILES + evidence + 1 question at a time | one `phase2_answer` per Q&A |
| 3 | Judge agreement | Gemma 4 31B | question + `phase1_answer` + `phase2_answer` (no evidence) | `verdict ∈ {agree, disagree, unclear}` + `judge_reasoning` |

The **soft rule** design: structural claims (scaffold, functional
groups, reactivity) must be SMILES-derivable; functional claims
(mechanism, metabolism, toxicity, engineering) may be evidence-supported
but never quoted or cited in the output. Compound identity is never
revealed to the models.

### What's already validated

| Signal | Result | What it means |
|---|---|---|
| Redaction leak rate (IUPAC / name holdout) | 0.03% / 0.07% | Phase 2 blind-answer premise holds. Negligible compound identity leakage. |
| Prompt adherence (forbidden phrases in 27,514 pilot answers) | 0.07% | Models are following the no-citation, no-quote rule. |
| Duplicate questions within compound (140K pair comparisons) | 0.00% | Anti-redundancy is emergent; no rule needed. |
| Ablation probe (scramble evidence), structural-answer Jaccard | 0.27 | Structural answers don't depend on the specific evidence sentences. |
| Ablation probe (scramble evidence), functional-answer Jaccard | 0.14 | Functional answers change when evidence changes. Evidence steers function. |
| SMILES-swap probe, structural tracks SMILES donor | 90% | Model genuinely reads each unique SMILES. |
| SMILES-swap probe, functional tracks evidence owner | 63% | Functional claims are evidence-driven. |
| Empty-evidence probe, functional Q&A volume drop | ↓74% | Model self-limits functional claims without evidence. |
| Canary agree rate vs main agree rate | 86.0% vs 89.7% (Δ = 3.7 pp) | Measured memorization footprint. Small but real. |
| Seed variance across 3 Phase-1 runs (answer Jaccard) | 0.34 ± | Content is ~34% overlap across seeds; counts are stable. |
| Fine-tune CIDEr (Qwen2.5-14B base vs finetuned, test=26,205) | 0.0017 → 0.2758 | Dataset meaningfully teaches medicinal chemistry Q&A. 160× improvement on a held-out test split. |

---

## 2. What the lab has already done (don't redo)

| Work | Path | Status |
|---|---|---|
| Full 4-phase pipeline run (main + canary) | `data/qa_pipeline/full_premium_kimi/` | Complete |
| Scaffold-based 70/15/15 split (Murcko, strict MoleculeNet) | baked into each compound record's `split` field; mapping at `full_premium_kimi/cid_to_split.json` | Complete |
| Redaction audit | `scripts/audit_redaction.py` | Run on main |
| Coverage / bias distribution | `scripts/analyze_coverage.py` → `coverage_report.json` | Run on main |
| Three design-validation probes (ablation, SMILES-swap, empty-evidence) | `scripts/*_probe.py` → `experiments/softrule_probes30/` | Complete |
| Seed-variance measurement | `scripts/measure_seed_variance.py` → `experiments/variance3x30/` | Complete |
| Contamination canary construction | `scripts/build_contamination_canary.py` → `canary_cids.txt` | Complete |
| Fine-tuning on 3 base models (Gemma3-12B, Llama3.1-8B, Qwen2.5-14B) | `/data/macaulay/ChemQA/ChemQA/checkpoints/` + `eval/results_full/summary.md` | Complete with CIDEr scores |

---

## 3. Dataset inventory — ONE source of truth

The canonical dataset for the hackathon lives here:

```
/data/luis/Chem2TextQA/data/qa_pipeline/full_premium_kimi/
├── dataset_gold.jsonl           ← PRIMARY. Use this for fine-tuning and
│                                  evaluation. 15,509 compounds, ~189K
│                                  agree-only Q&A, split field populated.
│
├── dataset_final.jsonl          ← All Q&A including disagree/unclear
│                                  (~210K). Use ONLY for label-noise /
│                                  verdict-analysis tasks.
│
├── canary/dataset_final.jsonl   ← 119 post-2024 compounds, ~1.5K Q&A.
│                                  Held-out contamination test set. Never
│                                  train on this.
│
├── cid_to_split.json            ← CID → split mapping (audit trail)
├── scaffold_split_report.json   ← Scaffold split statistics (paper-ready)
├── dataset_summary.json         ← Counts, agree rates, topic distribution
└── dataset_gold_summary.json    ← Same, for gold subset
```

### ⚠️ Older files — do NOT use unless specifically directed

| File | Why skip |
|---|---|
| `experiments/pilot1000_random200/` | Pre-soft-rule design. Prompts are older. Only use if comparing design versions. |
| `experiments/pilot100_improved/`, `pilot500_*`, `pilot20_softrule` | Superseded iteration artifacts. |
| `phase0_full_premium_v3/evidence_per_cid.jsonl` | This is the Phase 0 evidence — already embedded inside each compound record in `dataset_gold.jsonl`. Only read this directly if you specifically need raw evidence outside the Q&A context. |

### Record schema (each line of `dataset_gold.jsonl`)

```json
{
  "cid": 2244,
  "split": "train",                       // train | val | test | canary
  "name": "Aspirin",
  "iupac_name": "2-acetyloxybenzoic acid",
  "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
  "molecular_formula": "C9H8O4",
  "molecular_weight": 180.04,
  "inchi_key": "InChI=1S/...",
  "num_pmids": 42,
  "num_synonyms": 118,
  "num_evidence_sentences": 87,
  "evidence_sentences": [
    {"id": 1, "pmid": "12345", "source": "abstract",
     "text": "[COMPOUND] inhibits COX-1 ..."},
    ...
  ],
  "qa_pairs": [
    {
      "qa_index": 1,
      "topic": "mechanism",                  // freeform string
      "question": "What is the molecular target of the compound?",
      "phase1_answer": "The compound is an ...",
      "phase2_answer": "This compound acts as ...",
      "verdict": "agree",
      "judge_reasoning": "Both answers identify the same target ...",
      "evidence_ids": [1, 3]                 // sentence IDs cited by the model
    },
    ...
  ]
}
```

### Additional resources

- Full repo: `/data/luis/Chem2TextQA/`
- Pipeline code: `chem2textqa/qa_pipeline/`
- Probe scripts: `scripts/`
- Fine-tuned checkpoints: `/data/macaulay/ChemQA/ChemQA/checkpoints/`
- Fine-tune eval results: `/data/macaulay/ChemQA/ChemQA/eval/results_full/`
- Documentation: `DATASHEET.md`, `LIMITATIONS.md`, `CONTAMINATION.md`,
  `RESPONSIBLE_AI.md`, `EVALUATIVE_ROLE.md`, `USAGE_FINETUNING.md`
- Bulk baseline data (advanced use only): `data/bulk/pubmed_baseline/`,
  `data/bulk/CID-*.gz`

---

## 4. Open concerns from grant-reviewer sessions (map to tasks)

Each concern below is a real critique we heard. Every hackathon task
addresses at least one. If your task maps to one of these, write your
deliverable with the concern explicitly in mind — the output will go
straight into the paper's response.

| # | Concern | Tasks that address it |
|---|---|---|
| C1 | **Label reliability is circular.** The 89.7% agree rate is LLM consensus, not correctness. Two LLMs can agree on the same wrong fact. | T2 (multi-judge), T3 (human eval), T4 (grounding) |
| C2 | **Training-data contamination.** PubMed is in pretraining; the "gold" may partly reflect memorization. | T4 (canary deep-dive), T6 (leakage classifier) |
| C3 | **Soft-rule permits memorization.** Empty-evidence probe shows 27% of functional Q&A still produced without evidence. How many are training recall vs structural inference? | T1 (RDKit auto-verification), T4 (canary), T5 (grounding) |
| C4 | **Compound coverage skew.** Top 10% of compounds hold 65% of evidence. | T7 (diversity), T10 (case studies), T9 (per-topic stratification) |
| C5 | **Stochasticity and reproducibility.** Temp 0.3 produces ~34% token variance seed-to-seed. | T11 (reproducibility manifest) |
| C6 | **Positioning vs prior art.** ChemBench, ChemLLMBench, Mol-Instructions, SciQA, PubMedQA all adjacent. | T12 (related-work critique) |

---

## 5. Tasks

Each task is **self-contained**. Pick one and go. Expected effort per
task is noted. "Depends on" means the task needs outputs from another
task; start those in parallel.

---

### Task 1 — RDKit deterministic verifier

**Goal.** Automatically verify the numeric / structural claims in each
Q&A against RDKit ground truth derived from the SMILES. For claims like
"degree of unsaturation = 3", "has 2 chiral centers", "contains a
carboxylic acid group", RDKit can compute the right answer. We use this
to audit label quality without an LLM judge.

**Why it matters.** Addresses **C1** (is the label correct?) and **C3**
(are structure claims really SMILES-derivable?). Provides a
deterministic, LLM-free signal on at least the structural bucket of Q&A.

**Inputs.**
- `/data/luis/Chem2TextQA/data/qa_pipeline/full_premium_kimi/dataset_gold.jsonl`
- Per-Q&A `phase1_answer` text, plus the parent compound's `smiles`.
- Topic bucket helper: `/data/luis/Chem2TextQA/scripts/topic_bucket.py`

**Method sketch.**

1. For each compound, parse SMILES with RDKit. Compute a reference
   dictionary of verifiable facts:
   - heavy atom count
   - degree of unsaturation (double-bond equivalents)
   - aromatic ring count
   - rotatable bond count
   - H-bond donor / acceptor count (`rdMolDescriptors.CalcNumH*`)
   - set of functional group presences (carboxylic acid, ester, amide,
     nitro, halide, amine, hydroxyl, sulfonamide, ketone, aldehyde)
   - formal charge
   - number of stereocenters
2. For each Q&A, run lightweight extractors over `phase1_answer` and
   `phase2_answer`. Suggested patterns:
   - Numeric claim: `"degree of unsaturation of (\d+)"` → compare to
     RDKit reference.
   - Group presence: check for trigger words (`carboxylic acid`,
     `nitro group`, ...) → compare to RDKit reference.
3. For each match, emit a verdict: `verified` / `contradicted` /
   `not_applicable` (no checkable claim found).

**Output.** One JSON per compound:

```json
{
  "cid": 2244,
  "qa_verifications": [
    {
      "qa_index": 1,
      "checks": [
        {"claim_type": "aromatic_rings", "claimed": 1, "rdkit": 1, "status": "verified"},
        {"claim_type": "carboxylic_acid_present", "claimed": true, "rdkit": true, "status": "verified"}
      ]
    }
  ]
}
```

Aggregate report: fraction of structural Q&A that have ≥1 verifiable
claim; of those, fraction verified vs contradicted. Break down by split
(train/val/test/canary).

**Deliverable.**
- `scripts/rdkit_verifier.py`
- `outputs/rdkit_verification.jsonl` (per-compound)
- `outputs/rdkit_verification_summary.md` (overall + by-topic + by-split)

**Effort.** ~6 hours. Depends on: nothing (RDKit + SMILES + JSONL).

---

### Task 2 — Multi-judge LLM panel on the Phase-3 verdicts

**Goal.** Phase 3 used Gemma 4 31B as judge. Was it right? Re-judge a
stratified sample with 2–3 stronger judges; measure inter-judge
agreement.

**Why it matters.** Addresses **C1**. Gives a defensible number for how
often Gemma's verdict matches Sonnet's / GPT's / Gemini-Pro's on the
same (question, a1, a2) triple.

**Inputs.**
- `/data/luis/Chem2TextQA/data/qa_pipeline/full_premium_kimi/dataset_final.jsonl`
  (use this, not gold — we want to re-judge *all* verdicts including
  disagrees and unclears).
- `/data/luis/Chem2TextQA/chem2textqa/qa_pipeline/phase_3_validate/judge.py`
  (existing judge prompt — re-use it verbatim for apples-to-apples).
- `/data/debi/.env` for `OPENROUTER_API_KEY`.

**Method sketch.**

1. Sample 500 Q&A stratified by verdict: ~250 agree, ~200 disagree,
   ~50 unclear. Stratify further by topic bucket (structural /
   functional / other) so the sample covers the space.
2. For each, run the existing `JUDGE_SYSTEM_PROMPT` (same phrasing
   Gemma saw) against three new judges:
   - `anthropic/claude-sonnet-4.6`
   - `openai/gpt-5`
   - `google/gemini-2.5-pro`
3. Compute:
   - Per-judge verdict distribution (are the new judges harsher / more
     lenient than Gemma?)
   - Pairwise Cohen's kappa on verdict
   - Krippendorff's alpha across all four judges
   - % of cases where the majority of the three new judges **disagrees**
     with Gemma — this is the most important number. If it's >15%, the
     gold subset is noisier than we've been claiming.

**Output.**

```
outputs/multi_judge/
├── samples.jsonl               (the 500 sampled Q&A with all 4 judges' verdicts)
├── per_judge_distribution.json
├── agreement_matrix.json       (kappa between each pair of judges)
└── disagreement_analysis.md    (cases where Gemma diverges from panel majority)
```

**Deliverable.** JSON outputs + a markdown table like:

| Judge pair | Cohen's κ |
|---|---|
| Gemma × Sonnet | 0.72 |
| Gemma × GPT-5 | 0.68 |
| Gemma × Gemini-Pro | 0.74 |
| Panel avg | 0.71 |

**Effort.** ~8 hours. Depends on: OpenRouter API key + budget (~$20).

---

### Task 3 — Blinded human evaluation on fine-tune outputs

**Goal.** Expert humans compare base vs fine-tuned model outputs,
blinded. Establishes a ground-truth signal independent of all LLM
judges.

**Why it matters.** Addresses **C1** directly. "Did fine-tuning on
Chem2TextQA actually teach useful chemistry?" needs a non-LLM answer.

**Inputs.**
- Fine-tuned predictions: `/data/macaulay/ChemQA/ChemQA/eval/results_full/*/ft_chemqa.jsonl`
- Base-model predictions: `/data/macaulay/ChemQA/ChemQA/eval/results_full/*/base_chemqa.jsonl`
- The test split (same across both): 26,205 Q&A.

**Method sketch.**

1. Sample 100 Q&A from the test split, stratified by topic bucket.
   Also pull a parallel 50 from the canary for contamination-sensitive
   eval.
2. For each sampled Q&A, present to a human evaluator **blinded**:
   - Question
   - Model A answer (randomly = base or finetuned)
   - Model B answer (the other)
   - Ground truth (`phase1_answer` from the dataset)
   - SMILES + formula + MW
3. Evaluator scores each answer on three axes, 0–2:
   - Correctness (factual)
   - Completeness (covers the question)
   - Fluency / structure
4. De-blind after all scores collected; compute deltas.

**Output.** A CSV for the annotator + analysis script:

```
outputs/human_eval/
├── samples.csv                (blinded, for annotators)
├── annotations_filled.csv     (after humans)
├── analysis.md                (PT vs FT mean scores per axis, per topic, per split)
```

**Deliverable.** Paper-ready table:

| Topic bucket | n | Correctness (base) | Correctness (FT) | Δ |
|---|---|---|---|---|
| Structural | 40 | 1.2 | 1.8 | +0.6 |
| Functional | 40 | 0.8 | 1.7 | +0.9 |
| Canary (all) | 50 | 0.7 | 1.3 | +0.6 |

**Effort.** Prep ~4 hours; human annotation ~8 hours across the team.
Depends on: lab-member availability as annotators.

---

### Task 4 — Canary deep-dive + contamination analysis

**Goal.** The canary has 119 compounds, 1,304 agree-only Q&A.
Characterize *how* the canary differs from the main set beyond the 3.7pp
agree-rate delta.

**Why it matters.** Addresses **C2** and **C3**. If the canary gap is
structural-only (e.g., functional drops much more than structural), the
memorization footprint is concentrated in functional claims and we can
report that. If it's uniform across buckets, the story is different.

**Inputs.**
- `/data/luis/Chem2TextQA/data/qa_pipeline/full_premium_kimi/dataset_gold.jsonl`
  (main, for baseline stats)
- `/data/luis/Chem2TextQA/data/qa_pipeline/full_premium_kimi/canary/dataset_final.jsonl`
- Fine-tune eval outputs in `/data/macaulay/ChemQA/ChemQA/eval/results_full/`

**Method sketch.**

1. Compute per-bucket agree rates on main vs canary. Is the 3.7pp gap
   uniform across structural / functional / engineering?
2. Token-level diff: for compounds in canary, what fraction of
   `phase1_answer` tokens appear verbatim in any evidence sentence for
   that compound? Compare to main.
3. Run the fine-tuned models on the canary (they've been evaluated on
   the main test set; extend to canary) and report CIDEr / BLEU /
   correctness per topic bucket on canary vs main.
4. Deliverable answers: **"Does the fine-tuned model generalize to
   unseen-at-pretraining compounds at the same accuracy as to
   in-distribution test compounds?"**

**Output.**

```
outputs/canary_analysis/
├── per_bucket_agree_rates.json
├── evidence_token_overlap.json
├── finetuned_canary_metrics.json
└── canary_analysis.md
```

**Deliverable.** Markdown report with key finding in the abstract:
"Fine-tuned Qwen2.5-14B achieves CIDEr X on main test and Y on canary;
gap of Z points attributable to [bucket] claims."

**Effort.** ~8 hours. Depends on: Task 3 for human canary scores
(optional but strongly preferred).

---

### Task 5 — Phase 4 grounding check (claim-to-evidence alignment)

**Goal.** For each functional Q&A, decompose the answer into atomic
claims; for each claim, label it `stated_in_evidence` /
`implied_by_evidence` / `unsupported`. Scores the dataset on whether
functional claims are actually grounded.

**Why it matters.** Addresses **C1** and **C3** directly. The soft rule
PERMITS the model to write functional claims using training recall; we
never measured how often it does.

**Inputs.**
- `/data/luis/Chem2TextQA/data/qa_pipeline/full_premium_kimi/dataset_gold.jsonl`
- OpenRouter API key.

**Method sketch.**

1. Sample 300 Q&A, stratified by topic bucket, heavy on functional
   (mechanism, metabolism, toxicity, engineering).
2. For each, send Sonnet 4.6 or Gemini 2.5 Pro the Q&A + its parent
   compound's evidence sentences (by `evidence_ids` if non-empty; else
   all). Prompt:
   > "Decompose the answer into atomic claims. For each claim, output:
   > STATED (evidence sentence N directly says this) / IMPLIED (evidence
   > supports it by inference) / UNSUPPORTED (not in evidence)."
3. Aggregate: what fraction of functional claims are STATED / IMPLIED /
   UNSUPPORTED? Break down by whether `evidence_ids` was non-empty.

**Output.** Per-Q&A claim breakdown + summary:

```
outputs/grounding/
├── claims_per_qa.jsonl
└── grounding_summary.md
```

Key metric to report:
- "Fraction of functional claims STATED or IMPLIED by evidence: X%"
- "Fraction UNSUPPORTED (training-recall candidate): Y%"

If Y > 20%, we need to narrow the dataset's claim in the paper. If
Y < 10%, we can claim the soft rule is well-behaved.

**Effort.** ~10 hours. Depends on: OpenRouter API key + budget (~$30).

---

### Task 6 — Leakage classifier (evidence-to-answer text overlap)

**Goal.** Does the Phase-1 model ever borrow text verbatim from
evidence? Detect per-Q&A leakage via n-gram matching and embedding
similarity.

**Why it matters.** Addresses **C2**. The prompt explicitly forbids
quoting evidence. If the model does it anyway, that's a failure mode
worth documenting (and possibly filtering).

**Inputs.**
- `/data/luis/Chem2TextQA/data/qa_pipeline/full_premium_kimi/dataset_gold.jsonl`

**Method sketch.**

1. For each Q&A, compute against the parent compound's evidence
   sentences:
   - Longest common subsequence (LCS) length normalized by answer length
   - 5-gram overlap count
   - Max cosine similarity between answer embedding and any evidence
     sentence embedding (use `sentence-transformers/all-MiniLM-L6-v2`
     or similar light embedder)
2. Flag Q&A above thresholds: LCS > 40 chars, 5-gram overlap > 3,
   cosine > 0.85.
3. Sample 20 flagged cases; manually inspect.

**Output.**

```
outputs/leakage/
├── per_qa_leakage.jsonl   (one row per Q&A with all 3 metrics)
├── flagged_examples.md    (the sample inspection)
└── leakage_summary.md
```

Expected report: "N% of Q&A show LCS > 40 chars to evidence (i.e.
verbatim phrase reuse). M% show cosine > 0.85. After manual review of
20 flagged cases, X are actual leakage, Y are legitimate reuse of
technical terminology."

**Effort.** ~6 hours. Depends on: `sentence-transformers` (pip install).

---

### Task 7 — Lexical + semantic diversity analysis

**Goal.** How diverse are the questions and answers vs a templated
baseline? Is it 15K compounds × 14 questions but really 14 templates
repeated 15K times?

**Why it matters.** Addresses **C4**. Mol-Instructions uses templates;
we don't. Prove it.

**Inputs.**
- `/data/luis/Chem2TextQA/data/qa_pipeline/full_premium_kimi/dataset_gold.jsonl`

**Method sketch.**

1. **Lexical diversity:**
   - Type-token ratio (TTR) for questions separately, answers separately
   - Unique trigram count
   - Top-20 question prefixes (first 5 words) — demonstrate non-templated
   - Compare against a templated baseline: Mol-Instructions uses
     "Please generate a description of the molecule based on its SMILES."
     That sentence's TTR = 1.0 trivially; compute ours as the contrast.
2. **Semantic diversity:**
   - Sample 10K questions, embed with PubMedBERT
     (`microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract`)
   - Pairwise cosine similarity on 10K random pairs (not all N²)
   - Plot histogram; narrow peak near 1.0 = low diversity
3. **UMAP visualization** of question embeddings (colored by topic
   bucket) — makes a nice paper figure.

**Output.**

```
outputs/diversity/
├── lexical_stats.json
├── semantic_cosine_distribution.png
├── question_umap.png
├── top_question_prefixes.md
└── diversity_report.md
```

**Deliverable.** A figure + a table for the paper.

**Effort.** ~6 hours. Depends on: transformers + umap-learn + matplotlib.

---

### Task 8 — Per-topic performance breakdown of fine-tuned models

**Goal.** Macaulay ran CIDEr on the full test set (26,205 Q&A). Break
it down by topic bucket, split, and difficulty tier.

**Why it matters.** Addresses **C3** and **C4**. Reviewers will ask
"but WHICH kinds of Q&A improved?" — we need structural-vs-functional
deltas and a fair-warning "engineering claims improve less than pure
SMILES claims" kind of statement.

**Inputs.**
- `/data/macaulay/ChemQA/ChemQA/eval/results_full/{gemma3_12b,llama3_1_8b,qwen2_5_14b}/{base,ft}_chemqa.jsonl`
- `/data/luis/Chem2TextQA/data/qa_pipeline/full_premium_kimi/dataset_gold.jsonl`
  (to join topic and split metadata)

**Method sketch.**

1. For each prediction row, join on (cid, qa_index) to recover `topic`
   and `split`.
2. Recompute CIDEr / BLEU-4 stratified by topic bucket (structural /
   functional / engineering / other) and by split (train-holdout /
   val / test / canary).
3. Difficulty tiers via evidence count:
   - easy: compound has >50 evidence sentences
   - medium: 10–50
   - hard: <10

**Output.**

```
outputs/per_topic_finetune/
├── cider_per_topic.json
├── cider_per_split.json
├── cider_per_difficulty.json
├── heatmap_topic_x_difficulty.png
└── report.md
```

**Deliverable.** Paper Table: "Fine-tuning CIDEr improvement (base → FT)
per topic bucket and difficulty tier, on 26,205-compound test set."

**Effort.** ~6 hours. Depends on: pycocoevalcap (pip install — already
used by Macaulay).

---

### Task 9 — Difficulty / novelty stratified benchmarking

**Goal.** Beyond scaffold split, create a 3-tier novelty stratification
for the test set: common scaffolds, rare scaffolds, singleton
scaffolds. Measure fine-tune performance per tier.

**Why it matters.** Addresses **C4**. Claim "the model generalizes to
novel chemistry" needs a novelty gradient, not just aggregate test
numbers.

**Inputs.**
- `/data/luis/Chem2TextQA/data/qa_pipeline/full_premium_kimi/dataset_gold.jsonl`
- Macaulay's fine-tune eval outputs
- RDKit for scaffold computation

**Method sketch.**

1. Compute Murcko scaffold for every non-canary compound.
2. Count compounds per scaffold in the training split (70% of the data
   the FT model saw).
3. Classify each test compound:
   - Tier 1 (common): scaffold appears ≥10 times in train
   - Tier 2 (medium): 1–9 times in train
   - Tier 3 (novel): 0 times in train — truly unseen scaffold class
4. Stratify CIDEr / human eval scores by tier.
5. Compare against canary (post-2024 compounds, regardless of scaffold).

**Output.** Report comparing:
- Tier 1 (common scaffolds) FT performance
- Tier 3 (novel scaffolds) FT performance
- Canary (novel compounds, possibly seen scaffolds) FT performance

**Deliverable.** Paper table + a 3-bar chart.

**Effort.** ~5 hours. Depends on: Task 8 outputs (same underlying data).

---

### Task 10 — Qualitative case studies

**Goal.** Curate representative examples for the paper's qualitative
section. Six categories, 3–5 examples each.

**Why it matters.** Reviewers scan for this. Good case studies sell the
paper.

**Inputs.**
- `dataset_gold.jsonl` + `dataset_final.jsonl`
- Macaulay's fine-tune eval predictions

**Six categories.**

1. **Fine-tune win, structural:** large CIDEr gain base → FT, structural
   topic, both answers included.
2. **Fine-tune win, functional:** same but functional (mechanism /
   metabolism).
3. **Both models struggle:** low CIDEr for both base and FT; show the
   Q&A and speculate why.
4. **Pipeline caught a Phase-2 error:** disagree verdict where Phase 1
   was right and Phase 2 was demonstrably wrong (we saw a good example
   in CID 71 — dianion vs trianion).
5. **Leakage example** (from Task 6): Q&A where the answer borrows
   wording from evidence.
6. **Canary vs main** side-by-side on a compound with similar
   structural features.

**Output.** One markdown file with the 20–30 examples, formatted for
paste into the paper.

**Deliverable.** `outputs/case_studies.md`.

**Effort.** ~6 hours. Depends on: Tasks 1, 6, 8 outputs.

---

### Task 11 — Reproducibility manifest

**Goal.** Document every source of randomness, model version, and
configuration so the paper's reviewer can answer "is this
reproducible?" with yes.

**Why it matters.** Addresses **C5**. Track-specific requirement.

**Contents.**

1. Model IDs + version strings for Gemini 3 Flash preview, Kimi K2.5,
   Gemma 4 31B.
2. All seeds: evidence sampling (per-CID), Phase 1 temperature, split
   seed, canary cutoff date.
3. Prompt files — version-pinned copy of
   `chem2textqa/qa_pipeline/phase_1_qa/prompts.py` and the judge
   prompt.
4. Environment: conda env export.
5. Source data snapshot dates: PubMed baseline 2026, PubChem Q1 2026,
   PMC 2026-04.
6. Caveats: model availability (preview models may deprecate), rate
   limit effects, etc.

**Output.** `REPRODUCIBILITY.md` + `environment.lock.yml` + a
`prompts_frozen/` directory with dated copies of all prompts.

**Effort.** ~4 hours. Depends on: nothing.

---

### Task 12 — Related-work benchmark critique

**Goal.** Write the paper's "Related Work" section as a comparison
table.

**Why it matters.** Addresses **C6**. Track demands explicit
positioning.

**Datasets to cover.**
- ChemBench (Aldeghi et al.)
- ChemLLMBench (Guo et al.)
- Mol-Instructions (Fang et al.)
- SMolInstruct (used to train LlaSMol)
- PubMedQA
- SciQA

**Method.** For each, compare on: size, Q&A format, label provenance,
grounding strategy, SMILES-as-first-class input, cross-validation
signal, held-out contamination test, scaffold split, availability.

**Output.** `RELATED_WORK.md` with comparison table + 2-page draft
text.

**Effort.** ~6 hours. Depends on: reading.

---

## 6. Suggested RAG / downstream demos (nice-to-have)

Not primary hackathon tasks, but high-impact if time permits:

- **End-to-end case study demo.** Pick a well-known drug (e.g. metformin
  or warfarin). Show: Phase 0 evidence → Phase 1 Q&A → Phase 3 verdicts
  → fine-tune prediction → RDKit verification → human assessment. One
  visualization demonstrating the full pipeline output for one compound.
- **FDA-approved subset + RAG.** Filter `dataset_gold.jsonl` to
  FDA-approved compounds (via MeSH or DrugBank cross-ref). Build a RAG
  over Chem2TextQA that answers questions by retrieving evidence
  sentences and grounding an answer.
- **Clinical trial matching.** For drugs under investigation, show RAG
  matching patient cohort (from the evidence) to trial criteria.
- **RAG ablation.** Show a general LLM's accuracy improves when
  Chem2TextQA evidence is retrieved as context vs vanilla prompting —
  makes the dataset useful as a knowledge base, not just a fine-tune set.

---

## 7. Task ownership matrix

| Task | Start now? | Depends on | Effort | Picked by |
|---|---|---|---|---|
| 1. RDKit verifier | ✅ | RDKit (installed) | 6h | |
| 2. Multi-judge panel | ✅ | OpenRouter key | 8h | |
| 3. Human eval | prep now, annotate later | human annotators | 4h+8h | |
| 4. Canary deep-dive | ✅ | T3 preferred | 8h | |
| 5. Phase-4 grounding | ✅ | OpenRouter key | 10h | |
| 6. Leakage classifier | ✅ | sentence-transformers | 6h | |
| 7. Diversity analysis | ✅ | PubMedBERT | 6h | Avi (already in progress) |
| 8. Per-topic FT breakdown | ✅ | Macaulay's eval output | 6h | |
| 9. Novelty stratification | ✅ | T8 | 5h | |
| 10. Case studies | after T1, T6, T8 | those outputs | 6h | |
| 11. Reproducibility manifest | ✅ | — | 4h | |
| 12. Related-work critique | ✅ | reading | 6h | |

---

## 8. Hard rules for hackathon output

1. **One script, one deliverable.** Each task outputs to a
   `outputs/<task_name>/` directory. Don't dump in the repo root.
2. **Never modify the dataset.** Don't overwrite `dataset_gold.jsonl`.
   All task outputs are *analyses* of the dataset, not edits to it.
3. **Flag any surprising number directly to the team.** If your leakage
   rate comes back 30%, or your human eval kappa is 0.3, stop and ping
   — that affects the paper story before you finish analysis.
4. **Cite the document section.** When you write up your result, point
   to which concern (C1–C6) it addresses. Makes stitching into the
   paper trivial.
5. **Commit analysis code.** Even if exploratory. Reviewers will ask.

---

## 9. Questions / escalation

- **Data access problems** → Luis (dataset owner).
- **Fine-tune questions** → Macaulay.
- **Pipeline internals** → Luis for `chem2textqa/qa_pipeline/`,
  Robert for the inspiration / earlier chem pipeline.
- **Grant reviewer concerns / paper narrative** → PI.

---

## 10. Suggested one-day schedule

| Time | Activity |
|---|---|
| 09:00–09:30 | Kick-off. Luis walks through sections 1–4. Each person claims a task. |
| 09:30–12:30 | Morning sprint. Tasks 1, 2, 4, 5, 6, 7, 8, 11, 12 start in parallel. |
| 12:30–13:30 | Lunch. Mid-morning check-in: anyone hit data access / API issues? |
| 13:30–16:30 | Afternoon sprint. Tasks 9, 10 pick up outputs from morning. Task 3 annotators can start. |
| 16:30–17:30 | Integration. Paste key numbers into a shared Google Doc; quick slack-check on whether any result changes the paper's narrative. |
| 17:30–18:00 | Wrap-up. Assign follow-ups for any task not fully completed. |

End-of-day goal: each task has *at least* its `outputs/` directory
populated and a one-paragraph writeup in the shared doc. Full
polishing happens in the week after.
