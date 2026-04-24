# Datasheet for Chem2TextQA

Following Gebru et al. 2018, *Datasheets for Datasets*.

## Motivation

**For what purpose was the dataset created?**
To serve as an instruction-tuning training resource for medicinal-chemistry
reasoning. Each Q&A pair links a SMILES string with structural or
functional commentary about the compound, with the goal of teaching an
LLM to reason about drug-like molecules when given only the structure.

**Who created the dataset?**
*Anonymized for double-blind review.*

**Who funded the creation?**
*Anonymized for double-blind review.*

## Composition

**What do the instances represent?**
Each instance is a compound record containing:
- SMILES, molecular formula, molecular weight, InChI Key.
- A bundle of redacted literature evidence sentences (the compound's
  name is replaced with `[COMPOUND]` in every sentence).
- A set of Q&A pairs tagged by topic (structural, functional, or
  "other"), each with a question, an LLM-generated answer (Phase 1), a
  blind independent re-answer (Phase 2), and an agreement verdict
  (Phase 3).

**How many instances?**
- **Compound level:** 15,667 compounds in the v3 Phase 0 pool (of 22,438
  premium-tier compounds; 69.8% retention after redaction yields ≥1
  usable sentence).
- **Q&A level (projected for the full run on the v3 pool):** ~210,000
  raw Q&A pairs, ~185,000 in the agree-only subset at the ~88% agree
  rate observed in the 1000-compound pilot.
- **Validation pilot (already produced):** 998 compounds, 13,757 raw
  Q&A, 12,212 agree.

**Does the dataset contain all possible instances, or is it a sample?**
A sample, drawn from the premium filter tier of a joined PubChem ×
PubMed × PMC corpus. See `LIMITATIONS.md` §4 for coverage bias.

**What data does each instance consist of?**
See README for the output schema. In brief: structural identifiers plus
a list of Q&A objects with `{topic, question, phase1_answer,
phase2_answer, verdict, judge_reasoning}`.

**Is there a label or target?**
- `verdict` ∈ {agree, disagree, unclear} from Phase 3 acts as a
  weak-supervision label.
- **Important:** this is inter-model consistency, not correctness. See
  `LIMITATIONS.md` §1.

**Is any information missing from individual instances?**
- Compound identity is intentionally redacted from evidence text
  (`[COMPOUND]` placeholder).
- Author names and publication dates for evidence articles are
  preserved in the PMID metadata but not in the evidence text itself.

**Are relationships between individual instances made explicit?**
Yes: each Q&A pair carries its source compound's CID; within a compound,
each Q&A carries a `qa_index`. No cross-compound relationships are
encoded.

**Are there recommended data splits (train/validation/test)?**
No pre-defined split. Recommended:
- Train: all Q&A where `verdict == "agree"` and the compound is
  pre-2025-09-01 (main set).
- Dev / stress-test: the contamination canary subset (post-2025-09-01
  compounds), per `CONTAMINATION.md`.
- Held-out human-annotated: scheduled for Round 2.

**Are there any errors, sources of noise, redundancies?**
Yes, many — see `LIMITATIONS.md` in full.

**Does the dataset rely on external resources?**
Pipeline inputs are PubMed baseline XML, PMC open-access full text, and
PubChem bulk tables — all publicly redistributable. LLM dependencies are
OpenRouter-hosted models; response caching is scheduled (see
`LIMITATIONS.md` §7).

**Does the dataset contain confidential or sensitive data?**
No PII. Drug-related content is sensitive in a safety sense — see
`RESPONSIBLE_AI.md`.

## Collection process

**How was the data collected?**
See README § "Pipeline 1" and § "Pipeline 2". Briefly:
1. Bulk FTP download of PubChem CID tables, PubMed baseline XML, PMC
   open-access bundles.
2. Join into per-article records with linked compounds.
3. Seven-filter cleanup (language, length, retraction, publication type,
   linked-compound specificity, mention-actually-in-text, compound
   not-only-generic).
4. Per-compound redacted evidence extraction with synonym-based
   whole-word masking.
5. Four-phase LLM Q&A generation (generate → blind re-answer → judge →
   assemble).

**Over what timeframe?**
Source snapshots as of PubMed baseline 2026 release, PubChem Q1 2026,
PMC open-access current as of 2026-04.

**Were ethical review or compensation involved?**
No human subjects; automated pipeline over public-domain scientific
literature. No compensation. Human-evaluation round is scheduled (Round
2) and will require annotator compensation.

## Preprocessing / cleaning / labeling

**Was the data preprocessed?**
Extensively. Redaction to `[COMPOUND]`, sentence splitting, filtering,
per-compound sentence cap (500) with random sampling, per-CID-seeded RNG
for reproducibility.

**Was the "raw" data saved?**
The bulk FTP downloads are large and reproducible from public sources;
they are not redistributed. The filtered tier
(`drug_articles_v2_premium.jsonl`) is the effective raw input; local
regeneration via `run_build_dataset_v2.sh` reproduces it from the bulk
files.

## Uses

**Has the dataset been used for any tasks already?**
Validation probes only (ablation, SMILES swap, empty evidence) at the
30-1000 compound scale. A downstream fine-tune pilot is scheduled for
Round 2.

**Is there a repository that links to papers / systems using the dataset?**
Not yet.

**What tasks could the dataset be used for?**
- Instruction tuning for chemistry-aware LLMs (primary).
- Cross-model agreement studies.
- Ablation studies of evidence-grounding in LLM Q&A generation
  pipelines.
- **Not** a clinical decision support resource. **Not** a benchmark. See
  `EVALUATIVE_ROLE.md`.

## Distribution

**How will the dataset be distributed?**
Hugging Face Datasets Hub (scheduled), with a Croissant JSON-LD
metadata file including RAI fields. See `croissant.json` for the
current skeleton.

**Under what license?**
See `LICENSE-DATA.md`. Core dataset: CC BY 4.0, with upstream-imposed
restrictions.

## Maintenance

**Who maintains the dataset?**
*Anonymized for review.*

**Is there an erratum / update mechanism?**
Versioned releases on Hugging Face; CHANGELOG for each update.

## Comparison to prior work

| Dataset | Focus | Size | Label source | Differences |
|---|---|---|---|---|
| ChemBench | Curated chemistry benchmark, zero-shot | ~7K questions | Human expert | Benchmark, not training; broader chemistry not just drugs |
| ChemLLMBench | LLM chemistry evaluation | ~4K | Human curated | Evaluation-only; structural focus |
| Mol-Instructions | Molecule instruction tuning | ~2M molecule/text | Template-generated | Larger but templated; less functional-reasoning depth |
| SciQA | Broad science QA | ~200K | Crowdsourced | Not chemistry-specific; abstract-level |
| PubMedQA | Biomedical research Q&A | ~1K expert / ~200K silver | Expert + template | Biomedical but not SMILES-grounded |
| **Chem2TextQA** | **Drug structure + functional commentary, silently-evidence-grounded** | **~185K gold** | **LLM cross-model agreement** | **SMILES as first-class input; structural/functional split with per-pair provenance; engineering-question category** |

The distinctive elements of Chem2TextQA are (a) the
structural/functional bucket split with different grounding expectations,
(b) SMILES as a primary input in every Q&A (not an afterthought), and
(c) the engineering-question category for analog-design reasoning.
