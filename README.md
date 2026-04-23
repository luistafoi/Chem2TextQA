# Chem2TextQA

Training-data pipeline that links drug and metabolite chemical structures
(SMILES) to PubMed abstracts plus PMC full-text articles, then generates
cross-validated question/answer pairs tagged across mechanism, therapeutic
use, toxicity, metabolism, drug interactions, and chemistry.

Two pipelines live in this repo:

1. **Data building** — bulk FTP downloads of PubChem, PubMed, and PMC are
   streamed, joined, filtered, and tiered into quality-ranked JSONL.
2. **QA generation** — a four-phase LLM pipeline (evidence extraction,
   generation, independent re-answering, cross-validation) produces
   SMILES-grounded Q&A pairs from the filtered articles.

Canonical outputs:

- `data/filtered/drug_articles_v2_premium.jsonl` — ~320K articles,
  ~22K compounds at the top quality tier (higher standard/broad tiers
  also produced).
- `data/qa_pipeline/phase0_full_premium_v3/evidence_per_cid.jsonl` —
  per-compound redacted evidence bundles (15,667 compounds, ~500 sentences
  per compound max, randomly sampled across all of a compound's articles).
- `data/qa_pipeline/experiments/<run-name>/dataset_final.jsonl` — assembled
  Q&A records per compound, with per-pair verdicts from the judge.


## Install

```bash
conda env create -f environment.yml
conda activate chem2textqa
pip install -e ".[dev]"

# Configure API keys (OpenRouter is required for the QA pipeline)
cp .env.example .env   # then edit .env
```

Verify: `chem2textqa --help` should list `build-dataset`, `cleanup-dataset`,
`qa-extract-evidence`, `qa-generate`, `qa-independent`, `qa-judge`,
`qa-assemble`, plus the supporting commands.


## Pipeline 1 — Build the filtered article dataset

Downloads the PubChem bulk files, PubMed XML baseline, PMC open-access
bundles, joins them, filters, and produces quality tiers.

```bash
# Pulls PubChem CID-*.gz tables and curated source CID lists (DrugBank,
# HMDB, KEGG, ChEBI, BindingDB, ChEMBL) into data/bulk/. One-time ~15 GB.
bash run_pmc_download.sh

# Build the raw joined dataset from bulk files.
bash run_build_dataset_v2.sh

# Apply the 7-filter cleanup and produce the quality tiers.
bash run_cleanup_v2.sh
```

Filter order (configurable via `chem2textqa cleanup-dataset --help`):

1. English language
2. Abstract ≥ 500 chars
3. Not retraction / erratum
4. Not editorial / letter / comment / news
5. Has at least one linked compound
6. At least one compound is non-generic (excludes water, Na⁺, glucose,
   amino acids, etc.)
7. That compound is actually mentioned in title/abstract or tagged as a
   MeSH major topic

`data/filtered/filter_stats_v2.json` records the per-filter drop counts.


## Pipeline 2 — Generate cross-validated Q&A

Four phases; each is a separate CLI command and each phase's output is an
append-only JSONL so runs are resumable.

### Phase 0 — evidence extraction (no LLM, ~1–2 hours, $0)

```bash
bash run_phase0_full_premium.sh
```

For every compound in the premium tier:

- Collect synonyms (primary name, IUPAC, MeSH terms, PubChem
  `CID-Synonym-filtered.gz`).
- Compile a longest-match-first whole-word redaction regex.
- Seek into the tier JSONL via byte-offset index; for each article, split
  abstract + full-text into sentences, keep those matching the regex,
  redact all hits to `[COMPOUND]`, dedupe.
- Collect every matching redacted sentence across a compound's articles,
  then random-sample down to the per-compound cap (default 500) with an
  RNG seeded by CID.

Outputs `phase0_full_premium_v3/evidence_per_cid.jsonl` and
`retention_stats.json`. Around 15,667 of 22,438 premium compounds retain
evidence after redaction.

### Phase 1 — Q&A generation (LLM1)

System prompt follows a **soft-rule** design: structural claims must be
derivable from SMILES/formula/MW; functional claims (mechanism,
metabolism, therapeutic use, toxicity, drug interactions, ADME,
engineering/analog design) may be supported by the redacted evidence,
absorbed silently as background knowledge. Evidence is never quoted,
paraphrased, or cited with markers in the output. The model never names
the compound. Target Q&A count scales with evidence volume (5–7 for <10
sentences up to 35–50 for 300+).

### Phase 2 — blind independent re-answer (LLM2)

A different model family answers each question given only the SMILES and
the same evidence — it does not see LLM1's answer. This produces the
independent signal that Phase 3 judges for agreement.

### Phase 3 — judge (LLM3)

Gemma 4 31B classifies each `(question, answer1, answer2)` triple as
`agree` / `disagree` / `unclear`. A cheap local heuristic pre-filter
auto-classifies obvious agreements by token Jaccard to save LLM calls
(usually <5% hit rate; conservative, always escalates ambiguous cases).

### Full run

```bash
# Runs all four phases + assembly + an agree-only "gold" subset.
# Points at phase0_full_premium_v3 by default. ~12–15 hours, ~$750 at
# current OpenRouter prices (Gemini 3 Flash preview + Kimi K2.5 + Gemma 4
# 31B).
bash run_qa_full_premium.sh
```

### Smaller-scale / pilot runs

```bash
# 1000 random compounds from the evidence pool — use this to smoke-test
# before the full run. ~$50, ~30 minutes.
bash run_1000_pilot.sh
```

Outputs go to `data/qa_pipeline/experiments/<run>/`:
- `dataset_final.jsonl` — one record per compound with all Q&A + verdicts.
- `dataset_gold.jsonl` — `--agree-only` subset (Phase 3 = `agree`).
- `dataset_summary.json` — counts, agree rate, topic distribution.


## Design-validation probes

The three audit probes under `scripts/` verify that the soft-rule design
actually uses the evidence the way it claims to:

- **Ablation** (`run_ablation_probe.sh`): scramble each compound's
  evidence with random sentences from other compounds. Under the soft
  rule, structural answer Jaccard stays high (SMILES-driven) while
  functional Jaccard drops (evidence-driven). Split by topic bucket.
- **SMILES swap** (`run_smiles_swap_probe.sh`): keep compound A's
  evidence but substitute compound B's SMILES; compare to both real
  baselines. Expected: structural answers track the SMILES donor,
  functional answers track the evidence owner.
- **Empty evidence** (`run_softrule_probes.sh`, probe 3): replace
  evidence with a single non-informative placeholder. Functional Q&A
  should drop dramatically; structural Q&A is less affected.

`run_softrule_probes.sh` runs all three on a shared 30-compound sample
and emits per-bucket Jaccard metrics. Total cost ~$2.


## Project layout

```
chem2textqa/
├── cli.py                 # Click CLI (all `chem2textqa ...` commands)
├── config/                # Pydantic settings loaded from .env
├── models/                # Schema (used by deprecated scraper path)
├── processing/            # Bulk-FTP data builder (canonical)
│   ├── compounds.py       # streams PubChem CID-* tables
│   ├── sources.py         # fetches + caches curated CID lists
│   ├── pubmed_xml.py      # streaming iterparse of 50 GB XML
│   ├── mesh_local.py      # offline MeSH filter (replaces API)
│   ├── builder.py         # orchestrator
│   ├── cleanup.py         # 7-filter pass
│   └── fix_formula_mass.py
├── qa_pipeline/
│   ├── config.py          # paths + default model names + CAP constant
│   ├── openrouter.py      # async OpenRouter client
│   ├── phase_0_evidence/  # synonyms, redaction, extraction
│   ├── phase_1_qa/        # prompts + generate
│   ├── phase_2_independent/
│   ├── phase_3_validate/  # heuristic pre-filter + judge
│   ├── assemble.py        # merges all four phases
│   └── compare.py         # side-by-side experiment comparator
├── scrapers/              # DEPRECATED — live NCBI API path, preserved
│                          # for backward compatibility but the canonical
│                          # pipeline is 100% local bulk processing
├── storage/
├── filters/               # API-path MeSH category definitions
└── utils/

scripts/                   # Audit / probe helpers (not installed)
├── ablation_probe.py
├── smiles_swap_probe.py
├── empty_evidence_probe.py
└── topic_bucket.py

tests/                     # pytest suite — 163 tests
```


## Key gotchas

- **Stream, never load.** The PubChem CID-* files are 1–7 GB gzipped; the
  PubMed XML baseline is 50 GB. Use `iter_pubmed_articles` / the streaming
  `CID-*` readers. The canonical builder never holds a full table in
  memory.
- **Byte-offset indexing** for Phase 0: premium tier is 5.4 GB and each
  compound's articles are accessed by `seek()` rather than by scanning —
  do not rewrite this to a linear pass.
- **CID-Mass.gz is 4-tab-separated**, not 3. Use
  `processing.compounds._enrich_field_multi`; the single-column helper
  silently produces empty `molecular_formula` / `molecular_weight`.
- **MeSH major-topic asterisks** are preserved in stored data; matchers
  strip `*` before comparing but storage keeps the marker so downstream
  QA gen can prefer major topics.
- **Reasoning-token burn.** Hybrid models (Kimi K2.5, GPT-5 preview)
  consume their entire token budget on internal reasoning and emit empty
  content unless you pass `reasoning={"enabled": false}` in the
  OpenRouter payload. Phase 2 and Phase 3 already do this.
- **Compound identity stays redacted.** Phase 1/2 prompts forbid naming
  the compound; Phase 0 replaces every synonym hit with `[COMPOUND]`.
  Functional claims therefore come from evidence sentences rather than
  from the model recognising the compound by SMILES.


## Testing

```bash
pytest tests/ -v
ruff check chem2textqa/ scripts/
```

All 163 tests should pass against the current code. Tests under
`tests/test_scrapers/` and `tests/test_crawl_state.py` cover the
deprecated API path.


## Data gitignore

All `data/` subdirectories are gitignored (GB-scale outputs). The repo
ships code and scripts only. Regenerate data locally by running the
pipelines above.
