# Responsible AI analysis

This dataset contains claims about drug mechanisms, metabolism, toxicity,
drug interactions, and therapeutic use. Models fine-tuned on it can
produce confidently-phrased clinical-domain outputs. This document
discusses intended use, misuse risk, bias, safety, and mitigations, per
the NeurIPS E&D Track Croissant-RAI expectations.

## Intended use

**Research use only.** Chem2TextQA is intended as an instruction-tuning
resource for medicinal-chemistry reasoning research. It is **not**
intended for:

- Clinical decision support
- Patient-facing question answering
- Prescribing guidance
- Automated literature review that informs safety-critical decisions
- Any production deployment without downstream safety evaluation

Models fine-tuned on this dataset must not be deployed in contexts where
wrong answers can cause harm without additional safeguards (retrieval
grounding against curated sources, human expert review, explicit
uncertainty quantification).

## Misuse risks

1. **Medical misinformation at scale.** A fine-tuned model could produce
   plausible-sounding but incorrect drug-interaction or dosing claims.
2. **Fabricated mechanisms.** The soft-rule design permits the model to
   generate functional claims silently from training recall. These could
   be wrong in subtle ways not caught by the cross-model agreement check.
3. **Misleading structural claims** about controlled-substance analogs.
   A user asking a fine-tuned model "if I replace this methyl with
   ethyl, what changes?" could receive usable guidance for illicit
   synthesis. This engineering-question category is explicitly present
   in the dataset.

## Bias risks

### Representation bias

The compound pool is skewed toward FDA-approved drugs and well-studied
targets. Concrete implications:

- **Therapeutic area bias.** Oncology, cardiovascular, CNS, infectious
  disease are over-represented. Rare-disease and neglected-disease drugs
  are under-represented. A fine-tuned model will perform worse on
  conditions affecting under-resourced patient populations.
- **Geographic bias.** PubMed indexing skews toward English-language,
  Western-research-institution publications. Drugs primarily studied in
  Asian or African markets are under-represented.
- **Compound-class bias.** Small molecules are over-represented; biologics,
  peptides, and macrocyclic compounds are under-represented even when
  present in the source tier.

### Label bias

Phase 1 and Phase 2 models reflect the biases of their training corpora
and RLHF regimes. Claims that are widely endorsed in the training data
will receive high agreement; contrarian-but-correct claims may be
classified as disagreements and excluded from the gold subset.

### Availability bias

Compounds with abundant PubMed literature get more evidence sentences,
hence more Q&A per compound, hence more weight at fine-tuning time. A
model fine-tuned on this dataset will be disproportionately shaped by
well-studied drugs — the opposite of where novel chemistry research
benefit would be highest.

## Safety-critical claims

A spot-check of the agree-only subset from the 1000-compound pilot
surfaces claims about:

- Active metabolic pathways (CYP isoforms, major metabolites)
- Drug-drug interactions
- Contraindications in specific renal / hepatic impairment contexts
- Clinical dosing ranges

Any model deployed on these claims without a validation layer against
authoritative sources (FDA labels, Lexicomp, Micromedex) is a safety
risk. **Users fine-tuning on this data must not represent the resulting
model as clinically validated.**

## Mitigations

### Implemented in Round 1 (this repo)

- Evaluative-role document explicitly disclaims training for clinical
  use (`EVALUATIVE_ROLE.md`).
- Limitations document surfaces coverage, recall, and label-reliability
  gaps (`LIMITATIONS.md`).
- Contamination methodology proposes separating pre-cutoff and
  post-cutoff accuracy to stress-test functional claims
  (`CONTAMINATION.md`).
- Redaction-audit script provides measurable false-negative rate
  (`scripts/audit_redaction.py`).
- Coverage-analysis script quantifies therapeutic-area and
  molecular-property skew (`scripts/analyze_coverage.py`).

### Deferred to Round 2

- Human-evaluated accuracy on a safety-critical-claim sub-sample
  (scheduled).
- Phase 4 grounding check that verifies each functional claim is
  traceable to an evidence sentence (proposed; requires additional LLM
  compute).
- RAI review of the engineering-question category for synthesis-uplift
  risk (proposed).
- Dataset card fields per the Croissant RAI schema (skeleton provided in
  `croissant.json`; full population after full-run execution).

## Recommended downstream practice

For researchers fine-tuning on Chem2TextQA:

1. Exclude the engineering-question category if the target deployment
   involves novice or adversarial users.
2. Apply your own contamination check against your evaluation benchmark.
3. Do not report clinical metrics; use only chemistry-structural metrics
   for validation.
4. Include a model card that cites Chem2TextQA and repeats the
   intended-use restriction above.
