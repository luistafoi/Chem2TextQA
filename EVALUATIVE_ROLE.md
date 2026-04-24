# Evaluative role of Chem2TextQA

NeurIPS E&D Track requires that dataset submissions articulate their
evaluative role: what claims they support, under what assumptions, with
what limitations. This document pins the claim down.

## Primary role

**Chem2TextQA is an instruction-tuning resource for medicinal-chemistry
reasoning.** Specifically:

> *Fine-tuning a base language model on Chem2TextQA should improve its
> ability to answer structured questions about drug-like compounds given
> the SMILES string, spanning both structural (scaffold, functional
> groups, reactivity) and functional (mechanism, metabolism, therapeutic
> use, toxicity, drug interactions, engineering) topics.*

## What the dataset is NOT

- **Not a benchmark.** Agreement between two LLMs is not ground truth.
  The agree-only subset reflects cross-model consensus, which may codify
  shared biases. Evaluation of a trained model's correctness requires a
  held-out human-annotated test set (not provided here; see
  `LIMITATIONS.md`).
- **Not a comprehensive chemistry corpus.** Coverage is biased toward
  FDA-approved and well-studied drugs (see `DATASHEET.md` §
  "Composition").
- **Not a safety-critical knowledge source.** Drug-related claims here
  carry no clinical warranty. See `RESPONSIBLE_AI.md`.

## Scoped evaluative claim (testable)

*H1.* A 3B-parameter open model fine-tuned on the agree-only subset of
Chem2TextQA will outperform the same base model, zero-shot and few-shot,
on a held-out structural-reasoning Q&A benchmark covering functional
groups, scaffold identification, and rotatable-bond counting from SMILES.

*H2.* The same fine-tuned model will **not** reliably outperform on
purely functional claims (mechanism, clinical use) drawn from compounds
published **after** the base model's training cutoff.

*H1 is the use case the dataset is designed for. H2 is the failure mode
we expect and will document as a limitation, not hide.*

## Target downstream benchmarks (for validation)

Validation of H1 will use the following as downstream evaluation targets
(none of which Chem2TextQA was trained against):

| Benchmark | Focus | Why included |
|---|---|---|
| **ChemBench** | General chemistry reasoning | Covers structure, reactions, nomenclature; community standard |
| **ChemLLMBench** | LLM-specific chemistry tasks | Matches deployment pattern |
| **Mol-Instructions (mol subset)** | Structural reasoning | Closest domain match |
| **PubMedQA (held-out)** | Biomedical QA | Functional-claim transfer |

## Assumptions the claim depends on

1. The base open model being fine-tuned has a pretraining cutoff **after**
   the PubMed / PMC corpus was assembled (otherwise we measure memorization).
2. Held-out evaluation compounds are not present in Chem2TextQA training.
3. The target evaluation benchmark is not itself contaminated into the
   base model's pretraining.

Violations of 1-3 would confound the measurement. This is why
`CONTAMINATION.md` defines a canary set drawn from post-cutoff compounds
as a validation guard.

## What Round 1 explicitly does not claim

- That the dataset is SOTA for anything.
- That the agree-only subset is free of hallucinations.
- That Chem2TextQA is better than Mol-Instructions or ChemBench for any
  downstream task. (No head-to-head comparison run yet; scheduled for
  Round 2.)
