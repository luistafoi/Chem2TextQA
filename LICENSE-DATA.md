# Chem2TextQA dataset license and upstream terms

## Dataset release

Chem2TextQA, the derivative dataset produced by this pipeline, is
released under the **Creative Commons Attribution 4.0 International
License (CC BY 4.0)**.

## Upstream sources and their terms

Chem2TextQA derives from three publicly redistributable sources, whose
terms continue to apply transitively to any user of this dataset:

| Upstream | Terms |
|---|---|
| **PubChem** (NIH/NLM) | Public domain (no copyright claimed by NIH/NLM). Reuse requires attribution but no license compatibility is enforced. https://www.ncbi.nlm.nih.gov/home/about/policies/ |
| **PubMed** (abstracts via NIH/NLM) | No copyright claimed on the bibliographic database; individual abstract copyright is held by the original publishers. Use in research and aggregate is permitted; full redistribution of abstract text may be restricted. |
| **PMC Open Access Subset** | Articles carry their original author-selected licenses (CC BY, CC BY-NC, CC BY-SA variants). Chem2TextQA redacts and extracts sentence fragments; derivative-work status depends on fragment lengths and jurisdiction. |

## Important reservations

1. **Sentence fragments in evidence bundles.** Chem2TextQA's Phase 0
   evidence sentences are extracted verbatim from PubMed abstracts and
   PMC full-text articles (with compound names replaced by `[COMPOUND]`).
   Users redistributing the raw evidence bundles must respect the
   original publisher / author licenses. Where PMC articles are licensed
   CC BY-NC, derivative non-commercial use is required.
2. **Q&A pairs.** Phase 1-3 LLM outputs are model-generated derivative
   text inspired by the upstream evidence. These are released under CC
   BY 4.0 for Chem2TextQA as a whole, but users should be aware that
   they encode (in paraphrased form) content from the upstream sources.
3. **SMILES / structural metadata.** PubChem-sourced; public domain.

## Recommended attribution

Citations as a user of the dataset:

> Chem2TextQA: A SMILES-Grounded Q&A Dataset Derived from PubChem ×
> PubMed × PMC. [Authors anonymized for review.] 2026.

Include citations to PubChem, PubMed, and any specific PMC articles
surfaced via the PMID metadata preserved in each compound record.

## If you are a rights-holder with a concern

See the dataset contact in the Hugging Face dataset card (post-release).
Chem2TextQA will honor reasonable takedown / correction requests for
specific evidence records.
