"""Prompts for Phase 1 (Q&A generation).

Design philosophy
-----------------

The LLM receives SMILES + formula + MW plus a set of redacted literature
sentences describing the compound. It writes Q&A pairs that span two kinds
of claims:

  * Structural claims (scaffold, functional groups, descriptors, stereo,
    reactivity sites) — must be derivable from the SMILES.
  * Functional claims (mechanism, metabolism, therapeutic use, toxicity,
    drug interactions, design implications) — may be supported by the
    evidence sentences, used silently as background knowledge.

Output voice is a medicinal chemist synthesising structural and
pharmacological insight. Evidence is NEVER quoted, paraphrased, or cited
with markers — reader should not see the literature as literature.

This is a softening of an earlier design that required every claim be
SMILES-derivable alone; that rule made functional topics (metabolism,
mechanism, etc.) impossible to answer, which was off-target for this
dataset's stated categories (mechanism, therapeutic use, toxicity,
metabolism, drug interactions, chemistry).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StructuralTopic:
    """A topic inspiration — NOT a required bucket; the model may invent its own."""
    key: str
    description: str


# Inspiration topics. The LLM may use these as the `topic` tag, or invent
# its own (e.g. "resistance_mechanism", "prodrug_activation"). Validation
# accepts any non-empty string.
STRUCTURAL_TOPICS: list[StructuralTopic] = [
    StructuralTopic("composition", "molecular formula, MW, heavy-atom count, degree of unsaturation"),
    StructuralTopic("scaffold", "carbon skeleton, ring systems, fusion/spiro patterns, branching"),
    StructuralTopic("functional_groups", "functional groups, connectivity, modifiable sites"),
    StructuralTopic("electronics", "hybridization, aromaticity, conjugation, polarization, charges"),
    StructuralTopic("stereochemistry", "chiral centers, E/Z geometry, stereoisomer count"),
    StructuralTopic("shape_sterics", "3D shape, steric bulk, rotatable bonds, flexibility/rigidity"),
    StructuralTopic("physicochemical", "logP, PSA, pKa, solubility, molar refractivity"),
    StructuralTopic("reactivity", "electrophilic/nucleophilic sites, acidic protons, strained bonds"),
    StructuralTopic("pharmacophore", "H-bond donors/acceptors, aromatic rings, hydrophobic patches, charged groups"),
    StructuralTopic("mechanism", "molecular target, binding mode, mode of action"),
    StructuralTopic("metabolism", "metabolic pathways, soft spots, CYP involvement, major metabolites"),
    StructuralTopic("therapeutic_use", "indications and their structural rationale"),
    StructuralTopic("toxicity", "structural alerts, reactive metabolites, known adverse effects"),
    StructuralTopic("drug_interactions", "CYP inhibition/induction, transporter interactions, PK-level interactions"),
    StructuralTopic("adme", "absorption, distribution, BBB, clearance — from structure and literature"),
    StructuralTopic("druglikeness", "Lipinski/Veber rules, lead-likeness"),
    StructuralTopic("engineering", "bioisosteric substitutions, ring replacements, functional-group swaps and their predicted effects on logP, potency, metabolic stability, selectivity"),
    StructuralTopic("design_levers", "modifiable sites, scaffold variations, SAR handles"),
]


# Keep the old name around so legacy imports don't break.
QA_TEMPLATES = STRUCTURAL_TOPICS


SYSTEM_PROMPT = """You are an expert medicinal chemist producing Q&A training \
data about chemical compounds.

You will receive:
- The compound's SMILES, molecular formula, and molecular weight
- A numbered list of redacted literature sentences about the compound \
(PRIVATE topic hints — used silently, never quoted)

RULES for every question and every answer:
- Refer to the molecule only as "the compound". Never name it or guess its \
identity.
- Structural claims (scaffold, functional groups, descriptors, \
stereochemistry, reactivity sites, physicochemical properties) must be \
derivable from the SMILES / formula / MW.
- Functional claims (mechanism of action, metabolism, therapeutic use, \
toxicity, drug interactions, ADME, design implications) may be supported by \
the evidence sentences, used SILENTLY as background knowledge — you absorb \
them and write as an expert who happens to know these facts.
- Do NOT quote, paraphrase, or cite the evidence. Do NOT use phrases like \
"as reported", "according to the article", "in this study", "the paper \
discusses", "it was shown", "[E1]", etc. No markers, no pointers.
- The reader should not see the evidence as literature. The output should \
read as a knowledgeable chemist's analysis, not a literature review.
- Answers should be 1–4 sentences, specific, not encyclopedic.

TOPIC DIVERSITY
- Use the inspiration taxonomy in the user message as a menu, NOT a \
checklist. You may pick any subset.
- You may set `topic` to one of the taxonomy keys OR invent your own tag \
when the evidence warrants (e.g. "prodrug_activation", "resistance_mechanism").
- Two Q&A pairs may share a topic tag as long as they ask substantively \
different questions about different aspects. Avoid rephrasings that yield \
the same answer.

ENGINEERING QUESTIONS
- When the structure and evidence warrant it, include at least one \
analog-design question: "If the [group] at [position] were replaced with \
[alternative], what would be the likely effect on [property]?" Anchor the \
reasoning in structural intuition and (when applicable) the compound's \
known metabolism/mechanism/toxicity profile.

WORKED EXAMPLE (for a hypothetical salicylate-class compound, SMILES \
CC(=O)Oc1ccccc1C(=O)O):
{
  "qa_pairs": [
    {
      "topic": "functional_groups",
      "question": "What functional groups are present and how are they \
positioned on the scaffold?",
      "answer": "The compound contains a carboxylic acid and an aryl ester, \
ortho on a benzene ring. The ester oxygen bridges the aromatic ring and an \
acetyl group."
    },
    {
      "topic": "mechanism",
      "question": "What is the molecular target of the compound and which \
structural feature drives that interaction?",
      "answer": "The compound is an irreversible inhibitor of cyclooxygenase \
(COX-1/COX-2). The acetyl group transfers to an active-site serine, \
acetylating it and blocking the substrate channel — a covalent modification \
enabled specifically by the ester's acyl transfer capacity."
    },
    {
      "topic": "metabolism",
      "question": "Where in the structure is the major metabolic liability, \
and what pathway dominates?",
      "answer": "The ester linkage is hydrolysed rapidly by plasma and \
hepatic esterases to yield the free carboxylic acid (salicylate), which is \
then glucuronidated and partially converted to salicyluric acid. Hydrolysis \
is the rate-limiting structural vulnerability."
    },
    {
      "topic": "engineering",
      "question": "If the acetyl ester were replaced with a propionate \
ester, what would change?",
      "answer": "A propionate ester would be marginally more lipophilic \
(logP up ~0.5) and slightly more stable to esterase hydrolysis, blunting \
covalent COX acetylation. Analgesic onset would slow and the irreversible \
inhibition character would weaken, pushing the compound toward a competitive \
mechanism."
    }
  ]
}

OUTPUT FORMAT (JSON only — no prose before or after the JSON object):
{
  "qa_pairs": [
    {
      "topic": "<string: taxonomy key or a self-invented tag>",
      "question": "<question about the compound>",
      "answer": "<answer grounded in SMILES (structural) and/or evidence (functional)>"
    },
    ...
  ]
}"""


def _topic_taxonomy_text() -> str:
    return "\n".join(
        f"  - {t.key}: {t.description}" for t in STRUCTURAL_TOPICS
    )


def _target_count(n_hints: int) -> str:
    if n_hints < 10:
        return "5 to 7"
    if n_hints < 30:
        return "10 to 15"
    if n_hints < 100:
        return "15 to 25"
    if n_hints < 300:
        return "25 to 35"
    return "35 to 50"


def build_user_prompt(
    smiles: str,
    molecular_formula: str,
    molecular_weight: float | None,
    evidence_sentences: list[dict],
) -> str:
    """Build the user message for Phase 1."""
    mw_str = f"{molecular_weight:.2f}" if molecular_weight is not None else "?"
    hints_text = "\n".join(
        f"- {e['text']}" for e in evidence_sentences
    )
    n_hints = len(evidence_sentences)
    target = _target_count(n_hints)
    return (
        f"SMILES: {smiles}\n"
        f"Molecular formula: {molecular_formula}\n"
        f"Molecular weight: {mw_str}\n\n"
        f"[PRIVATE TOPIC HINT — DO NOT MENTION OR QUOTE — {n_hints} sentences]\n"
        f"{hints_text}\n"
        f"[END PRIVATE TOPIC HINT]\n\n"
        f"TOPIC INSPIRATION (menu, not a checklist — you may set `topic` to one "
        f"of these keys or a self-invented tag):\n"
        f"{_topic_taxonomy_text()}\n\n"
        f"Generate {target} Q&A pairs. Structural questions must be answerable "
        f"from the SMILES / formula / MW. Functional questions (mechanism, "
        f"metabolism, therapeutic use, toxicity, drug interactions, ADME, "
        f"engineering/analog design) should be grounded in the topic hints, used "
        f"silently — never quote, paraphrase, or cite them. Write as an expert "
        f"chemist synthesising structural and pharmacological knowledge. Include "
        f"at least one engineering / analog-substitution question when the hints "
        f"provide enough signal.\n\n"
        f"Output JSON only."
    )
