"""Shared helper — classify a Q&A's freeform `topic` tag as structural vs functional.

Under the soft-rule design:
  - Structural claims are derivable from SMILES/formula/MW.
  - Functional claims rely on evidence.

Bucketing a QA lets us measure each group separately in the ablation,
SMILES-swap, and empty-evidence probes, since the two are expected to
behave very differently (structural should be hint-independent;
functional should be hint-dependent).
"""
from __future__ import annotations


STRUCTURAL_KEYS = frozenset({
    "composition", "scaffold", "functional_groups", "functional_group",
    "electronics", "electronic", "stereochemistry", "shape_sterics",
    "shape", "sterics", "physicochemical", "reactivity", "pharmacophore",
    "druglikeness", "connectivity",
})

FUNCTIONAL_KEYS = frozenset({
    "mechanism", "metabolism", "therapeutic_use", "therapeutic",
    "toxicity", "drug_interactions", "drug_interaction", "adme",
    "engineering", "design_levers", "design", "pharmacokinetics",
    "bioavailability", "resistance_mechanism", "resistance",
    "microbial_metabolism", "environmental_chemistry", "prodrug_activation",
})

# Substring hints for freeform topics we haven't enumerated.
_STRUCTURAL_SUBSTRINGS = (
    "scaffold", "functional_group", "functional-group",
    "electronic", "stereo", "shape", "sterics", "steric",
    "physicochem", "reactiv", "pharmacophore", "druglik", "lipinski",
    "ring_system", "connectivity", "aromaticity", "conjugation",
    "hybridization", "logp", "psa", "pka",
)
_FUNCTIONAL_SUBSTRINGS = (
    "mechanism", "metabol", "therapeutic", "toxic", "interaction",
    "adme", "engineer", "design", "analog", "resistance",
    "pharmacokinetic", "bioavail", "cyp", "transporter", "excretion",
    "absorption", "distribution", "clearance", "indication",
    "clinical", "target", "binding", "receptor",
)


def bucket_topic(topic: str | None) -> str:
    """Return 'structural', 'functional', or 'other'."""
    if not topic:
        return "other"
    t = topic.strip().lower().replace("-", "_")
    if t in STRUCTURAL_KEYS:
        return "structural"
    if t in FUNCTIONAL_KEYS:
        return "functional"
    for sub in _STRUCTURAL_SUBSTRINGS:
        if sub in t:
            return "structural"
    for sub in _FUNCTIONAL_SUBSTRINGS:
        if sub in t:
            return "functional"
    return "other"
