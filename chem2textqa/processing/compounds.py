"""Load and filter PubChem compound metadata from bulk files.

The bulk files are large (123M lines for SMILES). We stream through them
once and only keep data for the compounds we care about (those with MeSH
drug/metabolite classifications by default).
"""
from __future__ import annotations

import gzip
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CompoundProfile:
    """A drug/metabolite compound with structure + identifiers."""

    cid: int
    name: str = ""
    iupac_name: str = ""
    molecular_formula: str = ""
    molecular_weight: float | None = None
    smiles: str = ""
    inchi_key: str = ""
    mesh_terms: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "cid": self.cid,
            "name": self.name,
            "iupac_name": self.iupac_name,
            "molecular_formula": self.molecular_formula,
            "molecular_weight": self.molecular_weight,
            "smiles": self.smiles,
            "inchi_key": self.inchi_key,
            "mesh_terms": self.mesh_terms,
        }


def _open_text(path: Path):
    """Open .gz or plain text file in text mode."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def load_compound_index(
    bulk_dir: Path,
    require_mesh: bool = True,
    extra_cids: set[int] | None = None,
) -> dict[int, CompoundProfile]:
    """Build the compound index from bulk PubChem files.

    Args:
        bulk_dir: Directory with CID-* files
        require_mesh: If True (default), include compounds that have MeSH
            terms (the strict drug/metabolite filter, ~89K compounds).
            If False, skip the MeSH load entirely.
        extra_cids: Additional CIDs to include (e.g. union of DrugBank + HMDB
            + KEGG + ChEBI from curated sources). These are added on top of
            the MeSH set if require_mesh is True.

    Returns:
        {cid: CompoundProfile} for the combined set.
    """
    bulk_dir = Path(bulk_dir)
    compounds: dict[int, CompoundProfile] = {}

    # Step 1: Load CID-MeSH (the baseline drug/metabolite target set)
    if require_mesh:
        mesh_file = bulk_dir / "CID-MeSH"
        if not mesh_file.exists():
            raise FileNotFoundError(f"Missing {mesh_file}")

        logger.info("Loading CID-MeSH (drug/metabolite filter)...")
        with _open_text(mesh_file) as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if not parts or not parts[0].isdigit():
                    continue
                cid = int(parts[0])
                mesh_terms = [t.strip() for t in parts[1:] if t.strip()]
                compounds[cid] = CompoundProfile(cid=cid, mesh_terms=mesh_terms)
        logger.info("  MeSH-classified compounds: %d", len(compounds))

    # Step 1b: Merge in extra CIDs from curated sources (DrugBank, HMDB, etc.)
    if extra_cids:
        added = 0
        for cid in extra_cids:
            if cid not in compounds:
                compounds[cid] = CompoundProfile(cid=cid)
                added += 1
        logger.info("  Added %d new compounds from curated sources "
                     "(total now %d)", added, len(compounds))

    if not compounds:
        raise ValueError(
            "No compounds loaded — enable require_mesh and/or pass extra_cids"
        )

    # Step 2: Enrich with name (CID-Title.gz)
    _enrich_field(
        compounds,
        bulk_dir / "CID-Title.gz",
        lambda profile, value: setattr(profile, "name", value),
        label="names",
    )

    # Step 3: Enrich with SMILES
    _enrich_field(
        compounds,
        bulk_dir / "CID-SMILES.gz",
        lambda profile, value: setattr(profile, "smiles", value),
        label="SMILES",
    )

    # Step 4: Enrich with IUPAC name
    _enrich_field(
        compounds,
        bulk_dir / "CID-IUPAC.gz",
        lambda profile, value: setattr(profile, "iupac_name", value),
        label="IUPAC names",
    )

    # Step 5: Enrich with InChI key
    _enrich_field(
        compounds,
        bulk_dir / "CID-InChI-Key.gz",
        lambda profile, value: setattr(profile, "inchi_key", value),
        label="InChI keys",
    )

    # Step 6: Enrich with molecular formula + molecular weight
    # CID-Mass.gz format: CID<TAB>Formula<TAB>ExactMass<TAB>MonoIsotopicMass
    def _set_formula_and_mass(profile: CompoundProfile, columns: list[str]) -> None:
        if len(columns) >= 1 and columns[0]:
            profile.molecular_formula = columns[0]
        if len(columns) >= 2 and columns[1]:
            try:
                profile.molecular_weight = float(columns[1])
            except ValueError:
                pass

    _enrich_field_multi(
        compounds,
        bulk_dir / "CID-Mass.gz",
        _set_formula_and_mass,
        label="molecular formula + weight",
    )

    logger.info("Built compound index: %d compounds with full metadata", len(compounds))
    return compounds


def _enrich_field(
    compounds: dict[int, CompoundProfile],
    path: Path,
    setter,
    label: str,
) -> None:
    """Stream a CID-* file and set a single field on the compound profile.

    The setter callable takes (profile, value) and mutates the profile.
    Lines are TSV: <cid>\\t<value> (with possibly more columns — first used).
    """
    if not path.exists():
        logger.warning("Skipping %s (file not found)", path.name)
        return

    matched = 0
    with _open_text(path) as f:
        for line in f:
            tab = line.find("\t")
            if tab < 0:
                continue
            cid_str = line[:tab]
            if not cid_str.isdigit():
                continue
            cid = int(cid_str)
            if cid not in compounds:
                continue
            value = line[tab + 1 :].rstrip("\n").split("\t")[0]
            setter(compounds[cid], value)
            matched += 1

    logger.info("  %s: matched %d / %d compounds", label, matched, len(compounds))


def _enrich_field_multi(
    compounds: dict[int, CompoundProfile],
    path: Path,
    setter,
    label: str,
) -> None:
    """Like _enrich_field but passes ALL post-CID columns to the setter.

    The setter receives (profile, [col1, col2, ...]) so it can pick whichever
    columns it needs. Used for multi-column files like CID-Mass.gz
    (CID<TAB>Formula<TAB>ExactMass<TAB>MonoIsotopicMass).
    """
    if not path.exists():
        logger.warning("Skipping %s (file not found)", path.name)
        return

    matched = 0
    with _open_text(path) as f:
        for line in f:
            tab = line.find("\t")
            if tab < 0:
                continue
            cid_str = line[:tab]
            if not cid_str.isdigit():
                continue
            cid = int(cid_str)
            if cid not in compounds:
                continue
            columns = line[tab + 1 :].rstrip("\n").split("\t")
            setter(compounds[cid], columns)
            matched += 1

    logger.info("  %s: matched %d / %d compounds", label, matched, len(compounds))


def build_pmid_to_cids(
    bulk_dir: Path,
    target_cids: set[int],
) -> tuple[dict[int, list[int]], int]:
    """Build a reverse index PMID → [CIDs] for the target compound set.

    Returns:
        (pmid_to_cids, total_links_seen)
    """
    bulk_dir = Path(bulk_dir)
    cid_pmid_file = bulk_dir / "CID-PMID.gz"
    if not cid_pmid_file.exists():
        raise FileNotFoundError(f"Missing {cid_pmid_file}")

    logger.info("Building PMID→CIDs reverse index for %d target compounds...",
                 len(target_cids))

    pmid_to_cids: dict[int, list[int]] = {}
    total = 0
    matched = 0

    with gzip.open(cid_pmid_file, "rt", encoding="utf-8") as f:
        for line in f:
            total += 1
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            try:
                cid = int(parts[0])
                pmid = int(parts[1])
            except ValueError:
                continue
            if cid in target_cids:
                pmid_to_cids.setdefault(pmid, []).append(cid)
                matched += 1

    logger.info("Reverse index: %d unique PMIDs from %d / %d total links",
                 len(pmid_to_cids), matched, total)
    return pmid_to_cids, total
