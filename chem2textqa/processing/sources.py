"""Download compound CID lists from PubChem source databases.

PubChem aggregates compounds from many curated sources. Rather than filtering
only by MeSH classification (89K compounds), we can expand the target set to
include vetted compounds from DrugBank, HMDB, KEGG, ChEBI etc.

These lists are fetched once via PUG REST and cached as plain CID text files
under <bulk_dir>/sources/<source>.txt, one CID per line.
"""
from __future__ import annotations

import logging
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

PUG_REST = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

# Curated source databases we can pull compound CIDs from.
# Keys are the short aliases we'll use; values are the exact PubChem source names.
CURATED_SOURCES: dict[str, str] = {
    "DrugBank": "DrugBank",
    "HMDB": "Human Metabolome Database",
    "KEGG": "KEGG",
    "ChEBI": "ChEBI",
    "BindingDB": "BindingDB",
    "ChEMBL": "ChEMBL",
}


def fetch_cids_from_source(
    source_alias: str,
    timeout: int = 120,
) -> list[int]:
    """Fetch all CIDs deposited by a curated source database via PUG REST.

    Uses /substance/sourceall/{source}/cids which returns all substance→CID
    mappings for the source. We flatten and deduplicate.
    """
    if source_alias not in CURATED_SOURCES:
        raise ValueError(
            f"Unknown source {source_alias!r}. Known: {list(CURATED_SOURCES)}"
        )

    source_name = CURATED_SOURCES[source_alias]
    url = (
        f"{PUG_REST}/substance/sourceall/"
        f"{requests.utils.quote(source_name, safe='')}"
        f"/cids/JSON?list_return=grouped"
    )

    logger.info("Fetching CIDs from %s (%s)...", source_alias, source_name)
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    data = r.json()

    info = data.get("InformationList", {}).get("Information", [])
    cid_set: set[int] = set()
    for entry in info:
        for cid in entry.get("CID", []) or []:
            try:
                cid_set.add(int(cid))
            except (TypeError, ValueError):
                continue

    cids = sorted(cid_set)
    logger.info("%s: %d unique CIDs", source_alias, len(cids))
    return cids


def save_cid_list(cids: list[int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for cid in cids:
            f.write(f"{cid}\n")


def load_cid_list(path: Path) -> list[int]:
    with path.open("r") as f:
        return [int(line.strip()) for line in f if line.strip().isdigit()]


def download_all_sources(
    bulk_dir: Path,
    sources: list[str] | None = None,
    force: bool = False,
) -> dict[str, int]:
    """Download and cache CID lists for each source.

    Returns: {source_alias: count}.
    """
    bulk_dir = Path(bulk_dir)
    sources_dir = bulk_dir / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)

    sources = sources or list(CURATED_SOURCES)
    counts: dict[str, int] = {}

    for alias in sources:
        out_path = sources_dir / f"{alias}.txt"
        if out_path.exists() and not force:
            existing = load_cid_list(out_path)
            counts[alias] = len(existing)
            logger.info("%s: using cached list (%d CIDs) at %s",
                         alias, len(existing), out_path)
            continue

        cids = fetch_cids_from_source(alias)
        save_cid_list(cids, out_path)
        counts[alias] = len(cids)

    return counts


def load_source_cids(
    bulk_dir: Path,
    sources: list[str],
) -> set[int]:
    """Load the union of CIDs across the requested sources."""
    bulk_dir = Path(bulk_dir)
    sources_dir = bulk_dir / "sources"

    all_cids: set[int] = set()
    for alias in sources:
        path = sources_dir / f"{alias}.txt"
        if not path.exists():
            logger.warning(
                "No cached CID list at %s — run download_all_sources() first",
                path,
            )
            continue
        cids = load_cid_list(path)
        all_cids.update(cids)
        logger.info("  %s: %d CIDs (union now %d)", alias, len(cids), len(all_cids))

    return all_cids
