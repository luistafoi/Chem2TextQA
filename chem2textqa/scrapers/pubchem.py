from __future__ import annotations

import json

import requests
from tqdm import tqdm

from chem2textqa.config.settings import Settings
from chem2textqa.models.document import CompoundRecord
from chem2textqa.scrapers.base import BaseScraper
from chem2textqa.utils.rate_limiter import RateLimiter
from chem2textqa.utils.retry import with_retry

PUG_REST = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PUG_VIEW = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view"
SDQ_URL = "https://pubchem.ncbi.nlm.nih.gov/sdq/sdqagent.cgi"

# Properties to fetch in bulk from PUG REST
COMPOUND_PROPERTIES = [
    "IUPACName",
    "MolecularFormula",
    "MolecularWeight",
    "CanonicalSMILES",
    "IsomericSMILES",
    "InChI",
    "InChIKey",
    "Title",
]

BATCH_SIZE = 100  # PUG REST allows up to 100 CIDs per request
SDQ_PAGE_SIZE = 10000  # SDQ can return up to 10k per page


# Curated source databases and their PUG REST names.
# Use these with --source to get vetted compounds instead of keyword search.
SOURCE_DATABASES: dict[str, str] = {
    "DrugBank": "DrugBank",
    "ChEMBL": "ChEMBL",
    "HMDB": "Human Metabolome Database",
    "ChEBI": "ChEBI",
    "KEGG": "KEGG",
    "BindingDB": "BindingDB",
}

# How many SIDs to retrieve per listkey page
SID_PAGE_SIZE = 10000


class PubChemScraper(BaseScraper):
    """Scrapes PubChem for drug/metabolite compounds and their linked PubMed articles."""

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self._rate_limiter = RateLimiter(settings.pubchem_rate_limit)
        self._session = requests.Session()
        self._session.headers["Accept"] = "application/json"

    @property
    def name(self) -> str:
        return "pubchem"

    # ------------------------------------------------------------------
    # Discovery: find CIDs from a curated source database
    # ------------------------------------------------------------------

    def cids_from_source(
        self, source_name: str, max_results: int | None = None
    ) -> list[int]:
        """
        Get compound CIDs deposited by a curated source database (e.g. DrugBank).

        Uses PUG REST:
          /substance/sourceall/{source}/sids → SIDs
          /substance/sid/{sids}/cids          → CIDs
        """
        pug_source = SOURCE_DATABASES.get(source_name, source_name)
        self.logger.info("Fetching CIDs from source: %s", pug_source)

        # Step 1: Get all SIDs from this source
        sids = self._fetch_sids_from_source(pug_source, max_results)
        if not sids:
            self.logger.warning("No SIDs found for source %r", pug_source)
            return []

        self.logger.info("Found %d SIDs from %s, converting to CIDs", len(sids), source_name)

        # Step 2: Convert SIDs → CIDs in batches
        cids = self._sids_to_cids(sids)

        self.logger.info("Got %d unique CIDs from %s", len(cids), source_name)
        if max_results and len(cids) > max_results:
            cids = cids[:max_results]
        return cids

    def _fetch_sids_from_source(
        self, pug_source: str, max_results: int | None
    ) -> list[int]:
        """Fetch SIDs from a PubChem source database using listkey pagination."""
        # First request gets a listkey
        url = (
            f"{PUG_REST}/substance/sourceall/"
            f"{requests.utils.quote(pug_source, safe='')}"
            f"/sids/JSON?list_return=listkey"
        )
        data = self._get_json(url)
        if not data or "IdentifierList" not in data:
            return []

        listkey = data["IdentifierList"].get("ListKey")
        total = data["IdentifierList"].get("Size", 0)
        if not listkey or not total:
            return []

        limit = min(total, max_results) if max_results else total
        self.logger.info("Source has %d SIDs total, fetching %d", total, limit)

        # Paginate through the listkey to get actual SID values
        sids: list[int] = []
        start = 0

        while len(sids) < limit:
            page_size = min(SID_PAGE_SIZE, limit - len(sids))
            page_url = (
                f"{PUG_REST}/substance/listkey/{listkey}/sids/JSON"
                f"?listkey_start={start}&listkey_count={page_size}"
            )
            page_data = self._get_json(page_url)
            if not page_data or "IdentifierList" not in page_data:
                break

            batch = page_data["IdentifierList"].get("SID", [])
            if not batch:
                break

            sids.extend(int(s) for s in batch)
            start += len(batch)

        return sids[:limit]

    def _sids_to_cids(self, sids: list[int]) -> list[int]:
        """Convert SIDs to CIDs via PUG REST. Returns deduplicated CID list."""
        cid_set: set[int] = set()

        for i in tqdm(range(0, len(sids), BATCH_SIZE), desc="SID→CID conversion"):
            batch = sids[i : i + BATCH_SIZE]
            sid_str = ",".join(str(s) for s in batch)
            url = f"{PUG_REST}/substance/sid/{sid_str}/cids/JSON"

            data = self._get_json(url)
            if data and "InformationList" in data:
                for info in data["InformationList"].get("Information", []):
                    for cid in info.get("CID", []):
                        cid_set.add(int(cid))

        return sorted(cid_set)

    # ------------------------------------------------------------------
    # Discovery: find CIDs matching a keyword
    # ------------------------------------------------------------------

    def search_cids(self, keyword: str, max_results: int = 10000) -> list[int]:
        """Search PubChem compounds by keyword via SDQ, return CID list."""
        self.logger.info("Searching PubChem for %r (max %d)", keyword, max_results)
        cids: list[int] = []
        start = 1
        total_available = 0

        while len(cids) < max_results:
            page_size = min(SDQ_PAGE_SIZE, max_results - len(cids))
            query = json.dumps({
                "select": "cid",
                "collection": "compound",
                "where": {"ands": [{"*": keyword}]},
                "order": ["cid,asc"],
                "start": start,
                "limit": page_size,
            })

            self._rate_limiter.wait()
            resp = self._session.get(
                SDQ_URL,
                params={"infmt": "json", "outfmt": "json", "query": query},
                timeout=30,
            )
            if resp.status_code != 200:
                self.logger.warning("SDQ returned %d for keyword %r", resp.status_code, keyword)
                break

            data = resp.json()
            output = data.get("SDQOutputSet", [{}])[0]
            rows = output.get("rows", [])
            if not rows:
                break

            for row in rows:
                cid = row.get("cid")
                if cid is not None:
                    cids.append(int(cid))

            total_available = output.get("totalCount", 0)
            start += len(rows)
            if start > total_available:
                break

        self.logger.info("Found %d CIDs for %r (total available: ~%d)",
                         len(cids), keyword, total_available)
        return cids[:max_results]

    def count_compounds(self, keyword: str) -> int:
        """Return the total number of compounds matching a keyword."""
        query = json.dumps({
            "select": "*",
            "collection": "compound",
            "where": {"ands": [{"*": keyword}]},
            "start": 1,
            "limit": 0,
        })
        self._rate_limiter.wait()
        resp = self._session.get(
            SDQ_URL,
            params={"infmt": "json", "outfmt": "json", "query": query},
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            return data.get("SDQOutputSet", [{}])[0].get("totalCount", 0)
        return 0

    # ------------------------------------------------------------------
    # Bulk property fetch
    # ------------------------------------------------------------------

    def fetch_properties(self, cids: list[int]) -> dict[int, dict]:
        """Fetch basic properties for a list of CIDs. Returns {cid: {prop: value}}."""
        results: dict[int, dict] = {}
        prop_list = ",".join(COMPOUND_PROPERTIES)

        for i in tqdm(range(0, len(cids), BATCH_SIZE), desc="PubChem properties"):
            batch = cids[i : i + BATCH_SIZE]
            cid_str = ",".join(str(c) for c in batch)
            url = f"{PUG_REST}/compound/cid/{cid_str}/property/{prop_list}/JSON"

            data = self._get_json(url)
            if data and "PropertyTable" in data:
                for prop in data["PropertyTable"].get("Properties", []):
                    cid = prop.get("CID")
                    if cid:
                        results[cid] = prop

        return results

    # ------------------------------------------------------------------
    # Linked PubMed articles for a compound
    # ------------------------------------------------------------------

    def fetch_linked_pmids(self, cid: int) -> list[int]:
        """Get PubMed article IDs linked to a PubChem compound."""
        url = f"{PUG_REST}/compound/cid/{cid}/xrefs/PubMedID/JSON"
        data = self._get_json(url)
        if data and "InformationList" in data:
            info_list = data["InformationList"].get("Information", [])
            if info_list:
                return info_list[0].get("PubMedID", [])
        return []

    def fetch_linked_pmids_batch(self, cids: list[int]) -> dict[int, list[int]]:
        """Get linked PMIDs for multiple CIDs. Returns {cid: [pmid, ...]}."""
        results: dict[int, list[int]] = {}

        for i in tqdm(range(0, len(cids), BATCH_SIZE), desc="PubChem→PubMed links"):
            batch = cids[i : i + BATCH_SIZE]
            cid_str = ",".join(str(c) for c in batch)
            url = f"{PUG_REST}/compound/cid/{cid_str}/xrefs/PubMedID/JSON"

            data = self._get_json(url)
            if data and "InformationList" in data:
                for info in data["InformationList"].get("Information", []):
                    cid = info.get("CID")
                    pmids = info.get("PubMedID", [])
                    if cid:
                        results[cid] = pmids

        return results

    # ------------------------------------------------------------------
    # Compound description / pharmacological actions
    # ------------------------------------------------------------------

    def fetch_description(self, cid: int) -> str | None:
        """Fetch the compound description from PUG View."""
        url = f"{PUG_VIEW}/data/compound/{cid}/JSON?heading=Record+Description"
        data = self._get_json(url)
        if not data:
            return None

        try:
            sections = data["Record"]["Section"]
            for section in sections:
                for sub in section.get("Section", []):
                    for info in sub.get("Information", []):
                        desc = info.get("Value", {}).get("StringWithMarkup", [])
                        if desc:
                            return desc[0].get("String", "")
        except (KeyError, IndexError):
            pass
        return None

    def fetch_pharmacological_actions(self, cid: int) -> list[str]:
        """Fetch MeSH pharmacological actions for a compound."""
        url = f"{PUG_VIEW}/data/compound/{cid}/JSON?heading=Pharmacological+Actions"
        data = self._get_json(url)
        if not data:
            return []

        actions: list[str] = []
        try:
            sections = data["Record"]["Section"]
            for section in sections:
                for sub in section.get("Section", []):
                    for info in sub.get("Information", []):
                        for markup in info.get("Value", {}).get("StringWithMarkup", []):
                            s = markup.get("String", "").strip()
                            if s:
                                actions.append(s)
        except (KeyError, IndexError):
            pass
        return actions

    # ------------------------------------------------------------------
    # Build full CompoundRecord
    # ------------------------------------------------------------------

    def build_compound_record(self, cid: int, properties: dict | None = None) -> CompoundRecord:
        """Build a CompoundRecord for a single CID, fetching all available data."""
        props = properties or {}

        # PUG REST returns SMILES as CanonicalSMILES, ConnectivitySMILES,
        # or IsomericSMILES depending on the compound. Check all.
        smiles = (
            props.get("CanonicalSMILES")
            or props.get("ConnectivitySMILES")
            or props.get("IsomericSMILES")
        )

        # MolecularWeight comes back as string from PUG REST
        mw = props.get("MolecularWeight")
        if isinstance(mw, str):
            try:
                mw = float(mw)
            except ValueError:
                mw = None

        return CompoundRecord(
            cid=cid,
            name=props.get("Title", f"CID-{cid}"),
            iupac_name=props.get("IUPACName"),
            molecular_formula=props.get("MolecularFormula"),
            molecular_weight=mw,
            canonical_smiles=smiles,
            inchi=props.get("InChI"),
            inchi_key=props.get("InChIKey"),
        )

    def scrape_compounds(
        self,
        keyword: str | None = None,
        source: str | None = None,
        max_compounds: int = 1000,
        fetch_pmids: bool = True,
        fetch_descriptions: bool = False,
        fetch_pharma_actions: bool = False,
    ) -> list[CompoundRecord]:
        """Full pipeline: discover CIDs → properties → linked PMIDs → CompoundRecords.

        Provide either ``keyword`` (SDQ search) or ``source`` (curated database).
        """
        if source:
            cids = self.cids_from_source(source, max_results=max_compounds)
        elif keyword:
            cids = self.search_cids(keyword, max_results=max_compounds)
        else:
            raise ValueError("Provide either keyword or source")

        if not cids:
            label = source or keyword
            self.logger.warning("No compounds found for %r", label)
            return []

        self.logger.info("Fetching properties for %d compounds", len(cids))
        props_map = self.fetch_properties(cids)

        pmids_map: dict[int, list[int]] = {}
        if fetch_pmids:
            self.logger.info("Fetching linked PubMed IDs")
            pmids_map = self.fetch_linked_pmids_batch(cids)

        records: list[CompoundRecord] = []
        for cid in tqdm(cids, desc="Building compound records"):
            props = props_map.get(cid, {})
            record = self.build_compound_record(cid, props)
            record.linked_pmids = pmids_map.get(cid, [])

            if fetch_descriptions:
                desc = self.fetch_description(cid)
                if desc:
                    record.description = desc

            if fetch_pharma_actions:
                actions = self.fetch_pharmacological_actions(cid)
                record.pharmacological_actions = actions

            records.append(record)

        label = source or keyword
        self.logger.info("Built %d compound records for %r", len(records), label)
        return records

    def collect_all_pmids(self, records: list[CompoundRecord]) -> list[int]:
        """Extract deduplicated PMIDs from a list of compound records."""
        pmid_set: set[int] = set()
        for rec in records:
            pmid_set.update(rec.linked_pmids)
        return sorted(pmid_set)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @with_retry()
    def _get_json(self, url: str) -> dict | None:
        self._rate_limiter.wait()
        resp = self._session.get(url, timeout=30)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()
