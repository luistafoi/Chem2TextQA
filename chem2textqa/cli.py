import json
from pathlib import Path

import click

from chem2textqa.config import get_settings
from chem2textqa.storage.jsonl import append_documents, count_documents
from chem2textqa.utils.logging import setup_logging


@click.group()
def main():
    """Chem2TextQA -- scrape PubChem, PubMed, and PMC for drug/metabolite AI datasets."""
    pass


# ------------------------------------------------------------------
# pubchem: discover compounds and their linked articles
# ------------------------------------------------------------------


@main.command()
@click.option("--keyword", "-k", default=None, help="Search keyword (e.g. 'drug', 'inhibitor')")
@click.option("--source", "-s", default=None,
              help="Curated source database (DrugBank, ChEMBL, HMDB, ChEBI, KEGG, BindingDB)")
@click.option("--max-compounds", "-n", default=100, show_default=True, help="Max compounds to fetch")
@click.option("--fetch-pmids/--no-fetch-pmids", default=True, show_default=True, help="Fetch linked PubMed IDs")
@click.option("--output", "-o", default=None, type=click.Path(), help="Output JSONL (default: data/pubchem.jsonl)")
def pubchem(keyword, source, max_compounds, fetch_pmids, output):
    """Search PubChem for compounds and extract linked article IDs."""
    if not keyword and not source:
        raise click.UsageError("Provide --keyword or --source")

    settings = get_settings()
    setup_logging(settings.log_level)

    from chem2textqa.scrapers.pubchem import PubChemScraper

    scraper = PubChemScraper(settings)

    label = source or keyword
    click.echo(f"Searching PubChem for: {label}")
    records = scraper.scrape_compounds(
        keyword=keyword,
        source=source,
        max_compounds=max_compounds,
        fetch_pmids=fetch_pmids,
    )

    if not records:
        click.echo("No compounds found.")
        return

    out_path = Path(output) if output else settings.output_dir / "pubchem.jsonl"
    written = append_documents(out_path, records)
    click.echo(f"Wrote {written} compound records to {out_path}")

    if fetch_pmids:
        all_pmids = scraper.collect_all_pmids(records)
        click.echo(f"Total unique linked PMIDs: {len(all_pmids)}")

        # Save PMID list for downstream use
        pmid_path = out_path.parent / "pmids_from_pubchem.txt"
        pmid_path.write_text("\n".join(str(p) for p in all_pmids))
        click.echo(f"PMID list saved to {pmid_path}")


# ------------------------------------------------------------------
# pubmed: fetch abstracts by query or PMID list
# ------------------------------------------------------------------


@main.command()
@click.option("--query", "-q", default=None, help="PubMed search query")
@click.option("--pmid-file", "-p", default=None, type=click.Path(exists=True), help="File with PMIDs (one per line)")
@click.option("--max-results", "-n", default=1000, show_default=True)
@click.option("--output", "-o", default=None, type=click.Path())
def pubmed(query, pmid_file, max_results, output):
    """Fetch PubMed abstracts by query or PMID list."""
    if not query and not pmid_file:
        raise click.UsageError("Provide --query or --pmid-file")

    settings = get_settings()
    setup_logging(settings.log_level)

    from chem2textqa.scrapers.pubmed import PubMedScraper

    scraper = PubMedScraper(settings)

    if pmid_file:
        pmids = [int(line.strip()) for line in Path(pmid_file).read_text().splitlines() if line.strip()]
        if max_results:
            pmids = pmids[:max_results]
        click.echo(f"Fetching {len(pmids)} articles by PMID")
        documents = scraper.fetch_by_pmids(pmids)
    else:
        click.echo(f"Searching PubMed: {query}")
        documents = scraper.search(query=query, max_results=max_results)

    if not documents:
        click.echo("No articles found.")
        return

    out_path = Path(output) if output else settings.output_dir / "pubmed.jsonl"
    written = append_documents(out_path, documents)
    click.echo(f"Wrote {written} documents to {out_path}")


# ------------------------------------------------------------------
# pmc: fetch full-text articles
# ------------------------------------------------------------------


@main.command()
@click.option("--query", "-q", default=None, help="PMC search query")
@click.option("--pmid-file", "-p", default=None, type=click.Path(exists=True), help="File with PMIDs to look up in PMC")
@click.option("--max-results", "-n", default=100, show_default=True)
@click.option("--output", "-o", default=None, type=click.Path())
def pmc(query, pmid_file, max_results, output):
    """Fetch full-text articles from PubMed Central."""
    if not query and not pmid_file:
        raise click.UsageError("Provide --query or --pmid-file")

    settings = get_settings()
    setup_logging(settings.log_level)

    from chem2textqa.scrapers.pmc import PMCScraper

    scraper = PMCScraper(settings)

    if pmid_file:
        pmids = [int(line.strip()) for line in Path(pmid_file).read_text().splitlines() if line.strip()]
        if max_results:
            pmids = pmids[:max_results]
        click.echo(f"Looking up {len(pmids)} PMIDs in PMC")
        documents = scraper.fetch_by_pmids(pmids)
    else:
        click.echo(f"Searching PMC: {query}")
        documents = scraper.search(query=query, max_results=max_results)

    if not documents:
        click.echo("No full-text articles found.")
        return

    out_path = Path(output) if output else settings.output_dir / "pmc.jsonl"
    written = append_documents(out_path, documents)
    click.echo(f"Wrote {written} full-text documents to {out_path}")


# ------------------------------------------------------------------
# count: estimate available data
# ------------------------------------------------------------------


@main.command()
@click.argument("file", required=False, type=click.Path(exists=True))
@click.option("--query", "-q", default=None, help="Count results for a PubMed/PMC query")
@click.option("--db", "-d", default="pubmed", type=click.Choice(["pubmed", "pmc"]))
def count(file, query, db):
    """Count documents in a JSONL file or results for a query."""
    if file:
        n = count_documents(Path(file))
        click.echo(f"{n} documents in {file}")
    elif query:
        settings = get_settings()
        setup_logging(settings.log_level)

        if db == "pubmed":
            from chem2textqa.scrapers.pubmed import PubMedScraper
            scraper = PubMedScraper(settings)
        else:
            from chem2textqa.scrapers.pmc import PMCScraper
            scraper = PMCScraper(settings)

        n = scraper.count(query)
        click.echo(f"{db.upper()}: {n:,} results for {query!r}")
    else:
        raise click.UsageError("Provide a FILE or --query")


# ------------------------------------------------------------------
# info: configuration summary
# ------------------------------------------------------------------


@main.command()
def info():
    """Print configuration summary."""
    settings = get_settings()

    click.echo("Chem2TextQA Configuration")
    click.echo("=" * 40)
    click.echo(f"Output directory: {settings.output_dir}")
    click.echo(f"Log level:        {settings.log_level}")
    click.echo()
    click.echo("API Keys:")
    click.echo(f"  NCBI/PubMed:    {'configured' if settings.ncbi_api_key else 'not set (free tier, 3 req/s)'}")
    click.echo(f"  NCBI email:     {settings.ncbi_email}")
    click.echo()
    click.echo("Data sources:")
    click.echo("  - pubchem   (PubChem compounds via PUG REST + SDQ)")
    click.echo("  - pubmed    (PubMed abstracts via Entrez)")
    click.echo("  - pmc       (PMC full-text via Entrez)")
    click.echo()
    click.echo("Curated compound sources (use with --source):")

    from chem2textqa.scrapers.pubchem import SOURCE_DATABASES
    for alias, pug_name in SOURCE_DATABASES.items():
        click.echo(f"  - {alias:<12} ({pug_name})")


# ------------------------------------------------------------------
# pipeline: full compound→article workflow
# ------------------------------------------------------------------


@main.command()
@click.option("--keyword", "-k", default=None, help="Compound keyword")
@click.option("--source", "-s", default=None,
              help="Curated source database (DrugBank, ChEMBL, HMDB, ChEBI, KEGG, BindingDB)")
@click.option("--max-compounds", "-n", default=100, show_default=True)
@click.option("--max-articles", "-a", default=1000, show_default=True)
@click.option("--full-text/--abstracts-only", default=False, show_default=True)
@click.option("--filter/--no-filter", "use_filter", default=True, show_default=True,
              help="Filter articles by MeSH QA categories (keeps only structure-relevant)")
def pipeline(keyword, source, max_compounds, max_articles, full_text, use_filter):
    """Run full pipeline: PubChem compounds → MeSH filter → PubMed abstracts → PMC full text."""
    if not keyword and not source:
        raise click.UsageError("Provide --keyword or --source")

    settings = get_settings()
    setup_logging(settings.log_level)

    from chem2textqa.scrapers.pubchem import PubChemScraper
    from chem2textqa.scrapers.pubmed import PubMedScraper
    from chem2textqa.scrapers.pmc import PMCScraper

    out_dir = settings.output_dir
    label = source or keyword

    # Step 1: PubChem compounds
    click.echo(f"\n=== Step 1: PubChem compounds from '{label}' ===")
    pubchem_scraper = PubChemScraper(settings)
    compounds = pubchem_scraper.scrape_compounds(
        keyword=keyword, source=source, max_compounds=max_compounds,
    )

    if not compounds:
        click.echo("No compounds found. Stopping.")
        return

    written = append_documents(out_dir / "pubchem.jsonl", compounds)
    click.echo(f"Saved {written} compounds")

    # Collect unique PMIDs
    all_pmids = pubchem_scraper.collect_all_pmids(compounds)
    click.echo(f"Found {len(all_pmids)} unique linked PMIDs")

    if not all_pmids:
        click.echo("No linked articles. Stopping.")
        return

    # Limit articles
    pmids_to_fetch = all_pmids[:max_articles]

    # Step 2: MeSH category filtering
    category_map: dict[int, list[str]] | None = None

    if use_filter:
        click.echo(f"\n=== Step 2: MeSH category filtering ({len(pmids_to_fetch)} PMIDs) ===")
        from chem2textqa.filters import MeSHCategoryFilter, QA_CATEGORIES

        mesh_filter = MeSHCategoryFilter(settings)
        category_map, counts = mesh_filter.categorize_with_counts(pmids_to_fetch)

        click.echo("  Category breakdown:")
        for cat in QA_CATEGORIES:
            n = counts.get(cat.name, 0)
            click.echo(f"    {cat.name:<25} {n:>8,} articles")

        # Only fetch articles that matched at least one category
        filtered_pmids = sorted(category_map.keys())
        click.echo(f"  Kept {len(filtered_pmids)} / {len(pmids_to_fetch)} "
                    f"articles ({len(pmids_to_fetch) - len(filtered_pmids)} filtered out)")
        pmids_to_fetch = filtered_pmids

    if not pmids_to_fetch:
        click.echo("No articles passed filtering. Stopping.")
        return

    # Step 3: PubMed abstracts
    step = 3 if use_filter else 2
    click.echo(f"\n=== Step {step}: PubMed abstracts ({len(pmids_to_fetch)} articles) ===")
    pubmed_scraper = PubMedScraper(settings)
    abstracts = pubmed_scraper.fetch_by_pmids(pmids_to_fetch, category_map=category_map)
    written = append_documents(out_dir / "pubmed.jsonl", abstracts)
    click.echo(f"Saved {written} abstracts")

    # Step 4: PMC full text (optional)
    if full_text:
        step += 1
        click.echo(f"\n=== Step {step}: PMC full text ===")
        pmc_scraper = PMCScraper(settings)
        full_texts = pmc_scraper.fetch_by_pmids(pmids_to_fetch)
        written = append_documents(out_dir / "pmc.jsonl", full_texts)
        click.echo(f"Saved {written} full-text articles")

    click.echo("\nPipeline complete.")


# ------------------------------------------------------------------
# crawl: iterative discovery loop
# ------------------------------------------------------------------


@main.command()
@click.option("--source", "-s", multiple=True,
              help="Curated source(s) for Phase 1 (e.g. -s DrugBank -s HMDB). Repeatable.")
@click.option("--max-compounds", "-n", default=500, show_default=True,
              help="Max compounds per source in Phase 1")
@click.option("--max-articles-per-round", "-a", default=5000, show_default=True,
              help="Max new articles to fetch per round")
@click.option("--max-expand-terms", "-t", default=50, show_default=True,
              help="Max new terms to query per expansion round")
@click.option("--max-rounds", "-r", default=3, show_default=True,
              help="Max expansion rounds (Phase 2 iterations)")
@click.option("--full-text/--abstracts-only", default=False, show_default=True)
@click.option("--filter/--no-filter", "use_filter", default=True, show_default=True)
@click.option("--max-disk-gb", default=None, type=float,
              help="Stop crawling if output directory exceeds this size in GB")
def crawl(source, max_compounds, max_articles_per_round, max_expand_terms,
          max_rounds, full_text, use_filter, max_disk_gb):
    """Iterative discovery: seed from sources, then expand via extracted terms.

    \b
    Phase 1: For each --source, fetch compounds and their linked articles.
    Phase 2: Use MeSH terms extracted from downloaded articles to search
             PubMed for more articles. Repeat until no new articles found
             or --max-rounds reached.

    State is saved to data/crawl_state.json so you can resume later.

    \b
    Example:
      chem2textqa crawl -s DrugBank -s HMDB -n 100 -a 1000 -r 3 --full-text
    """
    if not source:
        raise click.UsageError("Provide at least one --source (e.g. -s DrugBank)")

    settings = get_settings()
    setup_logging(settings.log_level)

    from chem2textqa.crawl_state import CrawlState
    from chem2textqa.scrapers.pubchem import PubChemScraper
    from chem2textqa.scrapers.pubmed import PubMedScraper
    from chem2textqa.scrapers.pmc import PMCScraper
    from chem2textqa.filters import MeSHCategoryFilter

    out_dir = settings.output_dir
    state_path = out_dir / "crawl_state.json"

    # Load or create state
    state = CrawlState.load(state_path)
    click.echo(f"Starting crawl. {state.summary()}")

    pubchem_scraper = PubChemScraper(settings)
    pubmed_scraper = PubMedScraper(settings)
    pmc_scraper = PMCScraper(settings) if full_text else None
    mesh_filter = MeSHCategoryFilter(settings) if use_filter else None

    # ================================================================
    # Phase 1: Seed from curated sources
    # ================================================================
    click.echo(f"\n{'='*60}")
    click.echo(f"PHASE 1: Seed compounds from {len(source)} source(s)")
    click.echo(f"{'='*60}")

    for src in source:
        click.echo(f"\n--- Source: {src} ---")
        compounds = pubchem_scraper.scrape_compounds(
            source=src, max_compounds=max_compounds,
        )
        if not compounds:
            click.echo(f"  No compounds from {src}")
            continue

        # Filter out already-seen CIDs
        new_compounds = [c for c in compounds if c.cid not in state.seen_cids]
        click.echo(f"  {len(compounds)} compounds, {len(new_compounds)} new")

        if not new_compounds:
            continue

        # Save compounds
        append_documents(out_dir / "pubchem.jsonl", new_compounds)
        state.mark_cids([c.cid for c in new_compounds])
        state.total_compounds += len(new_compounds)

        # Collect new PMIDs
        all_pmids = pubchem_scraper.collect_all_pmids(new_compounds)
        new_pmids = state.filter_new_pmids(all_pmids)
        click.echo(f"  {len(all_pmids)} linked PMIDs, {len(new_pmids)} new")

        if not new_pmids:
            continue

        # Limit per round
        batch = new_pmids[:max_articles_per_round]

        # Fetch and process articles
        _fetch_and_store(
            batch, state, pubmed_scraper, pmc_scraper, mesh_filter,
            out_dir, use_filter, full_text,
        )

        state.save(state_path)

        if _check_disk_limit(out_dir, max_disk_gb):
            state.save(state_path)
            return

    click.echo(f"\nPhase 1 complete. {state.summary()}")

    # ================================================================
    # Phase 2: Expansion rounds using discovered terms
    # ================================================================
    for round_num in range(1, max_rounds + 1):
        unused_terms = state.new_terms()
        if not unused_terms:
            click.echo("\nNo unused terms remaining. Stopping expansion.")
            break

        click.echo(f"\n{'='*60}")
        click.echo(f"PHASE 2, ROUND {round_num}/{max_rounds}: "
                    f"Expanding with {len(unused_terms)} unused terms")
        click.echo(f"{'='*60}")

        # Pick a batch of terms to query
        terms_batch = sorted(unused_terms)[:max_expand_terms]
        round_new_pmids: list[int] = []

        for term in terms_batch:
            state.queried_terms.add(term)

            # Build a MeSH query for this term
            query = f'"{term}"[MeSH Terms]'

            try:
                pmid_count = pubmed_scraper.count(query)
            except Exception as e:
                click.echo(f"  {term}: skipped ({e})")
                continue

            if pmid_count == 0:
                continue

            click.echo(f"  {term}: {pmid_count:,} results", nl=False)

            # Fetch just PMIDs (lightweight, no record download)
            fetch_limit = min(pmid_count, 10000)
            try:
                fetched_pmids = pubmed_scraper.search_pmids(query, max_results=fetch_limit)
            except Exception as e:
                click.echo(f" (error: {e})")
                continue

            new = state.filter_new_pmids(fetched_pmids)
            click.echo(f", {len(new)} new")

            if new:
                round_new_pmids.extend(new)

            # Don't exceed per-round limit
            if len(round_new_pmids) >= max_articles_per_round:
                round_new_pmids = round_new_pmids[:max_articles_per_round]
                break

        if not round_new_pmids:
            click.echo(f"\n  No new articles found in round {round_num}. Stopping.")
            break

        click.echo(f"\n  Round {round_num}: {len(round_new_pmids)} new articles to process")

        # Fetch and process
        _fetch_and_store(
            round_new_pmids, state, pubmed_scraper, pmc_scraper, mesh_filter,
            out_dir, use_filter, full_text,
        )

        state.rounds_completed += 1
        state.save(state_path)

        if _check_disk_limit(out_dir, max_disk_gb):
            break

    # ================================================================
    # Done
    # ================================================================
    click.echo(f"\n{'='*60}")
    click.echo("CRAWL COMPLETE")
    click.echo(f"{'='*60}")
    click.echo(f"  Compounds:   {state.total_compounds:,}")
    click.echo(f"  Abstracts:   {state.total_abstracts:,}")
    click.echo(f"  Full texts:  {state.total_full_texts:,}")
    click.echo(f"  Unique PMIDs:{len(state.seen_pmids):,}")
    click.echo(f"  Terms found: {len(state.discovered_terms):,} "
               f"({len(state.new_terms()):,} still unused)")
    click.echo(f"  Rounds:      {state.rounds_completed}")
    click.echo(f"  State saved: {state_path}")


def _check_disk_limit(out_dir: Path, max_disk_gb: float | None) -> bool:
    """Return True if the output directory exceeds the disk limit."""
    if max_disk_gb is None:
        return False
    total_bytes = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file())
    total_gb = total_bytes / (1024 ** 3)
    if total_gb >= max_disk_gb:
        click.echo(f"\n  Disk limit reached: {total_gb:.2f} GB >= {max_disk_gb} GB. Stopping.")
        return True
    return False


def _fetch_and_store(
    pmids, state, pubmed_scraper, pmc_scraper, mesh_filter,
    out_dir: Path, use_filter: bool, full_text: bool,
) -> None:
    """Shared logic: filter → fetch abstracts → extract terms → fetch full text."""
    category_map = None
    if use_filter and mesh_filter:
        category_map, counts = mesh_filter.categorize_with_counts(pmids)
        filtered = sorted(category_map.keys())
        click.echo(f"  Filtered: {len(filtered)}/{len(pmids)} articles have QA categories")
        pmids = filtered

    if not pmids:
        return

    # Mark PMIDs as seen (even before fetch, to avoid re-processing on retry)
    state.mark_pmids(pmids)

    # Fetch abstracts
    docs = pubmed_scraper.fetch_by_pmids(pmids, category_map=category_map)
    if docs:
        written = append_documents(out_dir / "pubmed.jsonl", docs)
        state.total_abstracts += written
        click.echo(f"  Saved {written} abstracts")

        # Extract new terms from downloaded articles
        new_terms_count = 0
        for doc in docs:
            extracted = state.extract_terms_from_document(doc.metadata)
            new = extracted - state.discovered_terms
            if new:
                state.discovered_terms.update(new)
                new_terms_count += len(new)

        if new_terms_count:
            click.echo(f"  Discovered {new_terms_count} new terms "
                        f"(total: {len(state.discovered_terms):,})")

    # Fetch full text
    if full_text and pmc_scraper:
        try:
            full_texts = pmc_scraper.fetch_by_pmids(pmids)
            if full_texts:
                written = append_documents(out_dir / "pmc.jsonl", full_texts)
                state.total_full_texts += written
                click.echo(f"  Saved {written} full-text articles")
        except Exception as e:
            click.echo(f"  PMC fetch failed (non-fatal): {e}")


# ------------------------------------------------------------------
# download-sources: cache curated source CID lists from PubChem
# ------------------------------------------------------------------


@main.command("download-sources")
@click.option("--bulk-dir", default="data/bulk", show_default=True,
              type=click.Path(), help="Where to cache source CID lists")
@click.option("--source", "-s", "sources", multiple=True,
              help="Source alias(es) — DrugBank, HMDB, KEGG, ChEBI, "
                   "BindingDB, ChEMBL. Repeatable. Default: all.")
@click.option("--force", is_flag=True, help="Re-download even if cached")
def download_sources_cmd(bulk_dir, sources, force):
    """Download compound CID lists from curated PubChem sources.

    Each source's CIDs are saved to <bulk_dir>/sources/<alias>.txt.
    Later passed to build-dataset via --source to expand the compound set
    beyond MeSH-classified compounds.
    """
    settings = get_settings()
    setup_logging(settings.log_level)

    from chem2textqa.processing.sources import (
        CURATED_SOURCES,
        download_all_sources,
    )

    src_list = list(sources) if sources else list(CURATED_SOURCES)
    for alias in src_list:
        if alias not in CURATED_SOURCES:
            raise click.BadParameter(
                f"Unknown source {alias!r}. Known: {list(CURATED_SOURCES)}",
                param_hint="--source",
            )

    click.echo(f"Downloading CID lists to {bulk_dir}/sources/")
    click.echo(f"Sources: {', '.join(src_list)}")
    click.echo()

    counts = download_all_sources(Path(bulk_dir), sources=src_list, force=force)

    click.echo("\n" + "=" * 50)
    click.echo("SOURCE CID COUNTS")
    click.echo("=" * 50)
    total_union = 0
    for alias, count in counts.items():
        click.echo(f"  {alias:<12} {count:>12,} CIDs")
        total_union += count
    click.echo(f"  {'(sum)':<12} {total_union:>12,} (before dedup)")


# ------------------------------------------------------------------
# build-dataset: local processing of bulk PubChem + PubMed files
# ------------------------------------------------------------------


@main.command("build-dataset")
@click.option("--bulk-dir", default="data/bulk", show_default=True,
              type=click.Path(exists=True, file_okay=False),
              help="Directory with CID-* files and pubmed_baseline/")
@click.option("--output", "-o", default="data/drug_articles.jsonl", show_default=True,
              type=click.Path(),
              help="Output JSONL path")
@click.option("--fulltext", default="data/bulk/pmc_fulltext.jsonl", show_default=True,
              type=click.Path(),
              help="Optional PMC full-text JSONL to merge in")
@click.option("--no-fulltext", is_flag=True,
              help="Skip full-text merging even if file exists")
@click.option("--require-abstract/--allow-empty-abstract", default=True,
              show_default=True)
@click.option("--require-category/--allow-uncategorized", default=True,
              show_default=True,
              help="Drop articles that don't match any QA category")
@click.option("--max-articles", default=None, type=int,
              help="Stop after N articles (for testing)")
@click.option("--source", "-s", "extra_sources", multiple=True,
              help="Curated source(s) to union with MeSH compounds "
                   "(DrugBank, HMDB, KEGG, ChEBI, BindingDB, ChEMBL). "
                   "Requires 'download-sources' first.")
def build_dataset(bulk_dir, output, fulltext, no_fulltext, require_abstract,
                  require_category, max_articles, extra_sources):
    """Build the final QA dataset from bulk PubChem + PubMed downloads.

    \b
    Pipeline:
      1. Load drug/metabolite compounds (CID-MeSH filter, ~89K compounds)
      2. Build PMID→[CIDs] reverse index from CID-PMID.gz
      3. Stream PubMed baseline XML files, extract matching articles
      4. Apply local QA category filter (mechanism, therapy, toxicity, ...)
      5. Tag each article with linked compound profiles
      6. Optionally merge in PMC full text
      7. Write to JSONL

    \b
    Example:
      chem2textqa build-dataset
      chem2textqa build-dataset --max-articles 1000  # quick test
      chem2textqa build-dataset --no-fulltext        # abstracts only
    """
    settings = get_settings()
    setup_logging(settings.log_level)

    from chem2textqa.processing.builder import build_dataset as run_build

    fulltext_path = None if no_fulltext else Path(fulltext)
    if fulltext_path and not fulltext_path.exists():
        click.echo(f"Note: full-text file {fulltext_path} not found, skipping merge")
        fulltext_path = None

    click.echo(f"Building dataset from {bulk_dir}")
    click.echo(f"Output: {output}")
    if fulltext_path:
        click.echo(f"Full text: {fulltext_path}")

    stats = run_build(
        bulk_dir=Path(bulk_dir),
        output_path=Path(output),
        require_abstract=require_abstract,
        require_qa_category=require_category,
        fulltext_path=fulltext_path,
        max_articles=max_articles,
        extra_sources=list(extra_sources) if extra_sources else None,
    )

    click.echo("\n" + "=" * 50)
    click.echo("BUILD COMPLETE")
    click.echo("=" * 50)
    click.echo(f"  Compounds loaded:           {stats.compounds_loaded:>12,}")
    click.echo(f"  Target PMIDs:               {stats.pmids_targeted:>12,}")
    click.echo(f"  XML files processed:        {stats.xml_files_processed:>12,}")
    click.echo(f"  Articles seen (matching):   {stats.articles_seen:>12,}")
    click.echo(f"  Articles dropped (no abstr):{stats.articles_dropped_no_abstract:>12,}")
    click.echo(f"  Articles dropped (no cat):  {stats.articles_dropped_no_category:>12,}")
    click.echo(f"  Articles written:           {stats.articles_written:>12,}")
    click.echo(f"  Full-text merged:           {stats.fulltext_merged:>12,}")
    click.echo(f"\nOutput: {output}")


# ------------------------------------------------------------------
# cleanup-dataset: filter the built dataset for quality
# ------------------------------------------------------------------


@main.command("cleanup-dataset")
@click.option("--input", "-i", "input_path", default="data/drug_articles.jsonl",
              show_default=True, type=click.Path(exists=True),
              help="Input JSONL from build-dataset")
@click.option("--output", "-o", "output_path",
              default="data/filtered/drug_articles_filtered.jsonl",
              show_default=True, type=click.Path(),
              help="Filtered output JSONL path")
@click.option("--stats", "stats_path", default="data/filtered/filter_stats.json",
              show_default=True, type=click.Path(),
              help="Filter statistics output path")
@click.option("--language", default="eng", show_default=True,
              help="Language filter (eng = English; pass '' to disable)")
@click.option("--min-abstract", default=500, show_default=True, type=int,
              help="Minimum abstract length in characters")
@click.option("--keep-retracted", is_flag=True,
              help="Keep retracted articles (default: drop)")
@click.option("--keep-non-primary", is_flag=True,
              help="Keep editorials/letters/news (default: drop)")
@click.option("--allow-generic-only", is_flag=True,
              help="Keep articles linked only to generic compounds")
@click.option("--skip-mention-check", is_flag=True,
              help="Don't require compound to be mentioned in title/abstract")
def cleanup_dataset_cmd(input_path, output_path, stats_path, language,
                        min_abstract, keep_retracted, keep_non_primary,
                        allow_generic_only, skip_mention_check):
    """Filter the built dataset to keep only high-quality, drug-relevant articles.

    \b
    Applies these filters:
      - Language (default: English only)
      - Minimum abstract length (default: 500 chars)
      - Not retracted / erratum
      - Not editorial / letter / news
      - Has at least one 'specific' (non-generic) linked compound
      - Compound actually mentioned in title/abstract or MeSH major topic

    \b
    Example:
      chem2textqa cleanup-dataset
      chem2textqa cleanup-dataset --min-abstract 300 --keep-retracted
    """
    settings = get_settings()
    setup_logging(settings.log_level)

    from chem2textqa.processing.cleanup import FilterConfig, cleanup_dataset

    config = FilterConfig(
        language=language or None,
        min_abstract_chars=min_abstract,
        exclude_retracted=not keep_retracted,
        exclude_non_primary=not keep_non_primary,
        require_specific_compound=not allow_generic_only,
        require_compound_mention=not skip_mention_check,
    )

    click.echo(f"Input:  {input_path}")
    click.echo(f"Output: {output_path}")
    click.echo(f"Stats:  {stats_path}")
    click.echo("Filter config:")
    click.echo(f"  language:                  {config.language}")
    click.echo(f"  min abstract chars:        {config.min_abstract_chars}")
    click.echo(f"  exclude retracted:         {config.exclude_retracted}")
    click.echo(f"  exclude non-primary:       {config.exclude_non_primary}")
    click.echo(f"  require specific compound: {config.require_specific_compound}")
    click.echo(f"  require compound mention:  {config.require_compound_mention}")
    click.echo()

    stats = cleanup_dataset(
        Path(input_path),
        Path(output_path),
        config=config,
        stats_path=Path(stats_path),
    )

    click.echo("\n" + "=" * 50)
    click.echo("FILTERING COMPLETE")
    click.echo("=" * 50)
    click.echo(stats.summary())
    click.echo(f"\nOutput:      {output_path}")
    click.echo(f"Stats file:  {stats_path}")


# ------------------------------------------------------------------
# patch-formula-mass: retrofit missing formula/mw fields
# ------------------------------------------------------------------


@main.command("patch-formula-mass")
@click.option("--input", "-i", "input_path", required=True,
              type=click.Path(exists=True),
              help="JSONL file with records to patch")
@click.option("--output", "-o", "output_path", required=True,
              type=click.Path(),
              help="Patched JSONL output")
@click.option("--cid-mass",
              default="data/bulk/CID-Mass.gz",
              show_default=True,
              type=click.Path(exists=True),
              help="CID-Mass.gz file for formula + mass lookup")
def patch_formula_mass_cmd(input_path, output_path, cid_mass):
    """Add molecular_formula + molecular_weight to linked_compounds.

    Useful when an earlier build didn't populate these fields correctly.
    Streams the JSONL once and fills in the missing fields. Much faster
    than rebuilding (~10 min vs ~7 hours).
    """
    settings = get_settings()
    setup_logging(settings.log_level)

    from chem2textqa.processing.fix_formula_mass import (
        load_cid_mass_map,
        patch_jsonl,
    )

    click.echo(f"Input:    {input_path}")
    click.echo(f"Output:   {output_path}")
    click.echo(f"CID-Mass: {cid_mass}")
    click.echo()

    cid_mass_map = load_cid_mass_map(Path(cid_mass))
    records, patched = patch_jsonl(
        Path(input_path), Path(output_path), cid_mass_map,
    )

    click.echo("\n" + "=" * 50)
    click.echo("PATCH COMPLETE")
    click.echo("=" * 50)
    click.echo(f"  Records written:        {records:>12,}")
    click.echo(f"  Compound entries fixed: {patched:>12,}")
    click.echo(f"\nOutput: {output_path}")


# ------------------------------------------------------------------
# merge-fulltext: fold pmc_fulltext.jsonl into an existing dataset
# ------------------------------------------------------------------


@main.command("merge-fulltext")
@click.option("--input", "-i", "input_path", required=True,
              type=click.Path(exists=True),
              help="Input JSONL with records that may be missing full_text")
@click.option("--output", "-o", "output_path", required=True,
              type=click.Path(),
              help="Output JSONL with full_text merged in")
@click.option("--pmc", "pmc_path",
              default="data/bulk/pmc_fulltext.jsonl",
              show_default=True,
              type=click.Path(exists=True),
              help="PMC full-text JSONL source")
def merge_fulltext_cmd(input_path, output_path, pmc_path):
    """Merge pmc_fulltext.jsonl into an existing filtered dataset.

    For records whose full_text is empty, looks up the PMID in
    pmc_fulltext.jsonl and fills in full_text / sections / pmcid / doi
    if available. Uses a byte-offset index to avoid loading the whole
    PMC file into memory.
    """
    settings = get_settings()
    setup_logging(settings.log_level)

    from chem2textqa.processing.merge_fulltext import merge_pmc_fulltext

    click.echo(f"Input:  {input_path}")
    click.echo(f"Output: {output_path}")
    click.echo(f"PMC:    {pmc_path}")
    click.echo()

    stats = merge_pmc_fulltext(
        Path(input_path), Path(output_path), Path(pmc_path),
    )

    click.echo("\n" + "=" * 50)
    click.echo("MERGE COMPLETE")
    click.echo("=" * 50)
    click.echo(f"  Total records:            {stats.total:>12,}")
    click.echo(f"  Already had full text:    {stats.already_had_fulltext:>12,}")
    click.echo(f"  Newly merged full text:   {stats.newly_merged:>12,}")
    click.echo(f"  No PMC match available:   {stats.no_pmc_match:>12,}")
    click.echo(f"\nOutput: {output_path}")


# ------------------------------------------------------------------
# make-subsets: generate quality tiers (premium/standard/broad)
# ------------------------------------------------------------------


@main.command("make-subsets")
@click.option("--input", "-i", "input_path", required=True,
              type=click.Path(exists=True),
              help="Input JSONL (broad tier)")
@click.option("--premium", "premium_path",
              default="data/filtered/drug_articles_v2_premium.jsonl",
              show_default=True, type=click.Path(),
              help="Output path for premium tier (major MeSH topic)")
@click.option("--standard", "standard_path",
              default="data/filtered/drug_articles_v2_standard.jsonl",
              show_default=True, type=click.Path(),
              help="Output path for standard tier (major topic OR title mention)")
def make_subsets_cmd(input_path, premium_path, standard_path):
    """Create relevance-tiered subsets from the merged dataset.

    \b
    Tiers:
      - premium:  compound is a MeSH major topic in the article (*)
      - standard: compound is a major topic OR appears in the title
      - broad:    the input file itself (pass-through, not duplicated)
    """
    settings = get_settings()
    setup_logging(settings.log_level)

    from chem2textqa.processing.subsets import make_subsets

    click.echo(f"Input:    {input_path}")
    click.echo(f"Premium:  {premium_path}")
    click.echo(f"Standard: {standard_path}")
    click.echo()

    stats = make_subsets(
        Path(input_path), Path(premium_path), Path(standard_path),
    )

    click.echo("\n" + "=" * 50)
    click.echo("SUBSETS COMPLETE")
    click.echo("=" * 50)

    def pct(n: int) -> str:
        return f"{100*n/stats.total:.1f}%" if stats.total else "0%"

    click.echo(f"  Total articles:  {stats.total:>12,}")
    click.echo(f"  Broad (input):   {stats.broad:>12,}  ({pct(stats.broad)})")
    click.echo(f"  Standard:        {stats.standard:>12,}  ({pct(stats.standard)})")
    click.echo(f"  Premium:         {stats.premium:>12,}  ({pct(stats.premium)})")


# ------------------------------------------------------------------
# qa-*: QA generation pipeline (Phase 0 → 3)
# ------------------------------------------------------------------


@main.command("qa-extract-evidence")
@click.option("--input", "-i", "input_path",
              default="data/filtered/drug_articles_v2_premium.jsonl",
              show_default=True, type=click.Path(exists=True),
              help="Tier JSONL to extract evidence from")
@click.option("--output", "-o", "output_path",
              default="data/qa_pipeline/phase_0_evidence/evidence_per_cid.jsonl",
              show_default=True, type=click.Path())
@click.option("--synonym-file",
              default="data/bulk/CID-Synonym-filtered.gz",
              show_default=True, type=click.Path(),
              help="PubChem CID-Synonym file (optional — missing = skip)")
@click.option("--target-cids", default=None, type=click.Path(exists=True),
              help="Optional file with CIDs (one per line) to restrict to — useful for pilots")
def qa_extract_evidence_cmd(input_path, output_path, synonym_file, target_cids):
    """Phase 0: build per-compound evidence bundles from a filtered tier."""
    settings = get_settings()
    setup_logging(settings.log_level)

    from chem2textqa.qa_pipeline.phase_0_evidence.extract import run_phase_0

    target_set = None
    if target_cids:
        target_set = set()
        with open(target_cids) as f:
            for line in f:
                line = line.strip()
                if line.isdigit():
                    target_set.add(int(line))
        click.echo(f"Restricting to {len(target_set)} CIDs from {target_cids}")

    click.echo(f"Input:  {input_path}")
    click.echo(f"Output: {output_path}")

    stats = run_phase_0(
        Path(input_path), Path(output_path),
        cid_synonym_file=Path(synonym_file),
        target_cids=target_set,
    )

    click.echo("\n" + "=" * 50)
    click.echo("PHASE 0 COMPLETE")
    click.echo("=" * 50)
    click.echo(f"  Compounds scanned:     {stats.compounds_scanned:>10,}")
    click.echo(f"  With evidence:         {stats.compounds_with_evidence:>10,}")
    click.echo(f"  Without evidence:      {stats.compounds_without_evidence:>10,}")
    click.echo(f"  Total sentences:       {stats.total_evidence_sentences:>10,}")


@main.command("qa-generate")
@click.option("--input", "-i", "input_path",
              default="data/qa_pipeline/phase_0_evidence/evidence_per_cid.jsonl",
              show_default=True, type=click.Path(exists=True))
@click.option("--output", "-o", "output_path",
              default="data/qa_pipeline/phase_1_qa/qa_pairs.jsonl",
              show_default=True, type=click.Path())
@click.option("--errors",
              default="data/qa_pipeline/phase_1_qa/errors.jsonl",
              show_default=True, type=click.Path())
@click.option("--model", default=None, help="Override default Phase 1 model")
@click.option("--workers", default=20, show_default=True, type=int)
@click.option("--api-key", default=None,
              help="OpenRouter API key (falls back to OPENROUTER_API_KEY env var)")
def qa_generate_cmd(input_path, output_path, errors, model, workers, api_key):
    """Phase 1: generate structured Q&A for each compound via LLM1."""
    import asyncio

    settings = get_settings()
    setup_logging(settings.log_level)

    from chem2textqa.qa_pipeline.config import DEFAULT_PHASE1_MODEL
    from chem2textqa.qa_pipeline.phase_1_qa.generate import run_phase_1

    model = model or DEFAULT_PHASE1_MODEL
    click.echo(f"Model:  {model}")
    click.echo(f"Input:  {input_path}")
    click.echo(f"Output: {output_path}")

    stats = asyncio.run(run_phase_1(
        Path(input_path), Path(output_path), Path(errors),
        model=model, workers=workers, api_key=api_key,
    ))

    click.echo("\n" + "=" * 50)
    click.echo("PHASE 1 COMPLETE")
    click.echo("=" * 50)
    click.echo(f"  Total compounds:       {stats.total:>10,}")
    click.echo(f"  Already done:          {stats.already_done:>10,}")
    click.echo(f"  Succeeded:             {stats.succeeded:>10,}")
    click.echo(f"  Failed:                {stats.failed:>10,}")
    click.echo(f"  Total QA pairs:        {stats.total_qa_pairs:>10,}")


@main.command("qa-independent")
@click.option("--input", "-i", "input_path",
              default="data/qa_pipeline/phase_1_qa/qa_pairs.jsonl",
              show_default=True, type=click.Path(exists=True))
@click.option("--output", "-o", "output_path",
              default="data/qa_pipeline/phase_2_independent/qa_independent.jsonl",
              show_default=True, type=click.Path())
@click.option("--errors",
              default="data/qa_pipeline/phase_2_independent/errors.jsonl",
              show_default=True, type=click.Path())
@click.option("--model", default=None)
@click.option("--workers", default=20, show_default=True, type=int)
@click.option("--api-key", default=None)
def qa_independent_cmd(input_path, output_path, errors, model, workers, api_key):
    """Phase 2: LLM2 answers each question independently (blind to LLM1)."""
    import asyncio

    settings = get_settings()
    setup_logging(settings.log_level)

    from chem2textqa.qa_pipeline.config import DEFAULT_PHASE2_MODEL
    from chem2textqa.qa_pipeline.phase_2_independent.independent import run_phase_2

    model = model or DEFAULT_PHASE2_MODEL
    click.echo(f"Model:  {model}")
    click.echo(f"Input:  {input_path}")
    click.echo(f"Output: {output_path}")

    stats = asyncio.run(run_phase_2(
        Path(input_path), Path(output_path), Path(errors),
        model=model, workers=workers, api_key=api_key,
    ))

    click.echo("\n" + "=" * 50)
    click.echo("PHASE 2 COMPLETE")
    click.echo("=" * 50)
    click.echo(f"  Total questions:       {stats.total_questions:>10,}")
    click.echo(f"  Already done:          {stats.already_done:>10,}")
    click.echo(f"  Succeeded:             {stats.succeeded:>10,}")
    click.echo(f"  Failed:                {stats.failed:>10,}")


@main.command("qa-judge")
@click.option("--input", "-i", "input_path",
              default="data/qa_pipeline/phase_2_independent/qa_independent.jsonl",
              show_default=True, type=click.Path(exists=True))
@click.option("--output", "-o", "output_path",
              default="data/qa_pipeline/phase_3_validate/validated.jsonl",
              show_default=True, type=click.Path())
@click.option("--model", default=None)
@click.option("--workers", default=20, show_default=True, type=int)
@click.option("--api-key", default=None)
@click.option("--no-heuristic", is_flag=True,
              help="Disable the token-overlap pre-filter; send every pair to the LLM.")
def qa_judge_cmd(input_path, output_path, model, workers, api_key, no_heuristic):
    """Phase 3: judge whether LLM1 and LLM2 answers agree."""
    import asyncio

    settings = get_settings()
    setup_logging(settings.log_level)

    from chem2textqa.qa_pipeline.config import DEFAULT_PHASE3_MODEL
    from chem2textqa.qa_pipeline.phase_3_validate.judge import run_phase_3

    model = model or DEFAULT_PHASE3_MODEL
    use_heuristic = not no_heuristic
    click.echo(f"Model:     {model}")
    click.echo(f"Input:     {input_path}")
    click.echo(f"Output:    {output_path}")
    click.echo(f"Heuristic: {'ON (cheap pre-filter)' if use_heuristic else 'OFF'}")

    stats = asyncio.run(run_phase_3(
        Path(input_path), Path(output_path),
        model=model, workers=workers, api_key=api_key,
        use_heuristic=use_heuristic,
    ))

    click.echo("\n" + "=" * 50)
    click.echo("PHASE 3 COMPLETE")
    click.echo("=" * 50)
    click.echo(f"  Total:                 {stats.total:>10,}")
    click.echo(f"  Already done:          {stats.already_done:>10,}")
    click.echo(f"  Agree:                 {stats.agree:>10,}")
    click.echo(f"  Disagree:              {stats.disagree:>10,}")
    click.echo(f"  Unclear:               {stats.unclear:>10,}")
    click.echo(f"  Failed:                {stats.failed:>10,}")
    processed = stats.heuristic_agree + stats.heuristic_unclear + stats.llm_calls
    if processed:
        auto = stats.heuristic_agree + stats.heuristic_unclear
        pct = 100 * auto / processed
        click.echo(f"  Auto-classified:       {auto:>10,}  ({pct:.1f}% of processed)")
        click.echo(f"    heuristic agree:     {stats.heuristic_agree:>10,}")
        click.echo(f"    heuristic unclear:   {stats.heuristic_unclear:>10,}")
        click.echo(f"  LLM calls made:        {stats.llm_calls:>10,}")


# ------------------------------------------------------------------
# qa-assemble: merge all 4 phases into a single dataset
# ------------------------------------------------------------------


@main.command("qa-assemble")
@click.option("--phase0", "phase0_path",
              default="data/qa_pipeline/phase_0_evidence/evidence_per_cid.jsonl",
              show_default=True, type=click.Path())
@click.option("--phase1", "phase1_path",
              default="data/qa_pipeline/phase_1_qa/qa_pairs.jsonl",
              show_default=True, type=click.Path())
@click.option("--phase2", "phase2_path",
              default="data/qa_pipeline/phase_2_independent/qa_independent.jsonl",
              show_default=True, type=click.Path())
@click.option("--phase3", "phase3_path",
              default="data/qa_pipeline/phase_3_validate/validated.jsonl",
              show_default=True, type=click.Path())
@click.option("--output-jsonl", default="data/qa_pipeline/dataset_final.jsonl",
              show_default=True, type=click.Path())
@click.option("--output-json", default="data/qa_pipeline/dataset_final.json",
              show_default=True, type=click.Path(),
              help="Pretty-printed JSON array (only written for <= 5000 compounds)")
@click.option("--summary", "summary_path",
              default="data/qa_pipeline/dataset_summary.json",
              show_default=True, type=click.Path())
@click.option("--agree-only", is_flag=True,
              help="Keep only QA pairs where the judge said 'agree' (gold subset)")
def qa_assemble_cmd(phase0_path, phase1_path, phase2_path, phase3_path,
                    output_jsonl, output_json, summary_path, agree_only):
    """Assemble all 4 phase outputs into one per-compound dataset."""
    settings = get_settings()
    setup_logging(settings.log_level)

    from chem2textqa.qa_pipeline.assemble import assemble_dataset

    stats = assemble_dataset(
        phase0_path=Path(phase0_path),
        phase1_path=Path(phase1_path),
        phase2_path=Path(phase2_path),
        phase3_path=Path(phase3_path),
        output_jsonl=Path(output_jsonl),
        output_json=Path(output_json),
        summary_path=Path(summary_path),
        agree_only=agree_only,
    )

    click.echo("\n" + "=" * 50)
    click.echo("ASSEMBLY COMPLETE")
    click.echo("=" * 50)
    click.echo(f"  Compounds:                 {stats.compounds:>8,}")
    click.echo(f"  Total QA pairs:            {stats.total_qa:>8,}")
    click.echo(f"  Agree:                     {stats.agree:>8,}")
    click.echo(f"  Disagree:                  {stats.disagree:>8,}")
    click.echo(f"  Unclear:                   {stats.unclear:>8,}")
    click.echo(f"  Missing verdict:           {stats.missing_verdict:>8,}")
    if stats.total_qa:
        click.echo(f"  Agree rate:                {100*stats.agree/stats.total_qa:>7.1f}%")
    click.echo()
    click.echo(f"  JSONL:   {output_jsonl}")
    if stats.compounds <= 5000:
        click.echo(f"  JSON:    {output_json}")
    click.echo(f"  Summary: {summary_path}")


# ------------------------------------------------------------------
# qa-compare: side-by-side comparison of two assembled datasets
# ------------------------------------------------------------------


@main.command("qa-compare")
@click.option("--baseline", "-a", required=True, type=click.Path(exists=True),
              help="First dataset_final.jsonl (e.g. Claude run)")
@click.option("--experiment", "-b", required=True, type=click.Path(exists=True),
              help="Second dataset_final.jsonl (e.g. DeepSeek run)")
@click.option("--label-a", default="baseline", show_default=True,
              help="Short name for the first run")
@click.option("--label-b", default="experiment", show_default=True,
              help="Short name for the second run")
@click.option("--output", "-o", "output_path", default=None,
              type=click.Path(),
              help="Optional JSON output path (prints to stdout if omitted)")
def qa_compare_cmd(baseline, experiment, label_a, label_b, output_path):
    """Compare two assembled QA datasets (agree-rate, per-topic, per-compound)."""
    settings = get_settings()
    setup_logging(settings.log_level)

    from chem2textqa.qa_pipeline.compare import compare, load_run

    a = load_run(Path(baseline), label_a)
    b = load_run(Path(experiment), label_b)
    report = compare(a, b)

    # Pretty print the headline comparison
    o = report["overall"]
    click.echo("=" * 60)
    click.echo(f"  {'Metric':<20} {label_a:>15} {label_b:>15}")
    click.echo("-" * 60)
    click.echo(f"  {'Compounds':<20} {o[label_a]['compounds']:>15,} {o[label_b]['compounds']:>15,}")
    click.echo(f"  {'Total QA pairs':<20} {o[label_a]['total_qa']:>15,} {o[label_b]['total_qa']:>15,}")
    click.echo(f"  {'Agree':<20} {o[label_a]['agree']:>15,} {o[label_b]['agree']:>15,}")
    click.echo(f"  {'Disagree':<20} {o[label_a]['disagree']:>15,} {o[label_b]['disagree']:>15,}")
    click.echo(f"  {'Unclear':<20} {o[label_a]['unclear']:>15,} {o[label_b]['unclear']:>15,}")
    click.echo(f"  {'Agree rate':<20} {100*o[label_a]['agree_rate']:>14.1f}% {100*o[label_b]['agree_rate']:>14.1f}%")
    click.echo(f"  Delta agree rate:  {100*o['delta_agree_rate']:+.1f}pp")

    click.echo()
    click.echo("Per-topic agree rate:")
    click.echo(f"  {'Topic':<25} {label_a + ' rate':>15} {label_b + ' rate':>15}")
    click.echo("  " + "-" * 58)
    for row in report["per_topic"]:
        ar = row[f"{label_a}_agree_rate"]
        br = row[f"{label_b}_agree_rate"]
        ar_str = f"{100*ar:6.1f}%" if ar is not None else "    —  "
        br_str = f"{100*br:6.1f}%" if br is not None else "    —  "
        click.echo(f"  {row['topic']:<25} {ar_str:>15} {br_str:>15}")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with Path(output_path).open("w") as f:
            json.dump(report, f, indent=2)
        click.echo(f"\nFull report: {output_path}")


if __name__ == "__main__":
    main()

