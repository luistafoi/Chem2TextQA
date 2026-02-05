from pathlib import Path

import click

from chem2textqa.config import get_settings
from chem2textqa.scrapers import SCRAPER_REGISTRY
from chem2textqa.storage.jsonl import append_documents, count_documents
from chem2textqa.utils.logging import setup_logging


@click.group()
def main():
    """Chem2TextQA -- scrape scientific papers and patents for drug development AI."""
    pass


@main.command()
@click.option("--source", "-s", required=True, help="Scraper name, comma-separated list, or 'all'")
@click.option("--query", "-q", required=True, help="Search query")
@click.option("--max-results", "-n", default=100, show_default=True, help="Max results per source")
@click.option("--date-from", default=None, help="Start date (YYYY-MM-DD)")
@click.option("--date-to", default=None, help="End date (YYYY-MM-DD)")
@click.option("--output", "-o", default=None, type=click.Path(), help="Output JSONL path (default: data/<source>.jsonl)")
def scrape(source, query, max_results, date_from, date_to, output):
    """Run one or more scrapers."""
    settings = get_settings()
    logger = setup_logging(settings.log_level)

    # Resolve source list
    if source == "all":
        source_names = list(SCRAPER_REGISTRY.keys())
    else:
        source_names = [s.strip() for s in source.split(",")]

    # Validate source names
    for name in source_names:
        if name not in SCRAPER_REGISTRY:
            available = ", ".join(SCRAPER_REGISTRY.keys())
            raise click.BadParameter(
                f"Unknown source '{name}'. Available: {available}",
                param_hint="--source",
            )

    total_docs = 0
    for name in source_names:
        scraper_cls = SCRAPER_REGISTRY[name]
        scraper = scraper_cls(settings)

        click.echo(f"\n--- Scraping {name} ---")
        documents = scraper.search(
            query=query,
            max_results=max_results,
            date_from=date_from,
            date_to=date_to,
        )

        if not documents:
            click.echo(f"  No results from {name}")
            continue

        # Determine output path
        if output:
            out_path = Path(output)
        else:
            out_path = scraper.default_output_path()

        written = append_documents(out_path, documents)
        total_docs += written
        click.echo(f"  Wrote {written} documents to {out_path}")

    click.echo(f"\nDone. Total: {total_docs} documents scraped.")


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
    click.echo(f"  USPTO:          {'configured' if settings.uspto_api_key else 'not set'}")
    click.echo(f"  EPO:            {'configured' if settings.epo_key and settings.epo_secret else 'not set'}")
    click.echo(f"  SerpAPI:        {'configured' if settings.serpapi_key else 'not set (HTTP fallback)'}")
    click.echo()
    click.echo("Available scrapers:")
    for name in SCRAPER_REGISTRY:
        click.echo(f"  - {name}")


@main.command()
@click.argument("file", type=click.Path(exists=True))
def count(file):
    """Count documents in a JSONL file."""
    n = count_documents(Path(file))
    click.echo(f"{n} documents in {file}")


if __name__ == "__main__":
    main()
