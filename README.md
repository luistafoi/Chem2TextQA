# Chem2TextQA

AI-augmented dataset pipeline for scraping scientific papers and patents related to chemical structures, drug mechanisms of action, metabolites, and drug development. Designed to produce training data for AI models.

## Data Sources

- **PubMed** -- papers and abstracts via NCBI Entrez API
- **USPTO** -- US patents via PatentsView API
- **EPO** -- European patents via Open Patent Services
- **Google Patents** -- broad patent search via SerpAPI or HTTP scraping

## Quick Start

```bash
# Create the conda environment
conda env create -f environment.yml
conda activate chem2textqa

# Copy and configure environment variables
cp .env.example .env
# Edit .env with your API keys (PubMed works without one)

# Verify installation
chem2textqa --help
```

## Usage

```bash
# Scrape PubMed for papers about drug mechanisms
chem2textqa scrape -s pubmed -q "drug mechanism of action" -n 100

# Scrape all sources at once
chem2textqa scrape -s all -q "chemical compounds" -n 50

# Scrape specific sources
chem2textqa scrape -s pubmed,uspto -q "metabolites drug development"

# Check configured API keys
chem2textqa info

# Count documents in a JSONL file
chem2textqa count data/pubmed.jsonl
```

## Project Structure

```
chem2textqa/
├── config/       # Settings from .env via pydantic-settings
├── models/       # ScientificDocument unified Pydantic schema
├── scrapers/     # One module per data source
├── storage/      # JSONL read/write utilities
└── utils/        # Rate limiting, retry, logging
```

## Output Format

All scrapers produce JSON Lines (`.jsonl`) files with a unified schema. Each line is a `ScientificDocument` with fields: source, title, abstract, authors, publication_date, identifiers, chemical_entities, keywords, and metadata.

## Development

```bash
# Run tests
pytest tests/ -v

# Lint
ruff check chem2textqa/

# Type check
mypy chem2textqa/
```
