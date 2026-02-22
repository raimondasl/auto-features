# RepoRadar

Monitor arXiv for papers relevant to your software repo and produce a ranked Markdown digest with actionable suggestions.

RepoRadar automatically profiles your repository (README, dependencies, docs), queries arXiv for matching papers, scores them by relevance, and generates a digest you can actually use.

## Features

- **Repo profiling** — extracts keywords via TF-IDF from README, docs, and dependency manifests (`requirements.txt`, `pyproject.toml`, `package.json`)
- **arXiv collection** — queries the arXiv API with auto-generated and user-defined seed queries
- **SQLite storage** — deduplicates papers across runs, tracks collection history
- **Heuristic ranking** — scores papers by keyword overlap, category match, and recency with configurable weights
- **Markdown digest** — three-tier output (Top Picks / Maybe Relevant / Muted) with score breakdowns and arXiv links
- **HTML output** — optional `--format html` for browser-friendly digests
- **Action suggestions** — template-based ideas grounded in paper abstracts (benchmarks, baselines, datasets, modules)
- **No API keys required** — uses only free, public APIs

## Installation

Requires Python 3.11+.

```bash
# Clone and install with uv
git clone <repo-url>
cd auto-features
uv pip install -e .

# Or with dev dependencies
uv pip install -e ".[dev]"
```

## Quick Start

```bash
# 1. Initialize RepoRadar in your repo
cd /path/to/your/repo
rr init

# 2. (Optional) Edit .reporadar.yml to add seed queries and categories

# 3. See what RepoRadar infers about your repo
rr profile

# 4. Fetch and score papers from arXiv
rr update

# 5. Generate a digest
rr digest

# 6. Open top papers in your browser
rr open --top 5
```

## CLI Commands

### `rr init [--path DIR]`

Creates `.reporadar.yml` config and `.reporadar/` storage directory.

### `rr profile [--config PATH]`

Prints the inferred topic profile: TF-IDF keywords, detected packages (anchors), and inferred domains.

### `rr update [--config PATH] [-v]`

Runs the full pipeline: profile repo, build queries, fetch papers from arXiv, store in SQLite, and score. Use `-v` for verbose logging.

### `rr digest [--config PATH] [--since 7d] [--run-id N] [-o PATH] [--format md|html]`

Generates a digest from the latest (or specified) run. Options:

- `--since 7d` — time window (currently informational; filtering by run)
- `--run-id N` — use scores from a specific run instead of the latest
- `-o PATH` — custom output file path
- `--format html` — output as HTML instead of Markdown

### `rr open [--config PATH] [-n 5]`

Opens the top N papers from the latest run in your default browser.

## Configuration

`.reporadar.yml` in your repo root:

```yaml
repo_path: .

arxiv:
  categories: [cs.LG, cs.CL]        # arXiv categories to search
  max_results_per_query: 50
  lookback_days: 14

queries:
  seed:                               # Your own search terms
    - "retrieval augmented generation"
    - "long context transformers"
  exclude:                            # Terms to penalize in ranking
    - "survey"
    - "benchmark"

ranking:
  w_keyword: 1.0                      # Weight for keyword overlap
  w_category: 0.5                     # Weight for category match
  w_recency: 0.3                      # Weight for recency

output:
  digest_path: ./reporadar_digest.md
  top_n: 15                           # Max papers in digest
```

## Digest Output

The digest groups papers into three tiers:

- **Top Picks** (score >= 0.5) — full details with score breakdown, abstract snippet, and action suggestions
- **Maybe Relevant** (score >= 0.2) — condensed details
- **Muted** (score < 0.2) — title and link only

Each top pick includes **Action ideas** — template-based suggestions like "Add evaluation on X", "Compare against Y baseline", or "Code may be publicly available". These are auto-generated from abstract patterns and clearly labeled as starting points.

## Development

```bash
# Install with dev dependencies
uv pip install -e ".[dev]"

# Run tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=reporadar --cov-report=term-missing
```

## Project Structure

```
src/reporadar/
  cli.py              # Click CLI entry points
  config.py           # YAML config loading/validation
  profiler.py         # Repo topic profiling (TF-IDF)
  collector.py        # arXiv API querying
  store.py            # SQLite storage + dedup
  ranker.py           # Heuristic paper scoring
  digest.py           # Markdown/HTML digest generation
  suggestions.py      # Template-based action suggestions
  templates/
    digest.md.j2      # Jinja2 Markdown template
    digest.html.j2    # Jinja2 HTML wrapper template
tests/
  test_config.py
  test_profiler.py
  test_collector.py
  test_store.py
  test_ranker.py
  test_digest.py
  test_suggestions.py
  fixtures/           # Sample READMEs, manifests for tests
```

## License

MIT
