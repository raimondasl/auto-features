# RepoRadar

Monitor arXiv for papers relevant to your software repo and produce a ranked Markdown digest with actionable suggestions.

RepoRadar automatically profiles your repository (README, dependencies, docs), queries arXiv for matching papers, scores them by relevance, and generates a digest you can actually use.

## Features

- **Repo profiling** — extracts keywords via TF-IDF from README, docs, and dependency manifests (`requirements.txt`, `pyproject.toml`, `package.json`)
- **arXiv collection** — queries the arXiv API with auto-generated and user-defined seed queries
- **SQLite storage** — deduplicates papers across runs, tracks collection history
- **Heuristic ranking** — scores papers by keyword overlap, category match, and recency with configurable weights
- **Markdown digest** — three-tier output (Top Picks / Maybe Relevant / Muted) with score breakdowns and arXiv links
- **HTML output** — optional `--format html` renders a real digest page (paper cards, score breakdowns, badges), not raw Markdown
- **GitHub Action + Pages** — a first-party action publishes a dated, ranked digest to GitHub Pages on a schedule; `rr archive` builds the browsable site
- **Action suggestions** — template-based ideas grounded in paper abstracts (benchmarks, baselines, datasets, modules)
- **No API keys required** — uses only free, public APIs

## Installation

Requires Python 3.11+. Dependencies: `click`, `pyyaml`, `scikit-learn`, `jinja2`, `arxiv`.

```bash
# Clone and install with uv
git clone <repo-url>
cd auto-features
uv pip install -e .

# Or with dev dependencies (pytest, pytest-cov)
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

Creates `.reporadar.yml` config and `.reporadar/` storage directory. Safe to run multiple times — skips files that already exist.

### `rr profile [--config PATH]`

Prints the inferred topic profile: TF-IDF keywords with weights, detected packages (anchors), and inferred domains.

### `rr update [--config PATH] [--foundational] [-v]`

Runs the full pipeline: profile repo, build queries, fetch papers from arXiv, store in SQLite, score, and display top 5 results. Use `-v` for verbose logging.

`--foundational` runs a one-time **seed-corpus sweep**: all-time, relevance-first (no recency window, recency weight dropped), so seminal foundational papers surface instead of only recent ones. Use it to seed the corpus; the default is the recent digest.

### `rr digest [--config PATH] [--since 7d] [--run-id N] [-o PATH] [--format md|html]`

Generates a digest from the latest (or specified) run. Options:

- `--since 7d` — time window (e.g. `7d`, `14d`)
- `--run-id N` — use scores from a specific run instead of the latest
- `-o PATH` — custom output file path
- `--format html` — a fully rendered HTML page (paper cards, score breakdowns, badges — not raw Markdown; auto-converts the `.md` extension to `.html`)

### `rr open [--config PATH] [-n N | --top N]`

Opens the top N papers from the latest run in your default browser. Defaults to 5.

### `rr archive [--config PATH] [--archive-dir DIR] [--date YYYY-MM-DD] [--since 7d]`

Publishes the latest run's digest as a rendered HTML page into a dated archive
(`<archive-dir>/<date>.html`, default dir `digests/`) and regenerates `index.html`
listing every edition newest-first. Re-running on the same date replaces that
edition. This is the content GitHub Pages serves for the [GitHub Action](#github-action-scheduled-digests--github-pages).

### `rr mcp [--config PATH]`

Runs RepoRadar as an **MCP server** (stdio) so coding agents — Claude Code, Cursor, VS Code, Windsurf — can query your repo-aware paper store conversationally. Unlike generic arXiv MCP servers, its tools are grounded in *this repository's* profile and ranking. Tools exposed:

- `get_repo_profile` — the repo's keywords / libraries / domains
- `get_ranked_papers(limit)` — top papers from the latest `rr update`, best-first
- `explain_relevance(arxiv_id)` — score-component breakdown + LLM actionability reason
- `rate_paper(arxiv_id, rating)` — record a 1–5 rating (feeds the feedback loop)

Requires the optional extra: `uv pip install -e ".[mcp]"`. Register it with your agent, e.g. Claude Code:

```bash
claude mcp add reporadar -- rr mcp --config /abs/path/.reporadar.yml
```

## GitHub Action (scheduled digests + GitHub Pages)

Run RepoRadar as scheduled team infrastructure with zero hosting: the
[`reporadar` action](action.yml) profiles the repo, fetches + ranks papers, and
publishes a dated, browsable **research radar** to GitHub Pages — the repo that
*is* the profile target runs its own radar.

**Setup**

1. Copy [`examples/reporadar.yml`](examples/reporadar.yml) to `.github/workflows/reporadar.yml`.
2. Enable Pages: **Settings → Pages → Source: GitHub Actions**.
3. (Optional) commit a `.reporadar.yml` to tune categories/queries; if it's absent the action creates a default with `rr init`.

Minimal workflow:

```yaml
permissions:
  contents: read
  pages: write
  id-token: write
jobs:
  radar:
    runs-on: ubuntu-latest
    environment: github-pages
    steps:
      - uses: actions/checkout@v4
      - uses: raimondasl/auto-features@main   # pin to a tag/SHA once released
        id: radar
        with:
          archive-dir: digests
      - uses: actions/upload-pages-artifact@v3
        with: { path: '${{ steps.radar.outputs.archive-dir }}' }
      - uses: actions/deploy-pages@v4
```

**Inputs** (all optional): `config` (default `.reporadar.yml`), `archive-dir`
(`digests`), `formats` (extra loose digest files — `md,json,…`; default none, HTML
archive only), `notify` (`slack,discord,email`), `github-issues` (`true` opens an
Issue per top pick — also needs `issues: write` + `GH_TOKEN`), `cache-db` (`true`),
`python-version` (`3.12`). See [`action.yml`](action.yml) for the full list and outputs.

**Secrets** never live in the committed config. Reference them by name —
`slack_webhook_url: ${SLACK_WEBHOOK}` in `.reporadar.yml` — and inject the value
in the workflow (`env: { SLACK_WEBHOOK: ${{ secrets.SLACK_WEBHOOK }} }`). Any
`${VAR}` in `.reporadar.yml` is expanded from the environment at load time (unset → empty).

**Run history** (the paper DB and ratings) is cached between runs via `actions/cache`.
Caches unused for 7 days are evicted; for durable history, commit the `.reporadar/`
directory to a branch instead of relying on the cache. If you set a non-default
`repo_path` in `.reporadar.yml`, point the action's `cache-path` input at
`<repo_path>/.reporadar` so the cache tracks where the DB actually lives.

## Configuration

`.reporadar.yml` in your repo root:

```yaml
repo_path: .                          # Path to the repo to profile (default: current dir)

arxiv:
  categories: [cs.LG, cs.CL]        # arXiv categories to search
  max_results_per_query: 50          # Max papers per query
  lookback_days: 14                  # Only fetch papers from this window

queries:
  seed:                               # Your own search terms (exact-match quoted)
    - "retrieval augmented generation"
    - "long context transformers"
  exclude:                            # Terms to penalize in ranking (0.5x per match)
    - "survey"
    - "benchmark"

ranking:
  w_keyword: 1.0                      # Weight for keyword overlap score
  w_category: 0.5                     # Weight for category match score
  w_recency: 0.3                      # Weight for recency score

output:
  digest_path: ./reporadar_digest.md  # Default output path
  top_n: 15                           # Max papers in digest
```

## How It Works

### Profiling

The profiler scans your repo for text to build a topic profile:

1. **README** (supports `.md`, `.rst`, `.txt` variants) and files in `docs/`
2. **Dependency manifests** — `requirements.txt`, `pyproject.toml`, `package.json`
3. **TF-IDF** — extracts up to 20 keywords (unigrams + bigrams) from the collected text
4. **Anchors** — package names from manifests, mapped to domain labels (e.g., `torch` → "deep learning")

### Query Building

Queries are built from two sources:

1. **Seed queries** from config — wrapped in exact-match quotes (e.g., `all:"retrieval augmented generation"`)
2. **Auto-generated** — top 5 profile keywords as individual queries (e.g., `all:transformers`)

All queries are scoped to your configured arXiv categories (e.g., `cat:cs.LG OR cat:cs.CL`).

### Scoring

Each paper gets a combined score from three components:

```
score = (w_keyword * keyword_score + w_category * category_score + w_recency * recency_score) * exclude_penalty
```

- **Keyword score** (0–1) — fraction of profile keywords found in paper title + abstract, weighted by TF-IDF weight
- **Category score** (0–1) — fraction of target categories that appear in the paper's categories
- **Recency score** (0–1) — linear decay from 1.0 (today) to 0.0 at the lookback boundary
- **Exclude penalty** — each matched exclude term multiplies the score by 0.5 (e.g., two matches → 0.25x)

### Digest Tiers

Papers are categorized into three tiers based on their combined score:

- **Top Picks** (score >= 0.5) — full details with score breakdown, abstract snippet, and action suggestions
- **Maybe Relevant** (score >= 0.2) — condensed details
- **Muted** (score < 0.2) — title and link only

### Action Suggestions

Top-scoring papers get up to 3 template-based suggestions, derived from pattern matching against the abstract:

| Pattern detected | Example suggestion |
|---|---|
| Benchmark/evaluation mentioned | "Add evaluation on {benchmark}" |
| Outperforms a baseline | "Compare your approach against {baseline}" |
| Proposes a new method | "Explore integrating the proposed {method}" |
| Dataset/corpus referenced | "Consider using the {dataset} dataset" |
| SOTA claim | "Claims SOTA on {task} — worth checking" |
| Open-source code available | "Code/data may be publicly available" |
| Modular/plug-in component | "Consider adding as a feature flag" |
| New loss/optimizer | "Try swapping your optimizer/loss for {name}" |

Suggestions are clearly labeled as auto-generated starting points.

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
  test_cli.py         # CLI integration tests
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
