# RepoRadar

Monitor arXiv for papers relevant to your software repo and produce a ranked Markdown digest with actionable suggestions.

RepoRadar automatically profiles your repository (README, dependencies, docs), queries arXiv for matching papers, scores them by relevance, and generates a digest you can actually use.

## Features

- **Repo profiling** — extracts keywords via TF-IDF from README, docs, and dependency manifests (`requirements.txt`, `pyproject.toml`, `package.json`)
- **arXiv collection** — queries the arXiv API with auto-generated and user-defined seed queries
- **Multi-source** — opt into Semantic Scholar, OpenAlex, **bioRxiv** (biology), and **DBLP** (systems/PL/DB) alongside arXiv via `sources:` — serves repos whose literature isn't on arXiv
- **SQLite storage** — deduplicates papers across runs, tracks collection history
- **Heuristic ranking** — scores papers by keyword overlap, category match, and recency with configurable weights
- **Markdown digest** — three-tier output (Top Picks / Maybe Relevant / Muted) with score breakdowns and arXiv links
- **HTML output** — optional `--format html` renders a real digest page (paper cards, score breakdowns, badges), not raw Markdown
- **GitHub Action + Pages** — a first-party action publishes a dated, ranked digest to GitHub Pages on a schedule; `rr archive` builds the browsable site
- **Local corpus search** — `rr search "<query>"` over every paper ever fetched: offline BM25, or `--semantic`/`--hybrid` embedding search backed by a cached, optionally sqlite-vec-accelerated index (also an MCP tool)
- **Action suggestions** — template-based ideas grounded in paper abstracts (benchmarks, baselines, datasets, modules)
- **Extends work you starred** — flags and boosts new papers that cite a paper you starred or rated highly (`ranking.w_citation_proximity`)
- **Learned recommendations** — your stars/ratings seed the free Semantic Scholar recommender; results are re-ranked locally before they reach the digest (`recommendations.enabled`)
- **SPECTER2 similarity** — citation-trained scientific embeddings (served free by Semantic Scholar, no local model) score how close each paper is to the work you starred/rated (`ranking.w_specter`)
- **Community attention** — optionally rank on Hugging Face Papers upvotes, log-scaled against the run's own maximum (`ranking.w_community`)
- **Withdrawal detection** — flags papers their own authors withdrew and demotes them out of Top Picks, so you never act on retracted work (on by default)
- **Hacker News attention** — badges papers that were discussed, with points and a link to the thread (`signals.hackernews`)
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

It also flags paper sources that match the repo but aren't in `sources:` — a repo built on `scanpy`/`anndata` gets pointed at bioRxiv, one built on `duckdb`/`rocksdb` at DBLP (`rr update` prints the same hint). Suggestions only: nothing is auto-enabled, since each source has costs worth opting into knowingly, and the hint says what they are.

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

### `rr search QUERY [--config PATH] [-n LIMIT] [--since 7d] [--semantic] [--hybrid] [--format text|json]`

Free-text search across **every paper RepoRadar has ever fetched** — the store
accumulates into a personal corpus, and this queries all of it offline. The
default is Okapi BM25 (no network, no embeddings). `--semantic` ranks by embedding
similarity and `--hybrid` fuses both via Reciprocal Rank Fusion; both need the
`embeddings` extra and encode + **cache** each paper's vector once (reused across
runs and by `rr update`). Install the optional `vectors` extra to back semantic
KNN with a sqlite-vec index on large corpora — it works without it (numpy fallback).

```bash
rr search "low-rank adaptation quantization"
rr search "efficient attention" --semantic
rr search "retrieval augmented generation" --hybrid -n 5 --format json
```

### `rr notify --channel shell|slack|discord|email [--config PATH] [--run-id N]`

Pushes a one-line summary of a run to a configured channel (`hooks:` in
`.reporadar.yml`: `on_digest` for a shell command, `slack_webhook_url`,
`discord_webhook_url`, `email`):

```
RepoRadar digest #12: 24 new papers, 5 top picks (40 scored) — 2 papers extend work you starred
```

The trailing clause appears only when papers in the run cite something you starred
or rated 4–5, so a citation alert reaches you instead of waiting to be noticed in
the digest. Shell hooks additionally get the whole summary as environment
variables: `RR_DIGEST_PATH`, `RR_RUN_ID`, `RR_PAPERS_NEW`, `RR_PAPERS_SEEN`,
`RR_TOP_PICKS_COUNT`, `RR_TOTAL_SCORED`, `RR_EXTENDS_STARRED_COUNT`, `RR_FORMAT`.

### `rr mcp [--config PATH]`

Runs RepoRadar as an **MCP server** (stdio) so coding agents — Claude Code, Cursor, VS Code, Windsurf — can query your repo-aware paper store conversationally. Unlike generic arXiv MCP servers, its tools are grounded in *this repository's* profile and ranking. Tools exposed:

- `get_repo_profile` — the repo's keywords / libraries / domains
- `get_ranked_papers(limit)` — top papers from the latest `rr update`, best-first
- `explain_relevance(arxiv_id)` — score-component breakdown + LLM actionability reason
- `rate_paper(arxiv_id, rating)` — record a 1–5 rating (feeds the feedback loop)
- `search_papers(query, limit)` — free-text BM25 search over the whole stored corpus

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
      - uses: raimondasl/auto-features@v1   # moving major tag (or pin @v1.0.0 / a SHA)
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
  w_citation_proximity: 0.0          # >0: fetch references + boost papers that cite work you starred/rated
  w_specter: 0.0                     # >0: SPECTER2 similarity to the papers you starred/rated highly
  w_community: 0.0                   # >0: rank on Hugging Face Papers upvotes (from cached enrichments)
  w_attention: 0.0                   # >0: rank on Hacker News points (needs signals.hackernews)
  withdrawn_penalty: 0.1             # multiplier for a withdrawn paper's score (1.0 disables)

enrichment:
  provider: huggingface               # or `off` to skip the HF Papers lookup entirely

signals:
  integrity: true                     # check whether a paper was withdrawn by its authors
  hackernews: false                   # look up Hacker News discussion (badge; see below)

recommendations:
  enabled: false                      # true: seed the free S2 recommender with your stars/ratings
  limit: 20                           # how many recommendations to request per run
  max_seeds: 50                       # cap on example papers sent

output:
  digest_path: ./reporadar_digest.md  # Default output path
  top_n: 15                           # Max papers in digest
```

`w_specter` also needs at least one **starred or 4–5-star** paper (it scores how close
each candidate is to that work, in citation-trained space). Vectors are fetched once
from Semantic Scholar and cached (~3 KB per paper); `rr update --rebuild-embeddings`
clears every cached vector. The component is skipped when a run's papers are too few
or too similar to rank meaningfully.

`w_community` scores each paper on its Hugging Face Papers upvote count, log-scaled
against the highest count in the same run (upvotes are heavy-tailed, so a linear
scale would flatten everything below the leader). It reuses the upvotes that
enrichment already stores — and because enrichment runs *after* ranking (it only
fetches for the papers that made the digest), the signal comes from **previous**
runs. A paper seen for the first time therefore has no community score yet and is
scored on the other components alone rather than penalized for it. Papers with zero
upvotes are likewise treated as "no signal", since a zero usually just means HF has
no page for that paper. Setting `enrichment.provider: off` therefore also stops the
community signal from ever refreshing (quoting it is not required — a bare `off` is
handled, even though YAML would otherwise read it as the boolean `false`).

### Withdrawal detection and Hacker News attention

`signals.integrity` (**on by default**) asks arXiv whether any of the run's papers
have been withdrawn by their authors. It is the one signal that defaults on, because
a withdrawn paper is the case where ranking can actively waste your time: it reads as
fresh and on-topic, and you can spend an hour on a result its own authors retracted.

A flagged paper's score is multiplied by `withdrawn_penalty` (0.1). That is a
multiplier, not another weighted component, so a withdrawn paper **cannot** reach Top
Picks by scoring well everywhere else — 1.0 × 0.1 = 0.1, below the Maybe Relevant
threshold. It is not dropped to zero, because a paper you may have seen elsewhere is
better shown flagged than silently missing; the digest gets a "Withdrawn by their
authors" section that appears regardless of tier.

arXiv has no withdrawal *field* — the notice is hand-written free text in the paper's
comment, so the matcher is the whole feature. It reads the comment liberally (the most
common real comment is the single word "Withdrawn") and prose conservatively, since an
abstract may legitimately discuss a drug withdrawn from the market, and it covers
"retracted" and "the authors withdrew this" as well as "withdrawn". Measured at
100% recall on notices phrased "withdrawn" and 83–85% on the other two verbs, with
no confirmed false positive over 600 ordinary papers. Papers are
re-checked at most weekly, capped per run, oldest first — withdrawal can happen days
after a paper is ingested, but chasing it every run would cost minutes of throttled
requests for a signal that fires for well under 1% of papers.

`signals.hackernews` (off by default) badges papers that were discussed on HN, with
points and a link. Be aware of what it is: across **340 random papers from the last
two weeks, zero had any HN story** — including 0/40 in cs.LG. HN surfaces a handful of
papers a week across all of science, so treat `w_attention` as a bonus that
occasionally fires, not a component you can rank on. Points are scaled against a fixed
reference rather than the run's maximum (the usual RepoRadar pattern), because with 0–1
discussed papers per run a relative scale would award 1.0 to a 12-point story.

Recommendations need at least one **starred or 4–5-star** paper (low-rated papers
become negative examples that suppress similar results; an explicit low rating
beats an implicit star from `rr open`). They're merged into the normal candidate
pool and **re-scored by RepoRadar**, and only those clearing the relevance bar are
shown — the API is repo-agnostic, so off-topic suggestions are dropped rather than
displayed, falling back to the local keyword recommender.

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
