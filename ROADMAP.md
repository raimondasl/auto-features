# RepoRadar Roadmap

Post-MVP feature plan. The MVP (Phases 1–5) is complete: `rr init → profile → update → digest → open` works end-to-end with heuristic scoring, SQLite storage, and Markdown/HTML output.

---

## Milestone 1 — Polish & Reliability

**Goal:** Make the tool pleasant for daily use and robust on real-world repos.

### 1.1 — CLI Quality of Life
- Add `--verbose` flag to `digest`, `profile`, and `open` commands (currently only on `update`)
- Colored terminal output via `click.style` — green for success, yellow for warnings, dim for muted papers
- Progress bar during arXiv fetching (`click.progressbar`) — shows query N/M and paper count
- `rr status` command — show last run time, paper count, DB size, config summary

### 1.2 — Diff Between Runs
- `rr digest --diff` — highlight papers that are **new since the previous run** vs. carried over
- Track per-paper "first seen in run N" so digests can show a "New this week" badge
- `rr history` command — list past runs with stats (date, queries, new/seen counts)

### 1.3 — Robustness
- Retry with exponential backoff on arXiv network errors (currently fails on timeout)
- Config validation — warn on unknown arXiv categories, invalid weight values, missing repo_path
- Schema migration system — version the DB schema, auto-migrate on open (for future schema changes)
- Graceful handling of corrupt/locked SQLite files

### 1.4 — CI & Code Quality
- GitHub Actions workflow: run tests on push/PR (Python 3.11, 3.12, 3.13, 3.14)
- `ruff` linter + formatter configuration
- `pre-commit` hooks for linting and formatting
- Type checking with `mypy --strict`

**Done when:** `rr update` survives flaky networks, output is colored and informative, CI is green.

---

## Milestone 2 — Smarter Ranking

**Goal:** Move beyond keyword overlap to semantic relevance, drastically reducing noise.

### 2.1 — Embedding-Based Similarity
- Add optional `sentence-transformers` dependency (e.g., `all-MiniLM-L6-v2`)
- Compute repo embedding from README + docs text
- Compute paper embedding from title + abstract
- New score component: `cosine_similarity(repo_embedding, paper_embedding)`
- Config: `ranking.w_embedding: 1.5` (new weight, higher default than keyword overlap)
- Fallback: if `sentence-transformers` not installed, skip embedding score (graceful degradation)

### 2.2 — Citation Signal (Optional)
- Integrate Semantic Scholar API (free tier, needs API key)
- Fetch citation count for each paper
- New score component: `log(1 + citations) / log(1 + max_citations)` normalized to [0, 1]
- Config: `ranking.w_citations: 0.2`, `semantic_scholar.api_key: "..."` (optional)

### 2.3 — Improved Query Generation
- Analyze top-scoring papers from previous runs to refine future queries (feedback loop)
- Bigram/trigram queries from profile keywords (e.g., "retrieval augmented" instead of separate words)
- Query deduplication — skip queries that returned 0 new papers in last N runs
- `rr queries` command — show auto-generated queries and let user approve/edit before fetching

### 2.4 — Scoring Calibration
- Normalize combined score to [0, 1] range (currently can exceed 1.0 with high weights)
- Add `--explain` flag to `rr update` — print score breakdown for each paper
- Log scoring distribution stats (mean, median, std) to help users tune weights
- Support per-category weight overrides (e.g., cs.CL papers weighted higher for NLP repos)

**Done when:** Embedding similarity is the primary ranking signal, false positives drop significantly, users can inspect and tune scoring.

---

## Milestone 3 — Enrichment & Integrations

**Goal:** Make digests more actionable by linking papers to code, datasets, and repo issues.

### 3.1 — Papers With Code Integration
- Query Papers With Code API for each top-ranked paper (by arXiv ID)
- Enrich digest entries with: linked datasets, reference implementations, task leaderboards
- Add "Code available" / "Dataset available" badges in digest output
- Store enrichment data in new `paper_enrichments` table

### 3.2 — GitHub Issue Export
- `rr gh-issues --top N` — create GitHub issues from top N papers using `gh` CLI
- Issue template: title, paper link, relevance summary, suggested actions
- Track which papers have been exported (avoid duplicates)
- `--dry-run` flag to preview issues without creating them
- Requires: GitHub CLI (`gh`) authenticated

### 3.3 — Multi-Source Collection
- Add Semantic Scholar search as secondary paper source (broader coverage)
- Add OpenAlex as fallback source (free, no key required for basic use)
- Unified paper schema — normalize results from all sources into the same format
- Config: `sources: [arxiv, semantic_scholar, openalex]`

### 3.4 — Export Formats
- `rr digest --format json` — machine-readable output for downstream tools
- `rr digest --format csv` — spreadsheet-friendly export
- `rr digest --format rss` — RSS feed for integration with feed readers

**Done when:** Top papers link to code/datasets, users can create GitHub issues directly, multiple paper sources are supported.

---

## Milestone 4 — Automation & Scheduling

**Goal:** Run RepoRadar unattended and integrate into development workflows.

### 4.1 — Scheduled Runs
- `rr schedule --cron "0 9 * * 1"` — register a cron job for weekly Monday runs
- Platform support: cron (Linux/macOS), Task Scheduler (Windows)
- `rr schedule --list` / `--remove` — manage scheduled tasks
- Alternative: document GitHub Actions workflow for scheduled runs on a repo

### 4.2 — Notification Hooks
- Config: `hooks.on_digest: "command to run"` — execute a shell command after digest generation
- Built-in hooks: email (via sendmail/SMTP), Slack webhook, Discord webhook
- Pass digest path and summary stats as environment variables
- `rr notify --channel slack` — send latest digest to configured channel

### 4.3 — Multi-Repo Support
- `rr workspace init` — manage multiple repos from a single config
- Shared paper database across repos (avoids re-fetching)
- Per-repo profiles with separate scoring
- Combined digest: "Papers relevant to your projects" with per-repo relevance tags

### 4.4 — Watch Mode
- `rr watch --interval 6h` — continuous monitoring with periodic updates
- Desktop notifications on new high-relevance papers (via `notify-send` / `osascript`)
- Rate-limit aware — respects arXiv's 3-second delay and daily query budget

**Done when:** RepoRadar runs weekly without intervention, sends notifications, and supports multiple repos.

---

## Milestone 5 — Advanced Profiling & Intelligence

**Goal:** Deeper repo understanding and smarter suggestions.

### 5.1 — Source Code Analysis
- Scan Python/JS/TS import statements for additional keyword signals
- Detect ML framework usage patterns (training loops, model architectures, loss functions)
- Extract function/class names as domain signals
- Config: `profiler.scan_source: true` (opt-in, can be slow on large repos)

### 5.2 — LLM-Powered Suggestions (Optional)
- Use local LLM (Ollama) or Claude API to generate contextual suggestions
- Input: paper abstract + repo profile + top source files
- Output: 2–3 specific, grounded suggestions per paper (replacing template-based)
- Config: `suggestions.provider: "template" | "ollama" | "claude"`
- Fallback to template-based if no LLM configured

### 5.3 — Trend Detection
- Track keyword frequency across runs — detect emerging topics
- "Trending in your field" section in digest — keywords appearing more frequently
- Alert when a new paper cites or extends a paper you previously starred/opened

### 5.4 — User Feedback Loop
- `rr rate <arxiv_id> [1-5]` — rate paper relevance
- Use ratings to adjust scoring weights automatically (simple logistic regression)
- "Papers like ones you rated highly" — collaborative-filtering-lite
- Store ratings in `paper_ratings` table

**Done when:** RepoRadar understands source code, generates intelligent suggestions, detects trends, and learns from user feedback.

---

## Priority Matrix

| Feature | User Value | Effort | Suggested Order |
|---------|-----------|--------|-----------------|
| Colored output + progress bar | Medium | Low | 1st |
| `rr status` + `rr history` | Medium | Low | 2nd |
| CI + linting | Medium | Low | 3rd |
| Diff between runs | High | Medium | 4th |
| Config validation | Medium | Low | 5th |
| Network retry/robustness | Medium | Medium | 6th |
| DB schema migrations | Low | Medium | 7th |
| Embedding similarity | High | Medium | 8th |
| Score normalization + `--explain` | Medium | Low | 9th |
| Query generation improvements | Medium | Medium | 10th |
| Papers With Code integration | High | Medium | 11th |
| JSON/CSV export | Medium | Low | 12th |
| GitHub issue export | High | Medium | 13th |
| Scheduled runs | High | Medium | 14th |
| Notification hooks | Medium | Medium | 15th |
| Source code analysis | Medium | High | 16th |
| Multi-repo support | Medium | High | 17th |
| LLM-powered suggestions | High | High | 18th |
| Citation signal | Low | Medium | 19th |
| User feedback loop | Medium | High | 20th |
| Trend detection | Medium | High | 21st |
| Multi-source collection | Low | High | 22nd |

---

## Version Targets

| Version | Milestone | Summary |
|---------|-----------|---------|
| **0.1.0** | MVP (done) | End-to-end pipeline with heuristic ranking |
| **0.2.0** | Milestone 1 | Polished CLI, run diffs, CI, robustness |
| **0.3.0** | Milestone 2 | Embedding-based ranking, better queries |
| **0.4.0** | Milestone 3 | Papers With Code, GitHub issues, exports |
| **0.5.0** | Milestone 4 | Scheduling, notifications, multi-repo |
| **1.0.0** | Milestone 5 | Source analysis, LLM suggestions, learning |
