# RepoRadar MVP Plan

## Overview

**RepoRadar** (`rr`) is a CLI tool that monitors arXiv for papers relevant to a
given software repo and produces a weekly Markdown digest with ranked results and
actionable suggestions.

This plan covers only the MVP: a working end-to-end pipeline from repo analysis
to digest output.

---

## Decisions & Constraints

| Decision            | Choice                     | Rationale                              |
|---------------------|----------------------------|----------------------------------------|
| Language            | Python 3.11+               | Matches .gitignore, ecosystem fit      |
| Packaging           | `pyproject.toml` + `uv`    | Modern, fast, lockfile support         |
| arXiv access        | `arxiv` PyPI package        | Handles rate limiting, retries, paging |
| Storage             | SQLite via `sqlite3` stdlib | Zero dependency, single file per repo  |
| Text extraction     | stdlib + `tomli`/`tomllib`  | Parse manifests; no heavy deps         |
| Keyword extraction  | TF-IDF via `scikit-learn`   | Lightweight, no API keys               |
| Templating          | Jinja2                      | Clean digest rendering                 |
| CLI framework       | `click`                     | Simple, well-known                     |
| Testing             | `pytest`                    | Standard                               |

**Not in MVP scope:** embeddings, Semantic Scholar, OpenAlex, Papers With Code,
GitHub issue creation, cron scheduling, any paid APIs.

---

## Project Structure

```
auto-features/
  pyproject.toml
  README.md
  src/
    reporadar/
      __init__.py
      cli.py              # Click CLI entry points
      config.py            # Load/validate .reporadar.yml
      profiler.py          # Repo topic profiling
      collector.py         # arXiv querying
      store.py             # SQLite storage + dedup
      ranker.py            # Heuristic scoring
      digest.py            # Markdown digest generation
      templates/
        digest.md.j2       # Jinja2 digest template
  tests/
    conftest.py
    test_profiler.py
    test_collector.py
    test_store.py
    test_ranker.py
    test_digest.py
    fixtures/              # Sample READMEs, manifests, arXiv responses
```

---

## Database Schema

Single SQLite file: `.reporadar/papers.db`

```sql
CREATE TABLE papers (
    arxiv_id    TEXT PRIMARY KEY,  -- e.g. "2401.12345v1"
    title       TEXT NOT NULL,
    authors     TEXT NOT NULL,      -- JSON array
    abstract    TEXT NOT NULL,
    categories  TEXT NOT NULL,      -- JSON array
    published   TEXT NOT NULL,      -- ISO 8601
    updated     TEXT,
    url         TEXT NOT NULL,
    pdf_url     TEXT,
    first_seen  TEXT NOT NULL,      -- ISO 8601, when we first fetched it
    last_seen   TEXT NOT NULL
);

CREATE TABLE runs (
    run_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_time        TEXT NOT NULL,
    queries_used    TEXT NOT NULL,   -- JSON array of query strings
    papers_new      INTEGER NOT NULL,
    papers_seen     INTEGER NOT NULL
);

CREATE TABLE paper_scores (
    arxiv_id        TEXT NOT NULL REFERENCES papers(arxiv_id),
    run_id          INTEGER NOT NULL REFERENCES runs(run_id),
    score_total     REAL NOT NULL,
    keyword_score   REAL,
    category_score  REAL,
    recency_score   REAL,
    matched_query   TEXT,            -- which query surfaced this paper
    PRIMARY KEY (arxiv_id, run_id)
);
```

---

## Config File Format

`.reporadar.yml` in the repo root:

```yaml
repo_path: .

arxiv:
  categories: [cs.LG, cs.CL]        # arXiv categories to search
  max_results_per_query: 50
  lookback_days: 14

queries:
  seed:                               # User-provided seed queries
    - "retrieval augmented generation"
  exclude:                            # Terms to penalize/filter
    - "survey"

ranking:
  w_keyword: 1.0
  w_category: 0.5
  w_recency: 0.3

output:
  digest_path: ./reporadar_digest.md
  top_n: 15                           # Max papers in digest
```

---

## Phases

### Phase 1 — Skeleton & Config
**Goal:** Runnable `rr` CLI with `init` and `profile` commands.

| #  | Task                                               | Output                             |
|----|----------------------------------------------------|------------------------------------|
| 1a | Set up `pyproject.toml` with dependencies          | Installable package via `uv`       |
| 1b | Implement `config.py` — load/validate YAML config  | Pydantic or dataclass config model |
| 1c | Implement `cli.py` — `rr init` creates `.reporadar.yml` + `.reporadar/` dir | Working init command |
| 1d | Implement `profiler.py` — parse README, `requirements.txt`, `pyproject.toml`, `package.json`; extract keywords via TF-IDF | Keyword list printed by `rr profile` |
| 1e | Write tests for profiler with fixture files         | `test_profiler.py` passing         |

**Done when:** `rr init` creates config, `rr profile` prints extracted keywords for a sample repo.

---

### Phase 2 — arXiv Collection & Storage
**Goal:** `rr update` fetches papers and stores them in SQLite.

| #  | Task                                               | Output                              |
|----|----------------------------------------------------|------------------------------------|
| 2a | Implement `store.py` — SQLite init, insert/upsert, dedup by arxiv_id | DB created on first run  |
| 2b | Implement `collector.py` — build queries from profile keywords + seed queries, call arXiv API | List of paper dicts        |
| 2c | Wire `rr update` in CLI — profile → collect → store | Papers in DB after `rr update`     |
| 2d | Write tests with mocked arXiv responses             | `test_collector.py`, `test_store.py` passing |

**Done when:** `rr update` on a real repo fetches papers from arXiv and stores them in `.reporadar/papers.db` without duplicates.

---

### Phase 3 — Ranking
**Goal:** Score stored papers against the repo profile.

| #  | Task                                               | Output                             |
|----|----------------------------------------------------|------------------------------------|
| 3a | Implement `ranker.py` — keyword overlap, category match, recency scoring | Scored papers list      |
| 3b | Store scores in `paper_scores` table per run       | Scores persisted                   |
| 3c | Add generic-term penalty (configurable via `exclude`) | Lower noise for broad terms     |
| 3d | Write tests with known paper/profile pairs          | `test_ranker.py` passing           |

**Done when:** After `rr update`, papers have scores in the DB, and top-N ordering is sensible for a test repo.

---

### Phase 4 — Digest Output
**Goal:** `rr digest` produces a readable Markdown file.

| #  | Task                                               | Output                             |
|----|----------------------------------------------------|------------------------------------|
| 4a | Create Jinja2 template `digest.md.j2`             | Template file                       |
| 4b | Implement `digest.py` — query DB for scored papers, render template | `digest.md` written      |
| 4c | Sections: "Top Picks", "Maybe Relevant", "Muted"  | Categorized output                 |
| 4d | Each entry: title, authors, abstract snippet, arXiv link, score breakdown, matched query | Complete entries |
| 4e | Wire `rr digest --since 7d` in CLI                 | Working CLI command                 |
| 4f | Write tests for digest rendering                    | `test_digest.py` passing           |

**Done when:** `rr digest` outputs a well-formatted Markdown file with ranked papers, links, and relevance explanations.

---

### Phase 5 — Suggestions & Polish
**Goal:** Add templated action suggestions; polish the end-to-end flow.

| #  | Task                                               | Output                             |
|----|----------------------------------------------------|------------------------------------|
| 5a | Add suggestion templates to digest (template-based, not LLM) | "Action ideas" per top paper |
| 5b | Implement `rr open top` — opens top N papers in browser | Working command                |
| 5c | Add `--format` flag (md / html) to digest          | Optional HTML output               |
| 5d | End-to-end manual test on 2-3 real repos            | Verify quality and noise levels    |
| 5e | Write README with usage instructions                | Updated README.md                  |

**Done when:** Full `rr init → rr update → rr digest` flow works on real repos, digest is useful, suggestions are clearly labeled as ideas.

---

## Acceptance Criteria (MVP)

- [ ] `rr init` scaffolds config and storage directory
- [ ] `rr profile` extracts and displays keywords from a repo
- [ ] `rr update` fetches papers from arXiv, deduplicates, and stores them
- [ ] `rr digest --since 7d` produces a Markdown file with ranked papers
- [ ] Each digest entry has: title, authors, link, abstract snippet, relevance explanation
- [ ] Top papers include 1-3 templated suggestion ideas
- [ ] All tests pass
- [ ] Works on repos with `requirements.txt`, `pyproject.toml`, or `package.json`
- [ ] No paid APIs or API keys required
- [ ] Single `uv pip install -e .` to install

---

## Key Risks & Mitigations

| Risk                                   | Mitigation                                         |
|----------------------------------------|----------------------------------------------------|
| arXiv queries return too much noise    | Start with user seed queries; add exclude terms; tune in Phase 3 |
| arXiv rate limiting (3 sec between calls) | `arxiv` library handles this; limit queries per run |
| Keyword extraction too generic         | TF-IDF with stop words; user can pin/edit via config |
| Digest too long / too short            | Configurable `top_n`; three-tier categorization    |
| Suggestions feel useless               | Keep them templated + grounded in abstract text; clearly label as ideas |
