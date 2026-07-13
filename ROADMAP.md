# RepoRadar Roadmap

**v1.0.0 shipped.** Milestones 1–5 of the original roadmap are complete: polished CLI (`status`/`history`/`--diff`), embedding + citation ranking, multi-source collection (arXiv/Semantic Scholar/OpenAlex), Papers With Code enrichment, GitHub issue export, five digest formats, scheduling, Slack/Discord/email notifications, watch mode, multi-repo workspaces, trend detection, source-code analysis, LLM suggestion engine, and a rating-driven feedback loop — backed by ~500 tests. The original milestone-by-milestone plan is preserved in git history (`git log ROADMAP.md`).

This document is the **next-generation roadmap (Roadmap 2.0)**, produced 2026-07-03 from a codebase audit plus six parallel web-research sweeps (discovery-tool landscape, paper-to-code agents, retrieval/ranking tech, ecosystem integration channels, data sources & signals, frontier AI capabilities). Every feature was **adversarially verified against live APIs and current documentation on 2026-07-03** — verification corrections (dead services, changed pricing, wrong module names) are folded into the text below.

---

## Tier 0 — Urgent repairs

The research surfaced things that were silently failing. **Most shipped in PR #13 (merged 2026-07-04)** — see status column.

| Repair | What was wrong | Status |
|---|---|---|
| **Papers With Code is dead** | paperswithcode.com was sunset July 2025 and redirects to Hugging Face; `paperswithcode.py` failed on every `rr update` | ✅ **Shipped** — replaced with HF Papers enrichment (`sources/hf_papers.py`); adds `[MODEL]` badge + upvote signal; `paperswithcode.py` deleted (Feature 1 core) |
| **OpenAlex now requires API keys** | Since 2026-02-13 keyless callers get a $0.10/day test allowance, then throttling; `sources/openalex.py` sent no key | ✅ **Shipped** — `openalex.api_key` config passed as `?api_key=`; `validate_config` warns when the source is enabled keyless (Feature 12 groundwork) |
| **Retired Claude model default** | `config.py`'s default `claude_model` (`claude-sonnet-4-20250514`) was retired 2026-06-15 — any wired Claude call 404s | ✅ **Shipped** — bumped to `claude-haiku-4-5` |
| **LLM path unreachable** | `llm_suggestions.py` (Ollama/Claude) is fully implemented but `cli.digest` never passed `suggestions_config`/`profile`, so `provider: claude` silently fell back to templates | ✅ **Shipped** — wired through `cli.digest` + the JSON path; failures are now logged, not swallowed (Feature 6 groundwork) |
| **Dead CLI option** | `rr digest --since` was accepted but unused | ✅ **Shipped** — now filters by publication date; opt-in (default: no filter) so existing digests are unchanged |
| **No CI** | `.github/workflows/` didn't exist despite ruff/mypy config | ✅ **Shipped** — `ci.yml`: pytest + ruff + `ruff format --check` + mypy on Python 3.11–3.13 |
| **Run-ordering bug** *(found while repairing)* | `get_runs`/`get_last_run` ordered only by `run_time`; runs sharing a timestamp tied and `get_last_run()` could return the **wrong** run | ✅ **Shipped** — added a `run_id DESC` tiebreaker |
| **Pipeline drift** | The collect→store→rank pipeline is duplicated across `cli.update` (~235 lines inline), `watcher.py`, and `workspace.py`; watch/workspace skip enrichment | ⏳ **Deferred** to Feature 14 (pipeline.py refactor) — duplication, not breakage; carries regression risk across three entry points |

> PR #13 also normalized the whole repo (ran `ruff format`, cleared all `ruff` + `mypy --strict` findings) so CI is genuinely green. 519 tests pass.

### Evaluation harness (`evals/`) — added 2026-07-04

A manually-run **benchmark** (not CI) that scores RepoRadar's ranking quality on realistic repos. It has two modes: an **offline** mode over frozen, real-arXiv labeled fixtures (deterministic, no keys) and a **live** mode that clones real repos and runs the full pipeline against real sources. This is the practical, standalone counterpart to Feature 11 (`rr eval`, which scores against a user's own ratings) and de-risks every ranking change on this roadmap. See `evals/README.md`.

---

## Feature overview

**Status legend** (as of 2026-07-12): ✅ shipped · 🟡 partial (core shipped, extensions pending) · ⬜ planned.
See [Implementation status](#implementation-status-2026-07-12) below for the shipped-vs-remaining breakdown.

| # | Status | Feature | Tier | One-line impact |
|---|--------|---------|------|-----------------|
| 1 | 🟡 | Hugging Face Papers enrichment | Certainly achievable | Repairs the dead PwC integration, adds live code/model/dataset links + community-buzz signal |
| 2 | ⬜ | RepoRadar MCP server | Certainly achievable | Puts repo-aware paper search inside Claude Code, Cursor, VS Code — the biggest 2026 distribution channel |
| 3 | ⬜ | GitHub Action + Pages digests | Certainly achievable | Turns a single-dev CLI into team-visible infrastructure with zero hosting |
| 4 | 🟡 | Hybrid retrieval core (BM25 + vectors + RRF) | Certainly achievable | Measurably better ranking, cached embeddings, and a local `rr search` over everything ever fetched |
| 5 | ⬜ | Semantic Scholar learned recommendations | Certainly achievable | Turns dormant ratings/stars into a server-side learned recommender at zero local ML cost |
| 6 | ✅ | Repo-aware LLM triage & reranking | High confidence | Wires the dormant LLM path; repo-conditioned relevance judgments no embedding can express |
| 7 | ⬜ | Scientific embeddings (SPECTER2) + CPU rerank | High confidence | The 2026-grade retrieval stack: citation-trained paper vectors + cross-encoder polish |
| 8 | ⬜ | Citation alerts + citation-graph digest section | High confidence | "A new paper extends work you starred" — finally makes starring do something |
| 9 | ⬜ | Attention & integrity signals (HN, OpenReview, Retraction Watch, Bluesky) | High confidence | "Is this paper real, reviewed, and talked about?" — a trust layer no paper tool ships |
| 10 | ⬜ | Domain source adapters (IACR ePrint, bioRxiv/medRxiv, DBLP) | Certainly achievable | Serves security/bio/systems repos whose literature is *not* on arXiv — biggest unserved segment |
| 11 | 🟡 | `rr eval` — recommendation-quality harness | Certainly achievable | Makes every other ranking upgrade falsifiable using ratings already collected |
| 12 | 🟡 | OpenAlex 2026 upgrade (keys, semantic search, Topics) | High confidence | Un-breaks the source; classifier-backed field watching instead of keyword guessing |
| 13 | ⬜ | Privacy guard (audit, redaction, local-only mode) | High confidence | Unlocks proprietary/enterprise codebases — currently an unexamined blocker |
| 14 | ⬜ | `rr deepscan` — agentic iterative search | Ambitious | Multi-round query-refine-expand loops, the flagship pattern of $12–20/mo commercial tools, free and repo-aware |
| 15 | ⬜ | `rr ask` — citation-grounded Q&A over your corpus | Ambitious | From alerting tool to research assistant, local-first |
| 16 | ⬜ | Technique fingerprinting ("supersedes what you import") | Ambitious | The category-defining alert: *did research just obsolete part of my codebase?* |
| 17 | ⬜ | Zotero / BibTeX bridge | Certainly achievable | Starred papers flow into the citation manager academics actually live in |
| 18 | ⬜ | Implementability & reproducibility scoring | Experimental | Answers "can I actually use this?" — a signal no free tool scores |
| 19 | ⬜ | Research-gap radar | Experimental | "Nobody has applied X to your Y" — from reading what exists to seeing what's missing |
| 20 | ⬜ | `rr apply` — paper-to-branch | Moonshot | One command from digest entry to a reviewable draft PR implementing the paper's technique |

---

## Implementation status (2026-07-12)

A benchmark-driven arc (Tier B eval → Feature 6 triage/rerank → all-time discovery → hybrid retrieval)
shipped the ranking-and-precision core. The 12-case Tier B benchmark now shows RepoRadar **net-positive
and competitive with an agentic Opus 4.8 baseline** (Top Picks mean net@2 **+1.42** vs **+1.75**, a 0.33
gap), winning on its ML home turf — see [`evals/RESULTS.md`](evals/RESULTS.md).

**✅ Shipped**
- **Tier 0 repairs** (PR #13): HF Papers (dead PwC), OpenAlex key support, retired model default, wired the
  LLM path, `--since` filter, CI, run-ordering fix. *(Pipeline-drift refactor still deferred → Feature 14.)*
- **Feature 6 — repo-aware LLM triage & reranking**: `triage.py` (0–3 actionability), shared `llm_client.py`,
  `TriageConfig`, store v7 `paper_llm_scores`, digest gating (abstains unless genuinely applicable), and
  **listwise rerank** by `llm_score`. The benchmark validated `min_actionable=2` as the default.
- **Feature 4 (core)** — **hybrid retrieval BM25 + RRF** in the production ranker (PR #37, `ranking.hybrid`):
  fuses the heuristic order with a lexical BM25 order via Reciprocal Rank Fusion; store v8 `rrf_score`.
- **Foundational / seed-corpus discovery** (`rr update --foundational`, PR #36) — the eval-validated all-time,
  relevance-first sweep that surfaces seminal work the recent window misses. (Realizes the "seed corpus"
  idea from Finding #2; closes most of the baseline's remaining benchmark edge.)
- **Two-tier evaluation harness** (`evals/`) — Tier A offline fixtures + **Tier B LLM-judged actionable-
  improvement benchmark** (12 cases, GPT-5.5 judge, Opus baseline). The standalone counterpart to Feature 11.

**🟡 Partial**
- **Feature 1** — HF Papers enrichment shipped; remaining: the `w_community` ranking component + `pwc-archive`
  offline fallback.
- **Feature 4** — BM25+RRF fusion shipped; remaining: the **`sqlite-vec` embedding cache** and a local
  **`rr search`** command over the whole stored corpus.
- **Feature 11** — the standalone `evals/` harness exists; the **in-CLI `rr eval`** over a user's own
  ratings/stars is not built.
- **Feature 12** — OpenAlex `api_key` groundwork shipped; semantic search, Topics, and full-text are not.

**⬜ Not started**
- Features 2, 3, 5, 7, 8, 9, 10, 13, 14, 15, 16, 17, 18, 19, 20.

---

## Certainly achievable

Proven technology, verified-live dependencies, clear path. Days-to-weeks each.

### 1. Hugging Face Papers enrichment (replace dead Papers With Code)

> **✅ Core shipped in PR #13** (`sources/hf_papers.py`, schema v6 with `models`/`upvotes`, `EnrichmentConfig`). Remaining: the `w_community` ranking component and the `pwc-archive` offline fallback.

**Verification: confirmed** (endpoints live-tested 2026-07-03).

Papers With Code was sunset in July 2025; `paperswithcode.py` fails on every run today. The free, no-auth Hugging Face Papers endpoints (`/api/papers/{arxiv_id}`, `/api/models?filter=arxiv:{id}`, `/api/datasets?filter=arxiv:{id}`, `/api/daily_papers`) restore code/model/dataset linkage and add community upvote counts as a new buzz signal. The per-paper endpoint already carries `linkedModels`/`linkedDatasets`/`linkedSpaces`, so one call covers most badge data.

**Capabilities**
- Working `[CODE]`/`[MODEL]`/`[DATA]` badges in digests again, pointing at live HF artifacts
- HF upvotes stored per paper as a community-attention ranking component (comment counts only exist on `daily_papers` — treat as best-effort)
- Optional offline fallback: the `pwc-archive` HF datasets hold the final PwC snapshot (~300k paper→repo link rows, CC-BY-SA-4.0 — query at runtime, don't bundle)
- An enrichment on/off config switch (today `update` always attempts the dead PwC call)

**Plan**
1. Create `sources/hf_papers.py` with the stdlib-urllib retry/backoff pattern from `openalex.py`
2. Swap the enrichment call in `cli.update`'s top-N loop; keep writing to `paper_enrichments` (columns map directly; add `upvotes` via a v6 entry in `store.MIGRATIONS`)
3. Add an enrichment config dataclass via the established `config.py` dataclass + `_dict_to_config` + `validate_config` triad
4. Feed upvotes into `ranker.score_paper` as optional `w_community`, following the `citation_score` pattern (touch `ranker.py`, `RankingConfig`, `feedback.py` weight_keys, digest templates)
5. Deprecate `paperswithcode.py` with a warning path

**Risks**
- HF Papers is curated and ML-heavy — thinner coverage than PwC for non-ML domains (mitigated by feature 10)
- Documented rate limits: 500 req/5min anonymous, 1000 with a free `HF_TOKEN` — ample for top-N enrichment, but add optional token config
- Community-documented API; endpoint shapes could drift
- Nothing replaces PwC's SOTA leaderboards

**Dependencies:** HF Papers public JSON API (free, no key).
**Sources:** [PwC sunset issue](https://github.com/paperswithcode/paperswithcode-data/issues/116) · [HF paper pages](https://huggingface.co/docs/hub/en/paper-pages) · [HF rate limits](https://huggingface.co/docs/hub/rate-limits) · [pwc-archive](https://huggingface.co/pwc-archive)

### 2. RepoRadar MCP server: repo-aware paper search inside coding agents

**Verification: confirmed** (SDK, registry, and precedents checked live).

MCP is the dominant 2025–2026 integration standard (~10,000 registered servers). Many arXiv MCP servers exist, but **none profiles a local repository and ranks papers against it** — exactly RepoRadar's differentiator. An `rr mcp` command exposing `profile_repo`, `get_ranked_papers`, `explain_relevance`, and `rate_paper` puts RepoRadar inside Claude Code, Cursor, VS Code, and Windsurf sessions.

**Capabilities**
- Query the ranked paper store conversationally from any MCP client
- Agents pull the repo profile and per-paper score explanations (`ranker.format_score_explanation`) as grounding context
- Agents submit ratings through MCP into the existing `feedback.py` weight-learning loop
- Distribution via the official MCP registry, Cursor one-click deeplinks, and (curated, best-effort) the Claude Code plugin directory

**Plan**
1. Add an optional `[mcp]` extra with the official MCP Python SDK (pin stable v1.x — a v2 beta is in flight); implement `mcp_server.py` wrapping `profiler.profile_repo`, `PaperStore` queries, ranker scoring, and `PaperStore.save_rating`
2. Register `rr mcp` in `cli.py` running stdio transport against `.reporadar/papers.db` via `_open_store`
3. Handle SQLite concurrency: WAL is already enabled; open read-mostly and serialize writes (a concurrent `rr update` could contend)
4. Publish `server.json` to the MCP registry (GitHub namespace verification); ship a Claude Code plugin (`.claude-plugin/plugin.json` + optional skills)
5. Tests mirroring `tests/test_cli.py` patterns per MCP tool

**Risks**
- MCP spec churn: the 2026-07-28 spec is at release-candidate stage; registry API is a v0.1 preview
- Long-lived server process + 5s-timeout SQLite store needs care alongside watch mode
- Crowded discovery (~16k mcp-server repos) — the repo-aware angle must lead the listing copy

**Dependencies:** MCP Python SDK (MIT); MCP registry publish flow.
**Sources:** [MCP release blog](https://blog.modelcontextprotocol.io/posts/2026-07-28-release-candidate/) · [registry](https://github.com/modelcontextprotocol/registry) · [adoption stats](https://www.digitalapplied.com/blog/mcp-adoption-statistics-2026-model-context-protocol) · [nearest competitor (search-only)](https://github.com/blazickjp/arxiv-mcp-server)

### 3. `reporadar-action`: GitHub Action + GitHub Pages published digests

**Verification: confirmed.**

The dominant open-source paper-alert pattern is a fork-and-configure GitHub Action publishing scored digests to Issues/Slack/Pages (zotero-arxiv-daily, 5.6k stars) — and none of them profile the host repository. A first-party `action.yml` wrapping the existing CLI gives teams a shared, linkable, always-current "research radar" page with zero hosting, where **the repo that is the profile target runs its own radar**.

**Capabilities**
- Zero-infrastructure scheduled digests via `on: schedule`
- Digest published as a browsable GitHub Pages archive and/or weekly GitHub Issue (reusing `gh_issues.py` + the `paper_exports` dedup ledger)
- Team-shareable URLs instead of local-only files
- Fixes the missing CI story as a side effect (the action's test workflow doubles as project CI)

**Plan**
1. Write `action.yml` + thin entrypoint invoking `rr update && rr digest`, inputs mapped to `.reporadar.yml` keys; cache `.reporadar/papers.db` via `actions/cache@v4` (v1–v3 are shut down)
2. Upgrade `digest.py`'s HTML path from markdown-in-`<pre>` to real rendered HTML — a proper Jinja2 template suffices (jinja2 is already a core dep); the `markdown` lib (BSD-3) is an optional extra
3. Add a Pages publish step writing dated digests to a `docs/` archive with an index (gpt_paper_assistant layout; Apache-2.0, safe to borrow from — zotero-arxiv-daily is AGPL, **pattern only, no code copying**)
4. Wire `notify.py` Slack/Discord senders as optional action inputs
5. Add `.github/workflows/` CI (pytest + ruff + mypy on 3.11–3.13)

**Risks**
- Scheduled workflows in public repos auto-disable after 60 days of repo inactivity (the digest-commit step itself acts as a keepalive)
- `actions/cache` eviction (7 days unused) loses run history/ratings — offer a commit-to-branch DB persistence option
- GitHub Pages is free **only for public repos**; private repos need Pro/Team (Issue/Slack outputs still work)
- arXiv's 3s politeness delay makes runs slow on large query sets; Actions minutes are finite on private repos

**Dependencies:** GitHub Actions, actions/cache@v4, GitHub Pages.
**Sources:** [zotero-arxiv-daily](https://github.com/TideDra/zotero-arxiv-daily) · [gpt_paper_assistant](https://github.com/tatsu-lab/gpt_paper_assistant) · [schedule trigger docs](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows)

### 4. Hybrid retrieval core: BM25 + RRF fusion with a sqlite-vec embedding cache

> **🟡 Core shipped (PR #37):** dependency-free BM25 + Reciprocal Rank Fusion in the production ranker
> (`ranking.hybrid`, `reporadar.retrieval`), persisted via store v8 `rrf_score`; validated on the Tier B
> benchmark (lifted Top-10 nDCG on every case). **Remaining:** the `sqlite-vec` embedding cache (one-time
> vectors instead of per-run recompute) and the local **`rr search`** command over the whole corpus.

**Verification: confirmed** (all libraries alive, MIT/Apache, Windows wheels checked).

Ranking is currently a heuristic weighted sum, and embeddings are recomputed for every paper on every run with no persistence. Adding a `bm25s` lexical index over stored abstracts, persisting vectors in a `sqlite-vec` virtual table inside the existing `papers.db`, and fusing lexical + dense rankings via Reciprocal Rank Fusion (~10 lines) is the proven, score-scale-agnostic retrieval upgrade that consistently beats either signal alone.

**Capabilities**
- Robust ranking on vocabulary-mismatch queries (repo jargon vs paper jargon) where TF-IDF overlap fails
- One-time embedding per paper — vectors cached in the DB, making embedding-enabled runs dramatically faster
- A local **`rr search <query>`** command over everything ever fetched — the store becomes a queryable personal corpus
- Optional cheap first-pass filter with Model2Vec static embeddings (up to ~500× faster than MiniLM on CPU)

**Plan**
1. Add `sqlite-vec` as an optional extra. **Do not create the vec0 table via the unconditional `store.MIGRATIONS` chain** — `_init_schema` runs migrations on every open and would break DBs for users without the extension. Create it lazily/conditionally, loading the extension (`sqlite_vec.load(conn)`) on every connection that touches it
2. Change `embeddings.py` to check-then-store vectors instead of recomputing in `ranker.rank_papers`' per-paper loop
3. Add `bm25s` (NumPy core; numba optional) and build an index over stored abstracts in a new `retrieval.py`
4. Implement `rrf_fuse(rankings, k=60)`; feed the fused rank into `ranker.score_paper` as `w_retrieval` via the `citation_score` optional-arg pattern
5. Expose `rr search` in `cli.py` (BM25 + vec0 KNN + RRF over the whole store)
6. Benchmark against stored ratings (feature 11) to prove fusion beats the current ranker

**Risks**
- sqlite-vec is pre-v1 (stable v0.1.9 is brute-force KNN; IVF/DiskANN exist in-repo but unreleased) with maintainer-availability gaps — version-pin and watch; fine at tens of thousands of papers
- RRF gains over a decent lexical baseline can be modest on keyword-friendly queries; the win is robustness
- Windows: win_amd64 wheel exists; no ARM64 wheel

**Dependencies:** bm25s, sqlite-vec (optional extras); Model2Vec optional.
**Sources:** [bm25s](https://github.com/xhluca/bm25s) · [sqlite-vec](https://github.com/asg017/sqlite-vec) · [RRF primer](https://opensearch.org/blog/introducing-reciprocal-rank-fusion-hybrid-search/) · [Model2Vec](https://github.com/MinishLab/model2vec)

### 5. Server-side learned recommendations via Semantic Scholar Recommendations API

**Verification: feasible-with-caveats** (API live-tested unauthenticated, HTTP 200).

`paper_ratings` and `paper_stars` are collected but only nudge heuristic weights. The free S2 Recommendations API accepts positive **and** negative example papers and returns fresh papers from a production-trained model (the engine behind S2 Research Feeds) — a learned, server-side recommender channel with zero local ML cost.

**Capabilities**
- A "Recommended for you" stream powered by a real learned model instead of the keyword-overlap recommender in `feedback.py`
- Negative signals (ratings ≤2) suppress similar papers — **requires ≥1 positive seed; negative-only calls return HTTP 400** (guard for it)
- Cold-start mitigation: seed with starred papers (`store.get_starred_papers` — written today, never consumed)

**Plan**
1. Add `sources/s2_recommendations.py` using the `semantic_scholar.py` `_request_json` retry pattern and optional `x-api-key`
2. In `cli.update`, after collection, call with positive ids (rating ≥4 + starred) and negative ids (rating ≤2); skip if no positives
3. Merge results through the paper-dict contract with `matched_query='recommendation'`; upsert and rank normally — **local ranker re-filtering is essential, not optional** (live tests returned off-topic general-science results for CS seeds)
4. Use API results in the digest "Recommended for You" section when available; local fallback otherwise
5. Add a recommendations config section via the standard dataclass triad

**Risks**
- The multi-example POST endpoint draws only from the **recent pool** (~last 60 days) — good for monitoring, but classic literature never surfaces via this path
- Unauthenticated shared pool (~5,000 req/5min globally, throttled under load); **S2 no longer issues API keys to free-domain emails or third-party apps** — design for keyless operation
- Only arXiv-native/S2-known papers have usable ids; `oa:` synthetic ids don't map

**Dependencies:** S2 Recommendations API (free; works unauthenticated).
**Sources:** [API docs](https://api.semanticscholar.org/api-docs/recommendations) · [release notes (limits/key policy)](https://github.com/allenai/s2-folks/blob/main/API_RELEASE_NOTES.md)

### 10. Domain source adapters: IACR ePrint, bioRxiv/medRxiv, DBLP

**Verification: proposed by completeness critique; APIs confirmed free.**

Every existing source assumes the arXiv-ML user, but RepoRadar's value proposition — *papers relevant to YOUR repo* — is strongest for the long tail of non-ML repos whose literature is **not on arXiv**: crypto/security publishes on IACR ePrint, bio tooling on bioRxiv/medRxiv, systems/PL/DB at USENIX/SOSP/VLDB (surfaced via DBLP, abstracts backfilled from OpenAlex/S2 by DOI). This is the single biggest unserved-user-segment gap.

**Capabilities**
- Automatic adapter activation from the repo profile (imports like `cryptography`/`biopython`, manifest keywords) with config override
- Security, bio, and systems repos get genuinely relevant digests for the first time
- Multiplies the addressable user base with zero new ranking machinery

**Plan**
1. Formalize the implicit source-adapter interface in `sources/` (S2/OpenAlex already follow it: `search_papers` + `collect_papers` → shared paper dict)
2. Add `sources/iacr.py` (free search/listing endpoints + RSS), `sources/biorxiv.py` (official JSON API at api.biorxiv.org, date-interval pagination, category filters), `sources/dblp.py` (venue mapping; abstract backfill by DOI)
3. Extend `PACKAGE_DOMAIN_MAP` in `profiler.py` so domain inference activates the right adapters
4. Category/venue config per adapter mirroring `arxiv.categories`

**Risks**
- Each new free API needs the graceful-degradation + retry pattern and a breakage watch
- DBLP has no abstracts — relevance scoring is title-only until backfill lands; backfill costs OpenAlex/S2 budget
- Venue mapping curation burden for the systems adapter

**Dependencies:** IACR ePrint endpoints/RSS, bioRxiv/medRxiv API, DBLP API (all free).
**Sources:** [bioRxiv API docs (via medrxivr)](https://docs.ropensci.org/medrxivr/) · [eprint.iacr.org](https://eprint.iacr.org/) · [dblp.org](https://dblp.org/)

### 11. `rr eval`: offline recommendation-quality harness and ranking regression gate

> **Related:** the standalone `evals/` benchmark harness (added 2026-07-04) already scores ranking quality on realistic repos with labeled arXiv fixtures. This feature is the *in-CLI* version that scores against a user's own accumulated ratings/stars; the two share the same metrics (P@k, R@k, nDCG@k, MRR).

**Verification: proposed by completeness critique; read-only over existing data.**

This roadmap ships four competing ranking upgrades (RRF, SPECTER2, LLM rerank, S2 recommendations) plus a weight-tuning feedback loop — with **no instrumentation to tell whether any of them helps**. `rr eval` treats accumulated `paper_ratings` (4–5 stars = relevant, 1–2 = not) and `paper_stars` as a labeled test set and computes precision@k, recall@k, nDCG@k, and MRR for the current ranker over the stored corpus.

**Capabilities**
- `rr eval` — offline metrics for the active ranking config; warns below ~20 judgments
- `rr eval --compare configA configB` — A/B any two ranker configs on identical data
- `--baseline` snapshots metrics to SQLite so CI (feature 3) catches ranking regressions after upgrades or feedback-weight tuning
- Every other ranking feature on this roadmap gets an acceptance test

**Plan**
1. Add `evaluation.py`: load rated/starred papers, re-score under a given `RankingConfig`, compute metrics (pure Python/NumPy — no new deps)
2. Register `rr eval` in `cli.py` with `--compare`/`--baseline`; store snapshots in a `metric_snapshots` table (v6 migration batch)
3. Report per-repo and pooled results in workspaces
4. Document the workflow: rate a while → eval → then trust ranking changes

**Risks**
- Rating sparsity: most users rate few papers; metrics are noisy below ~20 labels (surface the warning prominently)
- Ratings are biased toward papers the old ranker already surfaced (selection bias) — document; don't over-claim

**Dependencies:** none (existing store data).
**Sources:** [ranking-metrics guide](https://www.evidentlyai.com/ranking-metrics/evaluating-recommender-systems)

### 17. Zotero and BibTeX bridge

**Verification: proposed by completeness critique; pyzotero/Web API v3 confirmed.**

The dominant comparable tool (zotero-arxiv-daily) is Zotero-first — that's where the paper-alert audience manages papers. RepoRadar currently dead-ends starred papers in its own SQLite. A small, isolated output module with outsized adoption leverage for academics.

**Capabilities**
- `rr export bibtex` — well-formed BibTeX (arXiv/DOI/venue metadata is already stored) for any digest, star list, or rating filter
- Optional Zotero push via pyzotero (Web API v3 write requests), filing starred papers into a configured collection with tags like `reporadar` + repo name
- Dedup by DOI/arXiv ID so repeat runs are idempotent
- Composes with the MCP server (feature 2) for agent-driven library filing

**Plan**
1. Add `export_bibtex.py` beside `gh_issues.py`; entry generation from stored metadata
2. Add `rr export bibtex|zotero` command; Zotero config (API key, library id, collection) via the standard dataclass triad
3. Track exports in `paper_exports` (`export_type='zotero'`) for idempotency

**Risks**
- Zotero API key management UX; write rate limits on large first syncs
- BibTeX quality for non-arXiv synthetic-id papers (ss:/oa:) is metadata-dependent

**Dependencies:** pyzotero (optional extra); Zotero Web API v3.
**Sources:** [pyzotero](https://pypi.org/project/pyzotero/) · [Zotero write API](https://www.zotero.org/support/dev/web_api/v3/write_requests) · [zotero-mcp precedent](https://github.com/54yyyu/zotero-mcp)

---

## High confidence

Proven elsewhere; moderate integration risk or degraded-dependency caveats.

### 6. Repo-aware LLM triage and reranking (and actually wiring the LLM path)

> **✅ Core shipped.** `llm_client.py` (shared transport), `triage.py` (0–3 LLM actionability scoring), `TriageConfig`, schema v7 `paper_llm_scores`, `cli.update` triage stage, and digest tiering that **gates Top Picks on the LLM score** (abstains unless genuinely actionable) — directly targeting the precision/calibration gap the Tier B baseline exposed (`evals/RESULTS.md`). The `evals/run_judge_eval.py --rr-triage` flag measures the movement. Remaining: listwise reranking, storing scores to avoid re-paying inference across re-digests, and a HyDE query path.

**Verification: feasible-with-caveats** — and it fixes two live bugs.

`llm_suggestions.py`'s Ollama/Claude transports are fully implemented but unreachable: `cli.digest` never passes `profile`/`suggestions_config` (the profile-is-None guard at `suggestions.py:109` silently falls back to templates), and the default `claude_model` in `config.py` is **retired** (`claude-sonnet-4-20250514`, gone since 2026-06-15 — fixing the wiring without bumping it converts a silent fallback into a hard 404). Fix the plumbing, extract a shared `llm_client.py`, then use it for the proven pattern (gpt_paper_assistant, RankLLM): the LLM scores each candidate's relevance/novelty against a repo-derived interest statement and listwise-reranks the digest top ~20 with repo-aware instructions no embedding can express.

**Capabilities**
- Per-paper relevance + novelty scores with one-line justifications (~$0.03/day for ~20 papers on Haiku 4.5, ~$0.09/day on Sonnet 5; free via Ollama)
- Listwise rerank of the shortlist conditioned on the actual repo profile
- LLM action suggestions finally reachable from `rr digest` and `rr gh-issues`
- Surfaced errors instead of silent template fallback

**Plan**
1. Fix wiring: pass `cfg.suggestions` + profile through `cli.digest → write_digest`, `gh-issues` (passes config but not profile), **and the JSON digest path** (`digest.py:179` passes neither); bump the default model to `claude-haiku-4-5` (or `claude-sonnet-5`)
2. Extract `_call_ollama`/`_call_claude` into `llm_client.py` with `complete(prompt, cfg) -> str`, retry/backoff, non-silent errors; consolidate the three provider-dispatch sites
3. Add `llm_rerank.py`: interest statement from `RepoProfile` (keywords, domains, `source_analysis.py` ML patterns), send top-K titles+abstracts, parse the permutation, blend as final-stage rank adjustment
4. Store LLM scores in a `paper_llm_scores` table (v6 migration) so re-digesting doesn't re-pay inference
5. Gate behind `suggestions.provider`; template path stays default

**Risks**
- Hallucinated justifications — label clearly, ground in abstract text only
- EMNLP Findings 2025: LLM rerankers excel on familiar queries but generalization varies and cross-encoders are far cheaper — restrict to final top-K polish
- Small local models need tolerant permutation parsing and structured-output fallbacks
- API cost scales with digest frequency — per-run call budget

**Dependencies:** existing Ollama/Anthropic transports; reference impls: [rank_llm](https://github.com/castorini/rank_llm), [rerankers](https://github.com/AnswerDotAI/rerankers).
**Sources:** [gpt_paper_assistant](https://github.com/tatsu-lab/gpt_paper_assistant) · [LLM-reranker study, EMNLP Findings 2025](https://arxiv.org/abs/2508.16757)

### 7. Scientific embeddings upgrade: SPECTER2 + CPU cross-encoder rerank

**Verification: feasible-with-caveats** (S2 free tier degraded; design around it).

all-MiniLM is a generic sentence model; SPECTER2 is the de-facto scientific-paper embedder (citation-trained, beats SPECTER/SciNCL on SciRepEval). Semantic Scholar serves **precomputed SPECTER2 vectors** through the API — piggyback `fields=embedding.specter_v2` onto the existing `/paper/batch` call in `citations.py` at **zero extra requests** (500 papers/call, ~4MB, under the 10MB cap). Add a small cross-encoder rerank over the top ~50 candidates.

**Capabilities**
- Domain-tuned paper similarity without local paper-side inference (when S2 is reachable — see risks)
- A rerank stage that meaningfully reorders the digest top tier: default `bge-reranker-base` or an ms-marco MiniLM cross-encoder (note: `answerai-colbert-small-v1` is late-interaction, **not** a cross-encoder — optional backend via the `rerankers` package only)
- HyDE-style repo queries: embed an LLM-written hypothetical abstract of "the ideal paper for this repo" instead of a TF-IDF keyword bag — this also fits SPECTER2's 512-token title+abstract input format better than raw profile text

**Plan**
1. Add `embedding.specter_v2` to the existing `/paper/batch` request in `citations.py`; store vectors (plain BLOB columns work fine at this scale if feature 4 hasn't landed; sqlite-vec table if it has)
2. In `embeddings.py`, add a SPECTER2 option for embedding the repo profile query locally — requires the `adapters` library + the `allenai/specter2` proximity adapter on top of `specter2_base` (plain sentence-transformers cannot load it)
3. **Pick an explicit fallback strategy** — SPECTER2 is 768-dim, MiniLM is 384-dim; they cannot mix in one cosine space. Recommended: per-run whole-stage fallback to MiniLM when S2 vectors are missing
4. Add `rerank.py` applying the cross-encoder over top-K fused candidates before digest tiering (`ranking.reranker` config)
5. Implement HyDE in `collector.build_queries` as an optional `llm_client` call
6. Evaluate with `rr eval` (feature 11); expose timings

**Risks**
- **S2 free-tier access is the weak link**: live tests hit 429 on every unauthenticated call during saturation, and S2 no longer grants keys to free-domain emails/third-party apps — must degrade gracefully to the MiniLM path
- Cross-encoder CPU latency: fine for 33M-class models, optimistic for bge-reranker-base (278M) over 50 pairs — keep top-K small and configurable
- HyDE benefit partially reflects LLM parametric knowledge (ACL 2025 Findings "knowledge leakage" critique) — treat as additive
- `adapters` pins transformers versions — growing optional extra

**Dependencies:** S2 API `embedding.specter_v2` field; sentence-transformers + adapters; a reranker checkpoint.
**Sources:** [specter2_base](https://huggingface.co/allenai/specter2_base) · [SciRerankBench](https://arxiv.org/abs/2508.08742) · [HyDE critique](https://arxiv.org/abs/2504.14175)

### 8. Citation alerts for starred papers + citation-graph digest section

**Verification: feasible-with-caveats** (core mechanism proven by live API calls).

`paper_stars` is written by `rr open` but never consumed — the v1 roadmap's "alert when a new paper cites work you starred" was never built. Each update can check whether newly fetched papers cite starred/highly-rated ones (S2 `/paper/batch` with `references.externalIds` — live-tested; OpenAlex `filter=cites:W...` — live-tested), boost and badge them, and render a Connected-Papers-style co-citation section computed locally with no ML.

**Capabilities**
- "Extends work you starred" alerts in digests and notifications — high-precision personal relevance
- A `w_citation_proximity` ranking boost for papers citing or co-cited with liked papers
- An inline SVG citation-graph in the HTML digest connecting new papers to seed papers
- One-hop citation-trail expansion: references/citations of top hits become additional candidates

**Plan**
1. Consume `store.get_starred_papers` + rating ≥4 papers as the seed set in a new `citation_graph.py`
2. During `cli.update`, batch-fetch reference lists for new top papers (S2 batch pattern from `citations.py`; note these are **new endpoints** for the codebase, with new failure modes); intersect with seed ids; store edges in a `paper_citations` table (v6)
3. Fall back to OpenAlex `cites:` filters when S2 throttles (requires feature 12's key support; edge fetching fits easily in the ~10k free filter-calls/day budget)
4. Compute co-citation/bibliographic-coupling locally; add as optional ranker component
5. Digest template section (`{% if %}` guard-block pattern like trends) + inline SVG; fire `notify.dispatch_notification` on alerts

**Risks**
- Reference lists for brand-new preprints lag in S2/OpenAlex by days–weeks — alerts trail publication
- S2 unauthenticated is best-effort (429 on the second rapid request in live tests) — cache edges aggressively, prefer OpenAlex fallback
- Graph rendering must stay dependency-free (inline SVG)

**Dependencies:** S2 Graph API references/citations; OpenAlex `cites:` filter (keyed — feature 12).
**Sources:** [S2 API](https://api.semanticscholar.org/api-docs/) · [OpenAlex pricing blog](https://blog.openalex.org/openalex-api-new-features-and-usage-based-pricing/)

### 9. Composite attention & integrity signals: HN, OpenReview, Retraction Watch, Bluesky

**Verification: feasible-with-caveats** (all four sources confirmed free; two plan corrections applied).

With Crossref Event Data sunset (2026-04-23) and Altmetric keyed, no turnkey free altmetrics API exists — but the components are free: HN Algolia (10k req/hr, no key) for developer attention on arXiv URLs; OpenReview API v2 for actual reviewer scores and accept/reject decisions; Retraction Watch via the **production Crossref REST API** (`/works?filter=updates:{DOI}` — live-verified; the Crossref *Labs* endpoint is retired and stale, do not use); Bluesky for academic-social buzz. A trust layer no paper-alert tool currently ships.

**Capabilities**
- "Discussed on HN (245 points)" badges plus a composite `w_attention` ranking component
- "Accepted at ICLR 2026 (avg score 7.2)" / "Rejected" annotations that upgrade or demote preprints
- Retraction/withdrawal flags as a hard multiplicative penalty in ranking, excluded from trends
- All from free APIs

**Plan**
1. Add a `signals/` package: `hn.py` (Algolia search on arxiv.org/abs URLs), `openreview.py` (API v2 lookups by title/arXiv id), `retractions.py` (production Crossref `filter=updates:` with mailto polite pool)
2. Store results in a `paper_signals` key-value table (v6) — start breaking the rigid per-signal-column schema
3. Fold normalized attention/review components into `ranker.score_paper` via the `citation_score` pattern; retraction = hard penalty
4. Render badges in templates and JSON/CSV exports
5. Bluesky: use **authenticated `searchPosts` polling with a free app-password** (Jetstream cannot keyword-filter — tracking arXiv URLs via firehose means consuming every post; mark strictly best-effort)

**Risks**
- Title-based OpenReview matching is fuzzy (preprint vs camera-ready titles); rate limits undocumented — conservative client-side throttling from day one
- HN signal is heavily ML/systems-biased; near-zero for niche domains
- Three more free APIs to monitor for breakage; each needs graceful degradation

**Dependencies:** HN Algolia (free), OpenReview API v2, Crossref REST API (free, mailto), Bluesky account (optional).
**Sources:** [HN Algolia](https://hn.algolia.com/api) · [Retraction Watch in Crossref](https://www.crossref.org/documentation/retrieve-metadata/retraction-watch/) · [OpenReview docs](https://docs.openreview.net/) · [Crossref deprecations](https://www.crossref.org/deprecated/)

### 12. OpenAlex 2026 upgrade: API keys, semantic search, and Topics-based field watching

**Verification: confirmed** (pricing/endpoints checked against current docs).

OpenAlex made API keys effectively mandatory (since 2026-02-13; keyless = $0.10/day test allowance) with usage-based pricing ($1/day free credit per key; list/filter $0.0001, search $0.001, semantic search up to ~$0.01, content downloads $0.01/PDF). `sources/openalex.py` sends no key and degrades today. The same release shipped **semantic (embedding) search** accepting whole-abstract queries and **full-text downloads** for 60M OA works; and OpenAlex's ~4,516-topic hierarchy enables subscribing to classified *topics* instead of brittle keyword queries.

**Capabilities**
- Un-degraded OpenAlex source with key management and defensive free-allowance budgeting (read spend from response headers — OpenAlex's own semantic-search pricing docs are inconsistent, so don't hardcode prices)
- Semantic search using repo profile text directly as the query (`/works?search.semantic=` — beta; keep keyword fallback)
- Topic subscriptions with drift detection feeding `trends.py`
- Optional full-text PDF retrieval for top papers (count-capped: ~100 free PDFs/day), enabling RAG features downstream

**Plan**
1. Add `api_key` to `OpenAlexConfig` — passed as a **query parameter** (`?api_key=`), not a header; `validate_config` warning when enabled keyless; persist a daily-spend counter in the store
2. Add a semantic-search collect path as a config-selected alternative to keyword queries
3. Add `topics.py`: one-time repo-profile → topic-ID mapping, stored subscriptions, per-run topic-distribution snapshots (keyword_frequencies-style) for drift detection
4. Extend `cli.update`'s OpenAlex block (cli.py:201–218) with topics/semantic modes; surface topic drift in the digest trends section
5. Add `fulltext.py` fetching PDFs for top-N papers into `.reporadar/fulltext/` (opt-in, size- **and count-**capped)

**Risks**
- A misconfigured cron/watch loop can exhaust the $1/day allowance — budgeting must be defensive and visible
- Semantic search is explicitly beta ("don't rely on it for sensitive production workflows"); the Walden data rewrite (Nov 2025) means topic assignments may shift
- Full-text storage grows the local footprint quickly

**Dependencies:** free OpenAlex API key (user-supplied).
**Sources:** [pricing announcement](https://blog.openalex.org/openalex-api-new-features-and-usage-based-pricing/) · [developers.openalex.org](https://developers.openalex.org/) · [topic classifier (MIT)](https://github.com/ourresearch/openalex-topic-classification)

### 13. Privacy guard: query audit, term redaction, and fully-local mode

**Verification: proposed by completeness critique; local half already exists in the codebase.**

RepoRadar's core mechanic — deriving search queries from a repo's code and docs — is exactly what an engineer on a proprietary codebase cannot let leak into arXiv/S2/OpenAlex/Anthropic query logs. This roadmap adds *more* outbound surfaces (S2 recommendations send rated-paper lists; LLM triage sends profile context; OpenAlex keys tie queries to a billed identity). No feature addresses privacy today; this unlocks the industry/enterprise segment.

**Capabilities**
- `rr audit` — print every outbound query string, endpoint, and payload the current profile would generate: "exactly what leaves this machine"
- Redaction config: denylist terms/regexes (internal codenames), plus an option to profile only README/docs and skip source-import analysis
- `--local-only` mode: fetch from a periodically synced arXiv metadata snapshot (OAI-PMH or the Kaggle arXiv dump) ingested into SQLite, Ollama for suggestions, local embeddings — zero third-party calls after sync

**Plan**
1. Layers 1–2 are days of work: a query-plan dry-run function in `collector.py`/`sources/` + a redaction filter applied at query-build time
2. Layer 3: `sync.py` for bulk snapshot ingest (the moderately hard part — hence high-confidence, not certain); reuse the store/ranker unchanged
3. Document a privacy posture matrix per feature (what each integration sends where)

**Risks**
- Snapshot freshness vs the tool's "what's new this week" promise (daily OAI-PMH sync mitigates)
- Kaggle dump licensing/mirror reliability; OAI-PMH harvest takes hours on first sync
- Redaction gives a false sense of security if TF-IDF keywords still encode domain specifics — the audit view is the honest layer

**Dependencies:** arXiv OAI-PMH or Kaggle arXiv dataset; existing Ollama/embeddings/SQLite plumbing.
**Sources:** [arXiv OAI-PMH](https://info.arxiv.org/help/oa/index.html) · [Kaggle arXiv dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv)

---

## Ambitious

Emerging tech, real design risk; each has an unbuilt in-repo prerequisite.

### 14. `rr deepscan`: agentic iterative search with query refinement and citation trails

**Verification: feasible-with-caveats.**

The defining 2025–2026 shift is from one-shot search to agentic deep search: Undermind's multi-minute agent loops (its "10–50×" figure is marketing, but the pattern is real), Ai2's PaperFinder (pattern evidence only — the repo is a frozen snapshot needing four paid keys), and LangChain's Local Deep Researcher (MIT, 9.2k stars) proving the full loop runs on Ollama. RepoRadar builds static TF-IDF queries once; `rr deepscan [topic]` would have the LLM inspect first-pass results, rewrite queries to fill gaps, expand one hop along citations of strong hits, and emit a cited report — reusing the existing collectors as the search tool.

**Capabilities**
- Multi-round retrieval that recovers papers the static query builder structurally misses
- Reflection-driven query rewriting grounded in what was actually found
- Citation-trail expansion from high-ranked hits (depends on feature 8's edge-fetching code)
- A standalone deep-dive report ("state of X for this repo") distinct from the daily digest

**Plan**
1. **Prerequisite refactor:** extract the collect→store→rank core out of `cli.update`'s ~235-line inline orchestrator into a reusable `pipeline.py` — this also fixes the watcher/workspace pipeline drift (Tier 0)
2. Build `deepscan.py`: loop of build/refine queries (`llm_client`) → collect (existing collector + `sources/`) → rank → LLM reflection on coverage gaps → next queries, capped at N rounds and a call budget
3. Citation-hop expansion reusing `citation_graph.py` edges (feature 8 must ship first, or build reference-endpoint code here)
4. Report via a new Jinja2 template (the `digest.py` fmt-dispatch pattern) with per-claim paper citations
5. `rr deepscan [--rounds 3] [--budget]` with per-round progress output

**Risks**
- Budget enforcement is a **correctness requirement**: multi-round loops multiply arXiv's 3s politeness delay and OpenAlex's per-search pricing (a 3-round scan is ~$0.01–0.03 — small, but must be visible and capped)
- Local-model reflection quality is uneven; bad reflection wastes rounds
- 5–15 minute runtimes change CLI UX expectations — needs progress reporting; the sync architecture has no resumability
- The pipeline.py refactor carries regression risk across cli/watcher/workspace

**Dependencies:** `llm_client.py` (feature 6), pipeline refactor, feature 8 for citation hops, OpenAlex key (feature 12).
**Sources:** [local-deep-researcher](https://github.com/langchain-ai/local-deep-researcher) · [asta-paper-finder](https://github.com/allenai/asta-paper-finder) · [Undermind](https://www.undermind.ai/)

### 15. `rr ask`: citation-grounded Q&A over your paper corpus (local RAG)

**Verification: feasible-with-caveats.**

Every major 2025–2026 discovery tool ships paper-grounded chat (alphaXiv Ask AI, SciSpace, Emergent Mind), and PaperQA2 proves the open-source, citation-backed RAG recipe (RAG-QA Arena science SOTA). RepoRadar has the substrate — papers in SQLite, optional embeddings, Ollama/Claude transports — so `rr ask "how do these papers handle X?"` with per-claim citations is a large capability jump that stays local-first. **Ship an abstract-only v1 first**; the good experience needs full-text, which is the hard tail.

**Capabilities**
- Ask questions across everything RepoRadar has ever fetched, answered with per-claim citations to specific papers
- "Has anyone applied technique X to a system like this repo?" precedent queries conditioned on the repo profile
- "Ask about this paper" deep-links from the digest top picks
- Fully offline via Ollama + local embeddings

**Plan**
1. Build `rag.py`: chunk stored abstracts (v1) and `fulltext/` PDFs when present (v2, from feature 12; plain PDF-to-text chunking — TEI/GROBID is a Java service, name it explicitly if ever adopted); embed into the feature-4 vector store; retrieve via hybrid BM25+vec
2. Answer synthesis through `llm_client.py` (feature 6; note: existing transports are non-streaming — streaming is new work) with a strict cite-or-abstain prompt and a numbered source list mapping to arxiv_ids
3. `rr ask QUESTION [--paper ID] [--top-k]` in `cli.py`
4. Inject the repo profile as system context so answers are framed for this codebase
5. Evaluate on a small curated QA set before release; consider paper-qa as an optional extra (Apache-2.0; drags LiteLLM/tantivy/pydantic — too heavy to vendor into the lean stack)

**Risks**
- Hallucination in synthesis — abstain instructions + citation verification; small local models will still err
- Abstract-only corpora give shallow answers; full-text via OpenAlex is metered (~100 free PDFs/day; arXiv PDFs are the free fallback)
- Scientific PDF parsing/chunking quality varies wildly
- Multi-call RAG on CPU Ollama can take minutes per question — set UX expectations

**Dependencies:** features 4 (hybrid retrieval) and 6 (`llm_client`); feature 12 full-text optional.
**Sources:** [paper-qa](https://github.com/future-house/paper-qa) · [PaperQA2 announcement](https://www.futurehouse.org/research-announcements/paperqa2-achieves-sota-performance-on-rag-qa-arena-science-benchmark)

### 16. Technique fingerprinting: "this paper supersedes algorithm X you import"

**Verification: feasible-with-caveats** — and the open-lane claim survived adversarial search.

The JSS 2022 study of 10.3M GitHub repositories (4.8M READMEs analyzed) literally proposes this tool: *"tool support could notify developers when the SOTA in related research gets updated… cutting edge research that supersedes"* — and no shipping product does repo-fingerprint-to-paper alerting (verified by web sweep; nearest neighbors are paper-reproduction tools). Extend `source_analysis.py` from domain inference to a structured technique fingerprint, match papers against it, and use LLM claim-extraction to flag "improves on / replaces X" relationships with evidence quotes. This is the category-defining feature: it converts engineers-who-don't-read-papers into users.

**Capabilities**
- Alerts like "Paper claims 3× speedup over FAISS IVF-PQ — you import `faiss`," with the supporting abstract quote
- A technique-level relevance component far more precise than keyword overlap
- Optional GitHub-side adoption evidence (code search for the paper's arXiv ID in CITATION.cff/READMEs)
- Per-technique watchlists in `.reporadar.yml` (auto-seeded, user-editable)

**Plan**
1. Extend `source_analysis.py` to emit a structured fingerprint: imported packages + identifier extraction (both exist) + a curated technique alias table (e.g. `faiss` → ANN search, IVF-PQ) extending `PACKAGE_DOMAIN_MAP` (which lives in `profiler.py`)
2. Add `technique_match.py`: lexical alias match against title/abstract, then an `llm_client` claim-extraction pass classifying the relation (improves/replaces/extends/uses) with a quoted span
3. Store in `paper_technique_matches` (v6); surface as a "Supersedes something you use" digest section + `notify.py` channels
4. Optional GitHub code-search enrichment via `gh` (legacy search engine: default-branch only, punctuation tokenized — pair `filename:` qualifiers with client-side validation of matches; 10 req/min, top-N only, cached)
5. Feed user thumbs-up back into the per-repo alias table

**Risks**
- Claim-extraction hallucination: papers over-claim and LLMs misread comparisons — always show the quoted span, label unverified
- Alias-table curation burden; thin coverage outside ML/data-infra initially
- "Supersedes" precision/recall unproven — needs a labeled mini-benchmark before earning notification-level prominence

**Dependencies:** `source_analysis.py`/`profiler.py`; `llm_client` (feature 6); optional gh code search.
**Sources:** [Wattanakriengkrai et al., JSS 2022](https://arxiv.org/abs/2004.00199) · [GitHub CITATION.cff](https://github.blog/news-insights/company-news/enhanced-support-citations-github/)

---

## Experimental

Demonstrated only in research prototypes; ship behind flags with research-grade expectations.

### 18. Implementability & reproducibility scoring per paper

**Verification: feasible-with-caveats.**

PaperBench established rubric-decomposition for judging implementability; RECAP-style checklist pipelines (arXiv 2602.07059) extract reproducibility signals cheaply; alphaXiv is *building* (not yet shipping) implementation-ease ranking. RepoRadar can ship the cheap tier: a static per-paper score from observable facts (code released per HF linkage, repo activity, pinned deps, eval scripts, venue acceptance) plus an optional small-LLM checklist pass, rendered as an "effort to adopt" badge.

**Capabilities**
- An implementability badge ("code + pinned deps + eval scripts: low effort") on digest entries
- A ranking option prioritizing adoptable papers over merely interesting ones — the pragmatic engineer's sort order
- A code-fidelity caution flag when linked code looks divergent from the paper (advisory only — best models catch just 46.7% of real paper-code discrepancies per SciCoQA)
- Pre-filter input that makes the paper-to-branch moonshot (feature 20) safer

**Plan**
1. Add `implementability.py` computing static sub-scores from HF enrichment (feature 1), `gh api repos/...` metadata (new code — `gh_issues.py` only shells `gh issue create` today), and OpenReview acceptance (feature 9)
2. Optional LLM checklist pass (`llm_client`) over abstract/intro with a fixed rubric (method fully specified? hyperparameters given? datasets public?), yes/no/unclear per question, stored per paper
3. Persist in `paper_signals`; expose `w_implementability` via the standard ranking recipe
4. Compact badge + expandable rubric in templates and JSON/CSV
5. Calibrate against hand-labeled adoptions using `paper_ratings` as weak labels (SciCoQA's 81 real discrepancies are inspiration, not a calibration set)

**Risks**
- Static signals (stars, lockfiles) proxy quality imperfectly and are gameable
- Abstract-only LLM rubric answers are shallow; PDF-level checks need the full-text pipeline
- A wrong composite weighting quietly buries good papers — keep it a separate badge/sort before folding into default ranking

**Dependencies:** features 1, 9 (both unshipped — this stacks on them), `llm_client` (feature 6).
**Sources:** [PaperBench](https://arxiv.org/abs/2504.01848) · [SciCoQA](https://arxiv.org/abs/2601.12910) · [RECAP](https://arxiv.org/abs/2602.07059)

### 19. Research-gap radar: "nobody has applied X to your Y" alerts

**Verification: feasible-with-caveats** — precision expectations must stay research-grade.

GAPMAP (Oct 2025) showed LLMs — including Llama/Gemma via Ollama — can extract explicit and implicit knowledge gaps from papers (validated **only on biomedical literature**; CS transfer is untested), and OpenNovelty demonstrates retrieval-grounded claim comparison producing auditable novelty reports (deployed on 500+ ICLR 2026 submissions). Crossing gap statements mined from fetched papers against the repo's technique fingerprint yields a genuinely new alert type: unexplored intersections where this repo's domain meets an open problem. Note the third relevant benchmark, NovBench, finds LLMs *struggle* at novelty assessment — it supports the risk section, not the capability.

**Capabilities**
- A "Gap radar" digest section: open problems from recent papers that intersect the repo's fingerprint
- Precedent checks before alerting: cross-reference candidate gaps against the store + S2 search (batched, cached — unauthenticated S2 is unreliable per-call)
- Novelty-delta ranking: score papers by what they add relative to what the repo already implements
- Exportable "research opportunity" issues via `gh_issues.py`

**Plan**
1. Add `gap_mining.py`: `llm_client` extraction over top papers' abstracts/conclusions with a GAPMAP-style structured prompt (explicit/implicit/future-work), stored in `paper_gaps` (v6)
2. Match gap statements against the technique fingerprint (feature 16) with embedding similarity (feature 4 stack, or the in-memory embedding path as fallback)
3. Per candidate intersection, run a precedent search (S2 + local store) and attach evidence, OpenNovelty-style
4. Render as a clearly-labeled **speculative** section with source quotes; export with a `research-opportunity` label
5. Track accept/dismiss on gap alerts in a new table; use it for threshold/filter tuning (the existing `feedback.py` regression tunes ranking weights only — it cannot "tune prompts")

**Risks**
- False "nobody has done this" claims are embarrassing; GAPMAP's authors themselves recommend human verification — precedent search mitigates, cannot eliminate
- Implicit-gap extraction on small local models is noisy; acceptable precision may require the Claude API
- Several LLM calls per paper — top tier only, cached
- Value is diffuse for pure engineering repos; strongest for research-adjacent codebases

**Dependencies:** features 16 and 4 (or in-memory fallback), `llm_client` (feature 6).
**Sources:** [GAPMAP](https://arxiv.org/abs/2510.25055) · [OpenNovelty](https://arxiv.org/abs/2601.01576) · [NovBench (caution)](https://arxiv.org/abs/2604.11543)

---

## Moonshot

### 20. `rr apply`: paper-to-branch — agent-drafted implementation of a paper's technique in your repo

**Verification: feasible-with-caveats** — every benchmark citation checked out; the tier is honest.

The 2026 evidence defines a narrow but real corridor: full-paper replication is ~21% (PaperBench headline; ~26% only with 36h extended runtime), end-to-end experiments ~0.5% (EXP-Bench) — but **scoped, single-working-day technique application works**: agents beat human experts 4× on 2-hour research-engineering tasks and humans only pull ahead at 8+ hours (RE-Bench — this boundary is the eligibility gate, not optional polish), and a case study improved 11/11 published algorithm implementations in a day each (existence proof, not a rate). `rr apply <arxiv_id>` drives a coding agent with exactly the context research identifies as the bottleneck — repo profile, technique fingerprint, paper full-text — to draft a branch/PR implementing one well-specified technique. Always human-reviewed, never auto-merged.

Differentiation (corrected after verification): Paper2Agent turns papers into MCP-served agents ("apply this method to my dataset") and Paper2Code generates repos from scratch — but **no product ships the integrated flow: digest entry → implementability-gated → repo-grounded draft PR in *your existing codebase***. That narrower lane is genuinely open.

**Capabilities**
- One command from digest entry to a reviewable draft branch applying the paper's technique to this codebase
- Repo-side grounding (profile, fingerprint, conventions) supplied to the agent — the tacit-knowledge gap 2026 papers identify as the ~10% quality bottleneck
- Implementability pre-filter (feature 18): only well-specified, code-available, single-session-scoped papers get the Apply action; refuse otherwise with an explanation
- PR bodies with paper citation, claimed-vs-expected deltas, and a validation checklist; dedup via `paper_exports` (`export_type='apply_pr'`)

**Plan**
1. Add `apply.py`: assemble the apply-context bundle — paper full-text (arXiv PDF / OpenAlex), official code link (feature 1), `RepoProfile`, technique fingerprint (feature 16), target files from `source_analysis.py` import analysis
2. Drive Claude Code headless (`claude -p`) or the **Claude Agent SDK** in a fresh git worktree; hard-scope the prompt to one technique with acceptance criteria, plan-then-generate staging (Paper2Code lesson). This is a new subsystem — `llm_suggestions.py` is a one-shot helper with no tool loop
3. Use current model IDs — `claude-opus-4-8` ($5/$25 per MTok), `claude-sonnet-5` ($3/$15), or Claude Fable 5 for the hardest runs; never date-suffixed strings (the repo's old default already rotted once)
4. Gate eligibility on implementability score + a strict one-session size heuristic; create branch + draft PR via existing `gh` integration; never touch main
5. Ship an evaluation harness (ResearchCodeBench/CORE-Bench-style checks) on a curated demo set before promoting past an experimental flag
6. Collect a failure taxonomy (missing components / env misconfig / spec ambiguity, per EXP-Bench categories) to tighten the gate over time

**Risks**
- Success on scoped tasks is real but variable (expect the 20–40% scoped-benchmark range, not the 11/11 case study); users must expect **drafts, not merges** — framing failure is trust failure
- Requires Claude Code or the API — departs from the free-only posture (local models are not validated at this tier); dollars per run, needs explicit cost display + confirmation
- Security: generated/executed code from papers is a known risk (Sakana's own AI-Scientist README warns about it) — worktree isolation and no-network defaults required
- Agent harnesses churn fast; continuous re-validation as models change
- Three-deep dependency chain: full-text pipeline + features 16 and 18 must exist first

**Dependencies:** Claude Code headless / Claude Agent SDK; features 1, 16, 18; full-text pipeline (feature 12); `gh` CLI.
**Sources:** [PaperBench](https://arxiv.org/abs/2504.01848) · [EXP-Bench](https://arxiv.org/abs/2505.24785) · [RE-Bench](https://arxiv.org/abs/2411.15114) · [Paper2Code](https://arxiv.org/abs/2504.17192) · [11/11 case study](https://arxiv.org/abs/2604.13109) · [tacit-knowledge gap](https://arxiv.org/abs/2603.01801) · [Paper2Agent (adjacent)](https://arxiv.org/abs/2509.06917)

---

## Shared infrastructure (build once, used everywhere)

- **`llm_client.py`** — extract from `llm_suggestions.py` (features 6, 7, 14, 15, 16, 18, 19 all need it)
- **`pipeline.py`** — de-duplicate the collect→store→rank orchestration from `cli.update`/`watcher.py`/`workspace.py` (feature 14 prerequisite; fixes Tier-0 drift)
- **v6 schema migration batch** — `paper_llm_scores`, `paper_citations`, `paper_signals`, `paper_gaps`, `paper_technique_matches`, `metric_snapshots`, enrichment `upvotes` (the vec0 virtual table stays **outside** the migration chain — see feature 4)
- **Vector store** (feature 4) — consumed by features 7, 15, 19

## Suggested sequencing

| Version | Theme | Features |
|---------|-------|----------|
| **1.1** | Stop the bleeding | Tier 0 repairs · 1 (HF enrichment) · 12 (OpenAlex keys) · 6 step 1 (LLM wiring + model bump) |
| **1.2** | Retrieval foundation + honesty | 4 (hybrid retrieval + `rr search`) · 11 (`rr eval`) · 5 (S2 recommendations) |
| **1.3** | Reach | 3 (GitHub Action) · 2 (MCP server) · 17 (Zotero/BibTeX) · 10 (domain adapters) |
| **1.4** | Intelligence | 6 (LLM triage/rerank) · 7 (SPECTER2 + rerank) · 8 (citation alerts) · 9 (signals) |
| **2.0** | Assistant | 13 (privacy guard) · 15 (`rr ask`) · 14 (`rr deepscan`) |
| **2.x** | Experiments (flagged) | 16 (fingerprinting) · 18 (implementability) · 19 (gap radar) |
| **3.0** | Moonshot | 20 (`rr apply`) |

## Cross-cutting risks

- **Free-API fragility is the theme of 2026**: PwC died, Crossref Event Data died, OpenAlex went keyed/metered, S2 stopped issuing keys to individuals. Every integration needs graceful degradation, and the roadmap deliberately spreads across redundant sources (S2 ⇄ OpenAlex, HF ⇄ pwc-archive).
- **LLM claims need evidence anchors**: every LLM-derived assertion shown to users (suggestions, rerank justifications, supersedes-claims, gaps) must carry a quoted span and an "auto-generated" label.
- **Model-ID rot**: the retired-default-model bug will recur; prefer alias IDs, validate at startup, and surface 4xx errors instead of silent fallbacks.
- **Scope discipline**: features 14–20 are demos of agentic ambition; features 1–13 are what make the tool durably useful. Ship the boring tiers first — they are also what the experimental tiers stack on.
