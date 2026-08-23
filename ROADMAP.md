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
| **Pipeline drift** | The collect→store→rank pipeline was duplicated across `cli.update` (~700 lines inline), `watcher.py` and `workspace.py`; watch/workspace skipped the gate, the rescore, HyDE, fusion and every non-arXiv source, so `rr watch` served the **−8.12** configuration while the config said `triage.enabled: true` | ✅ **Shipped** (2026-08-16, two PRs) — `stages.py` discloses what an entry point skips, with an import-graph guard that refuses to drift in either direction; `pipeline.py` then made `rr update` and `rr watch` one implementation (`cli.update`: 728 → 84 lines). **`rr workspace update` is deliberately still separate** — one shared pool across many member repos under a single run id is a different shape, not duplicated code — and keeps the disclosure |

> PR #13 also normalized the whole repo (ran `ruff format`, cleared all `ruff` + `mypy --strict` findings) so CI is genuinely green. 519 tests pass.

### Evaluation harness (`evals/`) — added 2026-07-04

A manually-run **benchmark** (not CI) that scores RepoRadar's ranking quality on realistic repos. It has two modes: an **offline** mode over frozen, real-arXiv labeled fixtures (deterministic, no keys) and a **live** mode that clones real repos and runs the full pipeline against real sources. This is the practical, standalone counterpart to Feature 11 (`rr eval`, which scores against a user's own ratings) and de-risks every ranking change on this roadmap. See `evals/README.md`.

---

## Feature overview

**Status legend** (as of 2026-07-30): ✅ shipped · 🟡 partial (core shipped, extensions pending) · ⬜ planned.
See [Implementation status](#implementation-status-2026-07-30) below for the shipped-vs-remaining breakdown.

| # | Status | Feature | Tier | One-line impact |
|---|--------|---------|------|-----------------|
| 1 | 🟡 | Hugging Face Papers enrichment | Certainly achievable | Repairs the dead PwC integration, adds live code/model/dataset links + community-buzz signal |
| 2 | ✅ | RepoRadar MCP server | Certainly achievable | Puts repo-aware paper search inside Claude Code, Cursor, VS Code — the biggest 2026 distribution channel |
| 3 | ✅ | GitHub Action + Pages digests | Certainly achievable | Turns a single-dev CLI into team-visible infrastructure with zero hosting |
| 4 | ✅ | Hybrid retrieval core (BM25 + vectors + RRF) | Certainly achievable | Measurably better ranking, cached embeddings, and a local `rr search` over everything ever fetched |
| 5 | ✅ | Semantic Scholar learned recommendations | Certainly achievable | Turns dormant ratings/stars into a server-side learned recommender at zero local ML cost |
| 6 | ✅ | Repo-aware LLM triage & reranking | High confidence | Wires the dormant LLM path; repo-conditioned relevance judgments no embedding can express |
| 7 | 🟡 | Scientific embeddings (SPECTER2) + CPU rerank | High confidence | The 2026-grade retrieval stack: citation-trained paper vectors + cross-encoder polish |
| 8 | 🟡 | Citation alerts + citation-graph digest section | High confidence | "A new paper extends work you starred" — finally makes starring do something |
| 9 | 🟡 | Attention & integrity signals (HN, OpenReview, Retraction Watch, Bluesky) | High confidence | "Is this paper real, reviewed, and talked about?" — a trust layer no paper tool ships |
| 10 | 🟡 | Domain source adapters (IACR ePrint, bioRxiv/medRxiv, DBLP) | Certainly achievable | Serves security/bio/systems repos whose literature is *not* on arXiv — biggest unserved segment |
| 11 | ✅ | `rr eval` — recommendation-quality harness | Certainly achievable | Makes every other ranking upgrade falsifiable using ratings already collected |
| 12 | 🟡 | OpenAlex 2026 upgrade (keys, semantic search, Topics) | High confidence | Un-breaks the source; classifier-backed field watching instead of keyword guessing |
| 13 | 🟡 | Privacy guard (audit, redaction, local-only mode) | High confidence | Unlocks proprietary/enterprise codebases — currently an unexamined blocker |
| 14 | ⬜ | `rr deepscan` — agentic iterative search | Ambitious | Multi-round query-refine-expand loops, the flagship pattern of $12–20/mo commercial tools, free and repo-aware |
| 15 | ⬜ | `rr ask` — citation-grounded Q&A over your corpus | Ambitious | From alerting tool to research assistant, local-first |
| 16 | ⬜ | Technique fingerprinting ("supersedes what you import") | Ambitious | The category-defining alert: *did research just obsolete part of my codebase?* |
| 17 | ⬜ | Zotero / BibTeX bridge | Certainly achievable | Starred papers flow into the citation manager academics actually live in |
| 18 | ⬜ | Implementability & reproducibility scoring | Experimental | Answers "can I actually use this?" — a signal no free tool scores |
| 19 | ⬜ | Research-gap radar | Experimental | "Nobody has applied X to your Y" — from reading what exists to seeing what's missing |
| 20 | ⬜ | `rr apply` — paper-to-branch | Moonshot | One command from digest entry to a reviewable draft PR implementing the paper's technique |

---

## Implementation status (2026-07-30)

A benchmark-driven arc (Tier B eval → Feature 6 triage/rerank → all-time discovery → hybrid retrieval)
shipped the ranking-and-precision core. The 12-case Tier B benchmark shows RepoRadar **net-positive
and competitive with an agentic Opus 4.8 baseline** — Top Picks mean net@2 **+1.75** vs **+1.83**
(2026-07-31, a 0.08 gap, narrowed from 0.33 after the query-construction fix in PR #59). Read the
per-case table rather than the mean: the gain concentrates in the repos whose queries the fix
repaired, and `speech` regressed 10 points for reasons not yet understood — see
[`evals/RESULTS.md`](evals/RESULTS.md).

> **⚠ The binding constraint is retrieval, not ranking — and the mean hides it.** Measured 2026-08-01:
> RepoRadar fetched **2030 papers across 9 benchmark repos and reached 0 of the 24** papers the Opus
> baseline recommended *and* the judge confirmed actionable. Those two scores are near-tied while
> ranking **disjoint sets of papers**; net@2 cannot tell them apart, which is a limit of the metric
> rather than evidence of equivalence. Every ranking feature (F4 hybrid, F5 recommendations, F6
> triage, F7 SPECTER2) reorders a pool selected upstream of it, so no reranker can recover a paper
> that was never fetched. **Four candidate fixes were tested and three are negative results** —
> repo-bibliography seeding, LLM technique-phrase generation, and citation-sorted retrieval all
> recover ~0–8% on their own. Read
> [Candidate-pool diagnosis](evals/RESULTS.md#candidate-pool-diagnosis--what-reporadar-cannot-reach-and-why-2026-08-01)
> before starting retrieval work; it exists so the dead ends are not paid for twice.

**✅ Shipped**
- **Tier 0 repairs** (PR #13): HF Papers (dead PwC), OpenAlex key support, retired model default, wired the
  LLM path, `--since` filter, CI, run-ordering fix. *(Pipeline-drift refactor still deferred → Feature 14.)*
- **Feature 6 — repo-aware LLM triage & reranking**: `triage.py` (0–3 actionability), shared `llm_client.py`,
  `TriageConfig`, store v7 `paper_llm_scores`, digest gating (abstains unless genuinely applicable), and
  **listwise rerank** by `llm_score`. The benchmark validated `min_actionable=2` as the default.
- **Feature 4 — hybrid retrieval + local corpus search** (PRs #37, #43, + sqlite-vec cache): BM25+RRF fusion
  in the ranker (`ranking.hybrid`, store v8 `rrf_score`); **`rr search`** free-text BM25 over the whole corpus
  + a `search_papers` MCP tool; a **persistent per-paper embedding cache** (store v9, compute-once instead of
  per-run) that also powers **`rr search --semantic/--hybrid`**, KNN-accelerated by the optional `sqlite-vec`
  extra (numpy fallback).
- **Foundational / seed-corpus discovery** (`rr update --foundational`, PR #36) — the eval-validated all-time,
  relevance-first sweep that surfaces seminal work the recent window misses. (Realizes the "seed corpus"
  idea from Finding #2; closes most of the baseline's remaining benchmark edge.)
- **Two-tier evaluation harness** (`evals/`) — Tier A offline fixtures + **Tier B LLM-judged actionable-
  improvement benchmark** (12 cases, GPT-5.5 judge, Opus baseline). The standalone counterpart to Feature 11.
- **Feature 2 — RepoRadar MCP server** (PR #40): `rr mcp` (stdio) exposes `get_repo_profile`,
  `get_ranked_papers`, `explain_relevance`, `rate_paper` to coding agents; optional `[mcp]` extra.
  *(Remaining: MCP-registry publish + a Claude Code plugin — the distribution half.)*
- **Feature 3 — GitHub Action + Pages** (PR #41, released as `v1`): composite `action.yml` + `rr archive`
  publish a dated, ranked digest to GitHub Pages; a rendered HTML digest (replacing markdown-in-`<pre>`)
  and `${ENV}` config expansion landed with it.

- **Feature 5 — learned recommendations** (`sources/s2_recommendations.py`, opt-in `recommendations` config):
  starred/highly-rated papers seed the free Semantic Scholar recommender (low-rated ones become negative
  examples); results are merged as `matched_query="recommendation"` and **re-ranked locally**, and the digest's
  "Recommended for You" prefers them over the keyword recommender.

- **Feature 11 — `rr eval`** (`evaluation.py`, store v14 `metric_snapshots`): scores the ranker against your own
  ratings, with `--compare A.yml B.yml` re-scoring identical papers under two configs and bounding the
  difference by a bootstrap interval — so a change that the data cannot justify reads "NOT SHOWN" instead of
  looking like an improvement. Unjudged papers are removed rather than counted irrelevant, recency is scored as
  of each paper's `first_seen`, and the selection bias in rating-derived labels is printed on every run.

**🟡 Partial**
- **Feature 9 — attention & integrity signals**: a new `signals/` package. `signals/integrity.py` flags papers
  withdrawn by their authors (100% recall on notices phrased "withdrawn", 83-85% on "withdrew"/"retracted",
  no confirmed false positive over 600 ordinary papers) and applies a
  hard multiplicative `withdrawn_penalty`, so a withdrawn paper cannot reach Top Picks on other strengths;
  `signals/hn.py` badges Hacker News discussion behind an opt-in `w_attention`. Store v13 (`paper_signals` +
  `attention_score`). **Two of the four planned signals were disproven, not deferred for effort:** OpenReview's
  per-paper endpoint is 403-challenge-gated for anonymous clients (and 0.5% coverage of a 14-day run even via a
  local index), and Crossref indexes zero arXiv works, so the Retraction Watch layer cannot apply. Bluesky works
  keyless only through an undocumented host that bypasses a deliberate block. *Remaining:* the S2
  `publicationVenue` annotation, storing the arXiv comment at collect time, trends exclusion.
- **Feature 1** — HF Papers enrichment shipped, plus the `w_community` ranking component (opt-in, store v12
  `community_score`) scoring log-normalized upvotes from *previously cached* enrichments, since enrichment
  runs after ranking. *Remaining:* the `pwc-archive` offline fallback.
- **Feature 8 — citation alerts** ("extends work you starred"): store v10 `paper_citations`, a Semantic Scholar
  references fetch (`citations.fetch_references`), `citation_graph` (seed set from stars/high ratings +
  link-finding), a `w_citation_proximity` ranking boost, a digest "Extends work you starred" section +
  badge, and the count in `rr notify` (message tail + `RR_EXTENDS_STARRED_COUNT`), so the alert is pushed
  rather than only rendered. *Remaining:* the OpenAlex `cites:` fallback, a co-citation-graph SVG, and
  one-hop citation expansion.
- **Feature 10 — domain source adapters**: `sources/dblp.py` (keyword-search JSON, systems/PL/DB, title-only),
  `sources/biorxiv.py` (date-interval listing + local query filter, biology) and `sources/iacr.py` (ePrint
  HTML search, cryptography) — opt in via `sources:`, merged with arXiv priority. `sources/suggest.py` reads
  the repo profile and **suggests** a matching source in `rr profile` / `rr update`.
  Until 2026-08-12 none was actually functional: every non-arXiv source received arXiv boolean syntax
  instead of keywords (C-9), which DBLP answers with nothing and bioRxiv answers with its entire recent
  window. **Two are now validated end to end against a judge** (`evals/RESEARCH-scientific-software.md`
  §21, §39): `sources/europepmc.py` supplies **4.8 of the ~9.7 papers shown per case** over six biology
  repositories at a precision at or above the arXiv papers beside it under two judges, and
  `sources/openalex.py` supplies **1.83 per case** over six materials-science repositories — real journal
  literature (*Nature Machine Intelligence*, *Computer Physics Communications*, *npj Computational
  Materials*) — at a precision statistically indistinguishable from them. Neither is shown to *raise*
  digest quality: a source **displaces** 24–44% of the previous Top Picks rather than adding to them, and
  the net effect is unresolved at six cases. **DBLP, Semantic Scholar and IACR remain unvalidated**; IACR's
  only measurement is a two-case null too small to resolve a plausible effect. *Remaining:* re-measure
  DBLP now that it receives real queries, DOI abstract-backfill for DBLP, medRxiv/category config, and the
  half of the cross-scheme duplicate that OpenAlex's `locations` cannot close (§40.2).
- **Feature 7 — SPECTER2 similarity** (`specter.py`, `ranking.w_specter`): citation-trained vectors served free
  by S2, cached under a distinct `specter_v2` key in `paper_embeddings`, queried by the centroid of your
  starred/highly-rated papers (no local model), pool-normalized, persisted via store v11 `specter_score`.
  *Remaining:* the CPU cross-encoder rerank, local SPECTER2 for a profile query, and HyDE.
- **Feature 12** — OpenAlex `api_key` groundwork shipped; semantic search, Topics, and full-text are not.
- **Feature 13 — privacy guard** (PR #57): `privacy.py` (an enforced destination registry) and `rr audit`
  (`--json`) print every outbound destination and the exact query strings a config would transmit, sending
  nothing to find out; `privacy.redact` strips terms at two choke points — `build_queries` and
  `llm_client.complete` — wired via a config mirror so no call site can bypass it. *Remaining:* layer 3,
  `--local-only` against a bulk arXiv snapshot, deliberately deferred (see §13 for why).

**⬜ Not started**
- Features 14, 15, 16, 17, 18, 19, 20. (Feature 13 shipped its first two layers in PR #57 and is
  listed under 🟡 above; `--local-only` remains.)

---

## Research plan — derived from RESEARCH.md (2026-08-04)

An ordered set of experiments, not features. Each was proposed from the measured record in
[`RESEARCH.md`](RESEARCH.md), adversarially checked against that document's own negative
results (so nothing already-falsified is re-proposed), and carries a **prediction stated in
advance** and a **kill condition** — the two things RESEARCH.md §6.7 says made its own
falsified predictions more valuable than its confirmations. Costs are per experiment, not
per feature; almost everything runs offline against artifacts that already exist (the 602
labelled papers, the 18/24 known-good target list, the citation-hop machinery).

**The bet, in one paragraph.** The citation hop is the discovery engine — it is the only
channel with measured recall (18/24 where every other is 0–3/24) and it dies of density
(1:5,111). So the plan industrializes that single channel: persist the pool once with
per-candidate structure (P1), cut it structurally (P1), select from it semantically (P2),
extend it to the repos it structurally misses (P3–P4), and buy the labels that give the
cascade a precision measure it has never had (P5). Two experiments protect the epistemics
(P6–P7), because every correction in this project's history came from checking rather than
assuming. Gating — already at its document ceiling — gets exactly one direction (P8), the
only one that adds information the documents don't contain. Expected end state after ~2
weeks of experiments: a measured pool → filter → matcher → gate cascade at ~100–250
papers/repo with known recall (≥15/18) and a first-ever precision estimate.

**Evidence from running RepoRadar on itself (2026-08-04, ~$0.03).** Pointed at this
repository (foundational sweep, Haiku triage), RepoRadar fetched 171 papers from its
self-derived queries (`all:"papers mcp"`, `all:pytest`, …) and admitted 8. Three are
genuinely applicable — *vstash* (adaptive RRF fusion weighting; `retrieval.py` uses
fixed-k RRF), *Beyond Paper-to-Paper* (paper–reviewer matching, structurally the
repo–paper matching problem), *Method and Dataset Mining in Scientific Papers* (feeds
feature 16). Five are topical echoes of the keyword `mcp`. And the two papers most
relevant to this plan — *Discovering seminal works with marker papers* (co-citation
discovery from seeds, i.e. the citation-hop mechanism) and *Ranking Papers by their
Short-Term Scientific Impact* (the pool-selection problem) — were both **rejected by the
gate at score 1**, with reasons amounting to "proposes something different from what the
repo currently does". The system reproduced RESEARCH.md §1's register mismatch on itself
in one run: the gate reads "would improve this repo" as "resembles this repo". P8 and the
rubric's treatment of method-divergence are the direct response.

### P1. Persist the citation-hop pool with direction-aware coupling degrees; sweep the coupling filter offline

**Grounding:** §3.5 — the hop is the only channel with recall, unshipped at 92,014
candidates. RETRIEVAL_DESIGN Design 1's filter numbers (cv 14,867→2,115 keeping 3/3) are
REPORTED against a pool that matches neither the buggy nor the corrected state — treat as
hypothesis. Coupling degree is used as a **threshold only, never a sort** (the measured
ancestry warning). The backward direction provably contributes uniquely (Soft-NMS is
backward-only), so a forward-only filter is measurably lossy.

**Experiment:** two-pass fetch (lean ids+degrees first, then title/abstract/year/citations
at 500/chunk — the one-pass version would hit S2's ~10MB response cap *below* the existing
item-count truncation guard, the §6.5 failure class). Persist to
`evals/.work/hop_pool/<case>.jsonl`; leave-one-case-out sweep over forward-degree floor ×
backward band × cross-repo document frequency × citation floor, scored on retention of the
18 known-goods vs shrinkage. **$0, no LLM calls, 2–3 days.**

> **The target list is derived, never read from a file.**
> `evals/diagnose_pool.py::actionable_baseline_ids(case)` intersects the cached baseline
> picks with judge score ≥ 2 and yields **24** ids (18 of them inside the hop's reach). A
> stray `baseline_ids.json` in the repo root held **28** — it had skipped the judge filter
> and carried 4 phantom `webdev` "targets" on a negative control with zero actionable
> papers, plus one non-arXiv id. Scoring recall against it would have inflated the
> denominator by 17%. It is deleted; call the function.

**Prediction:** some threshold retains ≥15/18 while cutting the mean seeded-case pool
(~13,145) by ≥75%. **Kill:** if ≥5 of 18 targets have forward AND backward degree ≤1,
coupling cannot beat the noise floor at any threshold — close Design 1 as a corrected-pool
negative (the hop itself survives; its 18/24 is independent of any filter).

**Overlap:** this *is* feature 8's remaining "one-hop expansion", promoted to primary
discovery channel; feature 14 inherits the funnel. The artifact is the substrate for P2
and P5. Ships under `--foundational` only (§3.7).

### P2. Gap-phrase matching against the hop pool — dense + stemmed BM25, with a seed-centroid control

**Grounding:** §3.2 — the "lacks" prompt *aims correctly* and fails on phrasing (83%
zero-hit lexically); stemming and dense vectors collapse exactly that failure. Runs against
P1's pool because §3.5 says the pool contains the answers and §4.1 says the heuristic
ranker cannot select from it. The **seed-centroid control arm** matters as much as the
treatment: Design 3's REPORTED negative ("similarity surfaces ancestry, similarity is the
wrong relation") has never been MEASURED — this adjudicates it either way.

**Experiment:** embed the pool via the shipped MiniLM cache (CPU, offline); stemmed BM25
(one new small dependency — the shipped BM25 is unstemmed); regenerate "lacks" phrases
(~$0.01, cached) plus the summarizer's cached `improvement_areas` as a second phrase
source (§8.2, never tested on search). Score union-of-phrases top-200/repo: dense, BM25,
RRF-fused, and the seed-centroid control. **~$0.05, 2–3 days, overlaps P1's network waits.**

**Prediction:** ≥8/18 known-goods in fused top-200/repo (chance ≈ 0.4 — a ~20× lift);
control ≤4/18. Symmetric outcome: if the *control* wins, Design 3's negative was wrong and
plain similarity suffices — ship the simpler thing. **Kill:** all arms ≤4/18 even at
top-500 and after escalating to local SPECTER2 — close §8 items 2/4 as measured negatives.

### P3. Synthetic hop seeds from "uses"-phrase search — extend the hop to crypto, systems, diffusion

**Grounding:** crypto/systems are structural zeros (no arXiv-indexed bibliography) and
diffusion recovered 0/2 — five of the six hop misses. §3.2 measured "uses" phrases as
*accurate about the repo and mostly matching real papers*; the hop needs only accurate
anchors in paper space. Not contradicted by §3.2's 2/24 — that measured phrases as direct
target retrieval; here they retrieve *seeds* whose neighbourhoods are hopped. Untested.

**Experiment:** for the three repos: "uses" phrases → one quoted arXiv search each (≤10/repo,
PR #62 quoting fix, nowhere near the §3.4 rate-limit regime) → top-20 per phrase, cap 40
seeds → existing truncation-guarded hop → measure which of the 5 missing targets enter the
pool. Diffusion's synthetic pool measured alone, not unioned with its real seeds, for clean
attribution. **~$0.02, 1–2 days, parallel with P1.**

**Prediction:** ≥3/5 missing targets enter at ≤40k/repo. **Kill:** ≤1/5, or pools blow past
~50k from hub seeds — synthetic seeding is noise amplification; crypto/systems then depend
on P4 or feature 10's non-arXiv adapters.

> #### Scope and prediction RESTATED before running (2026-08-05)
>
> The 22-case re-run changed what P3 is for. It was written against 5 missing targets in 3
> repos; the hop's reach is now measured at **21/48 = 44%**, leaving **27 unreached targets
> across 12 cases** — and P3 is the only planned direction that addresses them:
>
> | group | cases | unreached |
> |---|---|---|
> | no bibliography at all (structural zero) | crypto, systems, storage, compiler, vectordb, columnar | **15** |
> | bibliography too thin (1–3 seeds) | db, numerics, llminfer, ann, speech | **10** |
> | bibliography present but missed | diffusion | **2** |
>
> **Restated prediction:** synthetic seeding puts **≥8 of 27 (30%)** unreached targets into a
> pool of **≤50k candidates per repo**. Two reference points bracket that number: a repo's
> *real* bibliography with ≥7 seeds reaches **89%**, and the same "uses" phrases used as
> *direct* arXiv queries reached **8%** (§3.2). P3 claims the mechanism is the hop, not the
> phrases, so it should land nearer the former than the latter.
>
> **Restated kill:** **≤2 of 27 (7%)** — indistinguishable from the direct-query baseline, so
> the neighbourhood hop adds nothing over the phrases themselves. Or any pool >50k/repo, which
> is noise amplification regardless of recall.
>
> The original 3/5 bar is kept above as the record of what was pre-registered when P3 was
> written; the run reports both.

### P4. Verify Design 2's four load-bearing dependencies ($0, half a day), then blind HyDE replication

**Grounding:** RETRIEVAL_DESIGN Design 2 is the largest single recall candidate (REPORTED:
HyDE-4 8/24 top-100 vs 1/24 for TF-IDF; covers crypto 2/2 and systems 1/1 — precisely the
hop's structural zeros; additive with the hop, union ≥17/24) and **every dependency is
explicitly unverified** — in a project where one REPORTED estimate was off 10× and another
measured a transport bug. Verification precedes any build.

**Experiment:** stage 1, $0: does the HF arXiv-embeddings dataset exist under the stated
licence; does columnar range-fetch work (pull one shard); is query latency ≤4× the reported
1.87s (4–10× = `--foundational` batch only; >10× = kill); are all 24 targets in the index.
Stage 2 only if 4/4 pass: blind HyDE-4 replication across **all nine** seeded cases
(hypotheses from repo profiles alone, ~$0.05, KNN via the shipped sqlite-vec path).

**Prediction:** stage 1 passes 4/4; stage 2 ≥8/24 in top-1k, median rank <5,000, crypto 2/2.
**Kill:** any stage-1 check fails — Design 2 dies at zero build cost (fallback dense
channel: feature 12's OpenAlex semantic search). Stage 2 ≤5/24 — the REPORTED numbers were
inflated; P1–P3 proceed alone. **Overlap:** feature 7's HyDE item; feature 13's
`--local-only` by another route (discovery would transmit nothing).

> #### RESULT (2026-08-06) — stage 1 4/4, stage 2 met on aggregate. **The first technique measured that beats the citation hop.**
>
> Scope restated before running, as with P3: P4 was written against 24 targets, the benchmark
> now holds **48**, so the gates were scaled by the same fractions — ≥16/48 in top-1k, kill at
> ≤10/48. `crypto 2/2` was kept verbatim.
>
> **Stage 1, $0** (`evals/verify_hyde_deps.py`): the dataset is
> `bluuebunny/arxiv_abstract_embedding_mxbai_large_v1_milvus_binary` — apache-2.0,
> **3,106,925 rows**, binary vector column. Columnar range-fetch is real (**15.9%** of a
> shard for `id`+`vector`). Latency **1.21 s**, *faster* than the reported 1.87 s. **48/48**
> targets present. Two corrections: the REPORTED "~370 MB sync" is 432 MB and only with
> column pruning (the full dataset is 2,542 MB), and each shard holds one row group, so C2
> measured column pruning rather than row-group pruning.
>
> **The check this plan did not think to name.** Nothing in C1–C4 establishes that a vector
> *we* compute is comparable to the ones in the index. Stage 2 now refuses to run until it
> reproduces stored vectors bit-for-bit — mxbai-embed-large-v1 over the abstract alone,
> normalised, binarised at >0: **Hamming 0/1024**. Had the publisher embedded title+abstract,
> every number below would have been noise that looked healthy.
>
> **Stage 2, ~$0.20** (`evals/hyde_replication.py`), blind, 48 targets, 17 cases:
>
> | arm | top-100 | top-1k | median rank | within 4k |
> |---|---|---|---|---|
> | hyde4-union | 5/48 | **27/48** | **837** | **42/48** |
> | hyde4-centroid | 4/48 | 17/48 | 2,805 | 27/48 |
> | hyde1 | 2/48 | 12/48 | 4,317 | 23/48 |
> | readme | 7/48 | 10/48 | 46,656 | 12/48 |
> | keywords | 0/48 | 3/48 | 32,582 | 9/48 |
>
> **The decisive number is the overlap, not the recall.** The hop reaches 21/48, HyDE 27/48,
> the **union 36/48 (75%)** — and **15 targets are HyDE-only**, i.e. structurally unreachable
> by any bibliography. Design 2's REPORTED union was ≥71%; measured on twice the targets, 75%.
> Density too: the hop's pools are 1 good paper per 5,224 candidates, HyDE-centroid 1 per
> 1,000.
>
> **What did not replicate:** `crypto` 1/2 and `systems` 0/1 — the specific two-repo claim
> that motivated Design 2 fails at the repo level, and the run script reports it separately
> so the aggregate cannot absorb it. `readme` is bimodal: 7/48 in the top 100, median 46,656.
>
> **What this does not settle:** every candidate still terminates in triage, which collapses
> on the ranker's top-10. P4 raises the ceiling; it does not deliver a better digest. The
> next question is P5's — what a filter over this pool costs in precision. Nothing ships from
> P4 yet; `--foundational` gating and the 432 MB sync are a separate build decision.

### P5. Label the pool where the filter will live — stratified judge sample + shipped-gate admit rates (~$10)

**Grounding:** the 602 labels cover only ranker/baseline-surfaced papers — any hop-pool
filter has recall targets but **no precision measure**. §4.1 is the precedent: ~$5 of
stratified labels converted the ranker from unmeasured to measured. Base-rate facts
already settled and reused: all 18 hop-reachable targets pass the shipped gate at
`min>=2`, only 6/18 at `min>=3` (so a strictness-only cascade is foreclosed); labelled
false-positive rate of the shipped prose-300 gate is 2.3% at `min>=2`.

**Experiment:** from P1's artifact: 4 free-feature strata per repo (forward degree,
backward membership, citation band, age band) × 10 papers × 7 repos = 280 wild candidates;
run **both** the GPT-5.5 judge (~$9; verdicts enter the shared cache and grow the labelled
set) and the shipped Haiku gate (~$0.05) over the same 280. Yields per-stratum actionable
base rates, gate wild-admit rates, and gate-vs-judge agreement off-distribution.

**Prediction:** top stratum ≥8% actionable vs bottom ≤1% (≥6× separation, non-overlapping
CIs); gate wild-admit ≤10%. Cascade arithmetic then closes: filter (≥4×) → gate (~10×) ≈
100–250 papers/repo at ~$0.20–0.35 of Haiku, keeping ≥15/18. **Kill:** all strata ≤2% with
overlapping CIs — sampled labelling cannot see filter quality at this density. Independently:
gate wild-admit >20% — the terminal stage must be redesigned before any cascade ships.

> #### RESULT (2026-08-06) — separation MET at the strict bar; the gate kill fired with its premise refuted
>
> Design restated before running (`evals/label_pool.py`): P5 predates P4, so its four hop-only
> strata became **six** — three rank bands of the fused HyDE ranking, two coupling bands of the
> hop pool, and a uniform draw from all 3.1M arXiv papers as the floor. The floor arm is the
> one that makes the rest readable; its absence is what made the `lacks` numbers unreadable in
> §7. 1,200 papers gated, 320 judged, **none of them a gold target** — the labels are fresh.
>
> | stratum | gate admits | judge ≥2 | judge =3 |
> |---|---|---|---|
> | `hyde-top100` | 38.0% | **58.0%** | **9.0%** |
> | `hyde-100-1k` | 24.5% | 43.3% | 0.0% |
> | `hyde-1k-10k` | 13.0% | 13.3% | 0.0% |
> | `hop-coupling3+` | 43.5% | **66.7%** | **26.7%** |
> | `hop-coupling1` | 13.0% | 33.3% | 0.0% |
> | `random-arxiv` | **0.0%** | 2.0% | 0.0% |
>
> **Separation:** 58% vs 2%, a 29× effect at p < 0.001. At the **≥3 bar — where the 48 gold
> targets were drawn** — it is 9.0% vs 0.0% (p = 0.0032) and all three pre-registered clauses
> are met, almost exactly on the predicted ≥8% vs ≤1%. At the ≥2 bar the floor clause misses by
> a single paper; the report prints the clauses separately so one miss cannot outvote the effect.
>
> **Gate:** the >20% wild-admit kill fires at 24.5% — but its premise does not hold. Precision
> is **0.97** (2 false positives in 66 admits), recall **0.60** (43 rejected actionable papers),
> and the band it fired on is 43% actionable, *above* its admit rate. The gate rejects 200/200
> random arXiv papers. It is not admitting junk; it is dropping actionable work. **Read the kill
> as aimed at recall** — and note that tightening the gate, the move every previous run
> suggested, now buys nothing and costs recall.
>
> **What this changes upstream.** "1 good paper per 5,224 candidates" measured distance to a
> known gold target, not the density of useful papers. By the judge's own bar the same pool's
> high-coupling band is 67% actionable. Every recall framing in RESULTS.md needs reading in
> that light.
>
> **Where the cascade breaks:** the gate is affordable everywhere (~$0.23/repo/run at 1,000
> candidates) and sufficient nowhere — the tightest operating point still admits ~38 papers for
> a ~10-paper digest. The missing stage is **ranking within the admitted set**, not a stricter
> gate. That is the 2026-07-06 all-time run's conclusion, now with numbers.

### P6. Temporal adoption ground truth — papers a repo actually cited later, mined from git history

**Grounding:** everything measured is agreement-with-GPT-5.5, and the 24 recall targets are
themselves Opus picks — doubly circular (§2, §8). §3.1's "a bibliography is a well-targeted
index of what the repo already does" supplies the contrapositive: an arXiv id present in
docs at HEAD but absent ~24 months earlier is a technique the project **actually adopted**
— a model-free actionability label as-of-T0. The judge-validity phase is the single
highest-value test in this plan: show GPT-5.5 the T0 repo and the papers it verifiably
adopted afterwards; if the judge doesn't call them actionable, every number downstream of
the judge inherits that.

**Experiment:** separate full-history clones (the `evals/.work` clones are depth-1 and
their state gates the verdict cache — never check out in them); `git log -p` over doc
paths for arXiv-id deltas; drop self-citations and papers <6 months old at citing time.
Structural zeros pre-registered: crypto, systems, cli, http, webdev. **$0, half a day** for
the count; 2–3 days for the retro-benchmark (T0 re-profile → T0 hop → judge pass).

**Prediction:** ≥30 usable adoptions across ≥6 of 7 arXiv-rich repos; T0 hop reaches ≥60%
of them; judge scores ≥70% actionable against the T0 repo — **if <40%, the judge is not
measuring the product's goal.** **Kill:** <10 usable adoptions or >80% self-citations —
ground truth must come from CHANGELOG/PR mining or new citation-rich benchmark cases.

> #### RESULT (2026-08-06) — the judge is not invalidated; 2 of 3 pre-registered bars met
>
> Scope restated before running: P6 was written for 12 cases and named 7 arXiv-rich repos.
> The arXiv-rich set is now measured rather than assumed. Clones are **blobless and separate**
> (`.work/fullclone/`, `--filter=blob:none --no-checkout`) — the `.work/<case>` clones gate the
> verdict cache and checking out an old commit in one would silently re-key every verdict.
>
> | | predicted | measured | |
> |---|---|---|---|
> | usable adoptions | ≥30 across ≥6 repos | **31 across 6** | MET |
> | retro-recall (T0 hop) | ≥60% | **21/31 = 68%** | MET |
> | judge calls them actionable | ≥70% | **19/31 = 61%** | below |
> | — the invalidation bar | <40% ⇒ judge invalid | **61%** | **NOT invalidated** |
>
> **The question P6 exists to ask is answered: the judge is measuring approximately the right
> thing.** It scores 61% of verifiably-adopted papers actionable against a 2% random-arXiv
> floor (P5). And retro-recall at 68% is the only recall figure in the project whose targets
> no model chose — higher than the 44% the hop reaches against the Opus-derived gold set.
>
> **The instrument has a noise direction nobody predicted.** All 12 judge misses were traced
> to the file they live in: `rl`'s 2 are on `docs/misc/projects.md`, a list of **downstream
> users** of stable-baselines3; all 5 of `graph`'s are background citations in one tutorial;
> one is a **broken arXiv link in the diffusers docs**. Only 4 are genuine disagreements. On
> the 23 verified adoptions the judge scores **83%** and retro-recall is **74%** — post-hoc,
> reported as such, and each exclusion checkable by opening the file.
>
> **Named next refinement:** a reverse-citation filter on paths matching
> `projects|showcase|used[-_ ]by`, the same shape as the existing self-citation filter. It
> would have removed `rl` entirely — and with it, one of the 6 repos, so the stricter labels
> would miss the yield bar. That trade is the real state of this ground truth.

### P7. Label-noise floor — second-judge kappa over 200 of the 602 labels

**Grounding:** every labelled-set decision, including shipping prose-300, rests on
single-sample GPT-5.5 verdicts deciding ±10-to-±21 differences, and nothing bounds their
noise. §5.2 varied the *gate* model, never the judge. Kappa bounds noise only, not
validity (two LLMs can share a famous-technique halo) — P6 is the validity test; they
compose.

**Experiment:** re-judge 200 (stratified by case and verdict) with Sonnet via the existing
`llm_client`, byte-identical rubric, inputs reconstructed and verified against the stored
`_prompt_hash`. Reweighted kappa on the actionable cut; re-score the +22 prose-300 headline
under second-judge labels using the cached per-paper gate verdicts. **~$2–5, 1 day.**

**Prediction:** kappa ≥0.6 and the +22 keeps its sign at ≥half magnitude. **Kill:** kappa
<0.4 — label noise swamps what the instrument decides; every labelled-set conclusion needs
adjudicated labels or noise-adjusted CIs before another arm is run.

> #### RESULT (2026-08-06) — kappa 0.51, below prediction, above the kill bar. The disagreement is a strictness offset.
>
> 200 labels re-judged by Sonnet, byte-identical rubric, stratified by case and verdict. **All
> 12 cases reproduced their stored `_prompt_hash` before any call was made**, so both judges
> answered the same question.
>
> | | |
> |---|---|
> | Cohen's kappa (≥2 cut) | **0.507** — below the ≥0.60 prediction, above the 0.40 kill |
> | quadratic-weighted kappa (0–3) | **0.727** |
> | base rate GPT-5.5 / Sonnet | **40% / 22%** |
> | prose300 − keywords on the 200 | **+2** under GPT labels, **+20** under Sonnet's |
>
> **The confusion matrix is almost entirely on or one step below the diagonal**, and GPT's 0s
> are Sonnet's 0s 58 times out of 58. Moving only the second judge's cut to ≥1 lifts kappa to
> **0.711 at 86% agreement**. So the two judges *rank* papers the same way and sit at different
> strictness — which decides the remedy: not label adjudication, but the recognition that a
> **paired difference between arms scored by the same judge largely cancels the offset**. That
> is the shape of nearly every conclusion in RESULTS.md.
>
> **What it changes upstream:** relative comparisons hold, absolute levels are judge-specific.
> At Sonnet's strictness P5's "58% of the HyDE top-100 actionable" reads ~32% and P6's "61% of
> adoptions" ~34%; the separations against their floors survive intact. **The prediction was
> missed, not met** — the labelled set is a noisier instrument than P7 assumed.
>
> The +22 test passes but **weakly**: the GPT-labelled delta on this verdict-stratified subset
> is only +2, so "≥half of +2" is nearly vacuous. The directional result is the strong one —
> under a stricter judge the prose-300 advantage is larger, not smaller.

### P8. Verbatim issue-tracker wants as gate context — the single gating bet

**Grounding:** the five purpose-statement arms converge at +85..+95; the rubric's score-3
band demands "directly addresses a known limitation" evidence **no document-derived arm
supplies** — a repo's issue tracker states that evidence verbatim, automatically. Correctly
distinct from the failed `improvement_areas` arm (+70): those were LLM-inferred *and
paraphrased*, and the supported diagnosis is paraphrase-vocabulary loss (verbatim beat
paraphrase by +21). Verbatim, externally-sourced want-statements are an untested cell. The
self-run's failure mode — rejecting the marker-papers and impact-ranking papers for
*differing* from current code — is exactly what stated wants would correct.

**Experiment:** top ~15 open issues by reactions for the 12 repos (GitHub REST, free);
append verbatim titles to the shipped prose-300 prompt as a new `--repo-context` arm; one
602-paper run (~$0.10); paired bootstrap vs prose-300, decision pre-registered on the CI.

**Prediction:** breaks out of the band — net@2 ≥+105 with CI excluding 0; `graph` recall
0.36 → ≥0.50; negative controls stay clean (their trackers are full of feature requests,
making them a sharp control). **Kill:** lands inside the band — then information *type* is
not the gate's constraint, and that negative directly calibrates item 0's triage arm
before item 0 spends on hand-authored goals.

> #### RESULT (2026-08-07) — KILLED. The worst arm ever measured, and the mechanism is the opposite of the one predicted.
>
> Top 15 open issues by reactions, verbatim, appended to the shipped prose-300 prompt. Paired
> on the same 602 papers:
>
> | | precision | recall | net@2 |
> |---|---|---|---|
> | `prose 300` | 0.92 | **0.68** | **+95** |
> | `wants` | 0.92 | **0.41** | **+57** |
> | delta | +0.00 | −0.27 | **−38**, CI **[−55, −21]**, P(Δ≤0) = 1.000 |
>
> 7 fixed, 49 broke. It is below the `keywords` control too (+73 → +57). The kill said "lands
> inside the +85..+95 band"; it landed **below every arm in the study**.
>
> **Precision is untouched and recall collapses**, so the block does not fool the gate — it
> makes it reject work it used to accept. Not because trackers are full of engineering
> requests (`peft` and `diffusion` have the most research-flavoured trackers and took the
> worst damage) but because **15 named wants replace the question**: the gate stops asking
> "would this improve the project" and starts asking "is this on the list". Damage scales with
> how much there was to lose — base rate vs recall loss correlates at **r = −0.61** across 10
> cases. `speech` surfaced zero issues, so its prompt is identical to prose-300's, and it is
> the one case that did not move (−0.05, one paper).
>
> **What it calibrates.** P8 was the single gating bet, and it says information *type* is not
> the gate's constraint. Per its own terms this "directly calibrates item 0's triage arm
> before item 0 spends on hand-authored goals" — a user stating what they want, fed to the
> **gate**, should be expected to narrow it the same way. Combined with P4/P5, the direction
> is: **stated wants belong in the query, not in the gate.**

### P9. An "extend this project" mode — NOT justified yet; the instrument exists

**Status: proposed, tested, and the test came back negative.** Recorded so the idea is not
re-proposed without new evidence, and so the instrument built for it is not rebuilt.

The proposal was that the "lacks" prompt is not failing — it finds papers that would *extend*
a repo, while the whole benchmark is built on a judge that asks whether a paper would
*improve* one ("directly addresses a known limitation or core capability"). Those two
hypotheses predict identical numbers in
[Negative result 7](../evals/RESULTS.md), so it was a real gap in that conclusion.

**Measured** (`evals/extend_vs_improve.py`, ~$3): no dissociation. Under a rubric written to
reward new capability and cap refinements at 1, `lacks` papers score **1.75 → 1.38**, i.e.
*lower*, and targets **2.75 → 2.00**. The rubric is not broken — it separates targets (2.00)
from a random pool sample (1.18). `lacks` retrievals are simply not extensions.

A second reading, that 68% of `lacks` papers scoring ≥2 meant the gold set was incomplete,
was also mostly refuted: **50% of uniformly random hop-pool papers score ≥2**. `lacks` beats
random by +0.38 (permutation p = 0.015) — real but modest, against targets' +1.38.

**What to keep:**

- `EXTEND_RUBRIC` in `evals/extend_vs_improve.py` is a working instrument. If extension
  discovery is ever wanted, that is how to measure it — and it would need **its own gold
  set**, since the current one was built end-to-end by an improvement judge and cannot score
  an extension channel fairly.
- The random-control pattern. Without it, "68% are actionable" reads as a strong result; with
  it, most of that is the judge's ≥2 bar being permissive on topically-adjacent papers.

**What would revive this:** evidence from real maintainers that they want capability
extensions more than component improvements — i.e. a preference signal from outside the
judge. That is a user-research question, not a retrieval one, and it sits closer to item 0
(user-stated goals) than to the retrieval plan.

### Dropped, with reasons (so they are not re-proposed)

- **Multi-sample gate voting** — its supporting evidence misattributed a variance source.
- **Ancestry-contrast ranking, listwise gating** — undecidable n at the current label count.
- **Thin-docs benchmark ablation** — the existing arms already bracket the answer; a real
  thin-docs *case* (RESEARCH.md §8.6) remains worth adding when benchmark cases are next
  revised.
- **Abstract-budget sweep (`abstract[:1500]`)** — real but third-order once the cascade
  shrinks the pool ~50× before the gate.

## Certainly achievable

Proven technology, verified-live dependencies, clear path. Days-to-weeks each.

### 0. Let the user say what they want to improve

**Verification: not started.** Proposed 2026-08-02, and promoted here because four
document-derived alternatives were measured and all hit the same ceiling.

Everything the gate knows about a repository today is *inferred from its documentation*:
extracted keywords, a README prefix, or an LLM reading of the docs. Measured on 602
labelled papers, those converge — any purpose statement is worth about +20 net@2 over none,
and no extraction strategy beats another (README prefix +95, LLM verbatim extraction +91,
tied at P = 0.778; see evals/RESULTS.md → "four ways to tell the gate what a repo is"). The
binding constraint appears to be **what the documents contain**, and no amount of cleverness
in reading them gets past it.

A maintainer's own statement of intent is not bounded that way. "We want to cut index memory"
or "we need better long-context retrieval" is ground truth about a goal, not an inference
about a codebase — and it is exactly what the rubric's score-3 band asks for ("directly
addresses a known limitation"). It is also the one input that can express something the
repository does *not* yet do, which no reading of its own docs can supply.

**Capabilities**
- `goals:` in `.reporadar.yml` — a short list of free-text improvement targets
- Goals in the triage prompt, and (probably more valuable) in **query construction**:
  retrieval is the measured bottleneck at 18/24 reach, and a stated goal is a search query
- `rr goals` to review/edit, so it does not require hand-editing YAML

**Plan**
1. `GoalsConfig` + validation; goals join the profile the same way `queries.seed` does.
2. Triage arm first — it is measurable offline for ~$0.10 against the existing 602 labels.
   Hand-write goals for the 12 benchmark repos from their *issue trackers*, not their
   READMEs, or the arm just re-measures document-derived description again.
3. Only then the retrieval arm, which needs new judge labels and is the expensive half.

**Risks**
- The benchmark cannot fairly evaluate a feature whose input does not exist for its 12
  cases; hand-authored goals are a proxy for real user goals and should be labelled as such.
- Stated goals are repo-derived text going to an LLM and to search APIs — `privacy.py` must
  declare them, and they are more sensitive than keywords because they describe unshipped
  intent.

### 1. Hugging Face Papers enrichment (replace dead Papers With Code)

> **✅ Core shipped in PR #13** (`sources/hf_papers.py`, schema v6 with `models`/`upvotes`, `EnrichmentConfig`). The **`w_community` ranking component now ships too** (opt-in `ranking.w_community`, store v12 `paper_scores.community_score`): `normalize_upvotes` log-scales each run's upvote counts against that run's own maximum, since raw counts are heavy-tailed. One design constraint worth recording — enrichment is **stage 9, ranking is stage 8**, because enrichment only fetches for the papers that made the digest. The plan's "feed upvotes into `score_paper`" is therefore impossible as written; the component reads the enrichments **cached by previous runs** instead, so a brand-new paper carries no community signal on its first run. Papers with zero upvotes (usually "HF has no page for it") are **omitted rather than scored 0**, so the absent-≠-zero rule keeps them from being handicapped. Two bugs the adversarial review turned up while wiring this: `enrichment: provider: off` was landing as the YAML 1.1 **boolean `False`**, so `!= "off"` passed and the documented off-switch did nothing unless quoted (now coerced in `_dict_to_config`); and an autouse `_no_network` fixture — added because the first version of these tests silently ran live enrichment — revealed **six pre-existing tests hitting huggingface.co** on every CI run (fixed by disabling enrichment in the shared repo fixture; the suite is now offline and ~3x faster under the guard). The guard asserts at *teardown*, since every adapter's `except Exception` would otherwise swallow a blocked request and let the test pass. **Remaining:** the `pwc-archive` offline fallback.

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

> **✅ Shipped in PR #40** — `rr mcp` (stdio) exposes `get_repo_profile`, `get_ranked_papers`, `explain_relevance`, `rate_paper`; optional `[mcp]` extra. Remaining: MCP-registry `server.json` publish + a Claude Code plugin (the distribution half).

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

> **✅ Shipped in PR #41 (released as `v1`)** — composite `action.yml` + `rr archive` publish a dated, ranked digest to GitHub Pages; a rendered HTML digest and `${ENV}` config expansion landed with it. Use `uses: raimondasl/auto-features@v1`.

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

> **✅ Shipped.** BM25 + Reciprocal Rank Fusion in the production ranker (PR #37, `ranking.hybrid`, store v8
> `rrf_score`); the local **`rr search`** command + `search_papers` MCP tool (PR #43); and a **persistent
> per-paper embedding cache** (store v9 `paper_embeddings`, `reporadar.embedding_cache`) that computes each
> vector once instead of per run and powers **`rr search --semantic/--hybrid`** — KNN-accelerated by the
> optional `sqlite-vec` extra (`reporadar.vec_index`), with an identical numpy fallback.

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

> **✅ Shipped.** `sources/s2_recommendations.py` + a `recommendations` config block (opt-in). Positive seeds = starred + rated ≥4 (stars first, so they survive the cap; an explicit low rating beats an implicit star), negative = rated ≤2; seeds accept real arXiv ids **and bare S2 paperIds** (so `ss:` papers found *via* recommendations can seed later runs) while unresolvable ids are dropped — one bad seed would 400 the whole call — and a call without positives is skipped rather than sent. Retries with backoff on 429/5xx (keyless pool throttles) and returns `None` on failure so the CLI reports *unavailable* rather than *no results*. Results merge as `matched_query="recommendation"` and are **re-scored locally, then held to the "maybe" tier bar** before the digest shows them (the API is repo-agnostic), with the keyword recommender as fallback; the section is exempt from `--since` since it's a user-seeded feed. *Inherent API limits (not gaps):* the recommender draws from a recent pool, so classic literature never surfaces via this path.

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

> **🟡 Core shipped.** The source-adapter contract is documented (`sources/__init__.py`), and two adapters ship: `sources/dblp.py` (keyword-search JSON, systems/PL/DB — title-only, `abstract=""`) and `sources/biorxiv.py` (date-interval listing bounded + locally query-filtered, biology). Both opt in via `sources:` and merge with arXiv priority; both degrade gracefully. `sources/suggest.py` closes the profile-driven half: it scores the repo profile (detected packages plus keyword/source-signal tokens) against each adapter's signal set and prints a one-line hint in `rr profile` and `rr update` when a source matches but isn't enabled. One matched package is decisive; bare keyword hits need corroboration, so a repo that mentions "protein" once is not nagged. Precision came from *removing* signals: the profiler's inferred domains are not used at all ("containers"/"data pipelines" mark deployment tooling, "distributed computing" comes from Ray/Dask, "databases" only from SQLAlchemy), ubiquitous infrastructure packages (redis, kubernetes, kafka, sqlalchemy) are not DBLP anchors, and words ML repos use in another sense ("distributed", "scheduler", "kernel", "runtime") are not DBLP terms — otherwise every Django app and every ML-systems repo would be told to enable DBLP. **It suggests rather than auto-activates, deliberately** — DBLP rate-limits hard and records publication *year* only, so silently enabling it would slow runs and mismatch the recency window; the caveat is printed with the suggestion instead. **Remaining:** the IACR ePrint adapter (RSS/HTML), DOI abstract-backfill for DBLP, and medRxiv + per-adapter category config.

> **Status 2026-08-23, correcting the block above.** Five adapters now ship, not two: `iacr.py` landed
> (so "Remaining: the IACR ePrint adapter" is done) and `europepmc.py` was added because
> `sources/biorxiv.py` is a **date listing rather than a search** and cannot answer a keyword query —
> prefer `europepmc`. Two of the five are validated end to end against a judge
> (`evals/RESEARCH-scientific-software.md` §21, §39): **Europe PMC** at 4.8 papers/case shown over six
> biology repositories, **OpenAlex** at 1.83/case over six materials-science repositories carrying real
> journal literature. Both at a precision indistinguishable from the arXiv papers in the same digests,
> under two independent judges; neither shown to *raise* digest quality, because a source displaces
> 24–44% of the incumbent Top Picks rather than adding to them. **DBLP, Semantic Scholar and IACR are
> still unvalidated.** Known open defect: with `openalex` on, a paper's journal version can appear beside
> its own arXiv preprint — §40 closes two of five measured cases at the id layer and the rest do not merge.

**Verification: proposed by completeness critique; APIs confirmed free.**

Every existing source assumes the arXiv-ML user, but RepoRadar's value proposition — *papers relevant to YOUR repo* — is strongest for the long tail of non-ML repos whose literature is **not on arXiv**: crypto/security publishes on IACR ePrint, bio tooling on bioRxiv/medRxiv, systems/PL/DB at USENIX/SOSP/VLDB (surfaced via DBLP, abstracts backfilled from OpenAlex/S2 by DOI). This is the single biggest unserved-user-segment gap.

**Measured 2026-08-23:** that gap is real and the channel reaches it. On six materials repositories OpenAlex's pool is **90.8% peer-reviewed journal literature** and **one paper in 1747 came from ChemRxiv** — so the long tail here is journals, not preprint servers, which is the opposite of what this item assumed.

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

> **✅ Shipped.** `evaluation.py` + `rr eval` (`-k`, `--compare A.yml B.yml`, `--baseline --label`, `--history`,
> `--format json`), store v14 `metric_snapshots`. The shared IR metrics moved into the installed package as
> `reporadar/metrics.py`, with `evals/metrics.py` re-exporting them — two copies of a metric definition drift,
> and the roadmap explicitly wanted the in-CLI and benchmark harnesses to agree.
>
> **Three methodological choices decide whether the numbers mean anything, and each is printed rather than
> buried in a docstring.** (1) Judgments are *incomplete* — tens of labels against thousands of stored papers —
> so unjudged papers are **removed** from the ranking (a condensed list), not counted as irrelevant; counting
> them would drown precision@10 in papers the user never looked at and a perfect ranker would score near zero.
> (2) The judged set is **not a random sample**: you can only rate what a digest showed you, and that digest was
> chosen by an earlier version of this ranker, so the metrics are conditioned on that selection and cannot be
> corrected offline. (3) Recency is scored **as of each paper's `first_seen`**, because against today's clock
> every stored paper is old, recency collapses to 0 for all of them, and the component silently disappears from
> every comparison (`score_recency` gained an optional `now=`).
>
> **`--compare` reports a bootstrap interval, not two point estimates.** With a few dozen labels a 0.05 nDCG gap
> is inside the noise; an interval straddling zero prints **NOT SHOWN** instead of inviting a decision the data
> cannot support. Worth recording *how* this nearly shipped broken: the first implementation resampled rank
> *positions*, which scrambles the ordering, so both configs collapsed to their base rate and every interval
> straddled zero — a statistical no-op that reads as rigor. A perfect ranking versus its exact reverse came back
> "cannot tell them apart". It now resamples judged *papers* and re-derives each config's order over the sample;
> a test pins that perfect-vs-reversed stays distinguishable.
>
> **The harness recommended a weight change on the strength of noise, and the review caught it.** SPECTER2's
> query is the centroid of the papers you starred or rated 4-5 — which is exactly the set `load_judgments` marks
> *relevant*. Scoring a judged paper against it puts that paper's own vector inside its own query, inflating
> every relevant paper by construction. Measured with purely random 768-d vectors: mean score **0.81 relevant vs
> 0.20 irrelevant**, and `rr eval --compare` duly reported nDCG 0.555 -> 1.000, 90% interval [+0.180, +0.751],
> "B is better (interval excludes zero)" — a *confident* false positive, in exactly the workflow the README tells
> you to use. Fixed by leave-one-out scoring (the centroid is a mean of unit vectors, so removing one is
> arithmetic, not a rebuild): 0/4 corpora now claim a win from noise, down from 4/4. The lesson is general — a
> feature derived from the same user signals the labels come from will leak unless something explicitly stops it.
>
> Also from the review: a stars-only user (no negative labels) got 1.000 on every metric — the most flattering
> output from the least informative data — and now gets a **DEGENERATE** headline instead; `hybrid` RRF turned
> out to be reproducible offline after all and is now measured rather than declared unmeasurable; and
> `--compare` warns when the two config files also differ outside the `ranking:` block, which it does not apply.
>
> **The optional components are measurable, which nearly did not happen.** The first cut passed no optional
> signals into `rank_papers`, so every one arrived as `None`, absent-is-not-zero (correctly) dropped it from the
> weighted sum, and a `--compare` differing only in `w_specter` reported "NOT SHOWN" on *every* corpus — a
> harness that looks like it works and answers nothing. `stored_signals` now rebuilds citation proximity,
> SPECTER2, HF upvotes, HN points and withdrawal flags from SQLite, with **no network** (`specter` gained an
> `offline=` flag rather than duplicating its centroid math), so a signal the user's runs never fetched stays
> absent instead of becoming a zero.
>
> Two acceptance tests for the whole feature. On a corpus where only keyword matching separates relevant from
> irrelevant, disabling `w_keyword` moves nDCG@10 from 1.000 to 0.555, interval [-0.729, -0.199]. And with all
> the Hacker News buzz placed on papers the user rated *badly*, `w_attention: 25` moves nDCG@10 from 1.000 to
> 0.000 — the harness reads the stored signal and reports that trusting it hurts. Interval calibration measured
> directly: two equal-quality random orderings are wrongly called different 7-10% of the time against a 10%
> nominal rate.
>
> **The CI gate is real, not just a stored number.** The review caught that `--baseline` recorded snapshots
> that nothing ever read back, while the README claimed "CI can catch a regression" — a capability asserted in
> docs and absent from the code. `--against latest|<id>` now compares a fresh run to a snapshot and exits 1 on a
> regression, tolerating movement under 0.02 (metrics shift whenever a new rating lands, and an exact-equality
> gate would be switched off within a week), refusing a baseline taken at a different `k` rather than
> differencing incomparable numbers, and flagging when the judged set itself changed.
>
> **Remaining:** per-repo and pooled reporting in workspaces; wiring `rr eval --against` into the Feature 3
> GitHub Action; and modelling `w_embedding`, `w_citations` and LLM triage, which a condensed list
> of judged papers cannot currently reproduce (`--compare` warns when a difference is confined to those rather
> than reporting a misleading null).

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

> **✅ Core shipped.** `llm_client.py` (shared transport), `triage.py` (0–3 LLM actionability scoring), `TriageConfig`, schema v7 `paper_llm_scores`, `cli.update` triage stage, and digest tiering that **gates Top Picks on the LLM score** (abstains unless genuinely actionable) — directly targeting the precision/calibration gap the Tier B baseline exposed (`evals/RESULTS.md`). The `evals/run_judge_eval.py --rr-triage` flag measures the movement. Remaining: a HyDE query path. (Listwise reranking shipped as `triage.rerank_by_actionability` behind `TriageConfig.rerank`; per-run scores already persist in v7 `paper_llm_scores`, so re-digests do not re-pay inference.)

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

> **🟡 Core shipped.** `specter.py` + `ranking.w_specter` (opt-in). **Re-probed live: S2 still serves `embedding.specter_v2` free and keyless (768-dim)** — the risk this feature hinged on. Vectors are fetched via the shared `_s2_batch_post` (500-chunked, retry/backoff, real arXiv ids only) and cached in the existing store `paper_embeddings` table under a distinct `specter_v2` model key, so **no schema change and no possible 768/384 mixing** with MiniLM. **The query side needs no local model** (so no `adapters`/transformers pin): it's the *centroid of the papers you starred or rated ≥4* in the same citation-trained space. Scores are **min-max normalized across the run's pool** — raw SPECTER2 cosines cluster ~0.87–0.93 even between unrelated papers, so unnormalized they'd carry almost no signal (verified on live vectors). Persisted via store v11 `paper_scores.specter_score` so stored explanations stay complete. **Remaining:** the CPU cross-encoder rerank (note F6's LLM triage already reranks the top tier, so this is partly redundant), local SPECTER2 for a profile-derived query, and the HyDE query path.

**Verification: feasible-with-caveats** (S2 free tier degraded; design around it).

all-MiniLM is a generic sentence model; SPECTER2 is the de-facto scientific-paper embedder (citation-trained, beats SPECTER/SciNCL on SciRepEval). Semantic Scholar serves **precomputed SPECTER2 vectors** through the API via `fields=embedding.specter_v2`. *(As shipped this is a **separate** batch sweep reusing the shared `_s2_batch_post` helper, chunked at 100 ids because each entry carries 768 floats — not piggybacked onto the citation-count call as originally planned. Merging the fields into one POST when several weights are enabled would restore the zero-extra-request idea and is a worthwhile follow-up. Vectors are cached, so the cost is one-time per paper.)* Add a small cross-encoder rerank over the top ~50 candidates.

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

> **🟡 Core shipped.** Store v10 `paper_citations`; `citations.fetch_references` (S2 batch `references.externalIds`, graceful); `citation_graph` (seed set from stars + ratings ≥4, version-insensitive link-finding); a `w_citation_proximity` ranking boost (opt-in, gates the reference lookups); and a digest **"Extends work you starred"** section + `[EXTENDS STARRED]` badge. `rr notify` now carries the count too — the message gains a "N papers extend work you starred" tail (only when N > 0) and shell hooks get `RR_EXTENDS_STARRED_COUNT` — so the alert is *pushed* instead of waiting to be noticed in the digest. The count and the digest section come from one shared `digest.find_extends_starred`, so they cannot drift. **Remaining:** OpenAlex `cites:` fallback, an inline co-citation-graph SVG, and one-hop citation expansion.

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

> **🟡 Core shipped — and two of the four planned signals were disproven by probing them.**
>
> **Shipped.** A `signals/` package (sibling to `sources/`: those *find* papers, these say something *about* one).
> `signals/integrity.py` detects papers withdrawn by their authors and applies a **hard multiplicative penalty**
> (`ranking.withdrawn_penalty`, default 0.1) rather than another weighted component — so a withdrawn paper cannot
> reach Top Picks on other strengths: 1.0 × 0.1 = 0.1, below the 0.2 Maybe threshold. `signals/hn.py` badges
> Hacker News discussion and feeds an opt-in `ranking.w_attention`. Store v13 adds a key-value `paper_signals`
> table plus `paper_scores.attention_score`. Digest gets a tier-independent "Withdrawn by their authors" section
> and `[WITHDRAWN]` / `[HN n]` badges.
>
> **The withdrawal matcher is the whole feature, because arXiv has no withdrawal field.** The notice is
> hand-written free text in `<arxiv:comment>`; a withdrawn paper's title is unchanged and its abstract is often
> the complete original. Measured over 300 real withdrawal comments sampled by the bare token (sampling by
> *phrase* biases toward whatever the regex already handles — an error worth avoiding): the phrasing arXiv's own
> help pages suggest catches **29%**, and the most common real comment is the single word "Withdrawn". The
> shipped field-aware matcher — liberal in the short comment, anchored in prose, so a pharmacology abstract about
> drugs withdrawn from the market is not flagged — measures **100% recall** on notices phrased "withdrawn" and
> **83–85%** on the "withdrew"/"retracted" phrasings, with **no confirmed false positive over 600 ordinary
> papers**. Those other two verbs are the lesson of this feature: the first two rounds of validation sampled by
> searching for the token *withdrawn*, so they structurally could not contain a paper that only ever says
> "retracted" — and reported 300/300 while missing that phrasing entirely. The adversarial review caught it by
> sampling independently. How the sample is drawn mattered more than how big it was.
>
> **Dropped: OpenReview.** The headline capability ("Accepted at ICLR 2026 (avg score 7.2)") is unreachable.
> `GET /notes` returns **403 ChallengeRequiredError** for anonymous clients on every documented form, on both
> `api2` and `api.openreview.net` — a browser challenge, not a rate limit, and solving it would be bot-detection
> circumvention. The roadmap's named risk (fuzzy preprint-vs-camera-ready titles) turned out to be a non-issue
> (29/29 titles matched byte-identically after normalization); the real blockers are the challenge gate and a
> structural 4–8 month lag between preprint and review. Even via a locally built 33k-note venue index, coverage
> of a default 14-day run is **0.5%** (47% for papers from a conference-deadline week — so revisit only as a
> backfill pass over already-stored papers).
>
> **Dropped: Retraction Watch via Crossref, outright rather than deferred.** Crossref indexes **zero** arXiv
> works (`filter=prefix:10.48550` → 0 results; `/works/10.48550/arXiv.*` → 404) because arXiv DOIs are DataCite.
> Reaching it needs an arXiv-id → S2-DOI → Crossref bridge, whose own ceiling measured 9/60 (15%) papers with a
> Crossref-resolvable DOI — and top ML venues (NeurIPS/ICLR/ICML) deposit no DOI at all, so coverage is worst
> exactly where this tool's corpus is. Two roadmap claims were also wrong: the prescribed `filter=updates:{DOI}`
> direction is the wrong one (batched `filter=doi:a,doi:b&select=updated-by` is one request for 100 DOIs), and
> the Labs bulk endpoint is *not* retired — it serves a 65.6 MB, 71,448-row CSV, 12 days fresh. Irrelevant
> anyway: 4 of those 71,448 rows reference arXiv at all.
>
> **Dropped: Bluesky.** It *does* work keyless — but only via undocumented `api.bsky.app`, while the documented
> `public.api.bsky.app` deliberately 403s that one method at its CDN. Shipping an apparent bypass of an
> intentional block is not a dependency worth having. Its coverage is also an artefact: per-category arXiv mirror
> bots post nearly every paper, so raw hit counts said 9/9 papers had "buzz" and 4/12 survived bot filtering. It
> throttles with **403 and no Retry-After**, which every existing adapter here would read as a dead endpoint.
>
> **Also measured, contradicting Feature 12:** keyless OpenAlex is *not* degraded today — it returns 200 with
> self-describing quota headers (`X-RateLimit-Limit-USD: 0.1`, `Cost-USD: 0.0001/call`). And keyless Semantic
> Scholar `/paper/batch` **is** degraded: it 429'd continuously for ~15 minutes of probing and defeated 8 backed-off
> retries, which is why the S2 `publicationVenue` venue annotation (the natural OpenReview replacement — it
> resolves DeepSeek-R1 → Nature, Attention → NeurIPS on a call `citations.py` already makes) is deferred rather
> than shipped.
>
> **Remaining:** the venue annotation above; storing `<arxiv:comment>` at collect time (v15 — v14 is `metric_snapshots`) so ingest-time
> detection is free and the network re-check becomes a top-up; excluding withdrawn papers from `trends.py`.

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

> **🟡 Layers 1–2 shipped.** `privacy.py` (a 16-entry destination registry + redaction) and `rr audit`
> (`--json`), plus `privacy.redact` wired into query building and LLM prompts. **Layer 3 (`--local-only`
> against a bulk arXiv snapshot) is deliberately deferred** — see below.
>
> **What makes the audit worth trusting is that neither half is hand-maintained.** The query strings come from
> the real `build_queries` call, so there is no second implementation to drift from the first. And the
> destination list is *enforced*: a test walks the package for modules making outbound calls and fails CI if
> any is undeclared. A privacy feature that rots is worse than none, because people act on it — so adding a
> source without documenting what it sends is a build error, not a quietly stale page. The detector is static
> and the test states its reach rather than implying completeness: it knows a list of request shapes and
> propagates through private helpers imported from an already-outbound module. Strengthening it during review
> immediately surfaced two undeclared destinations — `specter` (borrows `citations._s2_batch_post`, so it
> matches no request shape at all) and `embeddings` (downloads MiniLM weights from huggingface.co, outbound
> without being an API call).
>
> **Redaction is applied by construction, not per call site.** `privacy.redact` is mirrored onto `QueriesConfig`
> and `SuggestionsConfig` at config-load time, so it takes effect inside `build_queries` (which all four call
> sites and every text-search source share) and inside `llm_client.complete` (the last step before a prompt
> leaves the process). Threading a parameter through instead would have given four existing call sites — and
> every future one — a chance to forget. Two details were found by building it: terms are redacted *before*
> assembly into arXiv syntax, since filtering the finished string leaves `(all: ) AND (cat:cs.IR)` behind,
> which is a broken query rather than a private one; and entries are **literal unless prefixed with `re:`**,
> because `C++` is a valid Python regex (`++` is a possessive quantifier) that would compile happily and redact
> only the letter `C`.
>
> **The risk this section anticipated is real, so the tool states it rather than papering over it.** Redaction
> removes literal terms; TF-IDF keywords still encode your domain. `rr audit` prints a "what leaves regardless
> of redaction" section, and warns when configured patterns matched nothing — a user who configures redaction
> and gets silence would otherwise assume it worked.
>
> **Why layer 3 is deferred rather than attempted:** this section already rates it "the moderately hard part",
> and the two halves have different shapes. Layers 1–2 are honest reporting plus a filter — self-contained,
> and correct or not on their own terms. `--local-only` is an ingestion subsystem: an OAI-PMH harvest measured
> in hours, a sync scheduler, snapshot-freshness semantics that fight the tool's "what's new this week"
> promise, and a Kaggle mirror whose licensing and reliability are outside our control. Shipping it inside the
> same change would put a multi-hour data pipeline behind the same review as a print-what-leaves command.
> The audit is also the prerequisite: it is what tells you *whether* you need local-only mode.

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

> ## Items 14–20, re-derived against the measured record (2026-08-09)
>
> These seven were written in July 2026 from a **capability** sweep — what comparable tools
> ship and what the literature says is possible — before the measurement campaign existed.
> Five weeks of evidence now bears on most of them, in several cases decisively. Each is
> re-derived below; the original entries are left intact underneath, because what a proposal
> assumed is part of why it failed.
>
> | # | re-derived verdict |
> |---|---|
> | **14** `rr deepscan` | **Split it.** The query-refinement loop is measured-negative; the citation-trail half is the one channel with recall. |
> | **15** `rr ask` | **Unaffected.** Nothing measured bears on it — a product bet, not a research one. Judge it on demand, not on evidence. |
> | **16** technique fingerprinting | **Promoted 2026-08-09, then blocked 2026-08-16 [NR-39].** The premise holds — it starts from what the repo *has* — but a $0 probe found the relation cannot be *grounded*: 18 of 25 repositories produce no grounded claim at all, four carry 93% of them, and `replaces` fires on 8.9% of actionable abstracts. Needs a technique-alias layer first, with a coverage number for non-Python repos. **Condition withdrawn 2026-08-16 [P9, C-21]:** the "only anchors discriminate" finding it rested on is a case-mix artifact (−0.6pt per case, CI [−3.7, +1.4]; `peft` alone carries the pooled +13.7pt). **Closed 2026-08-17 [P11]:** built, shipped behind `profiler.typed_anchors`, measured at paired −0.32/case (rescued cohort −1.00 against a pre-registered +0 to +2). The grounding vocabulary is obtainable and inert. |
> | **17** Zotero/BibTeX | **Unaffected.** Pure integration. |
> | **18** implementability scoring | **Refuted as ranking.** Exactly these features measured AUC 0.585 (below bar). Survives only as a displayed badge. |
> | **19** research-gap radar | **Dead.** The same "name what it lacks" mechanism has now failed four independent times. |
> | **20** `rr apply` | **Unblocked, still a moonshot.** Its real precondition — input quality — is met for the first time. |
>
> ### 14 — split it
> The loop is "inspect results, rewrite queries to fill gaps, expand along citations". Those
> halves have opposite evidence. **Query rewriting is measured-negative three times**: LLM
> phrases recovered 2/24 (`uses`) and 0/24 (`lacks`); gap-phrase matching lost to *pasting the
> keyword profile*; and P2 proved the failure is not phrasing — with stemming and BM25 closing
> the morphological gap entirely, `lacks` queries still rank targets **worse than random**,
> because they name a plausible *different* agenda. More rounds of a mechanism that aims
> wrongly do not converge. **Citation-trail expansion, by contrast, is the single channel with
> demonstrated recall** (44%, and 89% on repos with ≥7 seeds). Ship the trail; drop the
> refinement loop unless someone brings a mechanism that closes the register gap.
>
> ### 16 — promoted, narrowed, then blocked on grounding [NR-39, 2026-08-16]
> **A $0 probe of the surviving half found the relation cannot be grounded on most
> repositories.** Repository *keywords* appear in 96.4% of actionable abstracts and 97.4%
> of non-actionable ones — coverage without discrimination. Only *anchors* (declared
> dependencies) separate anything, and 12 of 25 benchmark repos have none, because the
> anchor parser reads Python and JS manifests only. Four Python/ML repositories carry 93%
> of every grounded claim; `ann` misses its own `faiss` because it is a C++ project.
> `replaces` — the alert the feature is named for — fires on 8.9% of actionable abstracts.
> The product also already renders a one-line `llm_reason` per paper in both templates, so
> the increment is a typed label on ~12% of entries rather than an explanation where there
> was none. Reviving it starts with the technique-alias table, which the entry below calls
> a curation burden and which the probe shows is the whole feature. Full result in
> evals/RESULTS.md.
>
> ### 16 — the revival condition is void; the finding it rested on was a case-mix artifact [P9, C-21, 2026-08-16]
> **The "only anchors discriminate" result above is pooled, and it is one repository.**
> Computed per case over the same cached artifacts, the anchor channel is −0.6pt
> (Mantel-Haenszel), 95% CI [−3.7, +1.4], 2 cases better / 3 worse / 11 tied; removing
> `peft` takes the pooled +13.7pt to +0.2pt. So the condition NR-39 set for reviving this
> item — "build the technique-alias table and bring a non-Python coverage number" — was
> derived from a channel that does not discriminate within a repository. **Meeting it would
> prove nothing.** The condition is withdrawn rather than transferred.
>
> P9 tested the obvious way to build that table without curation: typed verbatim-span
> extraction from READMEs. It supplies terms on 9 of the 12 anchor-less repos with zero
> hallucination, and per case it does discriminate (+27.5pt M-H, CI [+21.5, +46.9]) where
> the manifest channel does not. But on the rescued repositories themselves the effect rests
> on two of them, two more extracted spans matching no abstract at all, and the whole
> magnitude is bounded above by a judge-circularity that cannot currently be priced —
> `assemble_repo_context` shows the judge the README and the manifests.
>
> **Item 16 stays blocked, on a different and weaker footing than before.** What it now
> needs is not an alias table but a judge-independent measurement: P6's 31 git-history-mined
> adoptions are the only model-free instrument in the project, and no repo-side channel has
> been scored against them. Until then nothing here is worth building or benchmarking.
>
> ### 16 — measured end to end, and it does not pay [P11, 2026-08-17]
> **The typed-span channel was built, shipped behind a flag, and measured: paired −0.32
> net@2/case over 25 live cases, 7 better / 8 worse / 10 tied, sign p = 1.0000.** The nine
> repositories it exists for — the ones with no parseable manifest — came in at **−1.00**,
> against a pre-registered prediction of +0 to +2. `ann`, the case with the strongest
> stage-1 signal of any rescued repo, lost 7 points by swapping three actionable papers for
> non-actionable ones.
>
> So the sequence closes: the grounding vocabulary NR-39 said was missing **can** be
> obtained (P9, 9 of 12 anchor-less repos, 0 of 209 spans hallucinated), it **does**
> discriminate where the shipped manifest channel does not (+27.5pt against −0.6pt), that
> discrimination is **not** an artifact of one judge or of the judge reading its own input
> back (P10, 0.87 retained under span redaction) — and none of it changes a digest.
>
> **`profiler.typed_anchors` stays False**, beside `profiler.scan_source`, which NR-36 left
> in the same state for the same reason. Item 16 is closed rather than blocked: its premise
> was that grounding vocabulary was the missing ingredient, and the ingredient turned out to
> be obtainable and inert. Reopening it needs a new mechanism, not a better alias table —
> the fourth null in a row (NR-33 +0.00, NR-35 +0.00, NR-36 −0.52, P11 −0.32) says the
> profile is not where the remaining headroom is; the gate and the rescore downstream
> already extract what it knows.
>
> ### 16 — the 2026-08-09 re-derivation, kept for the record
> "This paper supersedes algorithm X you import" seeds from the repo's imports — i.e. from
> what it *has*. Everything that ever worked has that shape (P2: *"every channel that works is
> one that starts from what the repo already does"*), and the judge's notion of actionable
> skews toward improving existing components rather than adding capabilities (P9 found no
> dissociation). So the premise is sound, unusually so. But its *retrieval* half is now
> redundant — HyDE and the hop already reach these papers — and what stays distinctive is the
> **relation classification** (improves / replaces / extends) with an evidence span. That is a
> presentation layer on retrieval we have, which makes it much cheaper than written and
> testable against the existing labels.
>
> ### 18 — refuted as a ranking signal
> E5 measured precisely this feature family — code availability proxy, citation count,
> influential citations, age — with leave-one-repo-out logistic regression: **AUC 0.585**,
> below its own pre-registered 0.60 bar, and the combined model lost to a single fine-scale
> score on every axis. This reproduces Lo et al.'s finding that practitioner relevance is
> uncorrelated with citation metrics. An "effort to adopt" **badge** may still help a reader
> decide; an "effort to adopt" **rank** is measured not to work.
>
> ### 19 — dead unless something new arrives
> "Nobody has applied X to your Y" is the *lacks* direction wearing a product hat. That
> direction has now failed four times independently: direct retrieval (0/24), gate context
> (below the no-description control), pool ranking (worse than random), and the extend-vs-improve
> control (P9: `lacks` papers are not extensions the improvement judge was blind to — they score
> *lower* under a rubric written to reward new capability). Four negatives on one mechanism is
> enough. Reviving it needs a preference signal from real maintainers that capability extensions
> are wanted over component improvements — a user-research question, not a retrieval one.
>
> ### 20 — unblocked by measurement, unchanged in ambition
> Nothing here bears on whether agents can implement papers. What *has* changed is the input:
> `rr apply` on a digest of mean net@2 **−11** would have drafted branches from junk. At **+3.18
> with 0.91 precision and zero net-negative repositories**, the precondition it always implicitly
> had is met for the first time. Still gated on item 18's badge (now the cheap version) and still
> a moonshot.

### 14. `rr deepscan`: agentic iterative search with query refinement and citation trails

**Verification: feasible-with-caveats.**

The defining 2025–2026 shift is from one-shot search to agentic deep search: Undermind's multi-minute agent loops (its "10–50×" figure is marketing, but the pattern is real), Ai2's PaperFinder (pattern evidence only — the repo is a frozen snapshot needing four paid keys), and LangChain's Local Deep Researcher (MIT, 9.2k stars) proving the full loop runs on Ollama. RepoRadar builds static TF-IDF queries once; `rr deepscan [topic]` would have the LLM inspect first-pass results, rewrite queries to fill gaps, expand one hop along citations of strong hits, and emit a cited report — reusing the existing collectors as the search tool.

**Capabilities**
- Multi-round retrieval that recovers papers the static query builder structurally misses
- Reflection-driven query rewriting grounded in what was actually found
- Citation-trail expansion from high-ranked hits (depends on feature 8's edge-fetching code)
- A standalone deep-dive report ("state of X for this repo") distinct from the daily digest

**Plan**
1. ~~**Prerequisite refactor:** extract the collect→store→rank core out of `cli.update`'s inline orchestrator into a reusable `pipeline.py`~~ — **done 2026-08-16** (Tier 0), and done for its own sake rather than as this feature's groundwork: the drift was shipping the −8.12 configuration under `rr watch`. `pipeline.run_pipeline` is the reusable core a deepscan loop would call
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

**Dependencies:** features 1 and 9 (both shipped — HF enrichment and `paper_signals`), `llm_client` (feature 6). Note feature 9 dropped OpenReview, so an acceptance sub-score is no longer reachable.
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
- **Schema migrations** — enrichment `upvotes` (v6), `paper_llm_scores` (v7), `paper_embeddings` (v9), `paper_citations` (v10), `paper_signals` + `attention_score` (v13), `metric_snapshots` (v14). `paper_gaps` / `paper_technique_matches` unbuilt; **next free version is 15**. The sqlite-vec `vec0` table sits outside the migration chain (see feature 4) (the vec0 virtual table stays **outside** the migration chain — see feature 4)
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
- **Capability sweeps date faster than measurements** (added 2026-08-09): items 14–20 were derived from what comparable tools ship and what the literature claims is possible. Five weeks of measurement then refuted two of them outright (18, 19), split a third (14), promoted a fourth (16) and left three untouched. A proposal grounded in *"this is demonstrated to be feasible"* is not grounded in *"this addresses our measured constraint"*, and the difference showed up as a 2-of-7 hit rate. Re-derive against the record before building from a sweep.
