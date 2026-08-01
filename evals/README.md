# RepoRadar evaluation benchmark

A **manually-run** benchmark (deliberately **not** in CI) that measures how well
RepoRadar surfaces papers for a repository. It profiles real repos with the
shipping profiler and ranks with the shipping ranker, so it measures the actual
code paths. There are **two tiers** that answer very different questions:

| Tier | Question | Judge | Keys | Cost |
|------|----------|-------|------|------|
| **A — domain sanity** (`run_eval.py`) | Does ranking separate on-topic from off-topic papers? | keyword/category labels | none | free |
| **B — actionable improvement** (`run_judge_eval.py`) | Would the returned papers *genuinely improve this code*, and does it correctly return **nothing** when there's nothing good? | a neutral LLM (GPT-5.5) | OpenAI + Claude Code | ~$1–2 per case; ~$15–30 for a full 12-case run (baselines + judge verdicts cache, so re-runs are far cheaper). Use `--case <name>` to run one. |
| **S — seeded preference** (`run_seeded_eval.py`) | Do the components that key off *your own* starred/rated papers actually improve ranking? | Tier A labels, train/test split | none (S2 is keyless) | free |

**Tier A is a weak bar and we don't overstate it.** Topic-match ("is this a RAG
paper for a RAG repo?") is easy and doesn't mean a paper is *useful*. Tier A is a
fast, free, deterministic regression gate — nothing more. **Tier B is the real
quality measure** (see [its section](#tier-b--actionable-improvement-llm-judged)).

## Tier A — domain sanity

Two modes:

| Mode | What it does | Network | API keys |
|------|--------------|---------|----------|
| **offline** (default) | Profiles each case's mini-repo (`repos/<case>/`) and ranks a **frozen pool of real arXiv papers** (`fixtures/<case>.json`) with known gold/distractor labels. Deterministic. | none | **none** |
| **live** | Clones each case's real GitHub repo and runs the full pipeline against **real sources** (arXiv, optionally OpenAlex / Semantic Scholar). | yes | optional (see below) |

Offline is the regression gate: same inputs → same numbers. Live is the "does it
run in the wild" check. Both only measure *topic separation*, not usefulness.

## Quick start (no keys, no setup)

```bash
# from the repo root
uv run python evals/run_eval.py
```

Example output:

```
=== RepoRadar offline benchmark ===
(k=10, embeddings=off, recency weight=0 for determinism)

  [rag] P@10=1.000  R@10=0.500  nDCG@10=1.000  MRR=1.000  MAP=0.882  sep=+0.148  (20 gold / 50)
  [cv]  P@10=0.900  R@10=0.450  nDCG@10=0.922  MRR=1.000  MAP=0.928  sep=+0.347  (20 gold / 50)
  [rl]  P@10=0.700  R@10=0.368  nDCG@10=0.801  MRR=1.000  MAP=0.759  sep=+0.141  (19 gold / 49)
  [webdev] negative control: PASS
        max_score=0.030 (threshold 0.5), top-tier papers=0, mean_top10=0.009

  --- mean over labeled cases ---
  P@10=0.867  R@10=0.439  nDCG@10=0.907  MRR=1.000  MAP=0.856  sep=+0.212
```

More commands:

```bash
uv run python evals/run_eval.py --case rag        # one case
uv run python evals/run_eval.py --k 5             # different cutoff
uv run python evals/run_eval.py --embeddings      # add the embedding signal (needs the extra, below)
uv run python evals/run_eval.py --live            # live mode, arXiv only (no keys)
uv run python evals/run_eval.py --live --sources arxiv,openalex
```

For the `--embeddings` flag, install the optional model once:
`uv pip install -e ".[embeddings]"` (downloads `all-MiniLM-L6-v2` on first use).

## The cases

| Case | Domain | Bucket | Live repo |
|------|--------|--------|-----------|
| `rag` | Retrieval-augmented generation / neural IR | ML | `stanford-futuredata/ColBERT` |
| `cv` | Computer vision / object detection | ML | `facebookresearch/detectron2` |
| `rl` | Deep reinforcement learning | ML | `DLR-RM/stable-baselines3` |
| `peft` | Parameter-efficient fine-tuning (LoRA) | ML | `huggingface/peft` |
| `diffusion` | Diffusion / generative models | ML | `huggingface/diffusers` |
| `graph` | Graph neural networks | ML | `pyg-team/pytorch_geometric` |
| `speech` | Automatic speech recognition | ML | `openai/whisper` |
| `crypto` | Cryptography — research on **IACR, not arXiv** | Research-adjacent | `pyca/cryptography` |
| `systems` | In-memory store — research at **VLDB/OSDI** | Research-adjacent | `redis/redis` |
| `webdev` | Web framework — **negative control** | Control | `pallets/flask` |
| `cli` | CLI framework — **negative control** | Control | `pallets/click` |
| `http` | HTTP client — **negative control** | Control | `psf/requests` |

Three buckets probe different behaviors:
- **ML** — arXiv-rich; RepoRadar should surface genuinely-actionable work here.
- **Research-adjacent** — real fields whose literature is *not* on arXiv (IACR /
  VLDB / OSDI). A healthy RepoRadar should mostly abstain, like a control.
- **Negative controls** — pure-engineering repos with no research to apply. If any
  starts scoring ML papers above 0.5 / passing the triage gate, the tool is
  over-firing.

The original 4 (`rag`/`cv`/`rl`/`webdev`) have offline Tier A fixtures in
`fixtures/`; the 8 added 2026-07-12 are **Tier B (live) only** until their
fixtures are built (`build_fixtures.py`).

## Interpreting the metrics

- **P@k** (precision@k) — fraction of the top-k that are genuinely relevant. Higher is better.
- **R@k** (recall@k) — fraction of all gold papers that made the top-k. (Capped by k/total, so ~0.5 when there are 20 gold and k=10.)
- **nDCG@k** — rank-aware quality; rewards putting gold papers *higher*, not just in the top-k.
- **MRR** — 1 / rank of the first gold paper. 1.0 means the #1 result is always relevant.
- **MAP** — mean average precision over the whole ranking.
- **sep** (separation) — mean gold score minus mean distractor score. The most interpretable single number: positive = it tells signal from noise.

**Reading live output:** live mode has no gold labels, so it reports
`domain_purity@k` (fraction of the top-k whose arXiv categories match the repo's
expected domain), the top score, and how many papers clear the 0.5 "Top Pick"
tier. Absolute scores run lower than offline because fresh papers only partially
match a repo's keyword profile and recency is off — judge live runs by
*purity* and by whether the top-3 titles are obviously on-topic, not by the raw
score. For the `webdev` negative control, expect `top-tier=0`.

## Tier S — seeded preference (personalization)

Three shipped ranking signals key off the user's **own** starred / highly-rated
papers — SPECTER2 similarity (Feature 7), citation proximity (Feature 8) and
learned recommendations (Feature 5). Neither Tier A nor Tier B can measure them:
both rank a candidate pool directly and **never build a store**, so there are no
stars or ratings and those components never fire. They shipped *tested but not
benchmarked*.

Tier S supplies the missing ingredient with a **train/test split** over the Tier A
labeled fixtures:

1. `--seeds` gold papers become "papers the user starred" (training). They are taken
   **round-robin across the fixture's `source_query` strata**, not as a file-order
   prefix — fixture order *is* query order, so a prefix draws every seed from one
   query, and if that query is a keyword homonym the whole "liked" signal is
   off-domain. (This is not hypothetical: it is what made Feature 8 look
   unmeasurable until the policy was fixed.)
2. Those seeds are **removed from the candidate pool**, so a component can't win
   by matching the very papers it was handed.
3. The remaining gold papers are **held out**, and each component is scored on how
   well it ranks *those* — at `k` and at the honest depth `k = n_heldout`.

```bash
uv run python evals/run_seeded_eval.py                       # all labeled cases
uv run python evals/run_seeded_eval.py --case rl --seeds 5
uv run python evals/run_seeded_eval.py --component specter --weight 0.5 -o out.json
uv run python evals/run_seeded_eval.py --fresh               # discard cached stores
```

**Deterministic** — no LLM judge, so unlike Tier B there is no noise floor to
fight. Vectors and reference lists come from Semantic Scholar (free, keyless) and
are cached in `evals/.work/seeded/<case>.db`, so re-runs are offline and identical.

Every run also prints **reference rankings** (id-order ≈ random, category-only,
keyword-only) and the **seed titles + categories**. Both exist so a reader can see
how hard the task really is and what "liked" actually meant, rather than taking a
delta on faith.

**What it does and does not prove.** Tier S inherits Tier A's bar: fixture
distractors come from *clearly different fields*, so this measures **topical
discrimination**, not actionability — and a coarse version of it (`category-only`
reaches nDCG@10 = 0.826 on `cv`, so a good share of the task is "does the arXiv
category match"). A component scoring well here has shown it generalizes from a few
liked papers to same-domain papers: necessary, not sufficient. Tier B remains the
quality measure.

Caveats worth knowing before reading results:

- **Report the weight sweep, not a point.** Component response is step-shaped
  (~0 below w≈0.25, saturated by w≈0.5), so a single weight is an arbitrary pick on
  a plateau. Note `w_specter == w_keyword` is *not* equal influence: `keyword_score`
  spans ~0.10–0.17 over a pool while a min-max-normalized component spans all of
  [0, 1].
- **Ceiling effects and concentration.** `rag` sits at nDCG@10 = 1.000 with no
  headroom, and one case can supply most of the mean — read per-case deltas.
- **nDCG@10 = 1.000 ≠ perfect ranking.** With 14–15 held-out gold, recall@10 is
  capped at ~0.7 by construction, so gold papers sit below the cut in every
  configuration. The runner reports `k = n_heldout` alongside for that reason.
- **The gold labels are noisy.** Fixtures come from keyword `gold_queries`, so some
  "gold" papers are homonyms (an RL case labels *"Explainable ML for Public
  Policy"* and *"Proximal Point Methods"* as gold). Deltas remain meaningful —
  both arms are scored against identical labels — but a component can be *more
  right than the labels*.
- **Learned recommendations (F5) are out of scope here.** That feature *adds*
  papers to the pool rather than re-ranking a fixed one, so a labeled pool can't
  score it — it needs the Tier B judge.

## Tier B — actionable improvement (LLM-judged)

This is the real quality measure: does a returned paper propose a method that
would **genuinely improve this specific code**, and is the tool willing to
**return nothing** rather than return noise? Run it with `run_judge_eval.py`.

**How it works (per repo):**

1. **RepoRadar** produces two views from its real ranking: its **Top Picks** tier
   (score >= 0.5 — the abstention-respecting output, often empty) and its **Top-10**
   (diagnostic — tells a too-conservative threshold apart from shallow ranking).
2. **Baseline** = **Opus 4.8**, prompted with *"fetch and summarize research papers
   that relate to the code and propose methods to improve it."* Two modes:
   - `--baseline api` (recommended) — the Anthropic Messages API with the server-side
     `web_search` tool. Needs `ANTHROPIC_API_KEY`; **no Claude Code CLI required**.
   - `--baseline cli` — Claude Code headless (`claude -p`) with web tools, run in the
     repo dir. Needs `claude` on PATH (set `RR_EVAL_CLAUDE_BIN` if it's elsewhere).
     Transient `claude` failures are retried; if `ANTHROPIC_API_KEY` is set it can
     shadow the CLI's claude.ai login and disable connectors — on that specific error
     the retry drops the key from the subprocess so the CLI uses its own login. **If
     `ANTHROPIC_API_KEY` is set, prefer `--baseline api`** (it's deterministic and
     avoids the CLI auth conflict entirely). Only successful baselines are cached, so a
     re-run reuses the ones that worked and retries the failures.
3. **Hallucination guard** — every proposed paper is resolved against the real arXiv;
   unresolvable references count as hallucinations and score 0. (Opus tends to invent
   plausible arXiv IDs; this keeps the comparison honest.)
4. **Neutral judge** — a pooled, blind LLM judge (**GPT-5.5**, so an Anthropic model
   isn't grading Anthropic output) scores the *union* of both lists 0-3 for whether
   each paper could improve **this** repo. 2+ = "genuinely actionable".

**Metrics (precision- and abstention-first):**

- **net@lambda** — `(# actionable) - lambda*(# non-actionable)` over what a system
  returned. With lambda > 1 a junk paper costs more than a good one earns, so
  **returning nothing beats returning noise**. Headline metric; reported at 1, 2, 3.
- **precision** — fraction of returned papers that are actionable (`n/a` when a system
  abstains — an empty result is not a precision-0 failure).
- **abstention_correct** — on repos whose pool has *no* actionable papers, returning
  nothing scores 1.0; any returned paper scores 0.0.
- **ndcg** (graded 0-3), **hallucination count**, and for the baseline a **recent-only**
  net value (it may cite older seminal papers; RepoRadar only fetches recent ones).

**Run it:**

```bash
# 1. Dry-run the whole pipeline with NO keys and NO spend (mock judge + baseline).
#    Still clones repos + hits arXiv (free); validates wiring end-to-end.
uv run python evals/run_judge_eval.py --mock --case rag

# 2. Install the clients + set your keys (see Keys below), then run for real:
uv pip install -e ".[evals]"
uv run python evals/run_judge_eval.py --case rag --baseline api   # one repo, API baseline
uv run python evals/run_judge_eval.py --baseline api              # all repos
uv run python evals/run_judge_eval.py --case rag --baseline cli   # Claude Code CLI baseline
uv run python evals/run_judge_eval.py --model o3 --baseline api   # cheaper judge

# Measure Feature 6: gate RepoRadar's Top Picks on LLM actionability triage
# (needs ANTHROPIC_API_KEY) instead of the heuristic 0.5 threshold. Compare the
# RepoRadar[TopPicks] net@2 against evals/RESULTS.md.
uv run python evals/run_judge_eval.py --case rag --baseline api --rr-triage

# Test the "paper-age artifact" hypothesis: let RepoRadar discover from ALL of
# arXiv (relevance-sorted, no 90-day window, recency weight dropped) so seminal
# older papers can surface and compete. NOTE: this surfaces papers not in the
# judge cache, so it incurs fresh OpenAI judge (+ triage) spend.
uv run python evals/run_judge_eval.py --baseline api --rr-triage --rr-all-time

# Listwise rerank: triage a deeper candidate pool (20) and reorder Top Picks by
# llm_score before the Top-10 cut, so a buried-but-actionable paper can surface.
# Implies --rr-triage. Combine with --rr-all-time for the full "closed both gaps"
# run. Incurs more triage (and, if new papers surface, judge) spend.
uv run python evals/run_judge_eval.py --baseline api --rr-rerank --rr-all-time

# Gate-precision sweep: report Top Picks metrics at every min_actionable threshold
# (1/2/3) in one run. FREE — triage scores are computed once, so re-gating is
# post-processing. Prints a cross-case rollup showing which threshold maximizes
# net@2 and eliminates false positives (e.g. the webdev negative-control leak).
uv run python evals/run_judge_eval.py --baseline cli --rr-rerank --rr-all-time --rr-sweep
# Then lock in the winner for a normal run, e.g. the stricter gate:
uv run python evals/run_judge_eval.py --baseline cli --rr-rerank --rr-all-time --rr-min-actionable 3

# Hybrid retrieval (roadmap #4): fuse the heuristic ranking with a BM25 lexical
# ranking via RRF before the Top-N cut, so a paper buried on vocabulary mismatch
# can surface. Measure-first — compare the RepoRadar columns with/without it.
uv run python evals/run_judge_eval.py --baseline cli --rr-rerank --rr-all-time --rr-hybrid
```

Requires **`OPENAI_API_KEY`** (the judge) and, for the baseline, either
**`ANTHROPIC_API_KEY`** (`--baseline api`) or **Claude Code** on PATH (`--baseline cli`).
For the CLI baseline, if your Claude Code version needs different headless flags, edit
`CLAUDE_FLAGS` in `evals/baseline.py` or set `RR_EVAL_CLAUDE_FLAGS`.

**Debugging a baseline that "recommended 0 papers":** the runner distinguishes a real
abstention from a failure. A failed baseline prints `!! BASELINE DID NOT RUN [status]`
(status is `missing_cli` / `no_key` / `error` / `timeout`) and its metrics row is marked
`** FAILED — did not run **`. The raw output of every run is cached at
`evals/cache/baseline/<mode>/<repo>.json` — inspect `raw` and `status` there. Caches are
**per-mode** (`api` / `cli` / `mock`), so a `--mock` run can never be served to a real
one, and **failures are never cached** (they retry next run).

**Honesty guarantees** (so infra hiccups can't silently favor a system): an arXiv
outage marks the baseline `arxiv_unverified` rather than counting real papers as
hallucinated; a judge error (empty/malformed/out-of-range score) **drops that paper
from the pool** instead of scoring it 0; a failed baseline emits **no metric numbers**
(so no aggregate can read a crash as a legitimate 0.0 net-value / 1.0 abstention); and
caches are invalidated when the model, prompt, rubric, or repo context changes.

**Reproducibility & cost:** successful judge verdicts and baseline outputs are cached
under `evals/cache/` keyed by (rubric version, model, repo, paper) / (mode, repo), so
re-runs are near-free and stable. A full uncached run is roughly **$10-30** (pool of
~30-60 papers x 4 repos judged on GPT-5.5, plus 4 Opus baseline runs). `evals/cache/`
and `evals/results/` are gitignored. Bump `RUBRIC_VERSION` in `judge.py` to invalidate
the judge cache; delete `evals/cache/baseline/<mode>/` to force baseline re-runs.

**Expected finding:** RepoRadar's Top Picks tier abstains often on live data (its
keyword scores rarely reach 0.5), and the Opus baseline will likely win on actionable
papers. That's the point — it's the evidence base for the roadmap's LLM-triage
(Feature 6) and retrieval upgrades, and gives us a number to move.

## Keys — what you need, how to get them, where to set them

**Tier A needs nothing** (offline, or live with arXiv only). **Tier B needs
`OPENAI_API_KEY`** (the judge) plus a baseline credential: `ANTHROPIC_API_KEY` for
`--baseline api` (recommended) **or** Claude Code on PATH for `--baseline cli`. Other
keys only unlock extra live sources.

| Env var | For | Cost | Get it from |
|---------|-----|------|-------------|
| `OPENAI_API_KEY` | **Tier B judge (GPT-5.5)** | **paid** (~$10–30/full run) | https://platform.openai.com/api-keys . Required for a real Tier B run; use `--mock` to dry-run without it. |
| `ANTHROPIC_API_KEY` | **Tier B `--baseline api`** (Opus 4.8 + web_search) | **paid** | https://console.anthropic.com/ . The no-CLI baseline path. |
| `OPENALEX_API_KEY` | live `openalex` source | **free** | https://openalex.org/ → sign in → **API key**. Recommended: since 2026-02-13 keyless callers are throttled to a tiny daily allowance. |
| `SEMANTIC_SCHOLAR_API_KEY` | live `semantic_scholar` source | free | https://www.semanticscholar.org/product/api . Works **without** a key on a shared pool (best-effort); keys aren't granted to free-domain emails since 2024, so most people run keyless. |
| `HF_TOKEN` | higher Hugging Face rate limits | free | https://huggingface.co/settings/tokens (a **read** token). Not used by the ranking eval — only enrichment — so usually unnecessary here. |
| Claude Code | **Tier B `--baseline cli`** (Opus 4.8) | **paid** (subscription or API key) | Only for the CLI baseline. Install from https://claude.com/claude-code ; `claude` must be on PATH (or set `RR_EVAL_CLAUDE_BIN`). Prefer `--baseline api` if you don't have it. |
| `GITHUB_TOKEN` | avoid clone rate limits | free | Only if `git clone` starts failing. Public repos clone with no token. |

### Where/how to set them

**Recommended — a `.env` file** (cross-platform, gitignored, nothing to remember):

```bash
cp evals/.env.example evals/.env
# then edit evals/.env and paste in the keys you have
```

`evals/.env` is already in `.gitignore` — it will never be committed. The runner
loads it automatically. Real environment variables always win over the file.

**Alternative — shell environment variables:**

PowerShell (Windows), current session:
```powershell
$env:OPENALEX_API_KEY = "your-key-here"
uv run python evals/run_eval.py --live --sources arxiv,openalex
```

PowerShell, persist for your user (new terminals):
```powershell
[Environment]::SetEnvironmentVariable("OPENALEX_API_KEY", "your-key-here", "User")
```

bash / zsh:
```bash
export OPENALEX_API_KEY="your-key-here"
```

The runner prints which keys it detected at the top of a live run, e.g.
`keys present: OPENALEX_API_KEY`.

## Regenerating the fixtures

The offline fixtures are frozen real arXiv papers committed under `fixtures/`.
Regenerate them (e.g. to refresh with newer papers) with:

```bash
uv run python evals/build_fixtures.py            # all cases (hits arXiv, ~2 min)
uv run python evals/build_fixtures.py --case rag # one case
```

This needs network but **no keys** (arXiv is open). Gold papers come from each
case's `gold_queries`; distractors from `distractor_queries` in `benchmark.yaml`.

## Adding a case

1. Add an entry to `benchmark.yaml` (name, `repo_dir`, `live_repo`,
   `expected_categories`, `gold_queries`, `distractor_queries`).
2. Create `repos/<name>/` with a realistic `README.md` and `requirements.txt`.
3. Run `uv run python evals/build_fixtures.py --case <name>`.
4. Run `uv run python evals/run_eval.py --case <name>`.

## Files

```
evals/
  README.md          this file
  benchmark.yaml     case definitions (repos, queries, categories)
  harness.py         profile a repo + rank a pool + shared live helpers
  metrics.py         Tier B only (net@lambda, abstention, graded nDCG). The shared IR
                     metrics (P@k, R@k, nDCG, MRR, MAP, separation) moved to
                     src/reporadar/metrics.py and are re-exported here, so this
                     benchmark and the in-CLI `rr eval` cannot drift apart
  seeded.py          Tier S: stratified seed/hold-out split + component builders
  run_seeded_eval.py Tier S runner
  build_fixtures.py  fetch real arXiv papers -> fixtures/ (run manually)
  run_eval.py        Tier A runner (--offline / --live)
  run_judge_eval.py  Tier B runner (LLM judge vs Opus baseline; --mock to dry-run)
  judge.py           neutral GPT-5.5 judge (rubric, caching, --mock scoring)
  baseline.py        Opus 4.8 baseline — --baseline api (Anthropic + web_search) or cli (Claude Code)
  diagnose_pool.py   free/keyless: is a miss a POOL or a SELECTION failure? Established that
                     RepoRadar fetched 2030 papers and reached 0 of 24 known-good ones
  diagnose_query_generation.py
                     ~$0.01: can an LLM emit phrases that reach what TF-IDF misses?
                     `--prompt uses` 2/24, `--prompt lacks` 0/24. Both negative; see
                     RESULTS.md "Candidate-pool diagnosis" for why, before retrying
  diagnose_citation_hop.py
                     free/keyless: one citation hop from the repo's own bibliography.
                     18/24 — the only approach measured that reaches the papers at all,
                     at 1 good paper per 5111 candidates (recall solved, precision not)
  verify.py          resolve proposed papers against real arXiv (hallucination guard)
  .env.example       template for API keys (copy to .env)
  repos/<case>/      realistic mini-repos profiled in Tier A offline mode
  fixtures/<case>.json   frozen labeled arXiv pools (committed)
  cache/, results/   Tier B verdict cache + run outputs (gitignored)
```
