# RepoRadar evaluation benchmark

## Which script re-derives which section of the paper

`paper/DRAFT.md` claims every quantitative result maps to a script here. This is that
mapping — written 2026-08-16, because the claim had been made for weeks against an index
that did not exist, which is the same unverifiable-assertion shape the paper spends §10
cataloguing. Each row is the script whose output the section's numbers come from; the
numbers themselves, with dates and costs, are in [RESULTS.md](RESULTS.md).

| paper section | scripts |
|---|---|
| §4.1 metric | `metrics.py` |
| §4.2 headline provenance | `run_judge_eval.py`, `noise_floor.py` |
| §4.3 benchmark, judge, baseline | `harness.py`, `judge.py`, `baseline.py`, `verify.py`, `build_fixtures.py` |
| §4.4 judge validity | `second_judge.py` (κ), `mine_adoptions.py` (adoption ground truth) |
| is the fine-scale stage right to withhold? (RESEARCH-scientific-software.md §19) | `second_judge_band.py` (~$3; Sonnet over all 324 score-2 band papers — the run that reversed §18.2's sign) |
| is a newly added source any good? (RESEARCH-scientific-software.md §21.2) | `second_judge_arm.py` (Sonnet over the papers an arm SHOWED, split by origin — within-arm, so no cross-session pairing) |
| why does a new source take half the digest? (RESEARCH-scientific-software.md §22.2) | `displacement_probe.py` ($0, judge-free; re-ranks a frozen pool under each absent-category mode) |
| is the 15-paper window too small? (RESEARCH-scientific-software.md §24) | `window_arm.py` (~$5-11; ranks 16-30 under both judges, kill check first) |
| why does gate-score 3 mean something different on scientific software? (RESEARCH-scientific-software.md §26) | `score3_mechanism.py` ($0 primary; refuted the tool-name mechanism §0 had asserted since §5) |
| does adding a source reorder the papers already there? (RESEARCH-scientific-software.md §27) | `rank_stability.py` ($0, judge-free; Kendall tau over the shared papers, with the cause isolated) |
| does the repo already citing a paper make it a dud? (RESEARCH-scientific-software.md §29) | `already_have.py` ($0; found the shipped already-cited rule is net-negative on the score-3 band) |
| §4.5 debugging the benchmark | `run_eval.py`, `harness.py` |
| §5.1–5.2 retrieval failure, register mismatch | `diagnose_pool.py`, `diagnose_query_generation.py`, `gap_match.py`, `extend_vs_improve.py` |
| §5.3 citation hop | `diagnose_citation_hop.py`, `build_hop_pool.py`, `hop_reach.py`, `sweep_hop_filter.py`, `synth_seeds.py`, `fill_pool_metadata.py` |
| §5.4 HyDE | `verify_hyde_deps.py` (stage 1), `hyde_replication.py` (stage 2) |
| §5.5 pool density | `diagnose_pool.py`, `label_pool.py` |
| §6.1–6.2 the gate | `compare_triage.py`, `diagnose_triage.py` |
| §6.3 gate context | `compare_triage.py`, `fetch_wants.py` |
| §6.4 gate depth | `gate_full_pool.py`, `run_judge_eval.py --rr-pool` |
| §6.5 near-binary distribution | `diagnose_triage.py`, `diagnose_ranker.py` |
| §7 E1–E5 band ranking | `band_testbeds.py` (shared), `exp_select.py`, `exp_finescale.py`, `exp_ensemble.py`, `exp_pairwise.py`, `exp_features.py` |
| §8.1–8.3 calibration | `exp_finescale.py`, `compare_finescale_baseline.py` |
| §8.4 live run | `run_judge_eval.py --rr-finescale` |
| §8.5 dense channel end to end | `run_judge_eval.py --rr-hyde` |
| §8.6 calibration audit | `calibrate_finescale.py` |
| §8.7 headline and its weather | `run_judge_eval.py`, `noise_floor.py` |
| §8.8 frozen pool | `noise_floor.py`, `run_judge_eval.py --rr-frozen-pool` |
| §8.9 two unmeasured defaults | `run_judge_eval.py` (`--rr-pool`/`--rr-window`), `audit_product_divergence.py` |
| §8.10 the third default, `w_embedding` | `run_judge_eval.py --rr-w-embedding`, `join_wemb_headline.py` ($0 prediction + the check that scored it) |
| roadmap 16 relation grounding (NR-39) | `relation_probe.py` ($0; reads cached pools, profiles and verdicts) |
| scientific software: the score-3 band (RESEARCH-scientific-software.md §9) | `probe_score3_band.py` (~$0.03; scores cached judge verdicts, no re-judging) |
| why a benchmark case returned nothing (RESEARCH-scientific-software.md §16.4, §17) | `why_case.py` ($0; reads a results artifact, the eval-side counterpart to `rr why`) |
| is the fine-scale map calibrated on scientific software? (RESEARCH-scientific-software.md §18.2) | `finescale_domains.py` ($0; reads `finescale_p` straight from a results artifact — `calibrate_finescale.py --analyse` cannot, its cache holds only the 22 legacy cases) |
| what shape are the gate's scores? (RESEARCH-scientific-software.md §18.4–§18.5) | `gate_shape.py` ($0, judge-free; the 37-case control that retracted §17.4) |
| typed README spans as an anchor channel (P9) | `nerdme_probe.py` (~$0.02 once to extract, then `--report` is $0; reuses `relation_probe.py`'s matching) |
| is P9's signal judge circularity? (P10) | `redacted_judge.py` (~$4; two Sonnet arms over the same papers, spans masked in one) |
| §9.1–9.3 the query bridge | `audit_query_transform.py`, `bigram_report.py` |
| §9.4 IACR ePrint | `verify_iacr_deps.py`, `source_ab_report.py` |
| §9.5 Semantic Scholar | `s2_yield.py` (stage 1), `source_ab_report.py` (A/B) |
| §9.6 absent-category bias | `run_judge_eval.py --rr-absent-category`, `ablation_report.py` |
| §9 OpenAlex | `openalex_yield.py` |
| §10 corrections, blast radius | `audit_product_divergence.py` |
| §12.1–12.2 thin documentation | `run_judge_eval.py --rr-ablate-docs`, `ablation_report.py` |
| source scanning (`profiler.scan_source`) | `scan_source_probe.py` (stage 1, $0), `run_judge_eval.py --rr-scan-source` |
| thin-docs detection (NR-37) | `thin_docs_detector.py` ($0; also carries the ablation-arm → budget mapping) |
| stated-intent experiment (NR-26) | `make_goals.py`, `run_judge_eval.py --rr-goals` |
| personalization (Tier S) | `seeded.py`, `run_seeded_eval.py` |

Scripts prefixed `verify_*` are the **$0 stage-1 dependency probes** that precede a paid
experiment; `diagnose_*` answer a question without changing anything; `exp_*` are the
pre-registered band-ranking experiments; `audit_*` are the free static checks that look for
a known defect shape on purpose.

---

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

### Cohort 2 (2026-08-05) — added because 12 was measurably too few

Jackknifing P1's headline exposed the problem: recomputing its leave-one-out result
with one repo removed moved a 70% pool cut to **11%** when that repo was `rag` (5 of
18 targets, smallest pool). Effective repo count over target share was **5.4 of 7**.
A result one repo can move 59 points describes the case set, not the mechanism.

Ten cases were added against four **measured** criteria — chosen to close named
blind spots, not to add more of the same:

| criterion | meaning | why |
|---|---|---|
| **T** | README < 3,000 chars **and** real applicable research | RepoRadar's actual target user. A thin *control* does not count — it cannot test the shipped `profiler.prose_chars: 300`, which may be an artifact of 12 well-documented repos. |
| **B** | zero arXiv ids in docs | structural zero for the citation hop; only 2 non-control cases had this, too few to measure P3's synthetic seeding |
| **N** | non-ML domain | 7 of the original 12 are ML |
| **C** | cites papers **and** is cited by them | P6's adoption ground truth needs citation-rich history |

| case | live repo | README | arXiv ids | T | B | N | C | targets |
|---|---|---|---|---|---|---|---|---|
| `db` | `duckdb/duckdb` | 3,480 | 1 | | | ✓ | ✓ | 3 |
| `storage` | `facebook/rocksdb` | **1,689** | 0 | ✓ | ✓ | ✓ | ✓ | 2 |
| `numerics` | `scipy/scipy` | 3,798 | 1 | | | ✓ | ✓ | 2 |
| `compiler` | `numba/numba` | **1,686** | 0 | ✓ | ✓ | ✓ | | 2 |
| `llminfer` | `ggml-org/llama.cpp` | 7,103 | 2 | | | | ✓ | 4 |
| `vectordb` | `qdrant/qdrant` | 11,530 | 0 | | ✓ | ✓ | | 4 |
| `linter` | `astral-sh/ruff` | 25,182 | 0 | | ✓ | ✓ | | 0 |
| `encryption` | `FiloSottile/age` | 11,618 | 0 | | ✓ | ✓ | | 0 |
| `ann` | `facebookresearch/faiss` | 6,364 | 3 | | | | ✓ | 3 |
| `columnar` | `apache/arrow` | 5,837 | 0 | | ✓ | ✓ | ✓ | 4 |

Coverage after: **T 4, B 11, N 13, C 11** across 22 cases (cohort 1 contributed
T 2, B 5, N 5, C 5).

**Effect on the concentration problem** — targets **24 → 48**, `rag`'s share
**21% → 10%**, effective repo count **5.4 → 15.2** of 17 contributing cases.

**Two predictions of mine that the measurement refuted**, kept here because the
labels were nearly shipped as asserted:

1. I expected `llminfer`, `vectordb`, `linter` and `encryption` to be thin-docs
   cases. **All four have substantial READMEs** — ruff's is 25,182 characters, four
   times peft's. "Rust CLI tool ⇒ sparse docs" was wrong. The only genuine T
   additions are `storage` and `compiler`, and the `criteria:` field in
   `benchmark.yaml` now carries the measured numbers inline so labels cannot drift
   from evidence.
2. I expected the six **B** cases to yield ~0 targets, reasoning that a repo citing
   no arXiv work has no arXiv work to recommend. **12 of the 22 new targets come
   from B cases.** ANN indexing, columnar compression and LSM compaction all have
   arXiv literature that those repos simply do not cite — which makes them the most
   interesting cases in the set: the hop is structurally blind there and the
   research exists anyway.

Two cases contribute **0 targets** and are kept: `encryption` (Opus abstained
entirely — correct for a repo whose literature is on IACR) and `linter` (Opus made 3
picks, the judge rejected all 3 — an over-firing case worth having).

`numerics` needed a second, longer run: the headless baseline takes >10 minutes and
$3.25 on a repo scipy's size and was cut short by the batch loop, not by any harness
limit. It contributes 2 targets. Budget wall-clock, not just dollars, when adding
large repos.

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

# The SHIPPED configuration, end to end (--rr-finescale). After the 0-3 gate, papers
# sitting exactly at --rr-min-actionable are rescored 0-9 and kept only above the
# calibrated P >= 2/3 — the second stage that fixes the near-binary gate. Runs through
# reporadar.finescale itself, never a local copy, because the probability map is
# calibrated to that exact prompt. Needs OPENAI_API_KEY (logprobs); ~$0.01/case on top.
# This is the command that produced the current headline (+3.18 vs the baseline's +1.82).
uv run python evals/run_judge_eval.py --baseline cli --rr-pool 50 --rr-rerank \
    --rr-all-time --rr-hybrid --rr-sweep --rr-finescale

# Hybrid retrieval (roadmap #4): fuse the heuristic ranking with a BM25 lexical
# ranking via RRF before the Top-N cut, so a paper buried on vocabulary mismatch
# can surface. Measure-first — compare the RepoRadar columns with/without it.
uv run python evals/run_judge_eval.py --baseline cli --rr-rerank --rr-all-time --rr-hybrid

# Triage-window depth: score N candidates instead of the default 20, then still cut the
# digest at 10 — so this measures SELECTION quality at a fixed digest size, not "return
# more papers". Cost scales linearly with N.
# MEASURED NEGATIVE at N=50: +0.67 net@2, 95% CI [-0.50, +2.00]. 4x the candidates bought
# 2 more actionable papers across 12 cases. See RESULTS.md -> "Negative result 5".
uv run python evals/run_judge_eval.py --baseline cli --rr-rerank --rr-all-time --rr-pool 50

# How much README the gate sees (profiler.prose_chars on the profile it is given).
# 0 is the pre-2026-08-02 behaviour and the CONTROL ARM for any prose measurement; the
# shipped default is 2000. The prompt is the shipped one either way — this used to build
# its own "README context" variant, which is how a result got published under the wrong
# name. See RESULTS.md -> "what the README variant actually sent".
uv run python evals/run_judge_eval.py --baseline cli --rr-triage --rr-prose-chars 0
```

**Decomposing a prompt change cheaply.** A 12-case Tier B run costs ~$11 and its paired CI
over cases is about ±2 net@2 — too wide to resolve a prompt tweak. `diagnose_triage.py`
scores every cached judge label (600+ papers) for ~$0.10 per arm, and `compare_triage.py`
pairs two arms per paper. Decide there first; spend the $11 confirming a winner.

```bash
uv run python evals/diagnose_triage.py --repo-context keywords   # control: no prose
uv run python evals/diagnose_triage.py --repo-context prose      # shipped: + real README
uv run python evals/diagnose_triage.py --repo-context tagline    # the historical variant
uv run python evals/compare_triage.py \
    evals/.work/diag_triage_keywords.json evals/.work/diag_triage_prose.json
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
| `OPENAI_API_KEY` | **Tier B judge (GPT-5.5)**, and the fine-scale rescore (`exp_finescale.py`, gpt-4o-mini) | **paid** (~$10–30/full run; the rescore is ~$1) | https://platform.openai.com/api-keys . Required for a real Tier B run; use `--mock` to dry-run without it. The rescore needs it for a different reason than the judge — token logprobs, which Anthropic does not expose. |
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
  diagnose_triage.py ~$0.10: scores all 428 cached judge labels with the shipping triage
                     gate. precision 0.81 / recall 0.78 vs a 32% base rate — NOT at chance,
                     correcting an n=10 reading. --repo-context {keywords,readme,both}
  diagnose_ranker.py ~$5: judges a RANK-STRATIFIED sample of the candidate pool, the only
                     way to score the ranker. Ranks 1-10 and 11-50 are indistinguishable
                     (31% vs 33% actionable) — the top-10 cut is arbitrary
  compare_triage.py  paired per-paper comparison of two --repo-context runs, with an exact
                     binomial on the discordant pairs so a noisy tie is not read as a win
  diagnose_citation_hop.py
                     free/keyless: one citation hop from the repo's own bibliography.
                     18/24 — the only approach measured that reaches the papers at all,
                     at 1 good paper per 5111 candidates (recall solved, precision not)
  verify_hyde_deps.py
                     free/keyless, P4 stage 1: the four load-bearing dependencies of
                     RETRIEVAL_DESIGN Design 2 — does the 3.1M-vector arXiv index exist
                     under a usable licence, is columnar range-fetch real, is the query
                     latency reproducible, are the targets in it. Gates stage 2 at 4/4
  hyde_replication.py
                     ~$0.20 + 432 MB, P4 stage 2: blind HyDE against that index. Refuses
                     to run until it reproduces the publisher's own vectors bit-for-bit.
                     `--build` fetches the id+vector columns; the run compares hypothesis
                     abstracts against README and keyword queries on the same index
  fetch_wants.py     free, P8: the top 15 open issues by reactions per repo, titles kept
                     VERBATIM (the failed improvement_areas arm was paraphrased, and
                     paraphrase-vocabulary loss was the diagnosis). Feeds the
                     `--repo-context wants` arm of diagnose_triage.py, which measured
                     +57 net@2 against prose-300's +95 — the worst arm in the study
  second_judge.py    ~$2, P7: the label-noise floor. Re-judges 200 of the 602 with Sonnet,
                     byte-identical rubric, after checking every case still reproduces its
                     stored _prompt_hash. kappa 0.51 on the shipped >=2 cut but 0.71 with
                     the second judge's cut moved one notch — the judges RANK the same and
                     differ in strictness, so paired arm-vs-arm differences largely cancel
                     the offset while absolute levels do not. Writes to .work/, never to
                     evals/cache/judge/

  already_have.py
                     $0: §28's tertiary and secondary. The eval harness never applies the
                     already-cited rule that the PRODUCT does, so the benchmark's score-3
                     problem is larger than a user's -- by 3 of 13 (23%). Uses the product's
                     own membership rule (digest.py:244) rather than a second copy.
                     RESULT (§29): being cited does NOT make a paper a dud -- 7 of the 10
                     cited score-3 papers are judged actionable, including chgnet's own
                     paper, PyG 2.0 for graph and FAISS for ann -- so the shipped rule drops
                     7 actionable to spare 3, net@2 -1 over 25 repos. Also caught that a bar
                     in percentage POINTS is not judge-comparable when base rates differ.

  rank_stability.py
                     $0, offline, judge-free: re-ranks the control and treatment pools and asks
                     whether adding a source changes the ORDER of the papers that were already
                     there -- which a bioRxiv preprint has no bearing on. RESULT (§27): mean
                     Kendall tau 0.906, but only 58/90 (64%) of the control top-15 survives in
                     the treatment's arXiv-only top-15, so the churn concentrates where a digest
                     is drawn from. Cause isolated: with hybrid RRF off the score is per-paper
                     and tau is exactly 1.0000, 0 discordant of ~140k. RRF is the whole of it --
                     a design consequence of rank fusion, not a bug, now measured.

  score3_mechanism.py
                     $0 primary (~$0.40 with --validity): tests WHY gate-score 3 behaves
                     differently on scientific software, over all 85 labelled score-3 papers.
                     RESULT (§26): the tool-name mechanism §0 and §6 G1 asserted since §5 is
                     REFUTED as stated -- among score-3 papers, naming the tool predicts
                     non-actionability at 0.119 vs 0.231, i.e. the OPPOSITE direction, under
                     both judges. Half of it survives: naming strongly predicts the gate
                     EMITTING 3 (21.1% vs 6.6%, p=1e-06). Also decomposes §18.5 and finds the
                     emission asymmetry is MATSCI (27.8%, p<0.0001), not scientific software
                     (bio 12.2%/13.3%, both non-significant across two draws).

  window_arm.py
                     ~$5-11: analyses a --rr-window 30 arm to answer whether output.top_n=15
                     truncates good material. Runs a KILL CHECK first and exits before
                     printing any number if it fails: the re-run must overlap the shipped
                     window by >=11/15, because the gate is sampled and exact reproduction is
                     not available. Per-paper endpoint (90 vs 90), both judges. RESULT (§24):
                     a sharp cliff at rank 15 -- 0.822 -> 0.344 under GPT, 0.489 -> 0.144
                     under Sonnet, Fisher p~0 both -- so 15 cuts where the quality does and
                     raising top_n costs about a point per paper added.

  displacement_probe.py
                     $0, offline, judge-free: re-ranks a frozen pool under each
                     ranking.absent_category mode and reports how much of the top-15 window
                     each source holds. Built to test whether the shipped 'omit' rule --
                     which score_paper's own comment says advantages uncategorised papers,
                     0.600 vs 0.567 -- explains §21.4's displacement. It does NOT: omit
                     gives Europe PMC 51% of the window and impute gives it 50%, keeping 85
                     of 90 slots. Hypothesis refuted for nothing, because the pools were
                     seeded to disk. 'zero' (19%) is not the corrected number -- it asserts
                     a bioRxiv paper has zero topical match rather than a different taxonomy.

  second_judge_arm.py
                     ~$0.60: second-judges the papers a benchmark arm actually SHOWED and
                     splits them by origin, so "is the new source any good" is answered
                     within one arm rather than across sessions. Built for the Europe PMC
                     arm (RESEARCH §21.2), where it found Europe PMC papers at precision
                     1.000 (GPT) / 0.724 (Sonnet) against arXiv's 0.897 / 0.586 in the same
                     digests -- higher under both judges, CIs overlapping under both, so the
                     claim is "not worse" and not "better". Wilson intervals, because a
                     precision of 1.000 has a zero-width Wald CI.

  second_judge_band.py
                     ~$3: Sonnet over all 324 score-2 band papers of the 37-case cohort-3
                     session, after verifying all 34 prompt hashes. Pre-registered bars, and
                     the result landed on the one the prediction ruled out. The withheld
                     papers are 26% actionable against a 67% break-even, but the SHOWN ones
                     are 57% against Sonnet's 22% base rate -- a +31 point separation, so the
                     map discriminates. Recomputing net@2 under each judge REVERSES the sign
                     of the stage's value (-1.250/case under GPT, +3.750 under Sonnet on the
                     scientific 12), which is why "true by construction" is not the same as
                     "durable". Judges the shown arm too, on purpose: a withheld-only number
                     sits at Sonnet's base rate and cannot be read. kappa on this band is
                     0.199 against 0.507 globally -- the band is where the judges part.
  mine_adoptions.py  $0 + ~$1, P6: ground truth no model produced. An arXiv id in a repo's
                     docs at HEAD and absent 24 months earlier is a technique it actually
                     adopted. 31 such adoptions across 6 repos; the judge calls 61% of them
                     actionable against the repo AS IT WAS at T0 (2% is the random-arXiv
                     floor), and the citation hop reaches 68% of them from the T0
                     bibliography alone. Clones are blobless and SEPARATE from .work/<case>,
                     whose working-tree state gates the verdict cache
  label_pool.py      ~$10, P5: the first labels drawn from the candidate pool itself rather
                     than from what the ranker surfaced. 1,200 papers through the shipped
                     gate, 320 through the judge, six strata plus a uniform-random-arXiv
                     floor. Established that the pool is dense (top band 58% actionable vs
                     2% floor) and that the gate's failure is recall (0.60), not precision
                     (0.97). `--dry-run` shows the sample and cost without calling anything
  gate_full_pool.py  ~$1.20: what the gate does on REAL candidate pools rather than on what
                     a ranker surfaced. 300 papers balanced across all 22 repos; 11% admit
                     rate at 0.73 precision. Refuses to cache an empty fetch — seven pools
                     were once cached empty after an arXiv 429 storm and scored as zeros

  --- ranking the score-2 band (2026-08-07/08) ---
  band_testbeds.py   $0, no LLM: the frozen testbeds every E-experiment below scores
                     against, so they share one definition of the data and the metrics.
                     Testbed A reconstructs per-paper gate bands POSITIONALLY from a run
                     file's sweep counts (the files record judge scores but not gate
                     scores); the reconstruction is verified against every case's own
                     recorded net values in tests/. Also owns `repo_context_block`'s only
                     eval-side caller, so the benchmark prompt cannot drift from the
                     shipped one — the fine-scale probability map is fitted to that prompt
  exp_finescale.py   ~$1, E2 — THE WINNER, now shipped as src/reporadar/finescale.py.
                     Scores each paper 0-9 and reads the EXPECTATION over the answer
                     token's logprob distribution (gpt-4o-mini; Anthropic exposes no
                     logprobs). Band AUC 0.84, and through a frozen 2-parameter logistic
                     at P >= 2/3 it takes mean net@2 +1.91 -> +3.14. `--arm haiku` is the
                     Anthropic-native alternative — 10 samples per paper instead of exact
                     logprobs — measured at AUC 0.59 and rejected: 44% of papers return
                     the same digit on 9+ of 10 draws, so sampling re-reads the mode
                     rather than the distribution
  verify_iacr_deps.py
                     $0, keyless: stage-1 dependency checks for the IACR ePrint source,
                     run BEFORE the adapter existed. Also where the two-case subset
                     (crypto, encryption) was PRE-REGISTERED — on the 25-case mean a
                     perfect adapter caps at +0.68, under the 1.04 floor, so the whole
                     benchmark cannot see this change however good it is

  noise_floor.py     $0, reads two completed runs of the SAME config: what is the
                     smallest effect this benchmark can resolve? Measured 2026-08-10 on
                     the shipped path — per-case sd 1.73, SE of a 22-case mean 0.37,
                     MINIMUM RESOLVABLE EFFECT 1.03 net@2/case. Size experiments against
                     that number BEFORE running them: the stated-intent goal arms (+0.44,
                     +0.12) were below the floor before they started, so their nulls cost
                     ~$30 and told us nothing. Refuses to compare a frozen-pool run with
                     a live one

  ablation_report.py $0, reads run files: the thin-docs dose response. Pair with
                     `run_judge_eval.py --rr-ablate-docs CHARS`, which builds RepoRadar's
                     profile from a repo whose README is capped at CHARS and whose docs/
                     is withheld, while the JUDGE still sees the real repo — ablate that
                     too and the ground truth degrades with the treatment, so a confused
                     system and a confused judge agree and the arm measures nothing.
                     Manifests are copied verbatim: a repo with no docs still declares
                     its dependencies. Exists because the benchmark's thinnest README is
                     1,639 chars against a 300-char prose budget, so nothing here has
                     ever run in the regime the target user lives in. At CHARS >= 300 the
                     gate's prose block is IDENTICAL to the control's, which is what
                     separates retrieval degradation from prompt degradation — the older
                     prose-budget arms could not, since docs were abundant either way
  calibrate_finescale.py
                     ~$0.30 (then $0 with --analyse): is the shipped probability map still
                     where it was fitted? Re-gates and re-scores the 220 top-10 papers of a
                     live run and checks the map against judge verdicts already on disk.
                     Reports the REPRODUCTION first — 117/121 shown papers, 97% — because a
                     reconstruction that does not reach the live decision is measuring
                     itself. Verdict 2026-08-09: decalibrated (under-confident by -0.129,
                     CI [-0.187, -0.067]) and worth +0.00 net@2 to fix under LORO, against
                     an oracle-threshold ceiling of +0.27. Cannot tune the threshold: 2/3 is
                     derived from 3p-2, so "the threshold that scores best here" is the
                     metric fitted to itself, and the LORO refit is the only honest arm
  bigram_report.py   $0, reads three run files: the phrase-query arms. `build_queries`
                     paired each keyword with its TF-IDF NEIGHBOUR and sent the pair as a
                     quoted phrase, with nothing requiring the two words to belong
                     together — "use page" for duckdb, "data cd" for redis. Verdict
                     2026-08-12 on 25 cases: `verified` (only phrases the repo actually
                     contains) +0.04 net@2/case vs the old behaviour, p = 0.55, INSIDE the
                     1.04 floor; `none` (no phrase queries) -0.48 and precision 0.914 ->
                     0.880, so deleting them is refuted rather than untested. The reason
                     nothing resolves is that arXiv's category clause keeps results in the
                     right field however meaningless the phrase — the benchmark measures
                     the one channel where the bug does not bite. Checks arm VALIDITY
                     (top-10 divergence) before believing any delta, so a flag that
                     changed nothing reads VOID rather than "no effect", and refuses an arm
                     whose run file records a different `bigram_mode` than its label
  source_ab_report.py
                     $0, reads two run files: does adding a paper source help? Verifies the
                     arms from their CONTENT (a source stamps `ss:`/`dblp:`/`iacr:` on what
                     it contributes, so a swapped pair is caught even though run files do
                     not record their --sources), then reports the mean THREE ways: all
                     cases, excluding negative controls, and controls only — read from
                     benchmark.yaml, never hardcoded. For the controls it prints the
                     judge's score distribution on exactly the papers the treatment added,
                     because `gold_n: 0` encodes "no gold ARXIV papers" and Tier B never
                     sees the label. Verdict 2026-08-13 for Semantic Scholar: -1.05
                     net@2/case on the 22 real cases, precision 0.908 -> 0.854, two repos
                     net-negative where the control had none. Distinguishes "past the
                     floor" from "established" — a mean beyond the MRE whose CI still spans
                     zero is suggestive, not resolved
  (arXiv cache)      Not a script: importing `harness` enables `reporadar.arxiv_cache`
                     into evals/.work/arxiv-cache, so every eval and diagnostic shares one
                     set of arXiv responses. A 25-case sweep is 174 queries, byte-identical
                     between runs, and on 2026-08-12 four sweeps in a day (~760 requests)
                     had their last two cases refused after 930s of waiting out throttles —
                     the rate limiter was correct throughout, nothing tracked VOLUME.
                     Measured on `rag`: cold 5 requests / 12.2s, warm 0 requests / 0.1s,
                     identical 150 papers. Keyed on query + max_results + sort_by;
                     lookback_days is NOT in the key because it filters after the fetch, so
                     one all-time response serves any window. OFF in the product — a
                     six-hour-old answer to a daily digest is an unmeasured behaviour change
  s2_yield.py        $0, no LLM, needs SEMANTIC_SCHOLAR_API_KEY: can S2's papers reach a
                     ranked top-10 at all? The stage-1 gate before the ~$26 judged A/B —
                     the same check that, run earlier, would have caught DBLP returning
                     nothing before four attempts to benchmark it. Verdict 2026-08-12 on
                     23 of 25 cases (arXiv throttled the other two, reported FAILED rather
                     than empty): S2 delivers ~211 new papers per case, ~175 of them
                     non-arXiv, and 73 reach a top-10 across 16 cases. But 22 of those 73
                     land in the three NEGATIVE CONTROLS — `webdev` took 10/10 slots,
                     `http` 9/10 — where the correct output is nothing. Optimistic by
                     construction: no HyDE (~100 more competing candidates) and no triage
                     rerank, so treat the counts as an UPPER bound
  openalex_yield.py  $0, no LLM, needs OPENALEX_API_KEY: the same stage-1 question for the
                     one source of five never measured in any form. Verdict 2026-08-14 on
                     25/25 cases (no refusals, 0 arXiv requests — the response cache served
                     all 174): OpenAlex delivers ~230 papers per case, ~229 of them
                     genuinely non-arXiv (only 32 across the whole sweep are a pool paper
                     re-badged under an `oa:` id), and **14 reach a ranked top-10 across 7
                     cases**. But 6 of those land in the three NEGATIVE CONTROLS and 5 in
                     `numerics`, whose arXiv pool (55) is a quarter of the median — leaving
                     THREE won on merit across 25 cases. Do not spend on the A/B.
                     Reports a case with any refused request as UNMEASURED, never zero:
                     the adapter returns [] for both a refusal and an empty answer, so this
                     wraps its request function and counts None returns. `--from-json`
                     re-derives the summary and verdict at $0 from a stored run, which is
                     how the FIRST verdict here got corrected — it read `cases_with >= n/4`,
                     cleared it by three quarters of a case, and said "a judged A/B is
                     justified" while 11 of the 14 slots sat where placing is worthless
  audit_product_divergence.py
                     $0, no network, no LLM: where does the benchmark stop measuring the
                     PRODUCT? C-9 (the query bridge at five call sites) and C-12 (the
                     version-strip fixed in cli.py, not in harness.py) are the same defect
                     — one invariant, two implementations — and both were found by
                     accident while looking for something else. Three passes: the WIRING
                     (every arXiv-id normalisation, source merge and digest-window
                     derivation, read out of the AST, with which of the competing rules it
                     uses), the CONFIGURATION (shipped
                     defaults against the benchmark's headline flags, where a difference is
                     fine and an UNDECLARED one fails a test), and the BLAST RADIUS (how
                     much of it reached a published number, read off evals/results/).
                     Verdict 2026-08-14: five divergences, of which one was a live product
                     bug (an ungated paper could reach Top Picks on the 0.5 heuristic —
                     the threshold Feature 6 replaced at net@2 -11) and one was C-12 again,
                     unfixed in run_eval.py because the guard named harness.py explicitly.
                     Zero published numbers move: the gate never once failed on a paper
                     across 87 runs and 6,420 top-10 records. Run it before believing a
                     benchmark number
  audit_query_transform.py
                     $0, no LLM, free APIs: what did the broken arXiv-to-keyword bridge
                     (C-9) actually do to each non-arXiv source? Runs REAL build_queries
                     output through the old and new transforms and asks DBLP, bioRxiv and
                     optionally S2 for both. Verdict 2026-08-12: DBLP returns 0 for every
                     malformed query (0 -> 1 and 0 -> 4 hits once repaired, at all-time
                     lookback so its year filter is not what is being measured), while
                     bioRxiv keeps its ENTIRE window — its local filter admits any query
                     word over two characters, and the surviving word was the boolean
                     operator AND, matching 90/90 abstracts where every real term matched
                     0/90. Enabling bioRxiv did not add biology papers, it turned the
                     topical filter off. Treats a refused request as unmeasured, never as
                     a zero: the first pass at this reported "0 vs 0" for DBLP after it
                     silently refused 12 of 18 requests.
                     `--sources s2` needs SEMANTIC_SCHOLAR_API_KEY in evals/.env (keyless
                     callers share one pool with every unauthenticated user and were
                     refused 20/20 twice). Verdict 2026-08-12 on the 12 original cases:
                     S2 returns 0 papers for the malformed query in 11 of 12, 20 for the
                     repaired one in 12 of 12, ZERO overlap — so RESULTS.md finding 3
                     ("adding Semantic Scholar did not help") measured an arm that
                     contributed no papers. VOID, not null
  compare_finescale_baseline.py
                     $0, calls nothing: the head-to-head against the Opus baseline on all
                     22 repos. The baseline's picks and their verdicts are already in the
                     frozen run file, so both systems are scored on the same candidates by
                     the same judge. +3.14 vs +1.82, paired +1.32, 10/6/6 — sign test
                     p = 0.45, so ahead on the mean and NOT established per repo. Uses the
                     same model family as the shipped map, deliberately (see its docstring)
  exp_select.py      ~$6 (Sonnet) / ~$0.50 (Haiku), E1: shuffled subset selection ("select
                     what maintainers should act on — possibly none", R=15 shuffles, vote
                     share as the score). NEGATIVE: AUC 0.64/0.62, policy below show-all.
                     Abstention fires correctly but both models are far too strict —
                     Sonnet selected nothing in 15/15 shuffles on two net-positive repos
  exp_ensemble.py    ~$3, E3: 10 Haiku votes per paper under 10 different reviewer personas,
                     each stating the strongest reason NOT to act, plus a verbalized
                     probability. KILLED by its own pre-registered bar at ECE 0.425 — and
                     in the direction the literature did not predict: chronic
                     UNDER-confidence, P-hat collapsing toward 0 regardless of label
  exp_pairwise.py    ~$3, E4: full round-robin pairwise comparison in BOTH orders (Claude
                     position bias is severe), Bradley-Terry by MLE, plus template anchors
                     meant to give the scale an absolute zero. Band AUC 0.64 misses the
                     0.70 bar; the anchors failed outright — 128/128 real papers beat the
                     borderline anchor, so it discriminates nothing
  exp_features.py    $0, E5: L2 logistic over free metadata (age, S2 citations, HyDE rank,
                     hop coupling), leave-one-repo-out throughout. Features alone reach
                     AUC 0.585 — below the pre-registered 0.60, confirming the
                     practitioner-relevance literature's warning that citation counts do
                     not predict what engineers act on. `--combined` adds E1-E4 as columns
                     and is beaten by E2 alone on every axis. NOTE: its inner-CV picks the
                     L2 strength by AUC, which is rank-only and therefore blind to where
                     P crosses a threshold — see RESULTS.md "Correction" before reusing it
                     to evaluate a thresholded policy

  why_case.py
                     $0, offline: where a benchmark case's ranked papers stopped, and where
                     the gate and the judge DISAGREE -- reported as disagreement, because
                     second_judge.py measured kappa 0.507 on the >=2 cut and only 8 of 48
                     papers GPT scored 2 were scored >=2 by Sonnet. `rr why` answers this
                     against a product store; the eval harness writes none, which is why
                     this adapter exists. Not available from an artifact: score_total and
                     rrf_score. finescale_p IS carried, for score-2 band papers -- an earlier
                     version of this file said otherwise; see RESEARCH §18.1.

  finescale_domains.py
                     $0, offline: is the fine-scale map still calibrated on the new domains?
                     Reads finescale/finescale_p from a results artifact and hands them to
                     calibrate_finescale.analyse, so no clone, no cache and no API key. Prints
                     the judge-free half and the judge-dependent half under separate headings
                     and projects every judged conclusion through second_judge.py's transition
                     table BEFORE stating it -- the check that would have caught RESEARCH
                     §17.2 before it was written.

  gate_shape.py
                     $0, judge-free: the gate's score distribution per case and its emission
                     rate per population. Exists as the control RESEARCH §17.4 never ran, and
                     refutes it -- the median case puts 73% of a ranked window in one bucket
                     and two cases are at 100%, so concentration alone is not a defect. What
                     survives needs no judge: the gate emits score 3 on 20.0% of scientific
                     papers against 8.0% of ML/CS ones (Fisher p=7.7e-05) and score 0 never.

  probe_score3_band.py
                     ~$0.03, pre-registered in RESEARCH-scientific-software.md §9: on six
                     scientific-software repositories the benchmark's finding about the gate
                     INVERTED -- its score-3 papers were actionable 25/36 while score-2
                     papers clearing the fine-scale rescore were 28/29. Two candidate fixes,
                     both scored against judge verdicts already cached (nothing is
                     re-judged, so the labels cannot drift to suit the answer). BOTH FAILED
                     their pre-registered bars: the rescore over the score-3 band ranks it at
                     AUC 0.710 against 0.84 on the band it was fitted to and drops 2 of 8
                     misses (bar 4) though it costs nothing; a rubric clause catches 8 of 9
                     misses and takes 14 of 52 actionable papers with it, killed by its own
                     >5 clause, worth +0.33 net@2/repo against a 1.03 floor for a 36% smaller
                     digest. The useful finding is that the gate and the rescore fail
                     TOGETHER -- MACE's own paper scores gate 3 and P 0.926 -- so the rescore
                     is not an independent second opinion on what the gate admits
  verify.py          resolve proposed papers against real arXiv (hallucination guard)
  .env.example       template for API keys (copy to .env)
  repos/<case>/      realistic mini-repos profiled in Tier A offline mode
  fixtures/<case>.json   frozen labeled arXiv pools (committed)
  cache/, results/   Tier B verdict cache + run outputs (gitignored)
```
