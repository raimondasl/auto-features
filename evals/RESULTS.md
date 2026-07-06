# Tier B benchmark — results

**Baseline run:** 2026-07-04, `main` @ PR #16 · judge = **GPT-5.5** · baseline = **Opus 4.8**
(`--baseline cli`). **Feature 6 run:** 2026-07-05, `main` @ PR #18 (`--rr-triage`, triage
model **claude-haiku-4-5**). **Baseline reference parsing corrected 2026-07-05** — see
[Harness fix](#harness-fix--baseline-reference-parsing-2026-07-05) below; the Opus responses
themselves are unchanged (cached), so the corrected numbers were re-derived with no new model
calls. Commands:

```bash
uv run python evals/run_judge_eval.py --baseline cli               # baseline (heuristic 0.5 gate)
uv run python evals/run_judge_eval.py --baseline cli --rr-triage   # Feature 6 (LLM triage gate)
```

Re-run and compare after a ranking change. (LLM-judged runs are non-deterministic; treat
±1 net@2 as noise. Judge verdicts and baselines are cached, so a re-run mostly re-uses them.)

## Results

Metrics defined in [`metrics.py`](metrics.py) / explained in [`README.md`](README.md#tier-b--actionable-improvement-llm-judged).
`net@2 = (#actionable) − 2·(#non-actionable)` over a system's returned papers; precision
is `n/a` when a system abstains. RepoRadar is shown two ways: its **Top Picks** tier
(score ≥ 0.5, the abstention-respecting output) and a **Top-10** diagnostic.

| Case (repo) | RepoRadar Top Picks | RepoRadar Top-10 | Opus baseline |
|---|---|---|---|
| **rag** (stanford-futuredata/ColBERT) | abstained | 2/10 · prec 0.20 · net **−14** | 3/3 · prec 1.00 · net **+3** |
| **cv** (facebookresearch/detectron2) | 0/2 · net −4 | 1/10 · prec 0.10 · net **−17** | 3/3 · prec 1.00 · net **+3** |
| **rl** (DLR-RM/stable-baselines3) | 0/10 · net **−20** | 0/10 · net −20 | 3/3 · prec 1.00 · net **+3** |
| **webdev** (pallets/flask) — negative control | 0/10 · net −20 | 0/10 · net −20 | abstained (0 refs · net **0**) |
| **mean** (rag/cv/rl, verifiable) | — | precision **0.10** · net **−17** | precision **1.00** · net **+3.0** |

## Findings

1. **Opus wins decisively on every verifiable case** (mean net@2 **+3.0 vs −17**, precision **1.00 vs 0.10**). RepoRadar's keyword+recency ranking surfaces topic-adjacent but non-*actionable* papers.

2. **The decisive nuance — `recent-only net@2 = 0.0` for the baseline on all three cases.** Every Opus pick was older/seminal (`recent=0/N`). Restricted to RepoRadar's 90-day fetch window, Opus recommends nothing actionable either. So:
   - **RepoRadar's fixable problem is precision, not discovery.** It confidently ranks 8–10 non-actionable papers into its top tier. Filtering those (LLM triage → correct abstention) targets **net@2 from −17 → ~0**.
   - **Opus's headline win is largely a paper-age artifact** — it cites timeless foundational work RepoRadar structurally can't see. Closing *that* gap needs a scope change (a seed/foundational-corpus mode), not a ranking fix.

3. **The 0.5 "Top Picks" threshold is miscalibrated.** On `rl` and `webdev`, RepoRadar put **all 10** candidates above 0.5 with **0 actionable** — overconfident, not conservative.

4. **The negative control fails in the actionable sense.** For a web framework, RepoRadar confidently returns 10 ML papers, none useful. (Tier A domain-sanity "PASS" doesn't survive an actionability judge under the live category fallback.)

5. **The negative control is a two-sided win (after the harness fix).** Opus's honest answer for Flask was to recommend nothing (its structured output was `[]`); RepoRadar now also abstains via triage. The earlier `arxiv_unverified` on `webdev` was **not** a real arXiv outage — it was a harness bug that scraped a ResearchGate URL out of Opus's "sources reviewed" prose as a bogus arXiv ID (`publication/2256929`), which 400'd on lookup and suppressed the baseline's (correctly empty) metrics. Fixed 2026-07-05 — see [Harness fix](#harness-fix--baseline-reference-parsing-2026-07-05).

## Feature 6 result — LLM triage gating Top Picks (2026-07-05)

`--rr-triage` replaces the miscalibrated 0.5 "Top Picks" threshold with an LLM
actionability gate (triage model **claude-haiku-4-5**): a paper is a Top Pick only if
the LLM judges it genuinely applicable (score ≥ 2). This changes RepoRadar's
**user-facing returned set**; the raw **Top-10** diagnostic (ungated ranking) is
unchanged by design — triage gates, it does not rerank.

**RepoRadar Top Picks — net@2, before vs after:**

| Case | Before (0.5 gate) | After (LLM triage) |
|---|---|---|
| **rag** | abstained · net 0 | **1/1 actionable · precision 1.00 · net +1.0** |
| **cv** | 0/2 · net **−4** | **abstained · net 0.0** |
| **rl** | 0/10 · net **−20** | **abstained · net 0.0** |
| **webdev** (negative control) | 0/10 · net **−20** | **abstained · net 0.0** |
| **mean** | **−11.0** | **+0.25** |

**What moved:**

1. **User-facing Top Picks: mean net@2 −11.0 → +0.25.** The gate eliminated the
   catastrophic false-positive dumps — `rl` and `webdev` went from **−20** (all 10
   confidently returned, 0 actionable) to a **correct abstention**.
2. **Calibration fixed.** The three repos with nothing genuinely applicable in-window now
   **abstain** instead of returning junk; precision on what RepoRadar *does* return is **1.00**.
3. **The negative control passes.** `webdev` (Flask) now returns nothing — as it should.
4. **Cost:** the judge pool is unchanged, so GPT-5.5 verdicts and the Opus baselines are all
   cache hits — the only new spend is ~10 Haiku triage calls per case (pennies).

**Caveats / next levers:**

- **Recall cost — and it is not a model-strength problem.** On `rag` the judge found 2
  actionable papers in the Top-10 but triage surfaced 1; on `cv` the judge found 1 but triage
  abstained. Re-running with `--rr-triage-model claude-sonnet-5` gave **metric-identical**
  results (rag 1 Top Pick; cv/rl/webdev abstain), so the gap is triage vs. GPT-5.5-judge
  *disagreement on ~1 borderline paper per case*, not a weak triage model. The real lever is
  the rubric/threshold (`triage.min_actionable: 1`), not a bigger model. The gate still never
  emits a false positive; it leaves some defensible papers in "Maybe".
- **Top-10 is unchanged** (−14/−17/−20) because triage only gates the returned set. Reranking
  the Top-10 by `llm_score` (the deferred half of Feature 6) would also lift the diagnostic.
- **The baseline still returns more actionable papers**, but all older/seminal (`recent=0/N`) —
  the foundational-corpus gap, which is a scope change, not a ranking fix.

## Harness fix — baseline reference parsing (2026-07-05)

The webdev baseline reported `arxiv_unverified` (no metrics) on *every* run — deterministic,
not a transient arXiv outage. Root cause was in how the harness extracts the baseline's
recommendations from its answer:

1. **Prose scraping overrode an explicit abstention.** `_parse_recommendations` parsed the
   baseline's structured ```json block, then *also* unioned in any arXiv-looking ID found in the
   surrounding prose. For Flask, Opus correctly recommended nothing (`[]`) and listed the papers
   it had reviewed-and-rejected under "Sources reviewed". The harness scraped those into 4
   phantom "recommendations".
2. **The old-style-ID regex matched a non-arXiv URL path.** One scraped "ID" was
   `publication/2256929`, pulled from `researchgate.net/publication/225692935_Hardened…`. It
   matched the `hep-th/9901001`-style pattern (`[a-z-]+/\d{7}`). That bogus ID returned **HTTP
   400** from the arXiv API → classified as an arXiv lookup failure → whole baseline marked
   `arxiv_unverified`, suppressing its (correctly empty) metrics.

Prose scraping also silently **over-credited `rag`**: it added `2304.01982` (XTR) and
`2505.11471` (CRISP) — papers Opus explicitly excluded from its JSON (*"it's really a
replacement model… not an improvement you graft onto the existing pipeline"*) — inflating the
baseline from its actual 3 recommendations to 5.

**Fix** (this PR): the ```json recommendation block is authoritative — an explicit `[]` is an
abstention, and prose is discussion, not recommendations. Prose scraping is used only as a
fallback when the baseline emits no structured block at all. The old-style-ID regex is
restricted to real arXiv archive prefixes. Cached baselines are re-parsed from their stored
`raw` on load, so this fix applies to the existing cache with **no new model calls**. Regression
tests in [`tests/test_eval_parsing.py`](../tests/test_eval_parsing.py).

**Corrected baseline column** (RepoRadar and triage columns are unaffected): `rag` **+5 → +3**
(drops the 2 rejected refs), `cv`/`rl` unchanged (**+3**), `webdev` **`arxiv_unverified` →
abstained (0 refs · net 0)** — the strong baseline *also* correctly recommends nothing for
Flask. Mean (rag/cv/rl) **+3.7 → +3.0**.

## All-time discovery experiment — is the baseline's edge a paper-age artifact? (2026-07-06)

Finding #2 argued the baseline wins largely because it cites **seminal older work** that
RepoRadar's 90-day fetch window structurally can't see (`recent=0/N` on every baseline pick).
`--rr-all-time` tests that directly: RepoRadar discovers from **all of arXiv, relevance-sorted,
recency weight dropped**, so old seminal papers can surface and compete.

```bash
uv run python evals/run_judge_eval.py --baseline api --rr-triage --rr-all-time
```

> **Baseline caveat:** this run used `--baseline api` (Anthropic API + web_search), a *different,
> noisier* baseline than the `--baseline cli` used in the tables above (it re-searches live, so
> `rl` landed **+3 → −3** and `cv` **+3 → +2**). So the baseline column here is this run's own
> reference, **not** comparable to the headline cli baseline. The clean comparison below is
> RepoRadar-vs-RepoRadar (triage = claude-haiku-4-5 in both; only discovery changed).

**RepoRadar, 90-day recency vs. all-time/relevance (both haiku triage):**

| Case | Top Picks 90-day | Top Picks all-time | Top-10 90-day | Top-10 all-time | api baseline (ref) |
|---|---|---|---|---|---|
| **rag** | 1/1 · net **+1.0** | 2/3 · prec 0.67 · net **0.0** | 2/10 · net −14 | 4/10 · net **−8** | 4/4 · net +4 |
| **cv** | abstained · net 0 | 2/3 · prec 0.67 · net **0.0** | 1/10 · net −17 | 5/10 · net **−5** | 2/2 · net +2 |
| **rl** | abstained · net 0 | abstained · net 0 | 0/10 · net −20 | 0/10 · net −20 | 1/3 · net **−3** |
| **webdev** | abstained · net 0 | 0/1 · net **−2** | 0/10 · net −20 | 0/10 · net −20 | abstained · net 0 |
| **mean** | **+0.25** | **−0.5** | −17.0 (rag/cv/rl) | **−11.0** (rag/cv/rl) | — |

**Findings:**

1. **Discovery hypothesis: confirmed.** All-time/relevance surfaced substantially more
   *genuinely actionable* papers into RepoRadar's candidate pool and Top-10 — actionable-in-Top-10
   went **rag 2 → 4, cv 1 → 5**, and Top-10 mean net@2 improved **−17.0 → −11.0**. The seminal
   older papers RepoRadar structurally couldn't see are now in the pool and the ranker floats
   several into the top-10. (The pre-merge live check had already shown it resurfacing
   `1511.05952`, one of Opus's own `rl` cites.) **The paper-age artifact is real and now
   addressable.**

2. **But the headline (abstention-aware Top Picks net@2) did *not* improve — it slightly
   regressed, +0.25 → −0.5.** The bottleneck moved from **discovery to precision**. On the larger,
   older, more on-topic pool the haiku gate (`min_actionable=2`) admits more borderline papers:
   Top-Picks precision fell **1.00 → 0.67** on `rag`/`cv`, and the negative control leaked its
   **first false positive** (`webdev`: 1 Top Pick, 0 actionable → net −2).
   - It is a recall/precision *retrade*, not a pure loss: `rag` Top Picks found **more** actionable
     papers (1 → 2) — discovery helped — but also admitted 1 non-actionable, so net went +1 → 0.

3. **`rl` stays hard, and RepoRadar's abstention was the correct call.** Only **1** paper in the
   entire `rl` pool was judged actionable; RepoRadar abstained (net 0) while the api baseline
   recommended 3 (1 actionable) for net **−3**. Here the conservative gate **beat** the baseline.

4. **`webdev` confirms all-time has a cost on out-of-domain repos.** More candidates → more chances
   to fool the gate. On a framework with nothing actionable on arXiv, all-time turned a correct
   abstention into a false positive. A discovery widening should probably be paired with a
   *stricter* gate, not a looser one, on low-signal repos.

**Conclusion / next lever.** All-time/relevance discovery is a genuine win at the *candidate* and
*Top-10* level (more actionable papers found, Top-10 net −17 → −11) and it validates the paper-age
hypothesis — but it does **not** yet convert into a Top-Picks headline win, because the
keyword-TF-IDF ranker + haiku gate can't cleanly separate the genuinely-actionable seminal papers
from the merely-relevant ones. The two deferred Feature-6 levers now matter most: **(a) rerank the
Top-10 by `llm_score`** so the actionable papers rise and the gate sees a cleaner head, and **(b)
calibrate the gate** (precision fell, so tighten the rubric / hold `min_actionable=2` — a *stronger*
triage model already showed no benefit). Discovery is solved; **precision on the enriched pool is
the remaining gap.**
