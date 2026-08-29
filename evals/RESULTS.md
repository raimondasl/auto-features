# Tier B benchmark — results

> ## ⚠ Every number below this line was measured at a digest window of **10**
>
> On **2026-08-15** the benchmark's returned-set cut moved 10 → 15, to match the shipped
> `output.top_n`, because widening it measured **+1.24 net@2/case** (CI [+0.48, +2.08]) —
> larger than any treatment effect published in this file. **91 of the 92 runs recorded
> before that date are window-10**, and every headline, precision figure and shown/actionable
> count below therefore describes a *narrower* system than the one that ships.
>
> What this does and does not invalidate:
>
> * **Levels understate.** Means, shown counts and the paired-vs-baseline advantage all
>   grow with the width. The one measured shift is +1.24/case on an arXiv-only frozen pool.
> * **Paired deltas stand.** HyDE +1.36, the rescore +1.36, gate depth +1.00,
>   absent-category +0.00, phrase queries +0.04, S2 — both arms of each were window-10, so
>   each is a sound estimate *at window 10*. None is re-run, and each is labelled rather
>   than restated.
> * **Nothing else moved.** `triage.top_k` 15 → 50 changed the product to match what the
>   benchmark already ran; C-13, the fine-scale scoping and the `paper_id` consolidation are
>   all outcome-neutral for the benchmark.
>
> Runs record `digest_window`, and the reports **refuse** to compare across it without an
> explicit `--across-windows`.

> **Headline (2026-08-09, 22 cases, live end-to-end): RepoRadar reaches mean net@2 +4.55
> against the Opus 4.8 baseline's +1.82 — paired +2.73, 15 cases better, 3 worse, 4 tied,
> sign test p = 0.0075.** The first result in this project's history that clears p < 0.05
> against the baseline; every previous headline was "ahead on the mean, not established".
> It now matches Opus's precision (**0.94**) while returning **2.5x as many papers**, with
> **zero net-negative repositories** against the baseline's one. See
> [HyDE measured end to end](#hyde-measured-end-to-end-the-first-result-that-clears-p--005-against-the-baseline-2026-08-09).
>
> Two shipped changes got it there, each worth about the same and neither significant alone:
> the **fine-scale rescore** (+1.36 over show-all, p = 0.109) and **HyDE dense discovery**
> (+1.36 over the rescore alone, p = 0.092, measured in the same session with HyDE the only
> variable). They compose rather than compete — precision *rose* 0.91 → 0.94 while the shown
> set grew 97 → 121 — because the rescore orders what a bigger pool admits. The same pool
> expansion without it was a
> [measured wash](#gating-the-whole-pool-end-to-end-a-wash-2026-08-07).
>
> **Recall, the standing open problem, is narrowed rather than closed.** HyDE was shipped to
> fix the six losses that were pure retrieval misses; three are fixed outright — `speech`
> 0 → +10, `graph` +1 → +10, and `rag` 0 → +4, the case that had been admitting *nothing* —
> while `llminfer` and `numerics` remain genuine losses.
>
> **A caveat that applies to both changes.** net@2 charges 2 for a false positive, so it
> rewards precision-preserving work — this metric flatters these results. And at n = 22 the
> per-case values move substantially between collections; the aggregate has now held across
> four runs, the individual repositories have not.
>
> **Previous headline (2026-07-31, 12 cases):** RepoRadar Top Picks mean net@2 **+1.75** vs the Opus
> baseline's **+1.83** — a **0.08 gap**, narrowed from 0.33. Measured after the query-construction
> fix (PR #59), which changed **two thirds of the queries** the benchmark transmits.
> **Read the per-case table, not the mean**: the improvement is concentrated where the bug was
> worst and one case regressed hard. See
> [Re-benchmark after the query-construction fix](#re-benchmark-after-the-query-construction-fix-2026-07-31).
> That regression is diagnosed below, but **read its conclusion about triage as superseded**:
> it inferred "no discriminative signal" from ~10 papers. Measured on all 428 labelled papers,
> triage runs at **precision 0.81 / recall 0.78 against a 32% base rate** — well above chance.
> What survives is narrower and sharper: triage collapses *specifically on the ranker's top-10*
> (0.33 vs 0.82 elsewhere), which is the only subset Tier B ever judges. See
> [Triage measured properly](#triage-measured-properly--it-is-not-at-chance-2026-08-02).
>
> **Headline (2026-08-02, 12 cases): RepoRadar Top Picks mean net@2 +2.75, up from +1.75.**
> Paired over cases **+1.00, 95% CI [+0.00, +2.00], P(Δ≤0) = 0.032**; 7 cases improved, 1
> worsened; it survives dropping the largest mover (+0.73). At an unchanged digest size the
> **junk papers a reader wades through fell from 9 to 5** across 12 repos, and precision at
> `min_actionable=2` rose 0.85 → 0.92. The cause is one prompt change: the gate is now told
> what the repo is *for*, using **300 characters** of its own README — the best-measured of
> four budgets, though **not a demonstrated optimum**: 300 vs 2,000 is +9 at P = 0.108, and a
> prefix fails outright on `graph`, whose first 300 characters are link badges. See
> [How much prose](#how-much-prose-does-the-gate-need-some-beats-none-the-amount-is-unresolved-2026-08-02).
> **Against Opus it is still parity** (+2.75 vs +1.83, but paired +0.92, CI [−0.67, +2.75]).
>
> Widening the triage window 20→50 remains a **negative result** — 4× the candidates bought
> 2 more actionable papers — see
> [Negative result 5](#negative-result-5--widening-the-triage-window-from-20-to-50-does-not-pay-2026-08-02).
>
> **`min_actionable=2` — three sweeps against, one for.** Three two-case sweeps had the
> strictest gate winning *by abstaining*, because triage found 0 of `rag`'s 3 actionable papers
> and 2 of `speech`'s 6. The 12-case sweep on 2026-08-02 goes the other way: `min>=2` wins at
> **+2.42** against −2.25 (`min>=1`) and +0.42 (`min>=3`), with 0 false positives. The 12-case
> evidence is the stronger of the two, but read the default as contested rather than settled.
> See [Two-case re-benchmark](#two-case-re-benchmark-after-the-quoting-fix-2026-08-02).
>
> **Previous headline (2026-07-29, 12 cases):** RepoRadar **net-positive** (Top Picks mean net@2
> **+1.50**) and **competitive with the Opus baseline** (+1.50 vs +1.83 — a **0.33 gap, unchanged**
> from 2026-07-12), *beating* it outright on the ML domains it's built for (diffusion,
> speech, peft). `min_actionable=2` is the decisively-correct gate. **Nine features shipped between
> the two runs and the validated configuration did not regress** — see
> [Re-benchmark](#re-benchmark-after-features-1-8-and-10--no-regression-2026-07-29) below.
> The [12-case benchmark](#12-case-benchmark--reporadar-is-net-positive-and-competitive-with-opus-2026-07-12)
> is the original of that measurement; the 4-case tables that follow are the earlier snapshots that
> got us there.

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

## Listwise rerank — does ordering by actionability convert discovery into Top Picks? (2026-07-06)

The all-time run above surfaced actionable papers into the *pool* but not the *returned set*: the
heuristic `score_total` order still decided who entered the Top-N window before the gate saw them.
`--rr-rerank` (PR #23) closes that — it triages a deeper pool (20 candidates) and reorders by
`llm_score` before the Top-10 cut, so a buried-but-actionable paper can reach Top Picks.

```bash
uv run python evals/run_judge_eval.py --baseline cli --rr-rerank --rr-all-time
```

(cli baseline this time — the strong one, mean net@2 **+3** on the three verifiable cases.
RepoRadar's own columns don't depend on baseline mode, so they compare directly to the run above.)

**RepoRadar Top Picks — all-time, rerank off → on (both haiku triage):**

| Case | rerank off | rerank on |
|---|---|---|
| **rag** | 2/3 · prec 0.67 · net 0.0 | 2/3 · prec 0.67 · net 0.0 |
| **cv** | 2/3 · prec 0.67 · net 0.0 | **3/4 · prec 0.75 · net +1.0** |
| **rl** | abstained · net 0 | abstained · net 0 |
| **webdev** | 0/1 · net −2 | 0/1 · net −2 |
| **mean** (rag/cv/rl) | 0.0 | **+0.33** |
| **mean** (4 cases) | −0.5 | −0.25 |

**Findings:**

1. **Reranking does exactly what it's for — where discovery left an actionable paper below the
   window, it surfaces it into the returned set.** `cv` went **net 0.0 → +1.0** (2 → 3 actionable
   Top Picks, precision 0.67 → 0.75): a third actionable paper, one that sat below the `score_total`
   top-10, was reordered into Top Picks by its `llm_score`. Mechanism confirmed.

2. **The remaining cap is gate *precision*, not ordering.** `rag` stayed at **net 0.0**: its
   actionable papers were already inside the window, so reranking had nothing to lift — its ceiling
   is the *one non-actionable paper the haiku gate admits* (a triage false positive: haiku scores it
   ≥2, GPT-5.5 scores it <2). Reranking can reorder papers but cannot un-admit one the gate wrongly
   passed. `webdev` is the same story (still −2).

3. **All-time discovery *needs* reranking to not be a headline regression.** Verifiable-case Top
   Picks mean net@2 traces **90-day +0.33 → all-time-no-rerank 0.0 → all-time+rerank +0.33**.
   Widening discovery alone diluted precision (more candidates, same window → the gate saw a noisier
   head); reranking recovered it by putting the actionable papers first. Together they **match the
   90-day headline while sourcing genuinely better picks** — seminal papers the 90-day window
   couldn't see (`cv` surfaced **4** actionable papers into its Top-10 this run vs **1** under the
   90-day window).

4. **The strong (cli) baseline still leads on net@2** — mean **+3** vs RepoRadar's **+0.33** on the
   three verifiable cases; on `rl` the baseline scored +3 while RepoRadar (correctly, given only 3
   actionable in a 13-paper pool and a conservative gate) abstained. Every baseline pick remains
   `recent=0/N` foundational work, so the residual gap is still part foundational-corpus scope, part
   gate precision.

**Conclusion — discovery and ordering are done; gate precision is the last lever.** All-time
discovery + listwise rerank close the discovery and ordering gaps: RepoRadar now *finds* seminal
papers and *floats them into Top Picks* (`cv` net 0 → +1 on foundational work), recovering the
90-day headline via strictly better picks. What's left is the **triage gate's own precision** — the
residual `rag`/`webdev` false positives are haiku scoring a paper ≥2 that the GPT-5.5 judge scores
<2. A stronger triage model already showed no benefit, so the final lever is the **rubric /
threshold**: tighten the actionability rubric, or raise `min_actionable` to 3 on low-signal repos
(which would also close the `webdev` negative-control leak). That is the single highest-impact
change left in this arc.

## Gate-precision sweep — which `min_actionable` threshold? (2026-07-12)

The last lever, measured directly. `--rr-sweep` re-gates the same triaged Top-10 at each threshold
in one run (free — triage scores are computed once), over all-time discovery + rerank + cli baseline.

```bash
uv run python evals/run_judge_eval.py --baseline cli --rr-rerank --rr-all-time --rr-sweep
```

**Cross-case rollup (4 cases):**

| `min_actionable` | mean net@2 | abstained | false-positive | mean precision |
|---|---|---|---|---|
| **≥1** (≈ungated) | −8.75 | 0/4 | 2/4 | 0.23 |
| **≥2** (current default) | −0.25 | 1/4 | 1/4 | 0.47 |
| **≥3** | **+0.25** | 3/4 | **0/4** | **1.00** |

**Per-case net@2 at each threshold** (returned · actionable in parens):

| Case | ≥1 | ≥2 | ≥3 |
|---|---|---|---|
| **rag** | −8 (10·4) | 0.0 (3·2) | 0.0 (abstain) |
| **cv** | −5 (10·5) | +1.0 (4·3) | +1.0 (1·1) |
| **rl** | −4 (2·0) | 0.0 (abstain) | 0.0 (abstain) |
| **webdev** | −18 (9·0) | −2.0 (1·0) | 0.0 (abstain) |

**Findings:**

1. **`min>=3` is the net@2-maximizing gate — perfect precision (1.00), zero false positives.** It
   **closes the `webdev` negative-control leak** (webdev abstains) and every paper it *does* return
   is genuinely actionable. This is the abstention-first ideal: never show a dud.

2. **But the win is narrow — it is entirely `webdev`.** The +0.5 mean gain from `≥2 → ≥3` is *only*
   webdev going −2 → 0. `rag` stays net 0, `cv` stays +1, `rl` stays 0. Raising the bar didn't make
   the good cases better; it silenced the one leak.

3. **The cost is recall, and net@2 hides it.** At `≥3`, **3/4 cases abstain** — `rag` returns
   **nothing** despite having **2 genuinely-actionable Top Picks at `≥2`** (and 7 actionable in its
   pool), because haiku scored them a confident **2, not 3**. net@2's λ=2 penalty makes `rag`'s "2
   actionable + 1 dud" (`≥2`, net 0) *tie* "abstain" (`≥3`, net 0) — so the metric is indifferent
   where a user, who would rather see 2 useful papers than nothing, is not.

4. **`min>=1` confirms the gate is essential** — precision 0.23, net −8.75, 2 false-positive cases.
   Without the actionability gate RepoRadar dumps noise; the gate is doing real work at any `≥2`/`≥3`.

5. **The residual driver is haiku's calibration, not the threshold.** It scored `webdev`'s dud a 2
   (should be 0–1) and `rag`'s real actionable papers a 2 (not 3). `min>=3` is a blunt-but-effective
   workaround; a better-calibrated rubric would let `≥2` keep `rag`/`cv` recall *without* the webdev
   leak. A stronger triage *model* already showed no benefit — so the lever is the **rubric**, not
   model size or threshold alone.

**Conclusion — the sweep maps a clean precision/recall frontier, and the choice is a product call:**

- **`min_actionable = 3`** — maximum precision (1.00), best net@2 (+0.25), no negative-control leak;
  aligned with the project's stated *"better to return nothing than papers that aren't genuinely
  relevant."* **Cost:** abstains even when genuinely-actionable papers exist (`rag` returns nothing).
- **`min_actionable = 2`** — more recall (surfaces `rag`'s 2 and `cv`'s 3 actionable papers) at the
  cost of occasional duds (precision 0.47) and the `webdev` leak.

Both are defensible; net@2 and the stated abstention-first philosophy both favor **3**. The deeper
fix that would beat *both* — recovering `≥2`'s recall *and* `≥3`'s precision — is tightening the
triage **rubric** so haiku stops scoring non-actionable papers a 2.

## Triage-rubric calibration attempt — the haiku ceiling (2026-07-12)

The rubric fix (PR #27): rewrite the triage rubric to encode the exact failure modes that fool a
lenient "2" (measurement-not-method, wrong-layer, application-level, general-tooling, wholesale-
replacement — the reasons the Opus baseline itself rejected the webdev papers) plus a **grounding
test** (to score 2+, name the concrete component of *this* repo the method changes). Goal: close the
`webdev` leak at `≥2` while keeping `rag`/`cv`'s genuine score-2 papers.

**Outcome — it missed** (`--rr-rerank --rr-all-time --rr-sweep`, `min>=2`, before → after):

| Case | before rubric | after rubric | verdict |
|---|---|---|---|
| **rag** | 3 · 2 act · net 0.0 | 3 · 2 act · net 0.0 | unchanged — dud persists |
| **cv** | 4 · 3 act · net +1.0 · prec 0.75 | 1 · 1 act · net +1.0 · prec 1.00 | **lost 2 genuine actionable papers** |
| **rl** | abstain | abstain | unchanged |
| **webdev** | 1 · 0 act · **net −2** | 1 · 0 act · **net −2** | **leak NOT closed** (its primary target) |
| **mean** | net −0.25 · prec 0.47 | net −0.25 · prec 0.56 | headline net unchanged |

It **did not close the `webdev` leak** — haiku still confidently scores one Flask paper ≥2 that the
judge scores 0. It **over-tightened `cv`** (dropped 2 genuinely-actionable papers to keep 1), and it
**flattened the score distribution so nothing scores a 3 anymore** — `min>=3` now abstains on **4/4**
cases (was 3/4), killing the high-precision tier. The only net effect was a marginal `cv` precision
bump; the actual defect is untouched.

**Conclusion — the gate levers are exhausted; this is haiku's capability floor.** Three independent
levers have now been tried on the triage gate and none closes the last-mile false positive:

1. **Threshold** (the sweep) — `min>=3` buys precision only by over-abstaining (silences real papers).
2. **Stronger model** (`--rr-triage-model claude-sonnet-5`, earlier) — metric-identical, no help.
3. **Rubric** (this attempt) — missed `webdev`, cost `cv` recall.

The `webdev` misjudgment needs domain reasoning haiku doesn't reliably have even when the rubric
spells it out — recognizing that a session-security paper doesn't apply to Flask *core* because Flask
*delegates* auth to extensions (the exact distinction Opus and GPT-5.5 make). Rubric/threshold/model
tuning on a small gate model has reached diminishing returns.

**Where the arc landed.** Triage + rerank + all-time discovery took RepoRadar's user-facing Top Picks
from a **mean net@2 of −11** (the pre-triage 0.5 gate) to **≈0** (−0.25 at `≥2`, +0.25 at `≥3`), with
precision **0.47–1.00** depending on the gate — a large, real improvement. The residual (one `webdev`
false positive, one `rag` dud, ~1 haiku-vs-judge disagreement per case) is a **model-capability
floor**, not a tuning target. The next genuine levers are **structural, not gate-side**: better
candidate quality via hybrid retrieval (roadmap #4) / SPECTER2 + cross-encoder (#7), or domain source
adapters (#10) for `webdev`'s real problem — its literature isn't on arXiv at all.

## Hybrid retrieval — BM25 + RRF fusion (2026-07-12)

`--rr-hybrid` (PR #30) fuses the heuristic ranking with a BM25 lexical ranking via RRF before the
Top-N cut, targeting papers the TF-IDF ranker buries on vocabulary mismatch. Run with vs without,
on top of all-time + rerank + cli baseline:

| Case | metric | without | with hybrid |
|---|---|---|---|
| **rag** | Top-10 | 4 act · net −8 · ndcg 0.48 | **6 act · net −2 · ndcg 0.73** |
| **rag** | Top Picks | 3 · 2 act · net 0.0 | 9 · 5 act · net **−3.0** |
| **cv** | Top-10 | 4 act · ndcg 0.64 | 4 act · ndcg 0.68 |
| **rl** | Top-10 | 0 act · ndcg 0.07 | 0 act · ndcg 0.10 |
| **webdev** | Top Picks | 1 · 0 act · net −2 (leak) | **0 · abstain · net 0** |
| **mean** | Top-10 net (rag/cv/rl) | −12.0 | **−10.0** |
| **mean** | Top Picks net (4) | −0.25 | **−0.5** |

**Findings:**

1. **Hybrid improves ranking — this is a real retrieval win.** `rag`'s Top-10 actionable count rose
   **4 → 6** and its nDCG **0.48 → 0.73**; nDCG improved on **all four** cases. BM25 surfaced
   genuinely-actionable papers the keyword ranker had buried (`rag`'s pool actionable count rose
   7 → 9). Top-10 mean net@2 improved **−12 → −10**.

2. **But it runs straight back into the gate ceiling, and slightly worsens the headline.** A richer
   candidate set feeds the imperfect haiku gate *more* borderline papers: `rag` Top Picks went from
   3 returned / 2 actionable (net 0.0) to **9 returned / 5 actionable / 4 duds (net −3.0)**. That one
   swing drags the Top Picks mean **−0.25 → −0.5**. More good candidates ⇒ the gate also passes more
   duds — the same precision ceiling, now fed a bigger pool.

3. **`webdev` leak closed as a side effect** (the reorder pushed its dud out of the Top-10), and
   **`rl` stayed stuck** (0 actionable in Top-10 — BM25 didn't surface its 3 pool-actionable papers).

**Takeaway — the pattern is now unmistakable, and 4 cases is too few.** Discovery ↑, ordering ↑, and
retrieval ↑ each improve the *candidate pool* (nDCG and Top-10 recall keep rising), but the
user-facing **Top Picks net@2 is pinned by the haiku gate's precision**, not by retrieval. And with
only four cases, a single case's swing (`rag` −3) flips the headline sign — the eval set lacks the
statistical power to separate a real ranking gain from per-case noise. **Expanding the benchmark set
is the prerequisite for trusting any further ranking comparison.**

## Tier S — personalization measured for the first time (2026-07-29)

The [re-benchmark](#re-benchmark-after-features-18-and-10--no-regression-2026-07-29) below closed by
admitting F5/F7/F8 were **tested but not benchmarked** — they key off the user's starred/rated
papers and no harness had a store. `run_seeded_eval.py` closes that: per case, seeds are drawn
round-robin across the fixture's `source_query` strata, **removed from the candidate pool**, and each
component is scored on how well it ranks the *held-out* gold.

> **A ranker bug was found while building this and it changed every number below.** `rank_papers`
> sorted stably on `score_total` alone, and `build_fixtures.py` writes all gold papers before all
> distractors — so every score tie silently resolved in gold's favour and **every Tier A/S baseline
> was a best-case ordering** (`cv` baseline nDCG@10 measured 0.931 in fixture order vs 0.842
> reversed). `ranker.py` now tie-breaks on `arxiv_id`. This is a shipping-code fix, not just an eval
> one: production ranking was fetch-order-dependent too.

**5 seeds, real mini-repo profiles, k=10 and at the honest depth k=n_heldout:**

| Case | baseline nDCG@10 | SPECTER2 w=0.25 | w≥0.5 | deep nDCG@n | proximity w≥0.5 |
|---|---|---|---|---|---|
| rag | **1.000** (ceiling) | +0.000 | +0.000 | — | inactive |
| cv | 0.915 | **+0.085** | +0.085 | 0.914 (+0.024) | inactive |
| rl | 0.599 | +0.000 | **+0.401** | 0.908 (+0.232) | **+0.128** |

**Reference rankings on the same pools** (what the task is actually worth):

| Case | id-order (≈random) | category-only | keyword-only | full baseline |
|---|---|---|---|---|
| cv | 0.224 | 0.826 | 0.915 | 0.915 |
| rl | 0.244 | 0.517 | 0.483 | 0.599 |

**Findings:**

1. **Both personalization components help, and the effect is weight-dependent — report the sweep,
   never a point.** SPECTER2's response is step-shaped: mean ΔnDCG@10 is **+0.028 at w=0.25** and
   **+0.162 at w≥0.5**, saturating thereafter. A single number would have been an arbitrary pick on
   a plateau. Note `w_specter == w_keyword` does **not** mean equal influence: `keyword_score` spans
   a narrow band over a pool (~0.10–0.17) while a min-max-normalized component spans all of [0, 1],
   so equal nominal weights give the component several times the score *range*.

2. **The mean is concentrated and partly censored — don't read it as three measurements.** `rag` sits
   at the ceiling (baseline 1.000) so its delta is structurally bounded at ≤ 0; of the remaining two,
   **`rl` supplies 82%** of the mean. n_informative = 2 of 3.

3. **nDCG@10 = 1.000 is not "perfect ranking".** With 14–15 held-out gold and k=10, recall@10 is
   capped by construction (0.667–0.714), so several gold papers sit below the cut in *every*
   configuration. At the honest depth k=n_heldout, SPECTER2 scores **0.914 / 0.908**, not 1.000 —
   and what it pushes below the cut are exactly the fixture's homonym false-golds
   (*"Proximal Point Methods"*, *"polynomial optimization"*), i.e. it is **more right than the
   labels**. (An earlier draft of this section claimed the opposite; that was wrong.)

4. **Citation proximity (F8) is measurable after all — the earlier "unmeasurable" call was my bug.**
   With seeds taken as a fixture-order prefix, no candidate cited a seed and F8 looked structurally
   untestable. Stratifying the seeds pulled in *Soft Actor-Critic* — a heavily-cited paper — and F8
   immediately scored **+0.128 nDCG@10** on `rl` (1449 reference edges over 44 candidates). The
   cause was the **seed-selection policy, not the fixtures**.

5. **The task is non-trivial, but less discriminating than the headline suggests.** An arbitrary
   (id-order) ranking scores ~0.23, so the baseline's 0.60–0.92 is real signal. But `category-only`
   reaches 0.826 on `cv`: a meaningful share of what these fixtures test is *"does the arXiv category
   match"*, a direct consequence of `benchmark.yaml` drawing distractors from clearly different
   fields.

6. **Read all of it as topical discrimination, not actionability** — and a *coarse* bar at that.
   Tier S inherits Tier A's construction. A component scoring well here has shown it generalizes
   from a handful of liked papers to same-domain papers: necessary, not sufficient. Tier B remains
   the quality measure. **Feature 5 stays out of scope** — it *adds* papers rather than re-ranking a
   fixed pool, so a labeled pool cannot score it.

**Honesty notes:** results are deterministic (no LLM judge; vectors/references cached in
`evals/.work/seeded/<case>.db`, `--fresh` to discard). Component helpers now return diagnostics, so
"no signal" is distinguishable from "Semantic Scholar was unreachable" — the F8 conclusion above
rests on that distinction. Coverage is 3 of 12 cases; only those have Tier A fixtures.





## Ranking the score-2 band: five pre-registered experiments; a fine-scale rescore wins (2026-08-07)

~$14 of ranker inference; **zero new judge calls** — every experiment scored offline
against the frozen labels in `results/judge-gpt-5.5-20260807T180938Z.json` (Testbed A:
22 cases, 220 shown papers, 105 in the score-2 band) and its companions (A300
replication arm, the 12-repo diag band B, the wild pools C), exactly as pre-registered
in [RESEARCH-score2-ranking.md](RESEARCH-score2-ranking.md) §4. Scripts:
`band_testbeds.py` (reconstruction + metrics, reconstruction verified against every
case's recorded sweep nets), `exp_select.py`, `exp_finescale.py`, `exp_ensemble.py`,
`exp_pairwise.py`, `exp_features.py`.

Baselines on this exact file: show-all (min≥2) **+1.91** mean net@2; score-3-only
**+0.82** with 14/22 abstentions. (The research doc quoted +0.50 / 16 abstentions for
score-3-only — that number was the 2026-08-02 run; +0.82 is the same policy on the
frozen 08-07 file all experiments are scored against.)

### The scoreboard

| Experiment | Band AUC (judge≥2, A) | Calibration | Policy net@2 (A) | Pre-registered verdict |
|---|---|---|---|---|
| **E2 fine-scale 0-9 logprob expectation** (gpt-4o-mini) | **0.841** (A300 0.761, B judge-3 0.760, C wild 0.949) | Brier 0.244 raw — over the 0.22 bar | +2.09 raw p_true | **Ordering: pass.** Raw calibration: miss |
| **E2 + 2-param LORO logistic map** | 0.838 | **Brier 0.126** | **+3.14** as shipped (+2.91 under the variant first recorded — see the correction below) | **The winner** — see below |
| E1 subset-selection share, Sonnet | 0.635 | share is not a P | +1.23 (10/22 ≥0) | **Fail** (bar: AUC≥0.65 AND beat +1.91) |
| E1 subset-selection share, Haiku | 0.616 | — | +1.18 (9/22, sign p=0.049 the wrong way) | **Fail** |
| E3 ensemble votes + verbalized P, Haiku | 0.676 | **ECE 0.425** | +0.82 | **Killed** (ECE > 0.3) |
| E4 pairwise + Bradley-Terry, Haiku | 0.643 (swap-inconsistency 0.322) | anchors: none | +2.00 (21/22 ties) | **Fail** (bar: AUC≥0.70); anchors non-discriminative |
| E5 free features, LORO logistic | 0.585 | Brier 0.214 | +1.50 | **Below bar** (0.60) — metadata is weak |
| E5 combined (features + E1-E4 columns) | 0.778 | Brier 0.147 | +2.50 | Worse than exp09 alone on every axis |

### What won: de-quantizing the gate, not restructuring it

The research pass's bet was that comparative structure (selection, pairwise) carries the
signal absolute scores discard. What actually carried it is much cheaper: **ask for the
same pointwise judgment on a 0-9 anchored scale and read the token distribution's
expectation instead of the sampled digit** (G-Eval/TrustJudge mechanism; needs logprobs,
so gpt-4o-mini — Anthropic exposes none). One call per paper, <$0.01 per repo.

The de-quantization is real, not cosmetic: the modal digit carries p>0.9 for only 23% of
papers (kill bar was 80%), and the motivating indistinguishable pair separates cleanly —

| band paper scores (exp09/judge, sorted) | |
|---|---|
| **vectordb** | 8.0/**3** 8.0/**3** 7.9/**2** · 7.1/1 6.7/1 6.3/1 6.0/1 3.8/1 |
| **diffusion** | 8.2/**3** 8.1/**2** 8.0/**2** 8.0/**3** 8.0/**2** 8.0/**2** 8.0/**2** 7.6/**2** 7.1/**2** · 5.4/**2** |
| **numerics** | 6.6/**2** · 6.0/1 5.7/**2** 5.4/**2** 5.4/1 5.2/1 4.6/**2** 4.2/**2** 4.0/**2** |
| **linter** | 4.1/1 |

vectordb's three actionable papers are its top three scores; diffusion sits uniformly
high; numerics and linter sit low. The 0.00-1.00 band-precision spread the coarse gate
could not see is visible at the finer scale.

**Validity checks, all passed.** (i) Same-family bias — the judge is GPT-5.5 and the
ranker is gpt-4o-mini — was the pre-registered landmine: on the 74 dual-judged rows of
`second_judge.json`, exp09 scores AUC 0.843 against the GPT judge and **0.896 against
the Sonnet judge**. If anything the signal is stronger out-of-family. (ii) It
replicates across testbeds it never touched during development: A300 0.761, B (judge-3
ordering, different label target) 0.760, wild pools 0.949. (iii) The degenerate-
distribution kill condition cleared.

### The policy: exp09 → 2-parameter logistic → global P ≥ 2/3

exp09 is a score, not a probability; the pre-registered path to a threshold is a
logistic map, LORO cross-fitted (fit on 21 repos, predict the held-out one — the map's
two parameters are the only fitted numbers anywhere in the winner). Digest = gate-3
papers + band papers with P ≥ 2/3:

| case | policy | show-all | Δ |
|---|---|---|---|
| compiler | **+2** | −5 | **+7** |
| vectordb | **+1** | −5 | **+6** |
| ann | **+8** | +4 | +4 |
| systems | **+5** | +1 | +4 |
| linter | **0** | −2 | +2 |
| numerics | **0** | −2 | +2 |
| rl | +4 | +2 | +2 |
| llminfer | +2 | +1 | +1 |
| diffusion | +9 | +10 | −1 |
| 13 others | unchanged | | |

Mean **+3.14 vs +1.91** for the map that ships (the +2.91 first recorded here scored a
more-regularised variant — see the correction below). Shown papers drop 132 → 102 while
actionable shown only drops 97 → 91: precision 0.73 → **0.89**. Every negative case is
rescued — including both members of the diffusion/vectordb pair, which was the exact
failure adaptive digest size could not touch — and the entire toll on the all-good tail
is **one paper on `diffusion`**.

**Replication on the other arm.** Freezing the map fitted on Testbed A and applying it
unchanged to the pool-300 arm (different run, different shown sets): **+3.09 vs +1.73**,
10+/3−, p = 0.092. Two independent draws, gaining **+1.23** and **+1.36** of mean net@2.

Neither arm alone clears p < 0.05 at n = 22 — read it as a strongly consistent
direction measured twice, not a certainty.

### What failed, and the pattern in the failures

**E1 (subset selection).** The abstention mechanism genuinely fires — numerics empty in
13/15 shuffles, encryption 15/15, linter 9/15 — but both models are far too strict:
Sonnet selected *nothing* in 15/15 shuffles on crypto (+3, all three papers actionable)
and cv (+4), Haiku's policy lands at +1.18 with the sign test significant in the wrong
direction. Share mean 0.16 (Sonnet) / 0.26 (Haiku); the predicted top-half-selection
prior never appeared — the opposite failure did. Operational note: Sonnet's
select-everything responses overflowed `max_tokens=2000` and the truncation was
correlated with the verdict (the max_tokens=500 judge bug by another door); raised to
4000 and re-run before reading any numbers.

**E3 (ensemble votes).** Killed by its own bar, ECE 0.425 — but in the direction the
literature did not predict: chronic *under*-confidence. Under skeptical personas plus
consider-the-alternative elicitation, Haiku's P̂ collapses toward 0 regardless of label
(all 10 linter papers at 0.03-0.07; policy +0.82). Same failure family as E1:
**Claude-family models under maintainer framing default to "no"** — restructuring the
judgment (selection, ensembles) amplifies strictness rather than extracting signal.

**E4 (pairwise + BT).** Swap-inconsistency 0.322 — position bias is real but under the
0.45 kill bar (both-order querying was doing its job). Band AUC 0.643 misses the 0.70
backbone bar, well under E2. The anchor mechanism failed absolutely: **128/128 admitted
papers beat the borderline anchor** — a real paper essentially always beats a synthetic
survey, so the anchor provides no threshold. The pre-registered paraphrase-stability
arm was not run: with zero discrimination there is nothing for paraphrase to
destabilize. Testbed B ran for the record: judge-3 ordering AUC 0.731 (swap 0.268) —
pairwise's best showing, still under E2's 0.760 on the same target at ~7× the calls.

**E5 (features).** Features-only LORO band AUC 0.585, below the 0.60 bar — citations,
age, HyDE rank and hop coupling are individually weak, as the practitioner-relevance
literature predicted. The combined model (features + all four method scores) reaches
0.778/Brier 0.147/+2.50 — and is beaten by exp09 *alone* (0.838/0.126/+2.91) on every
axis: at 22 queries, every added column is another way to overfit, even under LORO.

### HyDE, measured end to end: the first result that clears p < 0.05 against the baseline (2026-08-09)

```bash
uv run python evals/run_judge_eval.py --baseline cli --rr-pool 50 --rr-rerank \
    --rr-all-time --rr-hybrid --rr-sweep --rr-finescale --rr-hyde   # arm A
uv run python evals/run_judge_eval.py --baseline cli --rr-pool 50 --rr-rerank \
    --rr-all-time --rr-hybrid --rr-sweep --rr-finescale             # arm B (control)
```

Both arms, **same session, back to back, HyDE the only variable** — not a comparison against
the stored run from the day before, because that run demonstrated per-case swings of ±6 that
would be attributed to the flag. 22/22 cases, **zero degradations** (the harness prints
"this arm is NOT a clean HyDE measurement" and counts them; it printed nothing).

| | control | **+ HyDE** | Opus 4.8 |
|---|---|---|---|
| mean net@2 | +3.18 | **+4.55** | +1.82 |
| paired vs Opus | +1.36 (11/5/6, p = 0.21) | **+2.73 (15 w / 3 l / 4 t, p = 0.0075)** | — |
| paired vs control | — | **+1.36** (10+/3−, p = 0.092) | |
| digest precision | 0.91 | **0.94** | 0.94 |
| papers shown / actionable | 97 / 88 | **121 / 114** | 49 / 46 |
| abstentions | 7/22 | 4/22 | 4/22 |
| net-negative repos | 0 | **0** | 1 |

**This is the first time in the project's history that RepoRadar beats the baseline at
p < 0.05.** Every previous headline — 12-case parity, the 22-case dead heat, yesterday's
+1.36 — was "ahead on the mean, not established". At 15 wins to 3 losses the sign test is
**p = 0.0075**, and the system now matches Opus's precision (0.94) while returning **2.5×
as many papers**.

Read the attribution carefully: **HyDE's own increment is +1.36 at p = 0.092**, which is
*not* significant on 22 paired cases. What crossed the line is the cumulative system. The
honest statement is that HyDE is a consistent-direction improvement of the same size as the
fine-scale rescore, and that the two together clear a bar neither clears alone.

### It landed exactly where it was shipped to land

The pre-registered expectation, written before the run: HyDE addresses **recall**, so the
six recall-driven losses to Opus should narrow, and a mean gain without movement there
would not be the win it was shipped for.

| case | control | +HyDE | Opus | actionable papers in the judged pool |
|---|---|---|---|---|
| `speech` | 0.0 | **+10.0** | +3.0 | 5/10 → **12/12** |
| `graph` | +1.0 | **+10.0** | +3.0 | 6/13 → **13/13** |
| `rag` | 0.0 | **+4.0** | +3.0 | 3/13 → **11/13** |
| `llminfer` | +1.0 | +1.0 | +4.0 | 10/14 → 12/14 |
| `numerics` | 0.0 | +1.0 | +2.0 | 8/12 → 9/12 |
| `vectordb` | +5.0 | +5.0 | +4.0 | 12/14 → 11/14 |

**Three of the six are fixed outright**, including `rag` — the case that had been admitting
*nothing* for weeks and was the standing example of a pool that never contained the answer.
`speech` and `graph` went from near-abstention to a perfect ten. The other three moved
little; `llminfer` and `numerics` remain genuine losses, so the recall problem is narrowed,
not closed.

The pool numbers are the mechanism, not decoration: HyDE roughly **doubled to tripled every
candidate pool** (155→516, 296→687, and `numerics` 56→**452**), and the share of the judged
pool that is genuinely actionable rose on 15 of 22 cases. More candidates *and* better ones.

### Why this worked when gating the whole pool did not

[Gating the whole pool](#gating-the-whole-pool-end-to-end-a-wash-2026-08-07) was a **wash**
(−0.18 paired) and concluded: *"the bottleneck is not how many papers the gate sees. It is
that nothing ranks what it returns."* HyDE is another pool expansion, and it works — because
that sentence stopped being true. The fine-scale rescore now orders what the gate admits, so
a bigger pool converts instead of merely reshuffling which arbitrary admits fill ten slots.

The two changes are **complementary rather than additive by luck**: precision improved
(0.91 → 0.94) *while* the shown set grew 97 → 121. Without the rescore, doubling the pool
would have fed the old near-binary gate more borderline papers — exactly the failure BM25
fusion produced in July, when better nDCG made the headline worse.

Three cases returned *fewer* papers: `storage` 10→8, `diffusion` 10→9, `compiler` 2→1. All
three kept 100% precision on what remained, so these are the rescore declining newly-admitted
band papers, not lost recall.

**Cost:** ~112 new judge verdicts (1,375 → 1,487) across both arms, 22 hypothesis calls, and
the metadata fetches — roughly **$4**. The 417 MB index was already on disk from P4's
replication; a cold start adds a one-time 432 MB sync.

**Caveats.** The +1.36 increment is p = 0.092; only the cumulative comparison is significant.
net@2 charges 2 per false positive and so rewards precision-preserving expansions — it
flatters this result as it flattered the last one. And this is one paired session: the per-case
values will move again on the next collection, as they did between the previous two runs.

### IACR ePrint shipped and measured on two cases — no detectable effect, and a query bug that broke every non-arXiv source (2026-08-12)

```bash
uv run python evals/verify_iacr_deps.py           # $0 stage-1, run BEFORE the adapter existed
for S in arxiv arxiv,iacr; do
  uv run python evals/run_judge_eval.py --baseline none --case crypto,encryption --sources $S       --rr-pool 50 --rr-rerank --rr-all-time --rr-hybrid --rr-sweep --rr-finescale --rr-hyde
done
```

Cryptography's literature is largely not on arXiv, and `crypto`/`encryption` have been the
benchmark's steadiest under-performers. **The subset was pre-registered before the adapter
existed** (`evals/verify_iacr_deps.py`), chosen by domain: on the 25-case mean a perfect
adapter caps at +0.68, below the 1.04 floor and undetectable by construction; on the two
cases it serves, the MRE is 3.44 against 8.5 of headroom.

#### The first run was VOID, not null

Both arms scored identically, and the reason was that **zero IACR papers reached the top-10
in either case**. Reported as "no effect" that would have been a manufactured null. The
degraded-arm check — count whether the channel's papers actually arrived — is the same one
`--rr-hyde` carries, and it is the only reason this was caught.

#### The cause was not IACR

Queries arriving at ePrint were arXiv boolean syntax: `(all:"key cryptography") AND
(cat:cs.CR)`. Callers bridged arXiv grammar to keyword APIs with
`q.replace("all:", "").strip('"')` — written for an older query shape, and **silently
inert** once `build_queries` began wrapping queries in parentheses with a category clause.

**That transform feeds DBLP, bioRxiv, OpenAlex and Semantic Scholar too.** All four shipped;
all four have been receiving malformed queries. ePrint is simply the first source that
returns *zero* for it rather than degrading quietly, which is how a bug that has been live
across Features 10 and 12 finally surfaced. Fixed as one shared
`collector.to_plain_keywords`, beside the function whose output it consumes, routed through
all three call sites. IACR goes 0 → 5 papers on the identical query.

The new test caught a defect in **that fix**: stripping only *parenthesised* category groups
turned the no-keyword fallback `cat:cs.CR OR cat:cs.LG` into a literal search for
`cs.CR cs.LG`. The tests run against real `build_queries` output rather than hand-written
strings — a hand-written fixture is exactly what kept passing through the original drift.

> **Two corrections to the paragraphs above, from the follow-up audit (2026-08-12).**
> "Written for an older query shape" is **false** — git dates the parenthesised form to
> 2026-02-22 and the one-liner to 2026-02-23, so it never worked. And "routed through all
> three call sites" undercounts: there were **five**, and this PR routed two. See
> *C-9 audit* below.

#### The valid measurement

| case | control | +IACR | delta |
|---|---|---|---|
| `crypto` | +2.0 | +1.0 | −1.0 |
| `encryption` | +1.0 | +1.0 | +0.0 |
| **mean** | | | **−0.50/case** against MRE **3.44** |

Arm validity confirmed: **6 IACR papers reached the ranked top-10** (3 per case), one
reached a digest. **No detectable effect** — −0.50 is deep inside noise at n = 2.

**The mechanism is informative even though the number is not.** Of the six papers that
reached contention, **five were judged 1**: topically exact, not actionable. Only
*Cache-Timing Attacks on RSA Key Generation* scored 2. ePrint is dominated by papers
describing **attacks on** primitives rather than improvements a library should **adopt** —
the register mismatch of §1, reappearing in a source chosen specifically to fix a coverage
gap. And on this draw the three IACR papers per case *displaced* arXiv papers that scored
better, which is why `crypto` fell.

#### A sizing error of mine, worth recording

The subset was justified by comparing the MRE (3.44) against the **headroom** (8.5) — the
distance to a perfect score. That is the *ceiling*, not a plausible effect. A realistic
channel gain of +1 to +2 per case was never detectable at n = 2, and I sized the experiment
as though a perfect adapter were the expected outcome. The right bar is a *plausible* effect,
and by that bar two cases was never enough: detecting +1.5/case needs roughly **n = 11**.

**Shipping decision.** The adapter is opt-in (`sources: [iacr]`), off by default, and stays
that way: nothing here justifies enabling it, and nothing rules out a real effect below the
floor either. It is documented as **built and unvalidated**.

**Cost** ~$3.

### PRE-REGISTERED — the last unmeasured value that ships: `ranking.w_embedding` (2026-08-16)

**Why this one and not another ranking arm.** It is the only value still shipping to users
that no number covers. `RankingConfig.w_embedding` is **0.0** and every published result was
measured there — but `default_config_yaml()`, the file `rr init` writes, sets **1.5**, which
is larger than `w_keyword`'s 1.0 and therefore the heaviest weight in the config. Declared
in `audit_product_divergence.py` since 2026-08-15 as unmeasured *in either direction*; this
closes that.

**What makes it a better candidate than the last four.** NR-33, NR-35, NR-36 and NR-37 were
all *reorderings* of the same score components, and all came back null or inside the floor.
This adds a **component**, at the largest weight, changing what "score" means rather than
shuffling it. And there is a concrete mechanism for harm: the one arm that ever measured
this channel scored README embeddings as a *query* and found them **bimodal** — 7/48 at
top-100, occasionally rank-1, **median rank 46,656**. A signal that is usually terrible,
given the heaviest weight, is plausibly a real negative rather than a fifth null.

**Design.** `w_embedding` changes the score, not the candidates: the profile is untouched,
so queries and collection are identical. It is therefore a `RANKING_FLAG`, both arms share
`pool-depth`, and the floor is **0.74** (frozen, window 15) rather than the ~1.6 a live
comparison would carry.

```bash
COMMON="--baseline none --sources arxiv --rr-pool 50 --rr-rerank --rr-all-time --rr-hybrid
        --rr-sweep --rr-finescale --rr-hyde --rr-frozen-pool evals/.work/pool-depth
        --rr-window 15"
uv run python evals/run_judge_eval.py $COMMON                        # control, w_embedding 0.0
uv run python evals/run_judge_eval.py $COMMON --rr-w-embedding 1.5   # what `rr init` writes
```

**Void-not-null, enforced rather than hoped for.** Without the `embeddings` extra the ranker
scores every paper on keyword and category alone, so the treatment arm would silently be the
control and report "the weight does nothing" about a component that never ran — C-9 and
NR-30's exact shape. `rank_candidates` now **refuses** to run with `w_embedding > 0` when
the extra is missing or the repository yields no embeddable text, and a $0 probe confirms
the top-15 actually changes before any judge call is made.

**Prediction, written first.** **−1.5 to +0.5, most likely negative and possibly past the
floor.** The bimodal evidence above is the reason: at weight 1.5 a signal whose median rank
is 46,656 should drag good papers down more often than it lifts them. I put maybe 55% on a
negative point estimate, 25% on inside-the-floor-either-way, 20% on positive. My last three
primary predictions were wrong on sign twice, so this is a weak prior stated as one.

**What each outcome licenses.**
* **Negative past the floor** — the template is shipping a ranking degradation to every user
  with the extra installed, and `default_config_yaml` should set 0.0. A direct, user-facing fix.
* **Positive past the floor** — the *measured preset* sets 0.0 and is leaving value on the
  table; `BENCHMARK_HEADLINE` changes and every published headline is understated.
* **Inside the floor** — the declaration stays, but stops reading "unmeasured in either
  direction" and becomes "measured, unresolved", which is a weaker and more honest claim
  than the one it replaces.

**Cost** $6–10 of judge and gate calls, plus ~25 minutes of CPU: the treatment arm encodes
~580 candidates per case and the eval, unlike the product, has no embedding cache.

> #### RESULT (2026-08-16) — **+0.64/case, inside the floor.** Positive, and I predicted negative. **[NR-38]**
>
> One frozen pool (`pool-wemb`), 25/25 both arms, `w_embedding` the only variable.
>
> | arm | net@2 | shown | actionable | precision | net-negative |
> |---|---|---|---|---|---|
> | **0.0** — every published number | +5.16 | 195 | 173 | 0.887 | 0 |
> | **1.5** — what `rr init` writes | **+5.80** | 208 | **187** | **0.899** | 0 |
>
> **Paired +0.64/case, 95% CI [−0.28, +1.80], 8 better / 6 worse / 11 tied, sign p = 0.79 —
> inside the 0.74 floor.** Valid, not void: 25/25 cases changed, mean Jaccard 0.58.
>
> **My prediction was wrong, and wrong in an interesting direction.** I pre-registered
> **−1.5 to +0.5, most likely negative**, reasoning from the only prior measurement of this
> channel: README embeddings as a *query* are bimodal, occasionally rank-1 but with a
> **median rank of 46,656**, so weighting them at 1.5 should drag good papers down. The
> result is **+0.64 — outside my interval on the upside**. That is the third of my last four
> primary predictions to be wrong on sign.
>
> The reasoning failed because I transferred a *retrieval* property to a *ranking* role
> without checking that the transfer holds. As a query, a bad embedding match costs you the
> whole result set — the median rank is what you get. As one weighted component among four,
> scoring an *already-retrieved* pool, a mediocre signal is diluted rather than decisive,
> and its occasional rank-1 behaviour is exactly the tail that helps. Median rank is the
> wrong statistic for a component that only ever breaks ties near the top.
>
> **The composition improved on every axis**, which is unusual here: more papers shown
> (208 vs 195), more actionable (187 vs 173), *and* higher precision (0.899 vs 0.887).
> Compare NR-35, where fusion showed more and delivered more at *lower* precision for
> +0.00. Big movers: `rag` +2→+12, `cv` +4→+10, `llminfer` +4→+8, against `thin-kv` +7→+4
> and `speech` +6→+3.
>
> **What it licenses — exactly what the pre-registration said, no more.** Inside the floor
> means unresolved. The `DECLARED` entry stops reading "unmeasured in either direction" and
> becomes "measured once at +0.64, unresolved". **No default changes**: the template keeps
> 1.5 and the preset keeps 0.0, because a point estimate inside the floor cannot justify
> moving either. What *has* changed is the direction of concern — I opened this suspecting
> the template was shipping a degradation to users, and the evidence, such as it is, points
> the other way.
>
> **Worth one more draw.** +0.64 against a 0.74 floor is the closest any arm has come
> without resolving. Averaging a second paired draw on the same pool cuts the standard error
> by √2 — an effective floor near 0.52 — which would resolve this. It needs no collection
> (the pool exists), costs ~$8, and has a real decision attached: if it resolves positive,
> the *measured preset* is leaving value on the table and `BENCHMARK_HEADLINE` changes.
>
> **Cost** ~$9 plus one live collection (~1 h) forced by the frozen-pool break below.

#### PRE-REGISTERED — the second draw, and why it is weaker evidence than the first (2026-08-16)

Draw 1 gave **+0.64** against a 0.74 floor: the closest any arm in this project has come to
resolving without doing so. A second paired draw over the **same** frozen pool varies only
temperature-0 jitter in the gate and the rescore — which is exactly what the 0.74 floor was
measured over (two reuse passes) — so averaging the two draws' per-case deltas cuts the
standard error by √2 and gives an **effective floor of ≈0.52**.

**The decision rule, fixed before draw 2 is seen.** The estimate is the **mean of the two
draws' per-case deltas**, compared against **0.52**. Not "draw 2 alone", not "whichever
draw is cleaner", and not "keep drawing until it resolves".

* **|combined| ≥ 0.52 and the CI excludes 0** — resolved. If positive, the *measured
  preset* is leaving value on the table: `BENCHMARK_HEADLINE` changes to 1.5 and every
  published headline is understated. That is the expensive branch and the reason to run.
* **|combined| ≥ 0.52, CI spans 0** — suggestive, not established. Nothing moves.
* **|combined| < 0.52** — unresolved at two draws, and I stop. A third draw to chase 0.64
  past a shrinking floor is how a null becomes a finding by attrition.

**Why this is weaker than draw 1, stated plainly.** Draw 1's result is already known, so
this is *confirmatory*, not blind: I am running a second draw **because** the first was
encouraging, which is a selection effect no amount of arithmetic removes. Two consequences
I accept in advance. The p-value is not clean — this is a second look at the same question,
and I will report it without pretending otherwise. And had draw 1 come back at −0.64, I
would not be running this, which is exactly the asymmetry that makes the combined estimate
an upper bound on the evidence rather than a fair one.

**What would change my mind about the whole line.** If draw 2 lands materially negative
(say below −0.5), the honest reading is not "they average out" but "draw 1 was a favourable
draw", the same lesson C-7 recorded when a +4.55 headline re-ran at +3.91.

**Cost** ~$8, no collection — the pool exists and most judge verdicts are cached.

> #### RESULT (2026-08-16) — **+1.00/case, resolved.** The template was right and the *preset* is the one carrying the worse value.
>
> | | control 0.0 | treatment 1.5 | paired delta |
> |---|---|---|---|
> | draw 1 | +5.16 | +5.80 | +0.64 — inside the 0.74 floor |
> | draw 2 | +4.84 | +6.20 | **+1.36** — past it, CI [+0.40, +2.52] |
> | **combined** | | | **+1.00**, CI **[+0.14, +2.08]**, 12+/3−/10=, sign p = **0.035** |
>
> **Against the pre-registered floor of 0.52 the combined estimate is past it and the CI
> excludes zero: resolved, and positive.** Draw 2 is *larger* than draw 1, so this is not
> draw 1 having been a favourable draw — if anything it was the conservative one. The two
> control arms agree to 0.32 and the two treatment arms to 0.40, both comfortably inside
> single-draw jitter, which is the consistency check that makes averaging them legitimate.
>
> **My pre-registration was under-specified, and I would rather say so than quietly do
> something else.** I wrote that a positive result means "`BENCHMARK_HEADLINE` changes to
> 1.5 and every published headline is understated." The first half is wrong.
> `BENCHMARK_HEADLINE` records *what the published headline runs actually set*, and they
> set 0.0 — editing it to 1.5 would make it assert a run that never happened, which is
> precisely the C-17 error (a number attached to a configuration it did not describe) in a
> file whose whole job is preventing that. I did not notice the conflict when writing the
> rule because I was thinking of `BENCHMARK_HEADLINE` as "the configuration we endorse"
> rather than "the configuration we measured".
>
> **So nothing is edited, and the reason is not caution.** Moving the *preset* to 1.5 would
> break its defining property — that it reproduces the run behind the published +5.42/+5.12
> field by field. Moving `BENCHMARK_HEADLINE` would make it false. The gap is real,
> quantified, and **deliberately left open**: closing it requires re-measuring the headline
> at 1.5, roughly $25 with the agentic baseline, after which preset, headline and audit all
> agree again.
>
> > **Both halves of that last sentence were wrong, and the run below is what showed it.**
> > The `$25` is the *sunk* cost of the 25 cached Opus answers ($24.98, $1.00/repo), not the
> > marginal cost of a re-run: `baseline._discriminator` keys the cache on model, prompt,
> > mode and flags, so with the model unchanged the baseline re-runs at **$0**. The real
> > marginal cost was **$4.38**, and all of it went on one case whose cache entry was stale.
> > $25 is what *switching the baseline model* would cost — a different decision entirely,
> > and quoting one as the other made the cheap option look expensive for a day.
>
> **The inversion is the finding.** I opened this line suspecting the template was shipping
> a ranking degradation to every user with the `embeddings` extra installed. The opposite
> holds: **the default template's 1.5 is the better value, and the configuration we
> *recommend* is the one carrying the worse one.** Every headline in this project is
> understated by about a point for users who have that extra — and `.[hyde]`, which the
> measured setup instructs people to install, pulls it in.
>
> **The caveat I pre-registered, restated because it still applies.** This was a
> *confirmatory* second draw run because the first was encouraging, so the p-value is a
> second look at the same question and the combined estimate is an upper bound on the
> evidence rather than a fair one. What it is not is a fishing expedition: the decision rule
> and the stopping point were fixed in writing beforehand, and I am stopping at two draws
> as stated rather than chasing a third.
>
> **Scope.** The weight does nothing without the `embeddings` extra, so +1.00 describes
> users who have it. Six of 25 benchmark cases produce byte-identical profiles either way,
> so the per-case effect is concentrated in the cases where the signal exists at all.
>
> **Cost** ~$8, no collection.

### Roadmap 16's relation half: the claim cannot be grounded on 18 of 25 repos. **[NR-39]**

> **Superseded in part, 2026-08-16.** The coverage findings below stand. The two
> *discrimination* readings do not: "only the anchor channel discriminates" is a case-mix
> artifact (**[C-21]**) and "keywords are noise at +0.2pt" is a saturation artifact
> (**[C-22]**). Read this entry with both corrections, and note that the alias-table revival
> condition it sets has been withdrawn rather than met — see **[P9]**.

```bash
uv run python evals/relation_probe.py     # $0, no network, no LLM, no judge
```

Item 16 promises *"this paper claims 3× over IVF-PQ, and you import `faiss`"*. The
2026-08-09 re-derivation promoted it — it seeds from what the repository **has**, the only
register that ever worked here — and cut it to the **relation classification** with a
quoted evidence span, since HyDE and the hop already retrieve these papers. This probes the
assumption underneath the remainder: that the relation is there to be found and quoted. It
reads 1,237 judged papers against the 25 benchmark repos' own profiles, all from cache.

**Predictions were fixed in the script before it ran**, and three of four were right — the
fourth by less than it looks:

| | predicted | actual |
|---|---|---|
| Q1 judge's reasoning names a repo term | >60% | 94.0% by keyword, **22.9% by anchor** |
| Q2 the ABSTRACT names one (a span to quote) | 10–20% anchor | **20.5%** at score 3 ✓ |
| Q3 relation verbs discriminate | weakly | weakly ✓ (`improves` +16.8pt, rest ≤ +3.8) |
| Q4 grounded-claim coverage | 20–40% | **12.3%** — below the band |

**The high-coverage channel discriminates in the wrong direction.** Repository *keywords*
appear in 96.4% of actionable abstracts and **97.4%** of non-actionable ones. A badge built
on that fires on four papers in five and is very slightly more likely on the bad ones: it
says "this paper is about your topic", which showing it already implies. Only the *anchor*
channel — packages the repo actually depends on — separates anything: 20.5% / 14.9% / 6.7%
across score-3 / actionable / below.

**And the anchor channel does not exist for most repositories.** Median 2 anchors per case,
**12 of 25 with none at all**, because `_extract_anchors` parses `requirements.txt`,
`pyproject.toml`, `setup.py` and `package.json` — so a C++, Rust or Go repository has no
dependency list to read. The same structural limit already documented for
`profiler.source_extensions`, arriving in a second place.

| case | grounded claims | share of all |
|---|---|---|
| `peft` | 34 / 63 | **47.2%** |
| `graph` | 15 / 38 | 20.8% |
| `diffusion` | 9 / 40 | 12.5% |
| `rag` | 9 / 32 | 12.5% |
| **18 other cases** | **0** | **0%** |

Four Python/ML repositories carry **93%** of every grounded claim in the benchmark. The
canonical example from the roadmap entry is itself a miss: `ann` has `faiss` in its README
keywords and *not* in its anchors, because it is a C++ repository with no Python manifest.

**The alert the feature is named for is the rarest one.** `replaces` — supersedes,
alternative-to — fires on **8.9%** of actionable abstracts, +3.8pt over non-actionable.
`extends` is +2.0pt. `uses` is **−0.7pt**, i.e. noise. Only `improves` carries signal, and
"a paper that improves something" is close to a restatement of the gate's own question.

**What the probe found that was not the question.** The product already ships this. The
gate returns a one-line `llm_reason` alongside every score (`triage.py:42`), the store
persists it, and **both** digest templates render it —
`digest.md.j2:71` and `digest_page.html.j2:163`. On the eval side the judge writes a
`proposed_change` on **100%** of actionable papers, in exactly the register item 16 wants
(*"Add a compressed-code reranking option for IVF/PQ-style indexes…"*). So the feature's
true increment is not "explain the paper" — it is "replace a free-text sentence with a
typed label and a quoted span", on ~12% of entries, concentrated in four repositories.

**Verdict: do not build it as specified.** The premise survives — starting from what the
repo has is still the only thing that has ever worked — but the *grounding vocabulary* is
the wrong one. Reviving it needs a technique-alias layer that maps observable repository
facts to the names abstracts actually use (`faiss` → IVF-PQ, product quantization), and
that is precisely what the July entry called an "alias-table curation burden" and treated
as a detail. It is not a detail; it is the feature. Anyone proposing it again owes a
coverage number for that table on non-Python repositories first.

**Cost** $0.

> #### The probe shipped with the bug it exists to detect
>
> Its first run reported **0.0% keyword hits in all three strata**. That is a finding —
> "abstracts never mention these" — and it was false. `RepoProfile.keywords` is a list of
> `(term, weight)` pairs, and the `len(term) >= 3` filter was applied to the *pair*, whose
> length is 2, so every keyword was discarded before it was ever matched. Void read as
> null, inside a probe written to look for exactly that, one week after the pool scanner
> read 1,250 papers as 0.
>
> The tell was the shape: **0.0% in every stratum**, which is the same alarm as
> byte-identical pools across arms. What the repair adds is a guard that refuses to report
> a term class extracted as empty everywhere, rather than a comment saying to be careful —
> and `tests/test_eval_relation_probe.py` fires it in both directions.

### The judge is not date-biased. The retrieval is. **[NR-43]**

Stored data only — 4,019 cached GPT-5.5 verdicts and the 837 Claude Sonnet 5 verdicts P7 paid
for. No LLM calls, no judge calls, no new protocol. `evals/judge_date_stratify.py`.

**The hypothesis** (from LitLLMs): an LLM judge has *seen* pre-cutoff papers in training,
rewards that familiarity, and labels older work more kindly. If true, every recall and net@2
figure in this project inherits the bias.

**The design deliberately does not guess the cutoff.** Contamination produces a *step* at the
judge's own cutoff date. Paper age also predicts actionability honestly — a recent paper is
likelier to help a current codebase, and `recency` is one of the ranker's scoring components —
so a smooth trend is expected and is not the thing being looked for. Testing for a
discontinuity *anywhere* is strictly stronger than testing at a date we would have had to
assume, and it costs nothing extra.

**The trend is there and has no step in it:** 0.31 (2013) rising to 0.64 (2025), monotone
enough. Then the newest month with real volume falls off it entirely — **2026-07 scores 0.233
over 159 papers, CI [0.174, 0.304]**, against 0.46–0.68 for every other month of 2026. Not a
taper. One month.

#### The second judge settles it

| period | n | GPT-5.5 | Sonnet | agreement |
|---|---|---|---|---|
| 2024 and earlier | 500 | 0.720 | 0.410 | 0.662 |
| 2025 | 111 | 0.739 | 0.523 | 0.748 |
| 2026 to June | 89 | 0.685 | 0.461 | 0.775 |
| **2026-07** | 38 | **0.237** | **0.105** | **0.868** |

**Both judges collapse, and they agree *most* where they collapse.** A bias belonging to one
model's training data cannot appear in a second model trained differently — and if the newest
papers were the ones a judge could not assess, its verdicts there would be *noisier* and
agreement would fall. It rises to the highest of any period. **The hypothesis is refuted.**

#### Two confounds, checked rather than argued away

**Case mix.** The month over-represents `webdev`, `http` and `cli` — three of the four
repositories RepoRadar abstains on, which run low in every period. So the comparison is made
inside each repository: **10 of 11 cases fall, mean −0.221.**

**The index boundary.** The dense index's newest paper is in this same month, so July is split
between what the index holds and what only the live keyword channel could reach — and
keyword-only retrieval is the configuration measured at −8.12. That would have been a tidy
explanation, and it fails: the channel does matter (**0.156 outside the index against 0.333
inside**) but the in-index half still falls far below June's 0.510.

#### What is actually there: a 5.1× jump in outright rejections

| period | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| 2024–2025 | 0.103 | 0.259 | 0.423 | 0.215 |
| 2026 to June | 0.127 | 0.335 | 0.378 | 0.160 |
| **2026-07** | **0.522** | 0.245 | 0.164 | 0.069 |

A 0 means *"no relation to this repository"*. **Unfamiliarity does not have this shape** — an
unfamiliar paper draws a hedged 1 or 2, and here scores 1, 2 *and* 3 all fall while only 0
grows. Half the newest month is a flat rejection, five times the base rate, at the **highest
judged volume of any month** (159, against 100 in June and 80 in May).

Rising volume with collapsing precision, concentrated in the freshest slice, is what a ranker
that pays for recency looks like from downstream. **The freshest slice is where off-topic
material enters the pool — and RepoRadar is a freshness product.** That is a retrieval defect
sitting directly on the product's core claim, and it was found by a probe aimed at the judge.

#### The residual, recorded rather than buried

If **both** judges' training cutoffs fall in mid-2026, both would be unfamiliar with July
papers and both would mark them down — two models, one shared blind spot, the same prediction.
The *single-judge* story is refuted; the *shared-cutoff* story is not, and the score-0 shape
argues against it without excluding it. The artifact carries
`shared_cutoff_excluded: false` so the limit travels with the number.

**Cost** $0. Pinned by `tests/test_judge_date_stratify.py`.

> #### A decode bug in the first draft, caught before it reached a conclusion
>
> `_month` returns `year * 12 + month` with month in 1..12, so December leaves a remainder of
> 0 and a naive `m // 12` reports the **following** year. Every December was landing in the
> next year's bucket. Invisible in aggregate and directly on the boundary the whole probe is
> about — the first year table was quietly wrong at exactly the place the answer lives. The
> inverse is `((m-1)//12, (m-1)%12+1)`, and every table was recomputed rather than patched.
>
> Related: the repository already held **four** hand-rolled arXiv-id→date parsers, and they
> disagree. `cited_holdout._month` is correct (both id eras, 1990s handled, uses `dedup_id`);
> `exp_features` handles new-style ids only; `verify_hyde_deps._year_of` returns `None` for
> old-style ids on the stated grounds that they carry no YYMM prefix, which is false; and
> `synth_seeds._yymm` returns a bare 4-digit int, so a 1993 paper (`9304`) sorts *after* a
> 2022 one (`2201`). This probe imports the correct one rather than adding a fifth. The
> consolidation is filed, not done.

### The dense index has a recall gauge at last, and the prefix question is settled. **[PLANS item 4]**

The binary-quantized arXiv index sits under HyDE — +1.36 net@2 end to end, the project's first
p < 0.05 — and was **verified for identity and unmeasured for usefulness**. `verify_encoder`
proves our vectors reproduce the published ones bit-for-bit; nothing proved the index still
*finds* anything. Binarisation, column pruning, a bad yearly shard or an encoder swap could
have cost fifteen points of recall in silence.

`evals/litsearch_recall.py`, $0, no LLM or judge calls. 597 LitSearch queries
(arXiv:2407.18940), the shipped encoder, the shipped `hyde.search_index`.

| arm | R@5 | R@20 | R@100 | found in top 100 | median rank when found |
|---|---|---|---|---|---|
| **bare** (the shipped form) | 0.247 | 0.376 | **0.560** | 279 | 8 |
| prefixed | 0.259 | 0.396 | 0.530 | 264 | 5 |

498 of 597 queries scored. Recall lands between 0.25 and 0.56 — far enough from both 0 and 1
that a regression has somewhere to show, which is the property that makes a gauge worth having.

#### Why this measures the index rather than the corpus

**456 of 456 distinct gold arXiv papers are already in our shards.** 458 of LitSearch's 574
gold papers carry an arXiv id; every one of them is indexed. So a query that fails, fails at
*retrieval* — there is no coverage term to confound it with. The remaining 99 queries have no
gold paper the index could return; they are **excluded and counted, never scored**. Counting
them as misses would measure LitSearch's overlap with arXiv, drift upward every time arXiv
grows, and bury the signal underneath — void read as null, one more time.

The encoder is verified *before* anything is embedded, and the measurement refuses to run
without Hamming 0. Otherwise an identity failure would arrive disguised as a retrieval result:
recall would collapse, correctly, for a reason a recall number cannot name.

#### The prefix: the aggregates said one thing and the paired test said another

`mxbai-embed-large-v1` is an asymmetric retriever — documents bare, queries behind *"Represent
this sentence for searching relevant passages: "*. The index holds bare abstracts, so the
prefix belongs on the query side or nowhere. The aggregate table above says it wins at the top:
**+0.012 at k=5, +0.020 at k=20**. Exact McNemar over the same 498 questions:

| k | bare-only | prefixed-only | *p* | |
|---|---|---|---|---|
| 5 | 11 | 17 | 0.345 | not significant |
| 20 | 12 | 22 | 0.121 | not significant |
| **100** | **26** | 11 | **0.020** | **significant, favouring bare** |

**Both apparent wins are noise, and the only difference that resolves goes the other way.**
Two aggregate rates over the same questions are not a comparison, which is why the arms now run
together and their per-query ranks are stored — the first version of this script reported two
recalls and nothing else, and could not have answered the question it existed to raise.

The gauge freezes on `bare`, which is also the form the product uses: HyDE embeds hypothetical
abstracts with no prefix. That the measurement agrees is convenient, not the reason.

#### What it is not

**Not a net@2 claim, and it must never be quoted as one.** A researcher asking *"where can I
find work on X"* is a different register from a repository that needs a paper to act on — §5's
register-mismatch finding is precisely that the two do not transfer. This answers one question:
*does the index still retrieve what it retrieved before?* The `_comment` in the artifact carries
that caveat so a reader who greps `recall_at_20` finds it one line away.

**Not one of the six gates.** `--check` re-measures, which is ~40 minutes of CPU against the
six gates' two. It is a post-`rr sync-index` gate; `tests/test_litsearch_recall.py` reads the
frozen artifact, which is the part CI can afford. Not in the product CLI either — `rr
sync-index --verify` already answers the identity question, and this needs a dataset the
product does not ship.

**Cost** $0 (50 KB of LitSearch's `query` config and two S2 batch requests; the 1.26 GB and
1.6 GB corpus configs are not needed and not fetched).

> #### The script shipped with a ninth copy of the C-14 rule, and the guard caught it
>
> `litsearch_recall.py` needed "is this the same paper across a version suffix", and wrote
> `str(arxiv_id).split("v")[0]` rather than calling `paper_id.dedup_id`. C-14 recorded that
> exact expression at **eight** call sites; this was the ninth, written by someone who had
> read C-14 the same week.
>
> `tests/test_paper_id.py::TestOneRuleEverywhere` failed on the new file before it was ever
> committed, and the breakage is not hypothetical: **`solv-int/9304001` has a `v` in its
> archive name**, so the local rule returns `sol`. **844 index ids collapsed into that one
> string.**
>
> It did not move this measurement — the scored set is byte-identical under both rules, none
> of the 458 gold ids is affected, and no gold id is `sol` — and that was *checked* rather
> than assumed, because "it happens to agree today" is precisely what C-14 found to be false.
> The guard's value is that it does not depend on the bug having bitten yet.

### The relevance filter, priced and closed — and the defect that is actually there. **[NR-42]**

A $0 probe over artifacts already on disk: the three same-day source arms and the frozen pools
they ranked. No LLM calls, no judge calls, nothing re-run. The item is the "relevance condition
on non-arXiv results" that P22/P23 proposed, P24 retired, C-33 reopened in a narrower form and
C-34 then aimed at the wrong term.

**Verdict: do not build it.** Four independent reasons, any one sufficient.

**1. It filters a term that is already positive.** C-34: Europe PMC's own papers are worth
**+0.73/case**, OpenAlex's **+0.46**. A filter can only shrink a positive quantity unless it
discriminates almost perfectly.

**2. The only filter buildable today is net negative on both sources.**

| source | source term | restricted to gate-3 | the trade |
|---|---|---|---|
| Europe PMC | +0.73 | **+0.38** | loses 13 actionable to remove **0** non-actionable |
| OpenAlex | +0.46 | **+0.27** | loses 21 actionable to remove 7 |

**3. Neither instrument separates — and they are the two that solved this for arXiv.** The gate
and the fine-scale rescore are the system's only pre-judge signals. On OpenAlex's non-arXiv
papers the gate-3 rate is **0.588 among actionable and 0.588 among non-actionable** — identical
point estimates, heavily overlapping intervals. The rescore reaches **28 of 28** band papers
(it keys on `arxiv_id`, which non-arXiv papers fill with their DOI, so nothing excludes them)
and its mean P is **0.842 on actionable against 0.850 on non-actionable** — the wrong way
round. Scoped honestly: those are papers the rescore *admitted*, so the distribution is
truncated at its own threshold; this bounds its ordering within the admitted set and is not a
measurement over the papers it rejected.

**4. The cost is not in that term.** −1.22 for OpenAlex splits into **−0.78 slots and −0.44
quality**, and the slot loss is not digest competition: only 3–5 of 37 cases reach the 15-paper
window and the mean digest is 8.4. Papers lose their place upstream, in the gate's
`gate_depth: 50` input, which is **shared across sources** — so enabling a source spends
arXiv's gate slots. Europe PMC pays the same term (−0.28 slots, quality neutral at +0.09); the
difference is that its own papers cover it.

#### The defect that is actually there, and it is not relevance

| pool | arXiv abstracts | non-arXiv abstracts |
|---|---|---|
| core25 + Europe PMC | 100%, mean 1245 ch | **100%** of 17,511, mean 1976 ch |
| core25 + OpenAlex | 99.5%, mean 1244 ch | **73.5%** of 10,501, mean 915 ch |

A quarter of OpenAlex candidates arrive with **no abstract**. `finescale.build_prompt` reads
`paper.get("abstract", "")[:1500]` and the gate reads the same field, neither with a guard — so
those papers are scored on their titles. Among papers actually shown:

| OpenAlex shown | no abstract | mean length |
|---|---|---|
| actionable (51) | **1 (2.0%)** CI [0.003, 0.103] | 1393 ch |
| non-actionable (17) | **4 (23.5%)** CI [0.096, 0.473] | 729 ch |

The intervals separate, but barely, at n=17 — recorded as a defect in the scoring path, **not**
as a calibrated effect size. C-33 and C-34 were both cases of reading more from a small margin
than it held, and this margin is smaller than either.

What is solid independent of the interval: **a paper with no abstract is not an irrelevant
paper, it is an unmeasured one**, and scoring it anyway is void read as signal — this project's
most repeated failure (C-4, C-30, the 21% that measured nothing, the pool scanner that read
1,250 papers as 0). The product already takes the opposite stance one stage over: *"a paper
whose rescore call fails is omitted, never scored."* The gate scoring an abstract-less paper 2
or 3 is inconsistent with the system's own documented failure policy.

**And the fix is deliberately not proposed as a source strategy.** An evidence threshold is a
**complete no-op on Europe PMC** — 100% coverage, and it is the only currently net-positive
source. On OpenAlex it moves the source term +0.46 → +0.65 but the arm only −0.76 → −0.57,
because a display-time cut leaves displacement untouched. At 1000 characters, discarding 8
actionable papers, the arm is still negative.

#### What would reopen the item

The prize is real; the instrument is not. A perfect discriminator with zero displacement takes
OpenAlex to **+7.11 against the control's +5.73** — headroom **+1.38**, above the benchmark's
MRE of **1.04**. So this closes on the absence of an instrument, not the absence of value.

Everything realistic falls below the floor: Europe PMC's whole ceiling is +0.78; fixing its
slot term is worth +0.28; no evidence threshold makes OpenAlex positive. **Unmeasurable on this
benchmark even if built** — the selection rule refusing the work before a run is paid for.

**Cost** $0. Pinned by `tests/test_nonarxiv_evidence.py`.

#### The guard shipped, 2026-08-28

`src/reporadar/evidence.py`: `has_abstract` and `partition_by_evidence`, used by **both** LLM
stages. `triage_papers` and `finescale.score_papers` now skip a paper with no abstract instead
of scoring it on its title, and both say how many they skipped.

Four boundaries, each deliberate and each pinned by `tests/test_evidence_guard.py`:

- **Absence, not brevity.** The non-actionable non-arXiv papers averaged 729 characters against
  1393 for the actionable ones, and that is *not* turned into a threshold. A short abstract is
  evidence, merely less of it; picking a cutoff would be tuning the gate against net@2 through a
  back door, which is precisely what this entry declined to do one section above.
- **No backfill.** `top_k` is applied before the guard, so a skip shortens the batch rather than
  pulling the next-ranked paper into it. Backfilling would change *which* papers the gate sees —
  a separate decision needing its own measurement — and keeping this a pure removal is what
  makes its effect readable.
- **A skip is not a failed call.** `enough_scored(scored, attempted)` exists to notice that a
  whole stage broke and skip the fine-scale gate rather than abstain by accident. Its denominator
  is what was *attempted*, so the pipeline partitions the band first and passes the readable
  count. Passing `len(band)` would have let an abstract-poor band read as an outage and abandon
  the rescore for the papers that *were* readable — the void-as-null error one line below the
  guard against it, which is how this defect class keeps recurring.
- **Not configurable.** Every other stage's failure policy is an invariant, and a flag whose
  off-position restores "score papers you cannot read" is a footgun rather than a choice. A test
  asserts no such knob appears in `config.py`.

`triage_papers` had already promised this in prose — *"a paper whose scoring fails is omitted
(never scored 0), so downstream tiering treats 'couldn't judge' as 'not a confident Top Pick',
not as a confident rejection."* A missing abstract is the clearest case of not being able to
judge there is; the guard makes the promise true **before** the call rather than only after one
fails. It also saves the call.

**No published number moves.** Every benchmark run to date is arXiv-only or arXiv+EPMC, and both
are at ~100% abstract coverage — the guard is a no-op on all of them. That is the point: it was
shipped as a correctness fix, on the argument rather than on a benchmark win, and NR-42 priced
what it would buy as a source strategy at −0.76 → −0.57 on the one arm where it fires at all.

### The comparator arm, finished: the margin is shyness, and it crosses zero. **[P26, C-34]**

Six materials-science runs at the settings every other Opus 5 row used (v2 prompt, 30-turn
cap, `claude-opus-5`, effort unpinned, subscription auth). Draw 1 now covers all 37 cases.
All six returned `ok` with **zero** hallucinated, unjudgeable or lookup-failed picks — the
first cohort where every pick got a verdict. **$58.34**; the arm to date is **$351.40**.

| cohort | RepoRadar (arXiv+EPMC) | Opus 5 | paired | |
|---|---|---|---|---|
| core 25 | +6.16 | +4.20 | **+1.96** | 14W/10L |
| bio 6 | +7.50 | +5.83 | **+1.67** | 4W/1L |
| **matsci 6** | +5.50 | **+8.67** | **−3.17** | 2W/4L |
| **all 37** | **+6.27** | **+5.19** | **+1.08** CI [−0.97, +3.16] | 20W/15L |

**The margin was +1.90 over 31 cases and is +1.08 over 37, with the interval now crossing
zero.** Nothing was re-measured to produce that. Twelve cases the arm always intended to
cover were finished, and the figure moved. A margin that holds only while a third of its
cohort is missing is one to report with its interval attached.

The materials reversal is not a volume artefact, which was the first thing to rule out —
net@2 sums over what a system returns, and Opus 5 does return more there (12.7/case against
10.5). It also wins on the **rate**: 0.895 precision against 0.841, RepoRadar's worst cohort
and Opus 5's best. Everywhere else the ordering is the other way round (core 0.917/0.820,
bio 0.930/0.887). No source arm rescues it either — arXiv-only +5.00, +EPMC +5.50,
+OpenAlex +3.83, all below +8.67.

#### Where the margin lives, and what it is

| split | n | RepoRadar | Opus 5 | paired |
|---|---|---|---|---|
| Opus 5 over-answered (net@2 < 0) | 5 | −0.20 | −8.60 | **+8.40** |
| Opus 5 did not | 32 | +7.28 | +7.34 | **−0.06** |

**On the 32 cases where Opus 5 does not over-answer, the two systems are level.** All of the
+1.08 — 105% of it — comes from five cases, and four of those (`cli`, `http`, `linter`,
`webdev`) are ones where RepoRadar returns nothing at all and Opus 5 returns 5–20 papers,
averaging −7. That split alone is 70% of the total margin. It was already visible on the core
25 (21 cases, −0.29) and now replicates at n=32 with twelve cases that were not in it.

RepoRadar's advantage over Opus 5 is a **shyness advantage, not a retrieval advantage.**
net@2 charges 2 per false positive precisely to price shyness, so the advantage is real and
the gate is what produces it. It is not the claim "we find better papers", and §5 should not
be allowed to imply that it is.

> **A comparison run and discarded.** Capping both systems at `min(n)` per case gives −0.49,
> which reads as "the margin is just volume". It is not usable: the cap sets k=0 on exactly
> the four abstention cases, deleting the behaviour under examination. A number that answers
> a different question is worse than no number, because it looks like a check.

#### The published headline, restated against all three comparators

The margin the paper reports names one rival. Three exist now, and a single number cannot
show how much of the result is the rival's strength. Same 25 repositories, each system scored
on its own returned papers, both source arms carried side by side because arXiv+EPMC scores
higher everywhere and **is not what ships**:

| comparator | net@2 | /case | prec | abst | vs published +5.72 | vs arXiv +5.84 | vs arXiv+EPMC +6.16 |
|---|---|---|---|---|---|---|---|
| Opus 4.8, v1, 12 turns — **published** | +1.84 | 2.2 | 0.931 | 6 | **+3.88** *p*=0.0007 | +4.00 *p*<0.001 | +4.32 *p*<0.001 |
| Opus 4.8, v2, 30 turns | +2.16 | 4.2 | 0.838 | 3 | +3.56 *p*=0.019 | +3.68 *p*=0.027 | +4.00 *p*=0.007 |
| **Opus 5**, v2, 30 turns | **+4.20** | 9.1 | 0.820 | 0 | **+1.52** 12w/**13l** *p*=1.00 | +1.64 *p*=0.69 | +1.96 *p*=0.54 |

**The margin survives a harness upgrade and does not survive a model upgrade.** Every cell
against an Opus 4.8 comparator clears *p* < 0.05; not one cell against Opus 5 does, on any
arm. The published cell reproduces `restated_runs.json` to the digit — +3.88, CI [+2.24,
+5.60], 17w/2l/6t, *p* = 0.0007 — which is what licenses reading the rest of the table as
comparable rather than as a second computation of the same thing.

**The comparator was not under-resourced,** which is the obvious objection to a baseline
recommending 2.2 papers where we show 8.5. Decomposed: the harness step (v1→v2, 12→30 turns,
*same model*) is worth **+0.32**; the model step (Opus 4.8→Opus 5, *same harness*) is worth
**+2.04**. Six-sevenths is the model. +1.84 is a fair instantiation of the system it names.

The sign test against Opus 5 on the published arm is exactly **1.00** — 12 wins to 13 losses.
The mean is ahead and the case count is not, which a mean alone conceals. Over all 37 cases
the shipped arm is **18w/18l/1t**. Derived by `evals/restate_comparator.py` into
`evals/comparator_ladder.json`, pinned by `tests/test_comparator_ladder.py`.

> **One estimator, repo-wide.** `freeze_opus5_arm.py` had its own seeded bootstrap and
> reported [−0.97, +3.16] where `bigram_report.paired_bootstrap` gives [−0.97, +3.22] for the
> same deltas. A grid step apart, decision-irrelevant, and still two artifacts in one
> repository disagreeing about one quantity — C-25 recorded the same surprise about the same
> helper. The private bootstrap is gone; both artifacts now import the shared one and agree
> exactly.

#### Is Opus 5 winning on materials through non-arXiv sources? No — the opposite

The v2 prompt lets the baseline cite anything, and Opus 5 uses that freely. Share of its
picks that are not arXiv ids: **core 25 34%, bio 70%, matsci 6.6%** — 5 of 76 papers, worth
+0.33/case of a +8.67 result, under 4%. Materials science is where it reaches outside arXiv
*least*, and it is the one cohort it wins. Restricted to arXiv alone it is still ahead there
(0.901 against 0.841) and still behind on the core 25 (0.873 against 0.916).

So the answer is not "it has better sources". On materials repositories it picks better
arXiv papers than our ranker does, from a literature our pool already contains.

#### C-34: the arithmetic matched and the mechanism was still wrong

P25 attributed OpenAlex's −0.76 to its 17 non-actionable papers: 17 × −2 over 37 cases is
−0.92, near enough to look like an explanation, and `test_the_misses_roughly_account_for_the_loss`
asserted it. Decomposing the delta exactly — every non-arXiv paper is one the source
supplied; everything else is arXiv churn — gives:

| arm | delta | source's own papers | arXiv displaced |
|---|---|---|---|
| +EPMC, 37 | +0.54 | **+0.73** | −0.19 |
| +OpenAlex, 37 | −0.76 | **+0.46** | **−1.22** |

**OpenAlex's own papers are net positive.** The 17 misses arrive alongside 51 actionable
ones. The loss is 142 arXiv papers leaving the digest and 100 different ones arriving. The
two terms sum to the delta by construction, so this is a decomposition, not a model.

The materials six settle it: Europe PMC contributes **zero** papers there and the arm still
moves +0.50/case — 16 arXiv papers out, 16 in. **A source can move the score without
appearing in the digest at all.**

This reverses the remedy. A relevance filter on non-arXiv material — the item C-33 reopened
— addresses the term that is already positive. The term that costs is displacement, which
scales with how much a source is admitted and is paid by *both* sources; Europe PMC's own
papers merely cover it. That is also the real argument against stacking a third source.

The near-miss is the lesson worth keeping. −0.92 against an observed −0.76 is the most
dangerous kind of wrong explanation: close enough that nobody checks it. C-33 was
generalising a mechanism from one arm; C-34 is accepting a mechanism because its magnitude
happened to fit. Both were written into a passing test.

#### A gap closed on the way

`mat-phonon`'s pick `1703.03212` had been unjudged since the case was added — the sole entry
under `incomplete` in the gold set. Opus 5 picked the same paper, the judge scored it 2, and
the verdict cache is shared, so the `cli` baseline's long-standing pick finally resolved.
Gold set 75 → 76 targets, scisoft 19 → 20, **`incomplete` is empty for the first time**.
`benchmark25` is untouched at 56 targets / 20 cases: no published denominator moved.

The witness set grew 638 → 698 and regret held at exactly **+7.52**, which is what
`test_the_headline_regret_figures` predicted for growth outside the scored 25-case cohort.
Second confirmation, at 60 witnesses after the bio step's 47.

**Cost** $58.34.

### OpenAlex reaches the digest and costs -0.76: the gate is not uniformly robust. **[P25, C-33]**

Same control as P24, same flags, `--sources arxiv,openalex`. `pool_config` confirms the arms
differ in `sources` and nothing else.

| cohort | control | +europepmc | **+openalex** | openalex paired | 95% CI | W/L |
|---|---|---|---|---|---|---|
| core 25 | +5.84 | +6.16 | **+5.40** | **-0.44** | [-1.56, +0.72] | 7/10 |
| scientific 12 | +5.50 | +6.50 | **+4.08** | **-1.42** | [-3.58, +0.83] | 5/7 |
| all 37 | +5.73 | +6.27 | **+4.97** | **-0.76** | [-1.81, +0.30] | 12/17 |

Every interval crosses zero, so no single cohort is decisive. **The sign is consistent across
all three**, and the win record is 12W/17L against Europe PMC's 14W/6L. Consistency is the
evidence here, not any one delta.

**Reach was never the constraint — precision was.**

| | non-arXiv in digest | precision | non-actionable admitted |
|---|---|---|---|
| Europe PMC | 30 / 326 = 9% | **0.97** | **1** |
| **OpenAlex** | **68 / 337 = 20%** | **0.75** | **17** |

OpenAlex reaches the digest **more than twice as often** and is worth less for it. 17 misses
at -2 each over 37 cases is about **-0.92/case** against an observed -0.76 — not an identity,
since admitted papers also displace arXiv ones, but close enough that the penalty rather than
the displacement is the dominant term.

#### C-33: "the gate handles the collision" was true of one source, and was generalised

P24 retired the relevance-filter item on the evidence that **29 of 30** non-arXiv papers
reaching a digest were judged actionable. That evidence was Europe PMC only, and the
conclusion drawn from it — that the gate handles off-domain material, so no filter is needed —
does not survive a second source. OpenAlex admits **17** non-actionable papers where Europe
PMC admitted 1.

The corrected claim is narrower and more useful: **the gate rejects *obviously* off-domain
material and admits *near-domain* material.** Europe PMC's neurosurgery paper beside a linter
is easy to reject; OpenAlex's Engineering, Materials Science and Social Sciences results
beside a compiler are not — and that is precisely the borderline population NR-11 recorded as
making the headline *worse* when a wider pool met a near-binary gate.

The item is reopened in that narrower form. Note what it took to find: the first version of
this claim was made from one arm, in a PR, with the word "settled" in it.

**Stacking is now measured rather than cautioned against.** +0.54 and -0.76 on the same 37
cases; a three-source arm would most likely net negative.

**And P23 predicted the wrong thing about this, for a defensible reason.** OpenAlex measured
*cleaner* than Europe PMC on the collision probe — 48% off-domain against 68% — so it looked
like the better candidate. It is the worse one, because the probe measured what a source
RETURNS and the gate is what decides what survives. A source whose noise is obviously wrong is
safer than one whose noise is plausibly right. **A $0 retrieval-side probe cannot predict a
gate-side outcome**, which is worth remembering before the next one is used to rank candidates.

### Multi-source retrieval, against a matched control at last: +0.32 on the benchmark. **[P24, NR-41]**

Every earlier multi-source figure in this project compared a run with a source enabled against
a run collected on a **different day** — P21's bio **+4.00** among them. This arm holds
everything fixed but `sources`: two fresh frozen pools collected the same day, the same 37
repositories, same window, same `w_embedding`, same HyDE, same gate.

**The control reproduces the published headline.** +5.84 against the 2026-08-15 run's +5.72 on
the same 25 cases — **delta +0.12**, far inside this project's own noise floor (Jaccard 0.49
on the ranked top-10, its largest variance term). A fresh collection, a fresh HyDE pass and a
fresh draw land essentially on the frozen pool. That is what licenses reading the treatment
delta as an effect rather than as a redraw, and it independently confirms +5.72 was not a
lucky draw.

| cohort | control | treatment | paired | 95% CI | W/L | non-arXiv in digest |
|---|---|---|---|---|---|---|
| **core 25** | +5.84 | +6.16 | **+0.32** | [-0.24, +0.88] | 8/4 | **2 / 205 = 1%** |
| scientific 12 | +5.50 | +6.50 | **+1.00** | [-0.67, +2.50] | 6/2 | 28 / 120 = 23% |
| all 37 | +5.73 | +6.27 | +0.54 | [-0.08, +1.16] | 14/6 | 30 / 325 = 9% |

**NR-41: on the benchmark's own cohort, Europe PMC buys +0.32 and the interval spans zero.**
The mechanism is not what the collision probes implied. P22 measured 68% of Europe PMC's
results for these repositories as MeSH-indexed and warned about biomedical papers reaching
software digests. What actually happens is that **the channel supplies 2 of 205 shown
papers** — the gate rejects nearly all of it. Off-domain results are not admitted and
punished; they are discarded, exactly as the +0.00 for the one previously-measured channel
already said.

**P21's +4.00 does not survive the control.** Matched, the bio effect is +1.00 over twelve
scientific cases with an interval crossing zero. Most of the original figure was the
collection, not the source — which is what a matched control is for, and a caution about every
uncontrolled source comparison in this repository's history.

**The gate handles the collision, and the "relevance filter" item is retired.** Across all 37
cases, **29 of 30** non-arXiv papers that reached a digest are judged actionable; one scored 1
and none scored 0. There is nothing for a filter to remove. That item was proposed here from
the collision measurement *without checking the digests first*; checking refuted it.

**What this says about the paper.** §9's "one source, and a benchmark shaped like it" now has
a measurement behind it rather than an argument: multi-source retrieval, done properly, is
**+0.32 [-0.24, +0.88]** on the 25-repository benchmark. The judgement that these channels
have not been shown to help survives contact with a controlled experiment. The scientific
cohort is the one place the evidence points, and it is where the literature genuinely sits
off arXiv.

#### And against Opus 5, on the 31 cases it covers

| cohort | RepoRadar (arxiv+epmc) | Opus 5 | paired | 95% CI | W/L |
|---|---|---|---|---|---|
| core 25 | +6.16 | +4.20 | +1.96 | [-0.56, +4.56] | 14/10 |
| bio 6 | +7.50 | +5.83 | +1.67 | [-0.33, +4.50] | 4/1 |
| **25 + 6 bio** | **+6.42** | **+4.52** | **+1.90** | **[-0.13, +4.06]** | 18/11 |

Europe PMC accounts for about +0.55 of that (the arXiv-only control is +5.87 on the same 31).
The interval still touches zero and rests on a single Opus 5 draw whose own same-model
draw-to-draw spread was ±0.76 — so this is "probably ahead", not a result. But 18W/11L is a
markedly better record than the 12W/13L reported when the comparison was arXiv-only RepoRadar
against a baseline drawing a third of its wins from outside arXiv.

**The published margin is +3.88 against the v1/12-turn comparator and +1.90 against Opus 5.**
Roughly half of the headline does not survive a modern baseline. That is the number the paper
will have to state.

### OpenAlex collides too — less, and differently. **[P23]**

The same probe, the other source, run before any judge call. OpenAlex is a *general* index
rather than a biomedical one, so the prediction was that it might not collide at all. It does,
on nearly half of what it returns.

| | Europe PMC (P22) | **OpenAlex (P23)** |
|---|---|---|
| repositories returning nothing | 0 / 25 | **0 / 25** |
| hits | 1,721 | 1,740 |
| off-domain | **68%** (MeSH-indexed) | **48%** (field not CS-adjacent) |
| range across repositories | 57%–87% | **24%–84%** |

**The instrument is better here, and that matters.** OpenAlex labels every work with
`primary_topic.field` — its own 26-field taxonomy. No marker list, no field that can be
silently empty (the P22 failure), and it reports *which* discipline came back:

| field | share | |
|---|---|---|
| Computer Science | 34% | on |
| Biochemistry, Genetics and Molecular Biology | 15% | **off** |
| Medicine | 8% | **off** |
| Engineering | 6% | on |
| Physics and Astronomy | 5% | on |
| Social Sciences | 5% | **off** |
| Decision Sciences | 4% | on |
| Materials Science | 3% | **off** |
| Mathematics | 2% | on |
| Neuroscience | 2% | **off** |

**Computer Science is the largest single field and still a minority.** OpenAlex genuinely
reaches the ACM/IEEE/VLDB literature Europe PMC structurally cannot — 599 CS works — but
brings a great deal else with it.

**The wider spread is the finding.** 24% (`speech`) to 84% (`webdev`), against Europe PMC's
much flatter 57–87%. Repositories with distinctive technical vocabularies retrieve cleanly —
`speech` 24%, `cv` 29%, `crypto` 30%. Repositories whose profiles yield generic English do
not: `webdev` 84%, `systems` 73%, `thin-kv` 72%.

That is P22's conclusion arriving a second time by a different route: **the collision is a
property of the QUERY, not of the source and not of the repository.** A domain classifier
routing sources per repository cannot fix `webdev`, because `webdev` is a perfectly ordinary
software project whose queries happen to be common words.

**Verdict, both probes together.** No multi-source default and no per-domain routing. What
the evidence supports is a **relevance condition on non-arXiv results** — precisely what the
ranker's category component would supply if uncategorised papers did not escape it (§9's
measured bias, 18 of 32 papers moved). That is a ranking change, and it is the same fix for
both sources, which is the strongest argument that it is the right one.

**If a source is to be switched on anyway, OpenAlex is the one** — a third less noise, real
CS coverage, and a field label already attached to every result that a filter could read
without any new machinery.

#### A label that lied, caught before it was quoted

`report()` printed "repositories **Europe PMC** returned NOTHING for" and "a high
**biomedical** share" regardless of `--source`, so the OpenAlex run's output described itself
as a Europe PMC result. Nothing numeric was wrong; the frame around the numbers was. Fixed to
read the source, and the two artifacts are separate files (`europepmc_collision.json`,
`openalex_collision.json`) because they measure different quantities under different
taxonomies and a single file would invite exactly the averaging they cannot support.

### Europe PMC answers a compiler confidently, and 68% of what it says is biology. **[P22]**

P21 measured Europe PMC at **+4.00 net@2** on the six bio repositories, by *coverage* rather
than ranking: 54% of the shown digest came from it, at **0.97** precision. That is the
strongest multi-source evidence this project has, and the obvious next move was a
`arxiv,europepmc` default for every repository. This probe is why that would have been wrong.

**$0, no LLM and no judge calls.** Each core-25 repository's OWN queries — built by
`harness.profile_case_repo` and `collector.build_queries`, the shared implementations a real
run uses — sent to Europe PMC's free API. Classification by Europe PMC's own cataloguing, not
by a model: MeSH headings, assigned by NLM cataloguers, plus the Index Medicus subset.

| | result |
|---|---|
| repositories returning **zero** hits | **0 / 25** |
| total hits | 1,721 |
| **indexed as biomedical** | **1,176 = 68%** |
| range across repositories | 57% (`storage`, `compiler`) to 87% (`thin-gnn`) |

**Every repository is above half.** This is not an average dragged up by a few outliers.

The titles are the argument. Query `lint code` returns *"Occurrence of postoperative
pneumoencephalus in posterior fossa surgery using the semi-sitting position"*. Query
`arrow file` returns breast cancer, zebrafish telomerase, goat peripheral blood mononuclear
cells. Query `ruff server` returns plant phenotyping and delta opioid receptors.

**This is the expensive outcome, not the harmless one.** A source that stays silent outside
its domain costs nothing to enable everywhere. A source that answers *confidently and
off-domain* feeds the candidate pool of every repository — where net@2 charges **2 per false
positive**, and where non-arXiv papers additionally escape the ranker's category component, a
bias already measured moving 18 of 32 such papers into or out of the top-10.

**It is not uniformly noise, and the pattern says what the axis really is.** `crypto` gets
genuine post-quantum cryptography and blockchain papers; `compiler` gets GPU prefix-free
parsing and Python vectorisation. Domain-neutral CS terms retrieve real CS work. Generic
English does not: `key`, `arrow file`, `lint code`, `data`.

So **the collision is a property of the QUERY, not of the repository** — and per-domain
routing is therefore the wrong design as well as a premature one. `crypto` would be routed
"on" by any domain classifier and still receives biology from its own query `key`.

**Verdict: no multi-source default, and no per-domain routing on this evidence.** What the
data supports is a *relevance* condition on non-arXiv results — the same thing the ranker's
missing category component would have provided if uncategorised papers did not escape it.

#### The first run of this probe reported 21%, and it measured nothing

`_looks_biomedical` substring-matched journal names out of `journalTitle` and `pubType`. Both
are `None` on essentially every Europe PMC record — the real fields are `meshHeadingList`,
`subsetList`, and a nested `journalInfo`. So the flag compared against empty strings, returned
False for everything it could not see, and reported that 79% of the hits were **not**
biomedical.

It failed in the direction that reads as good news, which is the direction that gets believed.
Nothing about the run looked wrong: 25 cases, 1,721 hits, a plausible-looking 21%.

It was caught by reading the sampled titles instead of the counter — they were transparently
biological while the number said otherwise. The repair uses Europe PMC's own indexing rather
than a guess about words, and `tests/test_epmc_collision.py::test_the_flag_reads_fields_that_exist`
asserts that the two fields the broken version read are not sufficient on their own.

The artifact now stores the **query** beside each sampled hit, so the next reader can check
the collision rather than take 68% on trust — which is precisely what the first run's reader
could not do.

### The bio comparison, measured at a matched configuration — and `w_embedding` costs 1.33 there. **[P21, NR-40]**

The Opus 5 sweep ran the v2 prompt, so **it is not arXiv-limited**: on the bio cases **70% of
its picks and 68% of its certified targets are non-arXiv DOIs** (31 ACM, 10 Nature, 7 IEEE, 6
Oxford/NAR, 5 VLDB across the whole sweep). Comparing that against an arXiv-only RepoRadar was
not a like-for-like contest, and the first bio figure quoted here (+5.50 against Opus 5's
+5.83) was measured under exactly that asymmetry.

**Two RepoRadar runs already covered bio and differed by 5.5 points**, entirely on
configuration: 2026-08-20 at window 15 / arxiv-only gives +5.50, 2026-08-21 at window 30 /
arxiv+europepmc gives +11.00. Neither is the shipped configuration and neither answers the
question. So the missing arm was run: **window 15, sources arxiv+europepmc, `w_embedding`
1.5**, against the frozen `pool-epmc-treat` so no new collection draw enters the comparison.

| case | **w15 + epmc + wemb1.5** | w30 raw | w30 truncated to 15 | w15 arxiv-only | Opus 5 |
|---|---|---|---|---|---|
| bio-align | +5 | +10 | +10 | +0 | +7 |
| bio-kmer | +7 | +8 | +8 | +3 | +6 |
| bio-mdsim | +6 | +8 | +8 | +7 | +6 |
| bio-mdtraj | +7 | +4 | +4 | +7 | 0 |
| bio-scvi | +15 | +22 | +15 | +11 | +12 |
| bio-singlecell | +9 | +14 | +12 | +5 | +4 |
| **MEAN** | **+8.17** | +11.00 | +9.50 | +5.50 | +5.83 |

**RepoRadar +8.17 against Opus 5's +5.83, paired +2.33, 4W/1L.** The arXiv-only run said
-0.33; the reversal is the missing source, not the window.

**The 5.5-point gap decomposes cleanly**, which is the point of running the arm rather than
truncating:

| axis | effect on bio net@2 |
|---|---|
| window 30 -> 15 | **-1.50** |
| adding Europe PMC (at wemb 0.0) | **+4.00** |
| `w_embedding` 0.0 -> 1.5 | **-1.33** |

**NR-40: `w_embedding` 1.5 is worth -1.33 on the scientific cohort.** The value ships because
it was tuned on the arXiv-only core 25; on bio it costs a point and a third. One 6-case run,
so not conclusive — but it is the opposite sign from the assumption, and it was only visible
because the axis was varied rather than derived. Truncating the window-30 run would have given
window 15 + epmc at wemb **0.0** and quietly answered a different question.

**Window truncation IS valid, and was verified rather than assumed.** The digest is
`rerank_by_actionability(gated)[:top_n]` — a final cut on a list already ordered by
`llm_score`, and `--rr-window` is *refused* when it exceeds the candidate depth, so it cannot
silently reorder anything. Checked on the data: `llm_score` descends monotonically through all
22 stored picks of the window-30 run. So w30 -> 15 above is a free, sound derivation; it just
cannot reach the `w_embedding` axis.

#### C-9 is repaired, and the non-arXiv sources are testable again

C-9 recorded that every non-arXiv source had been sent arXiv boolean syntax as a keyword query
for the whole of the product's history, and **C-9b** that the repair was published as "routed
through all three call sites" when there were five and it routed two. Verified now against the
live code rather than the note: `pipeline` computes `plain = [to_plain_keywords(q) ...]` **once**
and every one of the six `KEYWORD_SOURCES` receives it — semantic_scholar, openalex, biorxiv,
europepmc, iacr, dblp. On a real query,
`(all:"vectorized execution" OR all:"columnar storage") AND (cat:cs.DB)` arrives as
`vectorized execution columnar storage`.

The design also defends itself: the fetchers keep real lazy `from ... import` statements rather
than a table of module strings driving `importlib`, because — as the comment says — that
"would have blinded `tests/test_stages.py`, which reads the import graph to prove the drift
warning tells the truth. A guard that cannot see the import cannot check it."

**So OpenAlex as a RETRIEVAL source is testable today**, and it is the obvious next probe: it
reaches the ACM/IEEE/VLDB literature holding 43 of Opus 5's non-arXiv targets, where Europe PMC
(biomedical) structurally cannot go — and P20 independently established that OpenAlex carries
those abstracts, because that is what the new verifier tier does. `pool-oa-treat` and
`pool-oa-control` are already on disk.

#### The frozen-pool guard earned its keep, and exposed a recording gap

The fingerprint guard **refused two attempts at this arm** and was right both times: the
command omitted `--rr-all-time` and `--rr-hyde`, both POOL_FLAGS. Its message names the failure
exactly — *"Reusing it would measure the old settings under the new run's name"* — and it
volunteers that ranking flags are not part of the fingerprint, so varying `--rr-window` and
`--rr-w-embedding` against a frozen pool is what freezing is for. Without it, a mislabelled arm
would have shipped as a result.

Recovering the right command required a **prose section of `RESEARCH-scientific-software.md`
(§14.4)**, because **`rr_hyde` and `rr_all_time` are pool-affecting flags that run artifacts do
not record**. The row stores `sources`, `bigram_mode`, `typed_anchors`, `scan_source` — not
these two — and the frozen pool's `pool_flags` stores flag *names*, not values. So no artifact
can answer "was HyDE on?" That is the same gap `digest_window` was recorded to close: *"an arm
cannot be reported under a window its own run file contradicts."*

### The abstract gap was the binding constraint, and one tier closed most of it. **[P20]**

The v2 sweep left 44 references unscoreable, every one a DOI and most of them ACM (P19). That
is a limit on the instrument, not the searcher: the papers were found, named, and proven to
exist, and nothing downstream could read them.

**Which source to add was measured, not chosen.** All 28 distinct unscoreable papers were
probed against both candidates before a line was written:

| | has an abstract |
|---|---|
| **OpenAlex** | **20 / 28** |
| Crossref | 14 / 28 |
| Crossref ∩ not-OpenAlex | **0** |

Every paper Crossref carries, OpenAlex carries too, and OpenAlex adds six more. The
half-hour probe is the reason this was one tier rather than two — "Crossref or OpenAlex" was
two options where the evidence supports one.

**The result.** Unscoreable references **44 → 13**; non-arXiv targets **36 → 61**; sweep
targets **196 → 221**; and **no ACM paper is unscoreable any more**. The searcher did not
change, so all 25 new witnesses are papers v2 had already named. Witness set **462 → 482**,
digest regret **+5.56 → +6.24** net@2/case.

`cli-v2@30`'s reach into `pool-wemb` *fell* as it grew, 0.141 → **0.123**, and pooled non-self
reach with it, 0.149 → **0.138**. That is the measure working: the papers the tier unlocked
are ones the shipped collection step holds even less often than the arXiv ones.

**The residual is 8 papers and is now a named floor**: five Springer book chapters
(chronically abstract-free), one Elsevier journal paper, and two fabricated DOIs that never
existed. Worth stating in the write-up rather than chasing.

#### `unjudgeable` is permanent only relative to the sources you asked

Adding a tier needed machinery, not just a function, and the reason is a genuine tension
between two corrections this project already paid for. `unjudgeable` is deliberately *not*
retryable — C-30 was the cost of a row that could never be finished being asked forever. But
that permanence is a property of the tier list, not of the paper: the moment the list grows,
every stored `unjudgeable` becomes a claim its evidence no longer supports. Left alone, the
31 recoverable references would have sat behind a predicate correctly refusing to re-ask a
question that had been settled.

`verify.TIER_SET` names the tiers; every judged row records it; `retryable` reopens a row when
the current set is a **strict superset** of the recorded one. Strictness is load-bearing in
both directions — reordering changes nothing findable, and losing a tier can only find less,
so neither should spend a call. `repair_row` stamps the new set, which is what makes the
clause terminate instead of firing on every future invocation. It is `prompt_version`'s lesson
one module over: a cached verdict has to carry the configuration that produced it, or it
quietly outlives its own justification.

Only rows with an `unjudgeable` verdict carry the stamp, and that asymmetry is deliberate: a
row whose references all resolved has nothing a new tier could reopen, and back-stamping it
with a tier set that did not exist when it ran would be a fabricated provenance claim.

#### Two things the tests caught that review did not

* **The stub fixture stopped covering every tier.** `tests/test_verify_widened.py` exists on
  the rule that its tests must run with no network — "a test that needs Semantic Scholar to
  be up is a test that reports the weather". Adding `openalex` to the resolution chain without
  adding it to the fixture turned two existing tests into live HTTP requests, silently, on
  exactly the paths that reach the new tier.
* **A new test passed for the wrong reason.** `test_it_rescues_what_the_earlier_tiers_miss`
  patched `fetch_work_by_doi`, but the fixture replaces `resolve_by_doi_openalex` wholesale,
  so the patch had no effect at all. It is driven through the fixture now.

`sources/openalex.py::_request_json` also carried the C-32 conflation already — `None` for
both a spent 429 and a 404 — so the new tier received the same `status` out-parameter
treatment at birth rather than reintroducing the bug one source over.

### The v2 prompt reaches past arXiv, and the instrument becomes the binding constraint. **[P19, C-32]**

`BASELINE_PROMPT_V2` permits journal, conference and bioRxiv/medRxiv papers and asks for an
arXiv id **or** a bare DOI. 75 draws over the benchmark25 cohort at a 30-turn cap, 2026-08-26,
**0 failed and 0 partial** — against 6/5/3 failures in the three 12-turn v1 draws.

| | picks | DOI picks | targets | DOI targets | precision |
|---|---|---|---|---|---|
| v1 @12 turns (23 cases) | 140 | 0 | 124 | 0 | 0.912 |
| **v2 @30 turns (25 cases)** | **270** | **97** | **196** | **36** | **0.867** |

**It reaches what it was written to reach.** 36% of picks and 18% of targets are non-arXiv —
papers v1 had no field to name in an answer, and so could not have contributed a witness
however well it searched. Precision falls, but the caps differ and that comparison is
confounded; the only clean control is the four cases carrying a v1@30 draw, where v2 returned
67 picks against 13 and 30 targets against 10.

**The limit has moved from the model to the instrument.** 44 of the 270 references could not
be scored, and **every one is a DOI**: 41 `unjudgeable` — real papers, proven to exist, whose
abstracts neither Semantic Scholar nor Europe PMC carries — and 3 `hallucinated`. Most are
ACM (`10.1145/…`): POPL, PLDI, OOPSLA, CACM, which is where a large part of the
software-engineering literature this system exists to surface is actually published. A
Crossref or OpenAlex abstract tier is now the highest-value verification work available.

**3 invented DOIs in 97 (3.1%)**, caught by the DOI Handle API rather than by the prompt. v2
was deliberately given no anti-fabrication instruction v1 lacks, so this is the unassisted
rate, and it is the number a comparator re-measurement would have to price.

**Pooled in, the witness set goes 385 → 462** and digest regret **+4.80 → +5.56** net@2/case.
Reach into `pool-wemb`: `cli` unmoved at 8/56, `cli-redraw` at 19/92, and `cli-v2@30` at
**19/135 = 0.141**, the lowest of the family. Pooled non-self reach therefore *fell*, 0.174 →
0.149 — the measure working, not regressing: the new source found papers the shipped
collection step is even less likely to have fetched. Chao1 for `cli-v2@30` is **≥ 252.3** from
135 observed with 88 singletons, so this searcher is no nearer exhaustion than the last one.

#### A refusal and a rejection are not the same failure **[C-32]**

This was the first run to send non-arXiv references through `resolve_by_doi_s2` in bulk, and
it exposed a defect the widening itself introduced. `_s2_batch_post` returns `None` for two
unrelated reasons — Semantic Scholar **refused** (429, retries exhausted) and Semantic Scholar
**rejected the id** (HTTP 400, no record) — and the DOI tier read both as refusal. The comment
there was explicit that a 429 must never harden into a verdict about the paper (C-4), and
correct; it simply had no way to tell the two apart.

The consequence was self-perpetuating rather than merely wrong. A real ACM paper with no
abstract in either source came back `lookup_failed` instead of `unjudgeable`, which marks the
row **retryable** — so every future invocation would re-ask a question that could never come
back differently. Five rows were in that state and had already been asked twice.

`_s2_batch_post` now takes an optional `status` out-parameter (the pattern `fetch_references`
already used) recording the HTTP code of a non-retryable failure. The product callers are
unchanged and still see "None means skip the batch"; only `verify.py`, which turns the answer
into a verdict, reads the code. A 4xx that is not 429 falls through to Europe PMC and then to
`unjudgeable`; everything else still raises.

Two smaller defects surfaced in the same run and are fixed with it:

* **A `partial` row was partial forever.** It carries `phase: "judged"`, so no re-invocation
  ever revisited it — the C-30 shape one layer over. `retryable` now derives the answer from
  the outcome counters rather than the status, and `repair_row` re-asks only the picks with no
  verdict. That is possible at all because `picks` survives on the row even though `raw_ids`
  is dropped: it holds the same references, already canonicalised. **19 of the sweep's 22
  lookup failures landed inside one window of serial judging in draw 1**, all recoverable —
  the retry took the sweep from 10 partial rows to 0.
* **A counter that grows every time you look at it.** The first `repair_row` accumulated
  `n_hallucinated` and `n_unjudgeable` while replacing `n_lookup_failed`, against its own
  comment. After two passes `1/linter` reported **31 unjudgeable references against 12 picks
  without a verdict**, and the sweep's total read 71 where the truth was 44. Nothing crashed
  and every number stayed plausible. The identity that catches it — *picks without a verdict
  equals hallucinated + lookup_failed + unjudgeable* — is now pinned for both sweeps in
  `tests/test_gold_spread_v2.py`.

No published figure was ever computed from the wrong values; all three defects were found and
corrected inside the run that produced them.

### Two phases, four workers, thirty turns: the witness generator gets 2.6x faster and stops failing. **[P18]**

```bash
uv run python evals/gold_spread.py --max-turns 30 --concurrency 4   # the faster variant
```

P17's sweep took **4h39m for 74 runs**, of which **114 minutes — 41% — went to runs that
produced nothing** (14 failures x 3 internal retries at ~2.7 min per attempt). The received
explanation was that the job could not be parallelised because arXiv would throttle. That is
true of *part* of the job, and the part is small.

**The phases are separable, and only one is rate-limited.** `run_baseline` shells out to
`claude` and parses the reply; arXiv and the judge are not touched until verification. Split
into phase A (agentic runs, concurrent) and phase B (verify + judge, strictly serial), a
four-case trial compressed **711 s of phase-A work into 275 s of wall clock** at concurrency
4 — with no arXiv throttling, because phase B never runs concurrently by construction rather
than by convention.

**Thirty turns rescued both chronic failures.** Cases chosen before running: two that
succeeded in all three 12-turn draws, and the two that failed in all three.

| case | draws 1–3 @ 12 turns | draw 4 @ 30 turns | `num_turns` | duration |
|---|---|---|---|---|
| `rag` | ok, ok, ok | ok — 5 targets | 16 | 142 s |
| `linter` | ok, ok, ok | ok — 1 target | 17 | 207 s |
| **`thin-lang`** | **error, error, error** | **ok — 2 targets** | 27 | 208 s |
| **`vectordb`** | **error, error, error** | **ok — 2 targets** | 17 | 154 s |

**0 of 4 failed, against 3-of-3 failure for the two hard cases at the shipped cap.** This is
the profile that makes the change safe rather than merely attractive: P15 measured the turn
effect on what a *successful* run finds as inside noise, so raising the cap does not mix two
populations — it converts failures into successes and leaves successes alone. `thin-lang`
reports `num_turns` 27, the highest observed, which is consistent with it genuinely needing
budget (with C-27's caveat that `num_turns` is not the quantity the cap bounds).

**Nothing shared moved.** `cache/baseline/cli/` is byte-identical after the trial, the
discriminator is still `da766b38114e`, and the gold set still re-derives. The script only
ever runs with `use_cache=False`: it is a *witness generator*, and the published comparator is
a separate question it cannot touch.

**Draws at different caps are different configurations**, and `report` now enforces that
rather than trusting anyone to remember: draws are discovered from the artifact instead of
assumed to be `1..k` (so a trial cannot be silently omitted), each draw's cap is recorded and
printed, and the P17 aggregate covers only the shipped cap with off-cap draws listed
separately. A test fails if an off-cap draw appears in the per-draw reproducibility block.

The 12-turn sweep also recorded no per-run timings, so P17's cost had to be reconstructed from
file mtimes. Rows now carry `duration_s`.

**Cost** 4 agentic runs. **Verdict:** for witness generation, 30 turns and concurrency 4 are
strictly better — fewer wasted runs, no measured distortion of what a successful run finds,
and the chronic failures (which the gold set had been representing only through their lucky
draws) start contributing.

### The gold set is a sample, and now we know from how large a population. **[P17]**

```bash
uv run python evals/gold_spread.py --report   # $0, re-read the 75-run artifact
```

Every published recall denominator divides by a gold set derived from **one** run of the
agentic baseline. P15 priced a re-run at the *pick* level (~59% disagreement) and left the
question the denominators actually depend on: picks are not targets, and the judge might
absorb the churn. Three more draws over the 25 benchmark cases, at unchanged flags, every
pick judged, `use_cache=False` throughout — the 34 stored answers verified byte-identical
afterwards.

#### The pre-registered prediction failed

> *Target-level reproducibility should EXCEED the pick-level 0.41, because the judge filter
> is a stable function applied to a noisy input.*

| | draw 1 | draw 2 | draw 3 | mean |
|---|---|---|---|---|
| frozen targets reproduced | 15/40 = 0.38 | 17/43 = 0.40 | 18/45 = 0.40 | **0.39** |

**0.39 against the pick-level 0.41 — the judge absorbs nothing.** It is a filter on *which*
papers, not a stabiliser of *whether*: a fresh draw finds different papers, and the judge
certifies those instead. The pre-registered rule ( < 2/3 ) fires: **the membership of the
denominator is not a stable quantity.**

#### But the size is stable, and — the part that matters — so is the estimate

Two things the reproducibility figure does *not* say. First, the denominator's **magnitude**
barely moves: 38, 39, 46 targets against the frozen 40, 43, 45 on the same cases. Second, and
decisively:

| target set | in the shipped pool | p | 95% CI |
|---|---|---|---|
| frozen (the published set) | 8/56 | 0.143 | [0.074, 0.257] |
| draw 1 | 6/39 | 0.154 | [0.072, 0.297] |
| draw 2 | 8/39 | 0.205 | [0.108, 0.355] |
| draw 3 | 12/46 | 0.261 | [0.156, 0.403] |

**Four independent gold sets, four overlapping estimates of the same quantity** (pooled
34/180 = 0.19, CI [0.14, 0.25]). This is what a *sample* behaves like: membership churn is
**variance, not bias**. A reach fraction measured against any exchangeable draw is an
unbiased estimate of the population reach rate, which is why the published figures survive —
as **estimates with intervals**, not as exact fractions:

| published as | read as |
|---|---|
| hop 21/56 | **0.38**, 95% CI [0.26, 0.51] |
| HyDE 34/56 | **0.61**, 95% CI [0.48, 0.72] |
| union 43/56 | **0.77**, 95% CI [0.64, 0.86] |

The intervals are ±0.11 or so — wide enough that "36/48 → 43/56" style comparisons between
adjacent figures were never resolving anything, and narrow enough that the qualitative claims
(keyword channels 0–8%, hop+HyDE ~75%) stand comfortably. One honest wrinkle: the four reach
estimates trend upward (0.143 → 0.261) and n does not resolve whether that is drift or noise.

#### The population is far larger than anyone assumed

Union of targets over the 11 cases present in every draw:

| | frozen | +draw 1 | +draw 2 | +draw 3 |
|---|---|---|---|---|
| distinct targets | 33 | 50 | 63 | **81** |

**No saturation whatsoever** — each draw adds 13–18 new targets and the curve is still
climbing. Chao1 over the four occasions: S_obs = 81, f1 = **57**, f2 = 9 → **≥ 262**. So on
eleven repositories the frozen set holds 33 of a cli-findable actionable population of at
least ~262. (With 70% singletons Chao1 is a loose and unstable lower bound — which cuts the
same way: the true figure is larger, not smaller.) P16 measured 4 sources as nearly disjoint;
this measures a *single* source as nowhere near exhausted by itself.

#### The failure rate is benchmark-wide, and it selects what the gold set contains

| draw | ok | partial | failed | rate |
|---|---|---|---|---|
| 1 | 19 | 0 | 6 | 24% |
| 2 | 20 | 0 | 5 | 20% |
| 3 | 21 | 1 | 3 | 12% |

**~19% of runs hit `error_max_turns`** at the shipped 12-turn budget, after three internal
retries — not a scientific-cohort phenomenon (C-28) but a property of the comparator
everywhere. And the pattern is not uniform: `thin-lang` and `vectordb` failed in **all three**
draws, `thin-kv` in two.

**Every one of those cases has a cached success in the gold set.** That is the selection
effect named in PLANS item 2, now measured: repositories whose agent habitually runs over
budget are represented in the denominator *only by their lucky draws*. `partial` rows
(unresolved or unjudged picks) are recorded but excluded from every count above, because a
floor read as a count would have manufactured instability out of an arXiv throttle.

#### What this settles

The union proposal that started this line of work is now measurable rather than intuitive: a
union **does** stabilise membership, and it does so by growing toward a population of ≥262 on
11 cases — so it converges to "everything this one searcher ever finds", not to ground truth.
The better answer is the one the numbers point at: **stop treating the gold set as a set and
treat it as a sample.** Report reach as a probability with an interval (P16's design, now
validated by four independent draws agreeing), and use additional draws to tighten intervals
rather than to chase a denominator that does not converge.

**Cost** 74 agentic runs (subscription) plus judging; the shared baseline caches were never
read or written, and the judge cache absorbed every repeated pick.

### The witness set v2: coverage was never recall, and the honest measures say something worse. **[P16]**

```bash
uv run python evals/witness_set.py --report   # $0, re-print from the committed artifact
uv run python evals/witness_set.py --check    # $0, re-derive and diff
```

The gold set is one draw of one searcher's discovery distribution, filtered by the judge
(P15 measured the draw noise at ~59% of picks). Its coverage was published as "recall", and
it is not: the judge marks ~2% of random arXiv papers actionable, so the true positive set is
order 10³–10⁴ per repository — unmeasurable, and not the quantity anyone wants. What the set
provides is per-member **certificates** (this paper is findable and judged actionable), i.e.
pooled relevance judgments with a pool of one system and one draw.

`evals/witness_set.json` is the pooled version, built entirely from already-judged material:

| source | witnesses | what it is |
|---|---|---|
| `cli` | 75 | the gold set, exactly (a test pins v2 to v1 — the first draft forked it, C-12 shape) |
| `api` | 50 | a different searcher entirely (P13) |
| `reporadar` | 189 | the headline run's own judged-actionable returns — in the set, excluded from grading (LOSO) |
| `adoption` | 19 | git-history-mined, the only model-free source (judged against the repo at t0) |
| **total** | **319** / 31 cases | overlap histogram: **306 singletons, 12 in two sources, 1 in three** |

Four discovery distributions, nearly disjoint — the single strongest fact in the artifact.
Each source is blind to most of what the others certify.

#### Reach, restated as a probability — and it is low

Reach = P(witness ∈ the shipped configuration's frozen candidate pool), per source, Wilson
CI, RepoRadar-sourced witnesses excluded from every denominator:

| into | `cli` | `api` | `adoption` | pooled non-self |
|---|---|---|---|---|
| `pool-wemb` (headline, 25 cases) | 8/56 = **0.14** [0.07, 0.26] | 7/28 = 0.25 | **1/19 = 0.05** | 13/93 = **0.14** [0.08, 0.23] |
| `pool-cohort3` (37 cases) | 15/75 = 0.20 | 15/50 = 0.30 | 1/19 = 0.05 | 29/133 = **0.22** [0.16, 0.30] |

**This does not contradict the published 43/56 = 77% — it completes it.** That figure measured
the hop and HyDE channels at top-1000 depth in isolation; witnesses sit at median rank 837 in
the dense index, and the shipped collection cuts far shallower. Both are true: the channels
*can* reach three-quarters of the witnesses at depth, and the pool the system actually ranks
contains one-seventh of them. And the headline +5.72 coexists with 14% witness reach because
the pool is dense in *other* actionable papers (58% in the HyDE top-100 stratum) — our own
data demonstrating that witness coverage and recall are different quantities.

**The adoption row is the sharpest.** 1 of 19 papers that repositories *demonstrably went on
to adopt* is in the shipped pool. NR-1 said repos do not cite what would improve them; this
says our collection cannot fetch what they later adopt either. The model-free source, the one
with no judge in its loop, is the one the system is blindest to.

#### At the digest: regret, not coverage

A missed witness costs only what it would displace: +1 filling an empty slot, +3 displacing a
shown paper judged < 2. Bounded by the window, so a growing witness set can only reveal
headroom, never inflate it. Against the headline run: **mean regret +3.48 net@2/case on top
of +5.72** — an oracle showing the best *known* 15 per case sits near +9.2. Concentrated:
`vectordb` +14, `speech` +9, `ann` +8, `llminfer` +7. Caveat that keeps it honest: at 14%
reach, most of that regret is a **discovery** deficit, not a selection one — the witnesses are
not in the pool for any selector to find.

#### How incomplete is the witness set itself?

The three cli draws (P15) as capture occasions, pick-level: S_obs = 24, f1 = 12, f2 = 7,
**Chao1 ≥ 34** — the cli-findable population on those six cases is at least ~40% larger than
three draws have shown, and the singleton fraction (half) says the set is nowhere near
saturation. The k = 3 spread runs planned in PLANS item 2 extend this curve for free.

**Semantics, fixed on purpose:** growing the set tightens the intervals instead of degrading
any number; a new source scoring lower *is* the pooling-bias measurement; and nothing here is
a recall claim. Pinned by `tests/test_witness_set.py` (artifact-internal arithmetic on any
machine; live re-derivation and an id-normalisation invariant where the caches exist),
mutation-verified in both directions.

**Cost** $0 — every member already had a verdict.

### The turn budget is not the problem. The comparator is barely reproducible. **[P15, C-27]**

```bash
uv run python evals/turn_budget_probe.py --report   # $0, re-read the stored arms
```

P14 left four `bio-*`/`mat-*` cases unmeasured against `error_max_turns` at `--max-turns 12`,
and the obvious fix — raise the cap — re-runs all 37 cases and re-derives the gold set. So the
question was whether raising it *changes anything on the cases that already succeed*: if not,
the re-run is a restatement; if so, a rebuild.

Six cases, three per cohort, chosen and written down before the first call. Three arms each:
the stored answer (**A**), a **fresh 12-turn control** (**B**), and a **30-turn treatment**
(**C**), B and C run back to back under one auth. Every arm ran with `use_cache=False`, so the
33 stored answers were neither read nor overwritten.

**The control arm is the whole design.** Comparing a 30-turn run against the *cache* cannot
separate "the cap mattered" from "it is a different draw". Only B–vs–C isolates the turn
change, and only A–vs–B says what a re-run costs by itself.

Case ids are benchmark labels, not repository names; both are given because the prose below
refers to the software, and a reader should not have to hold the mapping in their head.

| case | repository | cohort | cached | control (12t) | treat (30t) | J(A,B) | J(B,C) |
|---|---|---|---|---|---|---|---|
| `rag` | ColBERT | bench25 | 3 | **5** | 3 | 0.60 | 0.60 |
| `linter` | ruff | bench25 | 3 | **5** | 3 | 0.33 | 0.60 |
| `http` | requests | bench25 | 0 | 0 | 0 | n/a | n/a |
| `mat-descriptors` | dscribe | scisoft | 4 | 4 | **2** | 0.60 | 0.20 |
| `bio-align` | minimap2 | scisoft | 2 | **1** | **3** | 0.50 | 0.00 |
| `bio-singlecell` | **scanpy** | scisoft | **0** | **2** | 1 | 0.00 | 0.00 |

#### The turn question: no, and also unanswerable at this n

**The cap bound on nothing.** All six controls returned `ok` at 12 turns — and reaching
`--max-turns` is loud, not silent (it fails with `error_max_turns`, which is exactly how P14's
four cases present). So on these six, 12 turns was never reached.

**The turn effect is inside the noise.** Paired per case, `J(B,C) − J(A,B)` is **−0.13**,
bootstrap CI **[−0.38, +0.11]**, n = 5. Individually: `+0.00, +0.27, −0.40, −0.50, +0.00`. The
probe cannot separate a real turn effect from a different draw, and no amount of reading the
table harder will change that. **Verdict: inconclusive**, which at n = 6 against the noise
below is the outcome that should have been expected.

#### What the control arm actually found, which is worse

**A re-run of the identical configuration disagrees with the stored answer on ~59% of its
picks.** Mean J(cached, control) = **0.41** across five cases. Not a different prompt, not a
different model, not a different turn budget — the same configuration, run again.

The per-case detail is sharper than the mean:

* `rag` and `linter` each returned **5** picks where the cache holds 3.
* `bio-align` returned **1** where the cache holds 2; the 30-turn arm returned **3**.
* **`bio-singlecell` (scanpy) abstained in the cache and returned 2 picks on re-run.** Its
  stored answer is an explicit `[]` with prose explaining what scanpy already implements —
  the `webdev` shape, quoted in P14 as a *real* abstention. It is not a stable one.

**This is the finding that matters, and it is not about turns.** The gold set — 56 targets on
the benchmark25 cohort, the denominator under every published recall figure (21/56, 34/56,
43/56) — is derived from **one draw** of a process that reproduces about two picks in five.
Re-running the baseline at *any* setting moves it. The 2026-08-09 incident was read as "a
flag change invalidated the caches"; the truth underneath is that re-running at all would
have done nearly as much, and the flag change only forced the issue.

That reframes the 30-turn decision. There is no configuration under which the gold set is
stable and a turn change disturbs it. There is a gold set that is a **sample**, and it has
been treated as ground truth.

#### The auth question, answered as far as this can answer it

The benchmark25 caches were written under a signed-in CLI, the scientific ones under
`ANTHROPIC_API_KEY` (P14, and the question the auth change left open). If the auth path
changed what the agent recommends, A-vs-B similarity should be systematically worse for the
cohort whose auth changed.

| cohort | cached auth | mean J(cached, control) | n |
|---|---|---|---|
| benchmark25 | signed-in CLI | 0.47 | 2 |
| scisoft | `ANTHROPIC_API_KEY` | 0.37 | 3 |

Comparable, and both swamped by the nondeterminism above. **No evidence the auth path matters
— and this design could not have detected a modest effect if it did.** Reported as "not shown
to differ", not as "shown not to differ".

#### C-27 — the primary measure was measuring a different quantity

The pre-registration made **`num_turns` the primary measure**: a 30-turn arm that never
exceeds 12 turns would prove the cap was slack. The run disproved the measure, not the
hypothesis — every *control* arm, capped at 12, reported `num_turns` of **16, 17 or 9** and
still returned `ok`.

`--max-turns` is enforced, and `num_turns` is simply a different counter. Measured directly
against the CLI, outside our harness:

| invocation | outcome | `num_turns` |
|---|---|---|
| `--max-turns 2`, tool-using prompt | `error_max_turns` | 3 |
| `--max-turns 12`, the real baseline prompt in ColBERT | **success** | **15** |
| `--max-turns 30`, a two-fetch prompt | success | 6 |

So a run capped at 12 can succeed while reporting 15, which settles it: `num_turns` counts
something larger than whatever `--max-turns` bounds — plausibly tool-result steps as well as
model turns, though the payload does not define either field and we are not going to guess.
**The operational rule is the one that needs no theory: read `subtype`/`status` to learn
whether the cap was hit, never `num_turns`.** Read as pre-registered, the rule would have
printed *"6 of 6 cases exceeded the cap — ACTIVE CONSTRAINT"* and declared a rebuild.

The valid measure was in the same payload the whole time: **`status`**. Hitting the cap fails
loudly, so `ok` at 12 turns *is* the proof that the cap was not reached. The probe now reads
that, prints `num_turns` as an observation, and never thresholds it. Note the direction of the
correction — it makes the answer *less* decisive rather than more convenient, which is the
only reason a rule rewritten after seeing data deserves any credence at all.


#### The four "turn-limit" cases: two of them were never turn-limited **[C-28]**

The probe above covered only cases that already succeed, which measures what raising the cap
would *cost* and says nothing about what it would *buy*. Running the same three arms over the
four cases P14 could not measure — `bio-scvi`/scvi-tools, `mat-mlip`/MACE,
`mat-toolkit`/pymatgen, `mat-phonon`/phonopy — answers the other half, and contradicts P14:

| case | repository | fresh 12-turn control | 30-turn treatment |
|---|---|---|---|
| `mat-mlip` | MACE | **ok, 2 picks** | ok, 3 picks |
| `mat-phonon` | phonopy | **ok, 3 picks** | ok, 2 picks |
| `bio-scvi` | scvi-tools | `error_max_turns` | **ok, 2 picks** |
| `mat-toolkit` | pymatgen | `error_max_turns` | **ok, 0 picks** (abstains) |

**Two of the four succeed at twelve turns on a fresh draw**, under the identical
configuration that failed in P14. So P14's claim — *"a third of the scientific cohort
exhausts a turn budget that only 2 of the original 25 ever hit"*, offered as evidence that
the budget "does not transfer to large scientific codebases" — is **wrong in its mechanism**.
It is the same nondeterminism the control arm measured above, surfacing at the cap instead of
in the pick list. Half of that "domain effect" is weather.

What survives: `bio-scvi` and `mat-toolkit` did reproduce their failure at 12 and did
complete at 30. So the cap is real for *some* cases on *some* draws — a rate, not a property,
and the rate is not measured here.

**Two cheap consequences.** `mat-mlip` and `mat-phonon` can have `cli` baselines and gold
targets for the price of a re-run at the *unchanged* flags, taking the scientific cohort from
8/12 to 10/12 without touching the discriminator. And P14's +2.12 caveat — "excludes the four
largest repositories" — should be read as excluding four repositories that a re-run would
partly have included, not four the comparator cannot handle.

#### Three defects in this probe, found by running it **[C-29]**

The instrument needed three repairs, and one of them destroyed data:

**The `--out` flag reached the read path and not the write path.** A four-case rescue run
therefore **overwrote the six-case artifact** it was meant to complement, because the patch
adding `--out` replaced `OUT` in `--report` and left `OUT.write_text` alone — and the two
`str.replace` calls that should have caught it carried no assertion. This is *precisely*
lesson 4 of the methodology section — "partial runs overwriting whole-set artifacts", three
scripts before merge-by-key became the standard write pattern — reproduced in a brand-new
script in the same session that quotes it. Recovered from git (the artifact had been
committed), and repaired at the root: `merge_into` now merges fresh cases into the stored set
by case name, so the write is not destructive at all.

**The kill condition fired on the run it was pointed at.** It read "the 12-turn control
failed", which on the four cases with no successful cache is *the expected result
reproducing*, not an instrument fault. Now scoped to cases the cache records as succeeding.

**Void scored as null, in the noise statistic itself.** For a case with no cache,
`J(cached, control)` computes to 0.0 — an empty stored set against a non-empty control —
which reads as total disagreement when it is an absent measurement. Merging the four
cacheless cases in therefore dragged "what re-running alone costs" from **0.41 to 0.29**, and
the verdict line from *inconclusive* to *rebuild*, entirely on rows that had nothing to
compare. The A-vs-B statistics are now computed only over cases with a successful cache. No
published figure carried the wrong value; it existed for about ten minutes, in this session,
in a statistic written to price exactly this class of error.


#### Cashing in C-28: one of the two rescued cases did not stay rescued **[C-30]**

C-28 showed `mat-mlip` and `mat-phonon` succeeding at 12 turns where P14 had recorded
`error_max_turns`, so both looked collectable at unchanged flags. Run for real:

| case | repository | outcome | picks | gold targets |
|---|---|---|---|---|
| `mat-mlip` | MACE | **`error_max_turns` again** | — | — |
| `mat-phonon` | phonopy | ok, `num_turns` 25 | 3 | **+2** (both judged 2) |

**`mat-mlip` has now failed at 12 turns twice and succeeded once** — P14 fail, P15 probe
success, this run fail. One success in three draws. That is not a rescue, it is a coin, and
"we already know they succeed" was the wrong reading of a single draw; C-28's finding is that
the *failures* are draws, and the successes are draws in exactly the same way. The scientific
cohort goes 8/12 → **9/12**, not 10/12.

**The gold set grows 73 → 75, and `benchmark25` stays at 56** — the published denominator is
untouched, as it must be.

#### A cached case with unjudged picks, which looks identical to a rejected one **[C-30]**

`mat-phonon` also exposed a defect worth more than the case. Its baseline **succeeded** and
`run_baseline` cached it; then `resolve_references` hit **arXiv HTTP 429** on one of the three
picks, and `fill_cli_baseline` returned `arxiv_unverified` and judged *none* of them. The
driver keyed "done" on the existence of a cache file, so from that moment the case was:

* **finished**, as far as `fill_cli_baseline` was concerned — it would be skipped forever;
* **contributing 0 gold targets**, indistinguishable from a case whose picks the judge scored
  below the bar.

Void read as null, in the gold set. A survey of all 34 cached cases found it confined to
`mat-phonon` — created by this run — but only because nothing had thrown between the cache
write and the judge loop before.

Three repairs, all at the root rather than at the symptom:

* **`incomplete_cases` replaces `missing_cases`**, splitting candidates into *needs a
  baseline run* and *needs only judging*. The split is not cosmetic: a case with a cache must
  **never** be re-run, because that redefines ground truth, so it gets a `judge_only` path
  that does not touch `run_baseline` at all.
* **A lookup failure no longer abandons the whole case.** It judges everything that did
  resolve, returns `partial`, and says so — the C-15 rule (never score an unverified paper)
  was right, but bailing on the verified ones with it was not.
* **`gold_targets.json` records unjudged picks**, so the artifact cannot assert a
  completeness it does not have. `tests/test_gold_targets.py` pins that the field matches its
  count and that no case is both `incomplete` and `ids-only` — those labels mean opposite
  things about whether a pick was judged.

The remaining pick, `1703.03212`, is still unresolved: arXiv answers **429 for a known-good
control id too**, so this is a throttle rather than a hallucination, and the harness declining
to score it is C-4 working as intended. `mat-phonon` is frozen as **incomplete** until the
rate limit clears.

**Cost** 20 agentic runs across both arms (6 cases + 4 rescue cases x 2 arms),
subscription-billed. Nothing was written to the shared caches -- every arm ran with
`use_cache=False`, and the 33 `cli` answers were backed up to
`evals/.work/cache-backup-20260826-baseline-cli/` before the first call and verified
byte-identical after the last.

### The scientific cohort finally has the published comparator — and it is the strongest baseline we have measured. **[P14]**

```bash
uv run python evals/fill_cli_baseline.py --dry-run   # $0, what is still missing and why
uv run python evals/fill_cli_baseline.py --compare   # $0, cli vs api wherever both exist
```

P13 left the twelve `bio-*`/`mat-*` cases scored against `api`, a comparator it had just
shown to be a different system. Eight of them now have `cli` runs, at the **pinned
discriminator `da766b38114e`**, so nothing about the other 25 moved. `--compare` reproduces
P13's hand-computed 25-case figures to the digit (64/63/56, 0.889, +1.68 against 34/34/28,
0.824, +0.64), which is the check that licenses reading the new block:

| cohort | mode | picks | actionable | precision | net@2/case |
|---|---|---|---|---|---|
| benchmark25 (25) | `cli` | 64 | 56 | 0.889 | +1.68 |
| benchmark25 (25) | `api` | 34 | 28 | 0.824 | +0.64 |
| **scisoft (8)** | **`cli`** | **17** | **17** | **1.000** | **+2.12** |
| scisoft (8) | `api` | 18 | 16 | 0.889 | +1.50 |

**Three things follow, and none of them are comfortable.**

**The comparator on scientific software is harder than the one in the paper.** 17 picks, 17
actionable — the agentic baseline does not miss once on this cohort, and scores +2.12/case
against the +1.68 (+1.84 corrected, C-25) it manages on the benchmark the headline is built
from. Whatever we eventually claim here has a higher bar than the published one, and the
cohort's earlier `api`-scored appearance understated that bar by 0.62/case.

**The two modes share nothing at all here.** Zero overlapping picks across eight cases, and
seven cases where both returned papers and neither agreed on a single one. On the original 25
the modes at least shared 10 of 64. Whatever "the agentic baseline" denotes, it denotes
something even less stable off the ML/systems distribution.

**One case is a real abstention, not a gap.** `bio-singlecell` (scanpy) emitted an explicit
empty JSON array, listing in prose the methods scanpy has *already* implemented
(`flavor="pearson_residuals"`, HNSW, Leiden). That is the `webdev` shape, and C-25's
block-keyed fallback correctly leaves it at zero rather than resurrecting anything.

#### Four cases could not be measured at all, and raising the limit is not free

`bio-scvi`, `mat-mlip`, `mat-toolkit` and `mat-phonon` all returned
`subtype: error_max_turns` at `num_turns: 13`, against the `--max-turns 12` in `CLAUDE_FLAGS`
— after retries, so not transient. **A third of the scientific cohort exhausts a turn budget
that only 2 of the original 25 ever hit.** The budget was calibrated on ML and systems
repositories and does not transfer to large scientific codebases.

> **Corrected 2026-08-26 [C-28].** The mechanism above is wrong. Re-run at the *identical*
> flags, `mat-mlip` and `mat-phonon` **succeed at 12 turns** — the failures are draws, not
> properties of these repositories, and "does not transfer to large scientific codebases"
> attributes to domain what belongs to nondeterminism. `bio-scvi` and `mat-toolkit` did
> reproduce their failure and complete at 30. See **[P15, C-28]** above; "after retries, so
> not transient" was the claim to distrust, and retries within one session are not draws.

The obvious fix is the one that must not be applied casually: `_discriminator` hashes the
flags, so raising the limit **re-runs all 37 cases and redefines the gold set** — precisely
the 2026-08-09 incident, which moved `graph` from 3 targets to 4 and would have shifted every
published recall denominator. Running the four at 30 turns *without* re-running the rest is
worse still: `actionable_baseline_ids` reads `status`, not `_disc`, so the gold set would
silently mix two comparator configurations, which is the defect P13 exists to name.

So the four stay unmeasured, and the honest reading of the +2.12 above is that **it excludes
the four largest and most complex repositories in the cohort** (scvi-tools, MACE, pymatgen,
phonopy) — the ones the agent could not finish. The direction of that bias is unknown; the
fact of it is not.

#### The gold set is now two cohorts, and only one of them is published

`evals/gold_targets.json` grows **56 → 73 targets across 20 → 27 cases**, purely additively:
every case reports `lost=[]`. Because published recall figures (21/56, 34/56, 43/56) divide by
the 25-case set alone, the artifact now carries a per-cohort split and
`tests/test_gold_targets.py` pins **`benchmark25` at exactly 56**. Anyone computing recall
over all 73 and comparing it to 43/56 would be committing C-17 with our own data.

| cohort | targets | cases |
|---|---|---|
| `benchmark25` | 56 | 20 |
| `scisoft` | 17 | 7 |

**Cost** $7.04 recorded across 8 successful runs. True spend is higher and the instrument
cannot say by how much: `run_baseline` reports `cost_usd` only on success, so the four
turn-limit failures and one crashed attempt (C-26) were billed and never counted.

### A decode that does not raise where it fails: twelve subprocess captures, one of them load-bearing for ground truth. **[C-26]**

```bash
uv run pytest tests/test_subprocess_decoding.py    # $0, surveys every subprocess call we own
```

Filling the missing `cli` baselines (below) died on its first case, after that case had
already been billed for its answer. The traceback pointed at `json.loads`:

```
TypeError: the JSON object must be str, bytes or bytearray, not NoneType
```

**`claude --output-format json` emits UTF-8. `subprocess.run(text=True)` decodes with the
LOCALE codec** — cp1252 on this project's Windows box — and byte `0x81`, at position 1556 of
scanpy's answer, is undefined there. What makes this hard to read is what CPython does next:
the reader thread raises `UnicodeDecodeError`, and `subprocess.run` **prints that traceback
and swallows it**, returning a completed process whose `stdout` is `None`. So the decode
fails in one place and the program dies in another, four frames away, with an error message
about JSON. `tests/test_subprocess_decoding.py` demonstrates the swallow rather than
asserting it, because it is the part nobody believes.

**The survey found seven more, and one of them already carried the fix.** Reading every
`subprocess.run`/`check_output`/`Popen` in `src/` and `evals/` — not the one that failed —
turned up twelve captures that decode to text, seven with no `errors` handler:

| site | what it captures | why it matters |
|---|---|---|
| `evals/mine_adoptions.py` ×5 | `git grep`, `git ls-tree`, `git clone` | the **31 mined adoptions** — the project's only model-free ground truth |
| `evals/fetch_wants.py` | `gh api` issue titles | the arm whose whole point is keeping titles **verbatim** (P8) |
| `evals/prose_window_probe.py` | `git config remote.origin.url` | `.stdout.strip()` on `None` |

`mine_adoptions.py` is the instructive one: **line 157 already had `errors="replace"`**, added
when someone hit this on a blob read — and the five identical `git` calls beside it, in the
same file, did not. That is the C-9a/C-14b shape exactly: repaired where it was observed,
left everywhere it was not. `git grep`'s output is where the arXiv ids are extracted, so one
repository with a non-UTF-8 byte would have taken that repo's adoptions to zero.

**The handler is not one-size-fits-all, and `replace` would have been a data-loss bug.**
`scheduler._get_current_crontab` reads the user's **entire crontab**, filters our own lines
out, and writes the rest back. Under `replace`, one undecodable byte in an entry belonging to
somebody else comes back as U+FFFD and is then written over their crontab — strictly worse
than the crash it would be fixing. That pair uses **`surrogateescape`** at both ends, which
round-trips bytes exactly; a test asserts the two ends still agree, because a mismatch there
is a silent one-way corruption. Three handlers, by what the caller does with the bytes:

| bytes are… | handler | sites |
|---|---|---|
| a UTF-8 contract we can name | `encoding="utf-8", errors="replace"` | `claude`, `gh`, `git` |
| read, edited and **written back** | `errors="surrogateescape"` | the crontab pair |
| an arbitrary user command's output | locale codec, `errors="replace"` | `notify.run_shell_hook` |

Two second lines of defence, since the next unhandled thing will not be this one:
`_parse_cli_payload` now treats `None`/empty stdout as a **status** rather than an exception
(every other failure in that module already was one), and `fill_cli_baseline` catches per
case, so one bad repository cannot abort a paid batch on its first entry.

**No published number moves** — the 25 `cli` caches parsed fine, which is why this survived
six weeks. What it cost was one billed answer and a stalled batch.

### The comparator was understating itself, and the two baseline modes are different systems. **[P13, C-25]**

```bash
uv run python evals/freeze_gold_targets.py --check   # $0, pins every recall denominator
```

Preparation for letting the baseline recommend non-arXiv papers turned up two things about
the comparator that have nothing to do with arXiv, both measured at $0 from cached runs.

#### C-25 — three cases scored as abstentions while contributing seven gold targets

`run_baseline` re-parses the cached `raw` on every cache hit, deliberately, so that a parser
fix reaches already-cached runs. Three `cli` caches cannot survive that: `compiler`, `graph`
and `storage` hold a **128-character restoration note** where their transcript used to be,
after the 2026-08-09 30-turn re-run displaced the 12-turn entry and only their ids could be
recovered from a run record. Replaying a note parses to nothing.

So in **every headline run since**, those three reported `n_returned = 0, abstained = True`
— while `diagnose_pool.actionable_baseline_ids`, reading the `ids` field of the *same file*,
counted their seven targets. One cache, two consumers, opposite answers.

| | |
|---|---|
| published baseline | **+1.56 net@2/case** (sum +39 over 25) |
| forfeited on those three cases | 7 picks, **all judged ≥ 2** → **+7 = +0.28/case** |
| corrected baseline | **+1.84/case**, 58 shown / 54 actionable, precision 0.931 |
| published paired margin +4.16 | becomes **+3.88**, CI [+2.24, +5.60], 17 w / 2 l / 6 t |
| published sign *p* = 0.0004 | becomes ***p* = 0.0007** (`compiler` moves from a win to a tie) |

**The comparator was understated and RepoRadar's margin overstated, by 0.28/case — 7% of
the margin.** Significance survives (CI floor +2.24) but the level was wrong and is restated
here; `evals/restate_c25.py` re-derives both columns from the cited run file and
`tests/test_restate_c25.py` pins them. The fix is a narrow fallback: when a cached `raw`
contains **no recommendation block at all**, fall back to the stored ids.

**The blast radius is exact, because the damage has a date.** Four paired runs postdate
2026-08-09 and read the damaged caches; `restate_c25.py` finds and restates all four. (Runs
before that date are left alone: `storage` genuinely abstained on 08-07 and `graph` on
07-12, and "correcting" those would manufacture picks the run never had.)

| draw | paired, as measured | corrected | sign *p* | corrected |
|---|---|---|---|---|
| 08-10 (23 of 25 paired) | +2.26 | +1.96 | 0.0414 | **0.1153** |
| 08-11 (24 of 25) | +2.50 | +2.21 | 0.0001 | 0.0007 |
| 08-14 (24 of 25) | +3.79 | +3.50 | 0.0001 | 0.0007 |
| **08-15, the headline (25)** | **+4.16** | **+3.88** | 0.0004 | 0.0007 |

The comparator gains +0.28 to +0.30 per case in every one — a constant defect, not a
draw-dependent one. **The 08-10 draw loses significance outright** (p = 0.0414 → 0.1153): one
of four independent draws of the shipped system no longer separates from the baseline. It is
the draw §8.7 already flags as the weak one, so this sharpens an existing caveat rather than
raising a new one, but it belongs beside the headline. Two smaller consequences: the
baseline's precision rises with its picks (0.938 → 0.945 on the 08-14 draw), so the "four
times as many papers at lower precision" caveat widens from five points to nearly six; and
the three draws with a failed `thin-lang` baseline pair over 23–24 cases, not 25, which the
artifact now records per run instead of leaving to be rediscovered.

**The run artifacts never recorded which baseline mode produced them.** Identifying the four
above as `cli` required matching each run's picks back against both caches (45/45, 48/48,
48/48, 51/51 for `cli`; the 08-20 run matches `api` 34/34). Given P13 immediately below —
`cli` and `api` are different systems — a results file that cannot say which comparator it
ran against is a gap of the same family as the defect this entry is about.

**One thing the restatement turned up on the way.** Reproducing the published CI needed the
cases fed in *sorted* order: `bigram_report.paired_bootstrap` draws indices from a seeded
RNG, so its interval depends on the ORDER the deltas arrive in and not only on their values.
File order gives [+2.44, **+5.96**] where sorted order gives the published [+2.44, +6.00] —
one grid step, decision-irrelevant, and still two people computing "the same" CI from the
same run file and disagreeing. No published number moves, so this is not a correction; the
restatement sorts, and says so.

**Narrow is the whole design.** The obvious rule — "fall back whenever the replay yields no
ids" — is wrong, and `webdev` is why. It says *"My recommendation is to recommend nothing"*,
emits an explicit ```` ```json [] ````, and **still carries four ids** an older parser
scraped from its prose, including `publication/2256929`, a bare URL path. An empty array is
an *answer*; the absence of a fenced block is not. Keying on the block (`_has_answer_block`)
keeps `webdev` at zero and recovers only the three damaged caches. Both halves are pinned in
`tests/test_eval_baseline_replay.py`, including `webdev`'s real artifact.

**The gold set does not move.** `evals/gold_targets.json` now freezes all 56 targets with
per-id provenance, and `tests/test_gold_targets.py` pins the live derivation against it
(mutation-checked: removing one id fails the suite). Nine targets are labelled `ids-only` —
not reproducible from any surviving `raw`:

| case | ids-only targets |
|---|---|
| `compiler` | 1601.05400, 2004.03082 |
| `graph` | 2111.14522, 2202.13013, 2303.06147 |
| `rag` | 2304.01982, 2505.11471 |
| `storage` | 2311.15380, 2408.05625 |

They are frozen **at 56 rather than the 51 the parser yields**, because dropping them would
move every published denominator for a reason unrelated to any research question. The
weakness is now inherited knowingly. And it matters that `evals/cache/` is gitignored: until
this artifact, the gold set existed only on one machine. The *reasoning* behind those nine
is gone for good — the run record they were restored from keeps ids and verdicts, not the
model's answer.

#### P13 — `cli` and `api` are not two runs of one baseline

Both modes have cached runs over the same 25 cases, so this is free. It is also decisive:

| | picks | judged | actionable | precision | net@2/case |
|---|---|---|---|---|---|
| `cli` (agentic, Claude Code + web tools) | 64 | 63 | 56 | **0.889** | **+1.68** |
| `api` (Messages API + server tools) | 34 | 34 | 28 | 0.824 | +0.64 |

**Only 10 picks are shared**, and the only three cases with identical pick sets are the
three where *both* returned nothing (`cli`, `encryption`, `http`). There is **no case where
both found papers and agreed**. `cli` finds 1.9× as many papers at higher precision and
scores 2.6× the net@2.

**Consequence for the 12 `bio-*`/`mat-*` cases**, which have `api` runs only: they are
currently measured against a comparator roughly 2.6× weaker than the other 25, and their
25 picks (22 actionable) cannot be compared across that boundary. Any cross-case claim
mixing them with the main benchmark is comparing two different systems. They need `cli`
runs before they can carry gold targets on the same footing — which is what the three-arm
validation below is for.

**Cost** $0 (both from cached artifacts).

### The off-arXiv corpus question: the literature is there, the snapshot is two years stale. **[P12]**

```bash
uv run python evals/openscholar_yield.py            # ~1 min, S2 batch, no LLM, $0
uv run python evals/openscholar_yield.py --report   # $0 thereafter
```

PLANS item 1 proposed building a second dense index over OpenScholar's peS2o v3 datastore
(arXiv:2411.14199) to reach the literature the arXiv-only index cannot hold. **Its stated
stage-1 check was refuted before it ran, and the replacement returned a different answer
than the one the item was arguing about.**

#### The original probe was mis-specified, and finding that cost $0

The check as written asked whether the gold targets the shipped channels never reach are in
the OpenScholar datastore. They are already in **ours**: all **56 of 56** gold targets are
present in the local 3.1M-vector arXiv index (verified against the shard id files; the
sync-time dependency check independently recorded 48/48). The unreached ones sit at ranks
1,358 / 2,586 / 6,699 / 9,131 / 9,165 / **223,245**. Those are **ranking** failures, and a
larger corpus cannot fix one — Hamming top-k only gains competitors. The probe would have
returned "6/6 present", read as a pass, and established nothing.

**And the gold set cannot ask the question the item was really about.** A gold target is a
baseline pick judged ≥ 2, and `evals/baseline.py` demands the baseline answer as
`{"arxiv_id": ...}` — so **no non-arXiv paper can ever become one**. That structural blind
spot was invisible because the benchmark *does* surface off-arXiv value: **159 non-arXiv
papers judged, 79 actionable**, and **11 cases have actionable non-arXiv papers and zero
gold targets** — every `bio-*` and `mat-*` case, 59 of the 79 papers. `net@2` counts them;
gold-target recall is blind to them.

One documentation correction falls out (a second, attempted here, was itself wrong — see
the correction block below). 17 of 37 cases have no gold targets at all: 12 (`bio-*`/`mat-*`) never had a
baseline run, 3 (`cli`, `http`, `encryption`) had the baseline correctly return nothing, and
2 (`webdev`, `linter`) had every baseline pick judged below 2.

#### The replacement probe, and the pre-registered result

Ground truth: the 79 already-judged actionable non-arXiv papers — free, needing no baseline,
carrying no arXiv restriction. Question: what share would peS2o v3 contain, under its stated
inclusion rule (open access, in S2ORC, published by the October 2024 cutoff)?

> **RESULT (2026-08-17) — 43/79 = 54.4%. MARGINAL by the pre-registered rule, and the
> mechanism is not the one predicted.**
>
> | exclusion | n | share |
> |---|---|---|
> | **included** | **43** | **54.4%** |
> | after the Oct-2024 cutoff | **30** | 38.0% |
> | unresolved by S2 | 5 | 6.3% |
> | no S2 handle (one IACR paper) | 1 | 1.3% |
> | **not open access** | **0** | **0.0%** |
>
> Predicted 50–70%, kill below 50%, pass at 70% — so the number landed in band. **The
> reasoning behind it was wrong.** The prediction was that closed-access chemistry and
> materials journals (`10.1021`, `10.1016`, `10.1063`) would be the main loss. **Not one
> paper was excluded for access.** Every single exclusion that peS2o's own rule can explain
> is a *date*: 30 papers published after the snapshot, of which **28 are from 2025–2026**
> (2025: 17, 2026: 11). By cohort the post-cutoff losses are bio 19, other 7, mat 4.
>
> **So the corpus is not too narrow. It is too old.** peS2o v3 holds the right kind of
> literature — the access test excludes nothing — but OpenScholar's artifact was last
> modified 2024-11-18 and is frozen. For a product whose output is *"what is new that could
> improve your repo"*, a corpus that ends in October 2024 is structurally disqualified, and
> the deficit **grows with time** rather than shrinking. 54.4% today is the best this
> artifact will ever score.
>
> **What a PASS would have cost, priced before the run.** ~378 GB of passages against the
> arXiv index's 432 MB (≈875×), and its precomputed vectors come from OpenScholar's own
> retriever rather than `mxbai-embed-large-v1` — so adoption means **re-embedding the
> corpus**, not syncing an index. PLANS' original check (b), "reproduce a stored vector
> bit-identically", does not apply: that check exists for a corpus shipping *our* encoder's
> vectors, and this one ships someone else's.
>
> **Verdict: do not build on this artifact.** Not because the idea is wrong — the
> off-arXiv value is real and measurable (79 actionable papers prove it) — but because a
> frozen snapshot cannot serve a freshness product. What the probe establishes for any
> future proposal is a **requirement the original item never stated**: a second dense
> corpus must be *re-syncable*, like `rr sync-index` is, or it decays into exactly this.
>
> **Unverified, and stated as such.** The plan was to anchor the S2-rule inference by
> looking up a sample of corpus ids as `raw_id` in the datastore itself through the HF
> datasets-server. The endpoint returned "dataset index is loading", then 502s, then
> "Unexpected error" across three attempts. **The 54.4% is therefore inference from
> peS2o's stated inclusion rule, not a lookup in the artifact** — the `raw_id` field is
> confirmed to be an S2 corpus id, but no membership was directly observed. The direction
> of the finding does not depend on it: the 38% post-cutoff exclusion is a property of
> dates, not of the lookup.
>
> **Cost** $0.
>
> #### CORRECTION — the actionable-rate comparison was a denominator swap **[C-23]**
>
> This entry first read the non-arXiv cohort as *"79 actionable (~50%, above the ~30% arXiv
> base rate)"*, and the ~30% came from the **602-paper labelled set** — a different
> population from the judge cache the 79 were counted in. Recomputed within the same cases:
> **arXiv 48.4% (1259/2602) against non-arXiv 46.2% (84/182)**. Non-arXiv papers are
> marginally *less* actionable, not more.
>
> What the entry actually needed was the **count**, not the rate: 79 actionable papers that
> gold-target recall cannot see, 59 of them in cases with no gold targets at all. That claim
> is untouched. The rate comparison was decoration, and it was wrong in the direction that
> flattered the argument — the §6.1/C-7 shape, arriving through a denominator rather than a
> draw.
>
> #### CORRECTION — "the union is 64%" counted *unmeasured* as *unreached* **[C-24]**
>
> This entry also corrected the published union figure from 36/48 = 75% down to
> "**36/56 = 64%**", on the grounds that 8 thin-case targets entered the benchmark after the
> HyDE replication of 2026-08-06 and were never measured. The premise was right and **the
> arithmetic was not**: an unmeasured target is *unknown*, and this treated all 8 as misses.
> The entry said so in the same sentence and then published the number anyway.
>
> Re-running `evals/hyde_replication.py` over all 56 targets (2026-08-17, ~$0.01 of Haiku for
> the three missing cases) measures them: **7 of the 8 are reached**, `thin-kv` at ranks 4, 5
> and 14. The honest figures are **HyDE 34/56 in top-1k, hop 21/56, union 43/56 = 77%**,
> with 22 targets reachable only by HyDE — slightly *better* than the 75% this tried to
> correct downward. Filling a measurement gap improved the number; assuming its contents did
> not.

### Typed anchors end to end: the channel is real and it does not reach the digest. **[P11]**

```bash
COMMON="--baseline none --sources arxiv --rr-pool 50 --rr-rerank --rr-all-time --rr-hybrid
        --rr-sweep --rr-finescale --rr-hyde --rr-window 15 --rr-w-embedding 1.5"
uv run python evals/run_judge_eval.py $COMMON                      # control
uv run python evals/run_judge_eval.py $COMMON --rr-typed-anchors   # treatment
```

P9 measured that typed README spans discriminate where the shipped manifest channel does
not (+27.5pt Mantel-Haenszel against −0.6pt), P10 measured that 0.87 of that survives
redacting the spans from the judge's view, and the free cross-judge check reproduced the
direction on Sonnet. **None of it touches the product metric.** Anchors reach keywords,
therefore queries, therefore the pool (`profiler.py:554`), the BM25 bag
(`retrieval.py:37`) and the gate prompt (`triage.py:56-59`), so this must be live: a
frozen pool cannot be shared across an arm that changes the queries on 14 of 25 cases.

**Prediction, written before the run.** Primary, all 25 cases: **−1.0 to +1.5 net@2/case,
most likely inside the floor.** The stage-1 evidence is the strongest this project has
assembled for a repo-side channel, and I still expect close to nothing, for three reasons
that have each already happened here:

* NR-33 and NR-35 each reshuffled the digest substantially and moved the outcome +0.00.
  Everything downstream of the profile — the gate and the rescore — decides quality.
* NR-36 is the closest precedent by construction: it also enriched anchors, also won its
  $0 stage-1 probe, and measured **−0.52**.
* P9's own join says the discrimination lives in repositories whose manifests already
  parse. On the nine that gain anchors, three of six usable cases show any positive gap.

**What would change my mind, and it is not the mean.** The mechanism is anchor coverage,
so the pre-registered secondary is the **9 rescued repositories** (`ann`, `columnar`,
`compiler`, `db`, `encryption`, `linter`, `storage`, `systems`, `vectordb`). If typed
anchors work at all they work there, and n = 9 against a 1.04 floor cannot establish it —
reported as suggestive, never as a result, with the prediction recorded first so a
favourable draw cannot be read as confirmation afterwards. I predict this cohort moves
**+0 to +2**, i.e. better than the whole, and I am not confident of the sign.

**Gate-free secondary:** actionable papers reaching the ranked top-15 before gating. It
moved 0.00 in NR-33, −0.08 in NR-35, −0.20 in NR-36.

**A divergence between what was measured and what will run, stated first.** P9 scored
spans after applying `STOPWORD_ANCHORS` and `MIN_ANCHOR_LEN`, to keep its comparison with
the manifest channel fair. The shipped `typed_anchors` applies neither: it merges the raw
verbatim spans. So the treatment carries *more* terms than P9 scored — `ann` 2→21 rather
than 0→17, `peft` 16→31 — and the generic ones P9 filtered (`cosine similarity`, `cuda`,
`binary vectors`) are present. If this arm is negative, that difference is the first thing
to check, and it is cheap to check because the probe already has the filtered variant.

**A known false positive that will be in the pool.** P9's largest single false-positive
driver was the 3-character span `age` (from `age-plugin-pq` in `encryption`), which
matches ordinary English and fired on 16 non-actionable abstracts. It is not filtered out
of the shipped path either. `encryption` is one of the nine rescued repositories.

**Estimated cost** $20–30, two live 25-case arms, plus ~$0.02 of Haiku extraction.

> #### RESULT (2026-08-17) — **−0.32/case, inside the floor.** The pre-registered secondary went the other way.
>
> `…-20260817T052517Z.json` (control) and `…-20260817T064457Z.json` (`--rr-typed-anchors`).
> 25/25 both arms, live, window 15.
>
> | arm | net@2 | mean precision | abstained |
> |---|---|---|---|
> | control | **+6.20** | 1.00 on 8 cases | 6 / 25 |
> | `--rr-typed-anchors` | **+5.88** | 0.96 on 6 cases | 4 / 25 |
>
> **Paired −0.32/case, 7 better / 8 worse / 10 tied, sign p = 1.0000 — inside the floor.**
> The prediction was "−1.0 to +1.5, most likely inside the floor", and that is where it
> landed. It is the fourth time a mechanism has won a cheap probe here and moved nothing
> end to end: NR-33 +0.00, NR-35 +0.00, NR-36 −0.52, this −0.32.
>
> **The pre-registered secondary came in against its prediction.** I wrote that the nine
> rescued repositories would move **+0 to +2**, better than the whole, because anchor
> coverage is the mechanism and those are the repositories that gain anchors.
>
> | cohort | control | typed | paired | |
> |---|---|---|---|---|
> | all 25 | +6.20 | +5.88 | **−0.32** | 7+/8−/10= |
> | **rescued (9, pre-registered)** | +6.33 | +5.33 | **−1.00** | 2+/4−/3= |
> | other 16 | +6.12 | +6.19 | +0.06 | 5+/4−/7= |
>
> At n = 9 against a 1.04 floor this establishes nothing, exactly as pre-registered. It is
> recorded because the prediction was written first: the cohort the mechanism exists for
> did worse than the cohort it does not, and the sign is against the hypothesis.
>
> **The losses are displacement, not silence.** The two largest were `ann` (+12 → +5) and
> `db` (+14 → +11), and neither abstained:
>
> | case | control | typed |
> |---|---|---|
> | `ann` | 12 returned, **12** actionable, precision 1.00 | 11 returned, **9** actionable, precision 0.82 |
> | `db` | 14 returned, **14** actionable, precision 1.00 | 14 returned, **13** actionable, precision 0.93 |
>
> Enriched anchors change the keywords, therefore the queries, therefore the pool — and on
> these two the pool got worse. `ann` is the sharpest single fact in the entry: it was P9's
> best rescued case (+36.4pt discrimination, 4 of 11 score-3 abstracts naming a span) and
> it lost the most here.
>
> **A tempting explanation that this data cannot support.** Eight of 25 cases scored
> precision 1.00 in the control arm — four of them in the rescued cohort — and splitting on
> that gives −1.38 at the ceiling against +0.18 below it. The story writes itself: a repo
> already returning 12 of 12 has no headroom, so any retrieval change is downside. **It is
> not admissible.** The split is post-hoc, it was not pre-registered, and conditioning on
> control performance then measuring change produces exactly this pattern through
> regression to the mean alone. Recorded as a lead for a future pre-registration, not as a
> finding. C-7 is the entry that exists because this project already read a favourable draw
> as a property once.
>
> **The filtered variant was proposed and then refuted before it was run.** The
> pre-registration flagged that the shipped path applies neither `STOPWORD_ANCHORS` nor
> `MIN_ANCHOR_LEN`, and named that as the first thing to check on a negative. Checking it
> cost $0 and killed it: the probe's filter removes **7 spans of 196** across 5 cases
> (`l2`, `pandas`×2, `matplotlib`, `cmake`, `numpy`×2). It drops one span from `ann`, so it
> cannot explain −7.0 there, and it does not touch `age` at all — at 3 characters that span
> clears the length floor and is not a packaging word.
>
> No stricter length rule is available either. A floor of 4 removes `age` but also `ppo`,
> `dqn`, `sac`, `td3`, `jax`, `ia3`, `avx`; a floor of 5 additionally takes `lora`, `bert`,
> `cuda`, `llvm`, `hnsw`, `blas`. And `age` is not an extraction error: it is the real name
> of the encryption tool, colliding with an English word, which no length rule separates
> from `her` — Hindsight Experience Replay. **A $25 arm testing a 3.5% change in the span
> set against a 1.04 floor was not bought**, and that refusal is the entry's one saved cost.
>
> **Verdict: keep the code, ship nothing.** `profiler.typed_anchors` stays False, beside
> `profiler.scan_source`, which NR-36 left in the same state for the same reason. What the
> four probes bought is not a feature but a corrected understanding: the anchor channel the
> product ships does not discriminate (C-21), the keyword comparator that made it look
> special was saturated (C-22), typed spans genuinely do discriminate and survive both a
> second judge and span redaction (P9, P10) — and none of that reaches a digest, because
> the gate and the rescore downstream already extract what the profile knows.


### Typed README spans as an anchor channel: the channel works, the ledger it was aimed at does not. **[P9]**

```bash
uv run python evals/nerdme_probe.py            # ~$0.02 once: 25 Haiku extractions
uv run python evals/nerdme_probe.py --report   # $0 thereafter, cached spans
```

NR-39 blocked Roadmap 16 on a specific pair of facts: the *anchor* channel is the only
repository-side channel that discriminates, and it does not exist for 12 of 25 cases
because `_extract_anchors` reads Python and JavaScript manifests only. NERdME
(arXiv:2603.05750) proposes the mechanism that gap wants — typed, verbatim-span entity
extraction over READMEs — and this probe asks whether it supplies anchors worth having,
before anything is built. 209 spans over 25 READMEs, five types, scored against the same
1,237 cached verdicts and frozen pools `relation_probe.py` uses.

**Pre-registered, written before the first run.** Q1 coverage 10–12 of the 12 zero-anchor
cases gain spans. Q2 discrimination (score-3 hit % − below-2 %) lands at +6 to +12pt, with
a **kill condition below +7.0pt**. Q3 grounded coverage 12.3% → 20–32%. Q4 concentration
7 → 14–20 cases, top-4 share below 70%. Q5 the alias-table bridge: `method` spans on ≥15
cases **and** a method gap ≥ the library gap, both required.

| | pre-registered | measured | |
|---|---|---|---|
| Q1 coverage | 10–12 rescued | **9** (`cli`, `thin-kv`, `thin-lang` stay empty) | MISS |
| Q2 discrimination | +6 to +12pt | **+27.5pt** M-H, CI [+21.5, +46.9], 12/1/3 | PASS, far over |
| Q3 grounded | 20–32% | **30.3%** actionable, 38.0% at score 3 | PASS |
| Q4 concentration | 14–20 cases, <70% | **15** cases, top-4 **66%** | PASS |
| Q5 alias bridge | ≥15 cases AND gap ≥ library | **10/25** cases; +11.5 vs +10.9pt | **FAIL** |

**0 of 209 spans were non-verbatim.** Nothing was hallucinated to filter, and the three
cases that stay empty are explicable rather than failures: `cli` is a negative control
where abstaining is correct, and `thin-kv`/`thin-lang` have ~108-character READMEs — the
§5.3 ceiling, "what documents contain, not how they are read", arriving in a third place.

**Q2 passed by four times its predicted band, and that is the part to distrust.** Two
post-hoc repairs, both prompted by an adversarial audit of the first run, and neither
carrying the weight of a pre-registered result:

| channel | pooled | macro | macro n≥5 | **M-H** | 95% CI (cases) | +/−/= |
|---|---|---|---|---|---|---|
| manifest (shipped) | +13.7 | −0.9 | −0.8 | **−0.6** | [−3.7, +1.4] | 2/3/11 |
| typed spans | +36.3 | +33.7 | +24.4 | **+27.5** | [+21.5, +46.9] | 12/1/3 |
| union | +33.5 | +27.8 | +20.3 | **+22.8** | [+16.7, +40.7] | 12/1/3 |
| keywords | +0.2 | −1.3 | +0.7 | **−0.9** | [−6.0, +2.4] | 4/2/10 |

16 of 25 cases have papers in both strata and can carry a gap at all.

**The join the first version never made, and the verdict that governs.** Q1 was computed
over the zero-anchor repos and Q2 over all of them, and the two were never intersected —
so the +36.3pt headline was generated by repositories whose manifests already parse, which
are the repositories the feature is *not* for.

| repo set | cases | pooled | macro | macro n≥5 | M-H |
|---|---|---|---|---|---|
| all | 16 | +36.3 | +33.7 | +24.4 | +27.5 |
| already had anchors | 10 | +60.8 | +40.6 | +36.4 | +38.7 |
| **rescued — the feature's job** | 6 | **+3.0** | +22.3 | +9.9 | **+11.5** |

The verdict on the target subset swings from **+3.0pt (fail)** to **+11.5pt (pass)** on the
choice of estimator, and the per-case dump is why it should be read as *neither*:

```
ann       +36.4pt  (score3 4/11)   storage   +19.0pt  (score3 4/21)
columnar  +84.2pt  (score3 1/1)    systems    +0.0pt  (score3 0/7)
vectordb   +0.0pt  (score3 0/16)   db         -5.8pt  (score3 1/19)
```

**Three of six usable rescued cases show any positive gap**, three of the nine rescued have
no papers in both strata to measure, `columnar`'s +84.2pt rests on one abstract, and two
cases (`systems`, `vectordb`) extracted spans that match no abstract in either stratum.
Two repositories — `ann` and `storage` — carry the whole positive result on the subset that
motivated the work. That is not a channel that works where it was needed; it is a channel
that works twice.

**Judge circularity is unbounded and the obvious bound points the wrong way.**
`assemble_repo_context` gives the judge README[:3500] plus the manifests, so both compared
channels sit inside the labeller's own input. The natural experiment — spans beyond char
3500 the judge never saw — gives +3.0pt against +34.7pt for spans inside the window, but
it cannot separate "unseen" from "peripheral", because a README's first 3500 characters are
its overview and the rest is install steps and API reference. **Every discrimination number
in this entry, and in NR-39, is an upper bound of unknown tightness.** The model-free
instrument that could settle it is P6's 31 git-history-mined adoptions; it has not been run
against this channel.

**Structurally, this probe cannot speak to the profiler use at all.** Anchors are not an
inert label: `profiler.py:554` makes them an extra TF-IDF document, so they reach keywords,
`build_queries` and the arXiv pool; `retrieval.py:37` puts them in the BM25 bag;
`triage.py:56-59` prints them into the gate prompt. Merging the spans changes the keyword
list on 17/25 cases, the arXiv query set on 14/25, the BM25 query on 21/25 and the gate
prompt on 20/25 — new fetches include `all:ppo`, `all:"late interaction"`, `all:pandas`.
A frozen pool therefore cannot serve that arm, exactly as `rr_scan_source` could not in
NR-36. P9 measures the *labelling* use only.

**Verdict: do not build, do not pay for Tier B.** The extraction is cheap, clean and real —
that half is judge-independent and stands. The claim it was built to support does not: the
effect is absent or unmeasurable on two thirds of the repositories that motivated it, its
magnitude is bounded above by a circularity that cannot currently be priced, and its
pre-registered secondary failed outright. Spending $20–30 on a live pair now would buy a
number nobody could interpret. The next $0 step, if this is revived, is P6's adoption set.

**Cost** $0.02.

> #### The probe reported a pooled rate over 25 repositories with wildly unequal strata **[C-20]**
>
> Its first run led with **+36.3pt** and a PASS. That figure pools 1,237 papers into one
> rate, and `peft` alone supplies 38 of the 166 score-3 papers — 23% — at a 100% hit rate,
> while nine cases contribute 0 score-3 papers and 326 of the 652 below-2. The pooled number
> was close to a report about `peft`.
>
> The repair is the four-estimator table above, and it is not cosmetic: it reverses the sign
> of the shipped baseline and collapses the probe's own secondary. `method`-typed versus
> `library`-typed spans read **+28.3pt vs +4.8pt** pooled — the finding that "technique names
> are the discriminating part, package names are not" — and **+11.5pt vs +10.9pt** per case.
> The 6× contrast was `peft`'s LoRA vocabulary; all 38 of its score-3 papers hit a method
> span. Reported here because the pooled version was written up and sent before the audit
> caught it.
>
> The same shape as C-7 (a favourable draw read as a property) and the pool scanner that
> read 1,250 papers as 0: an aggregate that is really one case, presented as a system
> property. `_case_cells`/`_macro`/`_mh` and the sign column now make the composition
> visible in the output rather than in a reader's head.

> #### NR-39's "only the anchor channel discriminates" is a case-mix artifact **[C-21]**
>
> NR-39 reported anchors at **20.5% / 14.9% / 6.7%** across score-3 / actionable / below,
> read that +13.8pt gap as the one repository-side channel carrying signal, and built the
> Roadmap 16 revival condition on it. Computed per case with the same cached artifacts, the
> anchor channel is **−0.9pt unweighted, −0.8pt at n≥5, −0.6pt Mantel-Haenszel, 95% CI
> [−3.7, +1.4], 2 cases better / 3 worse / 11 tied.** Removing `peft` alone takes the pooled
> figure from +13.7pt to **+0.2pt**. Within a case the channel is flat at every boundary
> (3-vs-1 +0.5pt, 2-vs-1 +0.1pt).
>
> **The anchor channel does not discriminate within a repository.** NR-39's Q1/Q2 percentages
> stand as reported — they are correctly computed pooled rates — but the conclusion drawn
> from them does not, and the alias-table revival condition rests on it. What survives is
> narrower and still useful: anchors *cover* poorly (12 of 25 cases have none), and coverage
> was always the other half of that entry.

> #### NR-39's "keywords are noise at +0.2pt" is a saturation artifact **[C-22]**
>
> Keywords appear in 97.6% / 96.4% / 97.4% of abstracts across the three strata, and NR-39
> read the +0.2pt spread as "keywords say *this paper is about your topic*, which showing it
> already implies". The reading is right about the mechanism and wrong about the evidence:
> there are ~19 keywords per case against ~4 typed spans, so a **binary** hit metric is
> pinned at its ceiling and could not have shown a difference whatever the vocabulary did.
> The same vocabulary is plainly score-correlated once it is not binarised — mean keyword
> *count* per abstract is **5.25** at score-3 against **3.44** below-2.
>
> Sampled down to the treatment's size (200 seeded draws, |spans| keywords per case), the
> keyword channel scores **+3.3pt macro, 95% [−2.9, +10.4]** — still not a signal, but for a
> reason the +0.2pt figure never established. P9 inherited the flawed comparator in its own
> pre-registration ("well clear of keyword noise") and its Q2c control now replaces it. The
> honest increment of Haiku extraction over a size-matched keyword sample is **+30.4pt**.

### The headline re-measured at `w_embedding: 1.5` — the gap above is closed (2026-08-16)

```bash
uv run python evals/join_wemb_headline.py         # $0 -- the prediction, written first
uv run python evals/run_judge_eval.py --baseline cli --sources arxiv     --rr-pool 50 --rr-rerank --rr-all-time --rr-hybrid --rr-sweep --rr-finescale --rr-hyde     --rr-frozen-pool evals/.work/pool-wemb --rr-window 15 --rr-w-embedding 1.5 < /dev/null
```

`…-wemb1.5-20260815T225831Z.json`. Identical to draw 2's treatment arm except `--baseline
cli` replaces `--baseline none`: one variable.

**A free prediction was made first, and it is the reason this cost $4.38 instead of $13.**
The two existing draws already measured RepoRadar at 1.5; what they lacked was a
paired-vs-Opus column, because both ran `--baseline none`. That column does not need a new
baseline run, because **net@2 is a function of a system's own returned papers** —
`summarize_system` feeds `pool_gains` to `ndcg@k` and `pool_has_relevant` and to nothing
else. So the baseline's per-case net@2 measured on `pool-depth` is arithmetically the same
number it takes on `pool-wemb`. `join_wemb_headline.py` states that invariant, predicts the
paid run from it, and exists so the prediction can be scored rather than admired.

| | predicted (free) | actual (paid) |
|---|---|---|
| RepoRadar net@2, 24 cases | +6.23, range [+5.49, +6.97] | **+5.92** |
| paired vs Opus | +4.60 | **+4.29**, sign *p* = 0.0007 |
| baseline column, 24 shared cases | +1.62 · 48 shown · 45 actionable | **+1.62 · 48 · 45** |

**The invariant held to the digit** across two different pools — the join's assumption is
now demonstrated rather than asserted, and the miss on RepoRadar's own side (0.31) is well
inside the one-draw floor of 0.74.

**The headline, and it is a 25-case one for the first time.** `thin-lang`'s baseline had a
stale cache discriminator — it errored in the published run, and failures are never cached,
so it never got refreshed. It re-ran here and succeeded, which is where the $4.38 went. The
published +5.42/+5.12 split existed only because one baseline was missing; that split is now
gone.

| | RepoRadar (window 15) | baseline, as measured | baseline, corrected **[C-25]** |
|---|---|---|---|
| mean net@2, **25 cases** | **+5.72** | +1.56 | **+1.84** |
| shown / actionable | 212 / 189 | 51 / 47 | 58 / 54 |
| precision | 0.892 | 0.922 | **0.931** |
| net-negative repositories | 0 | 1 | 1 |

**As measured: paired +4.16, 95% CI [+2.44, +6.00], 18 w / 2 l / 5 t, sign *p* = 0.0004.**
**Corrected: paired +3.88, 95% CI [+2.24, +5.60], 17 w / 2 l / 6 t, sign *p* = 0.0007.**
Both columns from `evals/restate_c25.py`, pinned by `tests/test_restate_c25.py`. RepoRadar's
own column does not move — net@2 reads a system's own returned papers, so the whole
correction is a transfer from our margin into the comparator's.

**What this changes, and what it does not.** `BENCHMARK_HEADLINE`, the measured preset and
the audit now agree at 1.5; pass (d) reports all 39 measured fields reproduced. The
remaining divergence is dataclass-vs-template only, and it is deliberate — the dataclass
value is what the keyless −8.12 arm was measured at, and raising it would put an unmeasured
value under a published negative number. What has *not* changed is NR-38: this is a third
draw of the 1.5 arm and it is **not** averaged into the +1.00, which stays closed at the two
pre-registered draws. Its consistency with them (+5.72 against +5.80 and +6.20, every one
above every control draw) is a check, not additional evidence.

**A claim I made from the free join and then withdrew.** The two draws showed precision
0.899 and 0.913 against the published 0.888, and I reported that the precision gap to Opus
narrows at 1.5 — a real effect, in the direction that would have softened this project's
largest caveat. The paid draw reads **0.891**. Three draws of a two-draw pattern is how that
should have been read in the first place: the volume increase replicates (208, 208, 211
against 195, 193, 196), the precision change does not. It is not in the paper.

**Two per-draw properties reported as such.** "0 net-negative repositories" is this draw's
value, not the method's — the same configuration gave 1 and 2 on other draws, and C-7
records what happens when that distinction is dropped. Precision 0.892 against the
baseline's 0.922 leaves the headline caveat where it was: RepoRadar returns **four times as
many papers at three points lower precision**, and whether a maintainer prefers that is a
question net@2 does not answer.

**Cost** $4.38, one case's baseline. No collection: 24 of 25 baseline answers and most judge
verdicts were already cached, and the cache-hit rate was verified *before* launching rather
than discovered after.

#### Three guards fired during this experiment. Two were right, one was mine to fix. **[C-19]**

**The one that cost a collection, and it was my regression.** Adding `rr_scan_source` to
`POOL_FLAGS` on 2026-08-16 (NR-36) **invalidated every frozen pool in the project**: the
fingerprint hashes the flag set, so growing the set makes all stored pools unreadable. Both
arms refused immediately. That is the guard working — reusing them would have ranked the
old pool under the new run's name — but the frozen mode is what took the floor from 1.04 to
0.48/0.74, so the regression silently removed the project's cheap, sensitive experimental
mode until someone tried to use it.

I considered omitting default-valued flags from the hash so a new flag would be a no-op, and
**rejected it**: this project changes defaults (`bigrams` adjacent→verified, `top_k` 15→50,
`top_n` 10→15), so a pool collected under an old default would then silently match a run
under a new one — trading a loud failure for the exact silent-staleness bug the fingerprint
exists to prevent. The refusal is correct. What was weak was the *diagnosis*: two opaque
hashes, and working out the cause took three commands. Pools now record the flag set they
were fingerprinted over, and a mismatch names what was added or removed and why a defaulted
new flag still invalidates.

**The one that saved the experiment.** The void-not-null refusal added with `--rr-w-embedding`
fired on the first probe: without the `embeddings` extra the ranker scores on keyword and
category alone, so the treatment arm would silently *be* the control and report "the weight
does nothing" about a component that never ran. The cause was not a missing package — the
probe script lived in `evals/.work/`, Python prepends the script's directory to `sys.path`,
and that directory holds ~100 cloned repositories, one of which shadows the
sentence-transformers import chain. The identical script from `/tmp` runs fine. Without the
refusal this would have been ~$9 spent on a manufactured null.

**The one that was over-strict.** The report refused to compare the arms because one
*seeded* the pool (`frozen-seeded`) and the other *reused* it (`frozen:<fp>`) — despite
identical fingerprints and identical pool sizes on all 25 cases. Those are different
histories and `provenance` is right to say so; comparability is the narrower question of
whether the arms ranked the same candidates, which `same_pool` already answers. The fix went
to the comparability check, **not** to the shared `provenance` function — blurring that
would misdate the seeding run everywhere it is used. Left unfixed, the natural shape of a
frozen experiment (arm 1 seeds, arm 2 reuses) would need a throwaway collection every time.

**This overturns a decision recorded in August, and the reconciliation matters more than
the fix.** The absent-category experiment hit the identical refusal, paid for a discarded
seeding pass, and left a note in its runner reading *"they are right to."* That was right
**then**: with no fingerprint check at the decision point, mode equality was the only
evidence available, so the workaround was *sufficient* — but it was never *necessary*, and
nobody went back to ask which. What licenses the relaxation now is a specific argument
rather than convenience: the one way a seeding arm and a reuse arm can rank different
candidates is an **empty pool**, and `save_frozen_pool` refuses to store one (an empty pool
and a failed collection are the same bytes on disk). So that case is re-collected live by
the "reuse" arm, which makes the run's own modes mixed — and mixed provenance is already
refused. The divergence this check would have caught is caught anyway. The August runner's
note is now marked superseded rather than left to contradict the code.

### Documentation volume does not predict anything, and it closes the thin-docs axis (2026-08-16) **[NR-37]**

```bash
uv run python evals/thin_docs_detector.py     # $0, no network, no LLM
```

Three thin-docs remedies had failed (NR-25 similarity floor, NR-26 stated intent, NR-36
source scanning). The proposal here was not a remedy but a **detector**: §12.1 says the
system fails *coherently* on thin repositories — every internal signal looks healthy —
because queries, hypotheses, gate and rescore all read the same impoverished profile.
Documentation corpus size is the one signal from *outside* that loop: known before a
profile exists, deterministic, free, and structurally unable to be fooled by a
plausible-but-wrong profile. Measured with the profiler's own `_collect_text_corpus`, not
a re-implementation.

**On induced thinness the signal works.** Across the 24-point ablation grid (6 repositories
× 4 documentation budgets), the 8 materially degraded points have a **median corpus of 115
characters** against **1,076** for the 16 intact ones — a 9× separation.

**On real repositories it predicts nothing.**

| | |
|---|---|
| Pearson r(log₁₀ corpus, net@2), n = 25 | **+0.14** |
| Spearman ρ | **+0.20** |
| bottom quintile by corpus (< 5k chars, n = 5) | **+5.60** |
| the other 20 | **+5.25** |

The thinnest repositories score *slightly better*. The four thinnest, in order:

| repo | corpus | net@2 |
|---|---|---|
| `thin-lang` | 108 | +0.0 |
| `thin-gnn` | 1,073 | **+9.0** |
| `db` (DuckDB) | 1,857 | **+14.0** — the best case in the run |
| `numerics` | 2,205 | +1.0 |

`db` is the demonstration on its own: 1,857 profiler-visible characters, no parseable
manifest (§12.1 noted it profiles to almost nothing under ablation), and it scores the
**highest net@2 in the benchmark**. Meanwhile `webdev` (384,456 chars), `cli` (193,932),
`linter` (197,023) and `http` (77,873) all score 0.0. Corpus size points the wrong way at
both ends.

**Every abstention threshold either does nothing or destroys value:**

| threshold | flagged | delta |
|---|---|---|
| 500 | 1 | **+0.00** |
| 2,000 | 3 | **−0.92** |
| 10,000 | 8 | **−1.64** |
| 50,000 | 11 | **−2.20** |

**Weight the two results differently.** The sweep is *weak* evidence and was flagged as
such before running: this draw has zero net-negative repositories, so there is nothing for
abstention to rescue and the sweep can only lose. The **correlation is the substantive
finding**, and it does not depend on that — corpus size simply does not track outcome.

**What this actually closes.** Not one remedy — the axis. The ablation grid and real
repositories are **different populations**, and for the purpose of predicting which
repositories the system serves badly, the ablation is not a proxy at all. §12.2 validated
it as a proxy on *precision* and that stands; nothing licensed extending it to
identification, and this is what the extension would have got wrong. Ablating a rich
repository's prose strips what retrieval needs while the models still recognise the
repository — "the correct answer to the wrong question" (NR-25). A genuinely small
repository with 1,073 well-chosen characters describes itself perfectly well. **Volume is
not quality of description**, and only volume is what "thin documentation" ever measured.

The four repositories that score 0.0 in this run are the negative controls; the weak ones
after that are the arXiv coverage gap (`crypto`, `encryption`, `compiler`). Those are
properties of the *domain*, not of the documentation — which is where any further work on
"repositories we serve badly" should look.

**No product change.** The detector is refuted; nothing ships. The `--rr-ablate-docs`
budget is now recorded in run artifacts — the last `POOL_FLAG` that was not — because the
four grid arms could only be identified by matching their means against a derived summary
file, and `evals/thin_docs_detector.py` carries that mapping so it survives independently.
Past artifacts are left untouched: rewriting recorded runs to add a field would be editing
history, and the mapping is documented instead.

**Cost** $0.

### PRE-REGISTERED — should the profiler read source code? (2026-08-16)

```bash
uv run python evals/scan_source_probe.py       # the $0 stage-1 probe, run first
```

`profiler.scan_source` is shipped and **no benchmark arm has ever enabled it** — the class
of never-measured default that already produced +1.00 (gate depth) and +1.24 (digest
width). NR-26 pointed at it: whatever benefit lived in its richer arm "tracks the extra
*information* — source code the profiler never reads". And it is the obvious thin-docs
remedy, because `ablate_docs`'s own guard states the mechanism — such a repository is thin
in prose but **has code**.

#### Stage 1, $0: what it does to the profile

| | mean |
|---|---|
| top-20 keyword overlap, on vs off | **84%** (range 7/20 to 20/20) |
| implementation-vocabulary share | 2% → **3%** |
| cases gaining anchors | **24 / 25** |
| cases with **zero** anchors that gain them | **8** (`compiler` 0→43, `db` 0→35, `columnar` 0→32, `storage` 0→31, `linter` 0→19, `cli` 0→16, `systems` 0→11, `vectordb` 0→7) |

**The probe overturned the hypothesis that motivated it.** From `thin-lang` alone — prose
gives `programming language`, `native binaries`; scanning gives `vscode`, `net`,
`child_process` — I expected source code to *drown* conceptual vocabulary, the register
mismatch one level lower. Across 25 cases that is not what happens: keywords are 84%
stable and implementation vocabulary barely moves. One case was an anecdote.

**And it found a larger effect the hypothesis never mentioned.** The real change is in
**anchors**, not keywords: eight repositories have *no anchors at all* without scanning and
acquire 7–43 with it. Those are the repos whose dependency manifests the profiler cannot
parse — C, C++, Rust. My headline metric measured the wrong field, and only the per-field
dump showed it.

**The benefit looks inverted from where it was aimed.** For the thin cohort — the reason
this was proposed — scanning looks unhelpful or harmful: `thin-lang` gains 4 anchors while
its top terms become `vscode`/`child_process`, and `thin-kv` gains 2 while its terms become
`grafanalib`/`executor_panels`, which are its Grafana dashboards rather than its storage
engine. `thin-gnn` barely changes. The repositories that gain most are *rich* ones with
unparseable manifests.

#### Stage 2: the judged arms

Two arms, same session, **live** (scanning changes the profile → the queries → the pool, so
`rr_scan_source` is in `POOL_FLAGS` and a frozen pool cannot be shared; the live floor
applies, ~1.04 at width 10 and larger at 15).

```bash
COMMON="--baseline none --sources arxiv --rr-pool 50 --rr-rerank --rr-all-time --rr-hybrid
        --rr-sweep --rr-finescale --rr-hyde --rr-window 15"
uv run python evals/run_judge_eval.py $COMMON                     # control
uv run python evals/run_judge_eval.py $COMMON --rr-scan-source    # treatment
```

**Prediction, revised by the probe and written before the run.** Primary, all 25:
**−0.5 to +1.5, most likely a small positive, and quite possibly inside the floor** — the
eight zero-anchor repositories are the mechanism, and against them sits the fact that two
prior composition changes (absent-category NR-33, fusion NR-35) each reshuffled the digest
substantially and moved the outcome by **+0.00**. This would be the third instance of that
shape, and I would not be surprised.

**Secondary, pre-registered and openly underpowered: the thin cohort.** I predict it does
**not** help and may hurt, on the probe's evidence above. n = 3 with one dominant case
against a 0.74-plus floor cannot establish that either way — it is reported as suggestive,
never as a result, and the prediction is recorded so a favourable draw cannot be read as
confirmation after the fact.

**Gate-free secondary:** actionable papers reaching the ranked top-15 before gating. It
moved 0.00 in NR-33, −0.08 in NR-35, and 5.00 → 6.52 where the depth effect was real.

**A coverage limit worth stating before the number exists.** `profiler.source_extensions`
defaults to `.py/.js/.ts/.tsx/.jsx`, so a C, C++, Rust or Go repository is scanned only
through whatever tooling scripts it happens to carry. Six of 25 cases produce a
byte-identical profile either way. The treatment is unevenly applied by construction, and a
null could mean "source scanning does not help" or "it barely ran" — the per-case profile
dump distinguishes them.

**Estimated cost** $20–30, two live 25-case arms.

> #### RESULT (2026-08-16) — **−0.52/case.** Source scanning does not help, and it is worst where it was aimed. **[NR-36]**
>
> `…-20260815T041237Z.json` (control) and `…-scansource-20260815T053404Z.json`. 25/25 both
> arms, live.
>
> | arm | net@2 | shown | actionable | precision | net-negative |
> |---|---|---|---|---|---|
> | prose only (shipped) | **+5.32** | 187 | 169 | **0.904** | **0** |
> | `--rr-scan-source` | **+4.80** | 189 | 166 | 0.878 | 2 |
>
> **Paired −0.52/case, 95% CI [−1.72, +0.72], 6 better / 11 worse / 8 tied, sign p = 0.33 —
> inside the floor.** Valid, not void: 25/25 cases changed, mean Jaccard **0.34**.
>
> **Every measure points the same way, and none of them reaches the floor.**
>
> | measure | prose only | scanning | delta |
> |---|---|---|---|
> | net@2, all 25 | +5.32 | +4.80 | **−0.52** |
> | gate-free: actionable in the ranked top-15 | 9.00 | 8.80 | −0.20 |
> | thin cohort *(pre-registered, n = 3)* | +4.33 | +2.33 | **−2.00** |
> | other 22 | +5.45 | +5.14 | −0.32 |
>
> **The pre-registered secondary came in as predicted, and the primary did not.** I wrote
> "−0.5 to +1.5, most likely a small positive": the result sits at the extreme bottom of
> that interval with the sign opposite to my point estimate. But for the thin cohort I
> wrote, from the $0 probe, "I predict it does not help and may hurt" — `thin-gnn` 9→5,
> `thin-kv` 4→2, `thin-lang` 0→0. The probe's evidence (`thin-kv` acquiring `grafanalib`
> and `executor_panels` — its Grafana dashboards, not its storage engine) predicted the
> direction correctly. **At n = 3 this remains suggestive, exactly as pre-registered; it is
> recorded because the prediction was written first, not because three cases settle
> anything.**
>
> **The verdict is "no", not "harmful".** −0.52 is inside the floor, and the floor here is
> itself a lower bound (below). Consistent negative signs across three measures are weak
> evidence — they come from the same two runs and are not independent. What the arm
> establishes is that source scanning **does not pay for itself**, which is the question a
> default has to answer.
>
> **The shipped default is vindicated, and that is a first.** Every other never-measured
> default this campaign examined turned out to be wrong — `triage.top_k` too shallow by
> +1.00, `output.top_n` too narrow by +1.24. `profiler.scan_source: False` is right. Worth
> recording so the class is not over-generalised into "unmeasured defaults are always
> wrong": it is now two wrong and one right, and the only way to tell which is to measure.
>
> **What the $0 probe bought.** It cost nothing, overturned my own mechanism hypothesis
> (I expected source vocabulary to *drown* the concepts; keywords are 84% stable and
> implementation vocabulary moves 2% → 3%), found a larger effect I had not predicted
> (anchors: 24/25 cases gain them, eight from zero), and predicted the thin-cohort
> direction that the paid run then confirmed. The one thing it could not do was give the
> sign of the outcome — which is the thing that needed $26.
>
> **A coverage caveat that survives the result.** `profiler.source_extensions` is
> `.py/.js/.ts/.tsx/.jsx`, so six of 25 cases produce a byte-identical profile either way.
> This measures source scanning *as shipped*, not source scanning in principle: a version
> that read C, C++, Rust and Go might behave differently, and nothing here speaks to that.
>
> **Cost** ~$26, two live arms.

#### The floor guard had the same defect in the half I did not fix **[C-18]**

The report printed **"MRE = 1.04 — live collection, window 10"** for a comparison at
window 15. `mre_for` was made width-aware on 2026-08-15 to fix C-8 — and only its *frozen*
branch was. The live branch returned early, ignoring width, and hard-coded the label
"window 10".

The direction is the costly one. The frozen floor rose **0.48 → 0.74 (×1.54)** when the cut
widened; the live floor at 15 has never been measured, so 1.04 **understates** it, and
under-stating a floor turns noise into a finding. It changed nothing here — −0.52 is inside
1.04 and inside anything larger — but any live window-15 result between 1.04 and roughly
1.6 would have been reported as resolved when it was not.

Fixed by keying the live branch on width too, labelling an unmeasured width as
`UNMEASURED; 1.04 is the window-10 value and a LOWER BOUND`, and making the verdict line
refuse: "past the floor" against a lower-bound floor now prints **NOT resolved**. No
estimate was invented — scaling 1.04 by 1.54 would be an unmeasured number wearing a
measurement's authority.

Recorded as a correction, unlike the thin-docs wording of the same week, because this is an
*instrument* returning a confident wrong value with forward blast radius on every future
live comparison — not a characterisation in prose.

### Re-analysis, $0 — the thin-docs deficit is real, replicates, and rests on one repository (2026-08-16)

No new run. Three existing draws re-read before committing to a thin-documentation work
programme, because "handle thin repos better" is the kind of goal that is easy to adopt and
hard to measure.

| draw | thin-lang | thin-kv | thin-gnn | **cohort** | other 22 |
|---|---|---|---|---|---|
| shipped config, 2026-08-14 | −2.0 | +5.0 | +5.0 | **+2.67** | +5.45 |
| shipped config, 2026-08-16 | −2.0 | +5.0 | +2.0 | **+1.67** | +5.32 |
| no-fusion variant (NR-35, +0.00) | −4.0 | +5.0 | +5.0 | **+2.00** | +5.27 |

**Three things, and they point in different directions.**

**The deficit is real and it replicates.** ~3 net@2/case under the other 22, across three
draws. §12.2 reported +2.00 from a single session; a single draw of this benchmark is
weather (§8.7, C-7), and this is the first evidence the number is climate.

**It is not a failure.** The cohort is **positive** in every draw. The paper's Limitations
preamble said the system "fails on them silently" while the section it cited reported
+2.00 — corrected to "degrades sharply and silently". No published *number* was wrong, so
this takes no C-entry; inflating the correction count would be its own small dishonesty.

**One repository of three carries all of it.** `thin-kv` is +5.0 in all three draws and
`thin-gnn` +2.0 to +5.0 — both healthy. `thin-lang`, the 108-character case, is −2.0, −4.0,
−2.0. The cohort mean is one repository's problem averaged with two non-problems.

**What that settles about the work programme.** A thin-docs remedy **cannot be measured on
this cohort**: n = 3, one dominant case, against a 0.74 floor. Only a multi-point effect
would clear it, and a subgroup claim resting on a single repository is the fragility the
jackknife exposed at n = 12. So the deficit is recorded as real and its *remediation* as
currently unmeasurable — two claims a blanket "thin docs is broken" would have merged, and
the reason the next experiment is aimed at the whole benchmark rather than at these three.

**The lead it hands over.** NR-26 found that whatever benefit lived in its richer arm
"tracks the extra *information* — source code the profiler never reads". `profiler.scan_source`
is a shipped capability **no benchmark arm has ever enabled**, and `_ablate_docs`'s own guard
states the mechanism: a thin-docs repository is thin in prose but *has code*. `thin-lang` has
108 characters of prose and an entire compiler. That is the next experiment.

**Cost** $0 — re-read of runs already paid for.

### PRE-REGISTERED — does BM25-RRF fusion still earn its place? (2026-08-16)

**The question, and why it is not the one I said I would ask.** The `ranking.hybrid`
declaration in `audit_product_divergence.py` promised the *out-of-the-box arm* would settle
whether fusion should be on in the shipped default. Having run that arm and then read the
ungated code path, that experiment would mostly measure an artifact: `hybrid_reorder`
changes the order and deliberately leaves `score_total` intact, while the ungated Top Picks
tier admits the window on `score_total >= 0.5` — so RRF pulls lower-scoring papers into the
fifteen slots where they then fail the threshold, shrinking the shown set. At the default's
precision of 0.379 each shown paper is worth `3p-2 = -0.86`, so shrinking *helps*, for
reasons that have nothing to do with ranking quality. It would be a finding about the 0.5
tier rule reported as a finding about fusion.

The question worth the money is the one nobody has asked: **fusion has been ON in every
headline since PR #30 and has never been ablated inside the measured configuration.** Its
keep decision rests on NR-11's *pre-rescore* argument — better nDCG everywhere, lower
headline — with the headline cost assumed recovered once the rescore began ordering what
the gate admits. §8.5 claims the stages compose. That claim has never been tested for this
component.

**Design.** Two arms, same session, **one frozen pool** (`pool-depth`, 25 cases, 14,552
candidates), hybrid the only variable. `rr_hybrid` sits in `RANKING_FLAGS` and is
deliberately absent from the pool fingerprint, so both arms provably rank the identical
pool — this is the case freezing was built for.

```bash
COMMON="--baseline none --sources arxiv --rr-pool 50 --rr-rerank --rr-all-time
        --rr-sweep --rr-finescale --rr-hyde --rr-frozen-pool evals/.work/pool-depth
        --rr-window 15"
uv run python evals/run_judge_eval.py $COMMON --rr-hybrid     # control (the shipped preset)
uv run python evals/run_judge_eval.py $COMMON                 # treatment: fusion removed
```

**Floor.** 0.74 net@2/case (frozen pool, window 15) — the width-aware floor, not the 0.48
that C-8 misapplied.

**Prediction, written first.** Fusion is worth **−0.5 to +1.5**, most likely small and
positive (~+0.5), and **quite possibly inside the floor**. The composition argument says it
should help: better nDCG puts more actionable papers in the top 50 the gate sees, and the
rescore now sorts the borderline ones NR-11 complained about instead of dumping them. The
counter is that the rescore only runs on the threshold band — papers above it are trusted
on the gate's word — so extra borderline candidates are only partly rescued.

**Secondary, gate-free.** Actionable papers reaching the ranked top-15 *before* gating. This
tests the mechanism directly: if fusion does not put more actionable papers in front of the
gate, no downstream story can rescue it. The same measure moved 0.00 in §9.6 and correctly
predicted a null there, and moved 5.00 → 6.52 in the gate-depth experiment.

**What each outcome licenses.**
* **Past the floor and positive** — the keep decision finally has a post-rescore
  justification instead of a pre-rescore one, and NR-11's headline cost is confirmed
  recovered.
* **Past the floor and negative** — the *recommended* configuration contains a component
  that hurts. It comes out of the preset, out of `BENCHMARK_HEADLINE`, and every headline's
  provenance row gains a footnote. This is the outcome that would cost the most to act on
  and is the reason the arm is worth running.
* **Inside the floor** — unresolved, and I will say so rather than reading the sign. That
  outcome is itself worth recording: it means the configuration we recommend to users
  carries a component we cannot show earns its place, which is a weaker position than the
  paper currently implies.

**Estimated cost** $10–20. Both arms are frozen-pool, so no collection; the control's papers
are almost entirely in the judge cache, and the bill is Haiku + rescore for both arms plus
judge verdicts on whatever the no-fusion arm surfaces that has never been judged.

> #### RESULT (2026-08-16) — **+0.00/case.** Fusion reshuffles 59% of the digest and moves nothing. **[NR-35]**
>
> `…-20260814T234847Z.json` (control) and `…-nohybrid-20260815T002741Z.json` (treatment),
> one frozen pool, 25/25 cases both arms.
>
> | arm | net@2 | shown | actionable | precision | abstained | net-negative |
> |---|---|---|---|---|---|---|
> | `hybrid` (shipped preset) | **+4.88** | 197 | 172 | 0.873 | 3 | 2 |
> | `no-hybrid` | **+4.88** | 206 | 178 | 0.864 | 3 | 2 |
>
> **Paired: +0.00/case, 95% CI [−1.00, +0.96], 8 better / 9 worse / 8 tied, sign p = 1.000
> — inside the 0.74 floor.**
>
> **Valid, not void, and this is the part that makes the null informative.** The flag is
> doing a great deal: **25/25 cases changed their returned top-10, mean Jaccard 0.41** — 59%
> of the shown papers are different — and per-case scores swing hard in both directions
> (`ann` 6→11, `llminfer` 5→9, `vectordb` 4→8 against `speech` 6→0, `rag` 7→2, `crypto`
> 2→0). Seventeen of 25 cases move. They cancel to two decimal places.
>
> **The pre-registered gate-free measure agrees, and it is the one that settles the
> mechanism.** Actionable papers reaching the ranked top-15 *before any gating*:
> **8.80/case with fusion, 8.72 without — a delta of −0.08** (220 against 218 papers), with
> 20 of 25 cases differing on the count. Fusion was kept on NR-11's argument that it ranks
> better; the composition claim that argument implies — more actionable papers in front of
> the gate — is measured at essentially zero. The same measure moved 0.00 in the
> absent-category null and 5.00 → 6.52 where the depth effect was real, so it is not a
> measure that refuses to move.
>
> **Prediction check.** I pre-registered −0.5 to +1.5, point estimate ~+0.5, and wrote
> "quite possibly inside the floor." The interval held and the hedge was right; the point
> estimate was high. Unlike the out-of-the-box arm, this one I did not get wrong.
>
> **Control sanity.** The control arm reads +4.88 against the +5.12 the same configuration
> scored on the same pool on 2026-08-14 — a gap of 0.24, a third of the floor, which is what
> temperature-0 jitter across two reuse passes is supposed to look like.
>
> **What this licenses, and what it does not.** Per the pre-registration: the outcome is
> **unresolved, not zero**, and I am not reading the sign of a +0.00. But the secondary is
> not floor-limited in the same way — it is a direct count, and it says the mechanism by
> which fusion was supposed to earn its place is absent. So the honest position is:
> **the configuration we recommend to users carries a component we cannot show earns its
> place, whose stated mechanism measures null.**
>
> **It stays in the preset anyway, and the reason is not inertia.** Every published headline
> was measured with fusion on; the preset's entire value is that it reproduces that
> configuration, asserted field-by-field. Dropping a component on an unresolved result would
> make the recommended configuration differ from every number recommending it — trading a
> documented uncertainty for an undocumented divergence. What changes is the record: this is
> now a known-unjustified component with a starting point for a future simplification, not
> a stage the paper can keep citing as part of a composing whole.
>
> **The second instance of a shape worth naming.** NR-33 found the absent-category rule
> "real and large in composition, +0.00 in outcome". This is the same result for a different
> ranking-stage change, measured independently: **two components now reshuffle the digest
> substantially and move the outcome by nothing.** The conclusion §13 already draws — that
> the value sits downstream of the heuristic ranker, not in it — has stopped being an
> interpretation of one experiment.
>
> **Cost** ~$6: no collection (frozen pool), most judge verdicts cached, the bill was Haiku
> gate + rescore on both arms plus verdicts on what the no-fusion arm newly surfaced.

### PRE-REGISTERED — what does the out-of-the-box configuration actually score? (2026-08-16)

**Why this is being run at all.** PR #134 shipped a README table row reading
"mean net@2 on the 25-repo benchmark | **−11** | **+5.42**", and **−11 was never measured
on the 25-repo benchmark.** It is the mean of four cases from 2026-07-05 — `rag` 0,
`cv` −4, `rl` −20, `webdev` −20 — one of which (`webdev`) is a negative control, i.e. a
repository with no applicable literature, contributing a quarter of that mean. It also
predates all-time discovery (NR-5), the C-9 query-bridge repair, the C-10 phrase fix,
verified bigrams, and the 10 → 15 digest width. The two numbers in that row share a metric
and nothing else. This run replaces the left column with something that belongs beside the
right one.

**The arm.** The configuration `rr init` writes, at the width it writes, on all 25 cases:

```bash
uv run python evals/run_judge_eval.py --baseline none --sources arxiv \
    --rr-all-time --rr-bigrams verified --rr-absent-category omit \
    --rr-pool 15 --rr-window 15 < /dev/null
```

No `--rr-triage`, so the returned set is the product's own ungated tiering — the harness
imports `TOP_THRESHOLD` from the shipped `digest.py` and returns papers scoring ≥ 0.5,
which is exactly what `rr update` does with the gate off. No HyDE, no hybrid, no rescore.

**Two fidelity limits, stated before the number exists.**

1. `--rr-pool 15` is **required**, not cosmetic. With the gate off, the harness's candidate
   depth defaults to 10, so `--rr-window 15` would have cut the digest at 10 while
   recording `digest_window: 15` — an artifact asserting a width the run did not have.
   The harness now refuses that combination instead of warning. No recorded run was
   affected (all five window-carrying runs have gate depth 50).
2. The default template sets `ranking.w_embedding: 1.5` and the harness **cannot reproduce
   it** (`evaluation.UNMODELLED_KNOBS`). So this measures the default *as installed without
   the `embeddings` extra* — the common case, not every case. That gap is the same
   install-dependent divergence the config audit flagged, now costing a measurement.

**Prediction, written first.** The gate is worth more than everything else combined, so the
ungated arm should land **well below +5.42** — I expect somewhere in **−2 to +2**, not −11.
Reasoning: the −11 era had a 90-day recency window that structurally excluded the gold set
and a broken query bridge, both since repaired, so retrieval is far better; but the digest
is now 15 wide instead of 10, and at low precision each extra paper costs 2. I expect
those to partly cancel, leaving a number that is *bad* but not *catastrophic*.

**Kill / surprise conditions.** If the arm scores **above +3**, the gate's contribution is
much smaller than this project has claimed for six weeks and the claim needs re-examining,
not just the README. If it scores **below −5**, my "retrieval got better" reasoning is
wrong and the −11 was closer to right than I thought. Either outcome is reportable; the
prediction above is what I am betting.

**What the number will and will not license.** It is a *level* from a single live draw, so
it carries the ±0.6 whole-run drift documented in §8.7 and is not paired with the +5.42
run (which was frozen-pool). Given an expected gap of several points that is immaterial;
if the arm lands close enough to +5.42 for session drift to matter, a paired same-session
re-run is the follow-up, and I will say so rather than quietly compare across sessions.

**Estimated cost** $10–15 — no Haiku and no OpenAI at all, since nothing gates or rescores.
The entire bill is judge verdicts on papers the cached pool has not seen.

> #### RESULT (2026-08-16) — **−8.12**. My prediction missed and my own surprise condition fired.
>
> `judge-gpt-5.5-bigrams_verified-20260814T194558Z.json`. 25/25 cases, 375 papers pooled
> and judged, **0 judge failures**, live pool.
>
> | | **default (`rr init`)** | **measured (`rr init --measured`)** |
> |---|---|---|
> | mean net@2, 25 cases | **−8.12** | **+5.12** |
> | papers shown | 235 | **197** |
> | of those, actionable | 89 | **174** |
> | precision | **0.379** | **0.883** |
> | net-negative repositories | **19 / 25** | 2 / 25 |
> | abstentions | 3 | — |
>
> **Gap: +13.24 net@2 per case.** (The +5.42 quoted elsewhere is the *24*-case figure from
> the vs-baseline run, restricted to cases where the Opus baseline completed; on all 25 the
> same arm is +5.12. Both arms here are window 15. The default is live and the measured one
> frozen, so this compares **levels, not a paired delta** — legitimate for a 13-point gap
> against ±0.6 whole-run drift, and it is why no CI is quoted.)
>
> **I predicted −2 to +2 and wrote "below −5 means my reasoning is wrong."** It came in at
> −8.12. The condition fired; the reasoning was wrong.
>
> **Where it was wrong, precisely.** I reasoned about the *pool* and forgot the *display
> rule*. Retrieval really did improve since July — this arm finds **89 actionable papers**,
> which the −11-era configuration could not have done. But net@2 pays `3p − 2` per shown
> paper, so at p = 0.379 **every paper shown costs 0.86 on average**, and the ungated
> digest has no way to stop: the 0.5 heuristic admitted all 15 in 17 of 25 cases. Better
> retrieval fed a display rule that cannot decline, and the extra papers were charged for.
>
> The single line that captures it: **the measured configuration shows fewer papers (197 vs
> 235) and delivers nearly twice as many actionable ones (174 vs 89).** The gate's value is
> not finding papers. It is declining to show them — which is what §6.1 said in July
> ("mostly by converting false-positive floods into correct abstentions"), and I still
> predicted wrong, because I was thinking about the numerator.
>
> **The stale number was better than my reasoning about it.** The 2026-07-05 figure I set
> out to correct (−11, four cases, one a negative control) sits **2.9** from the true
> 25-case value. My prediction's midpoint sits **8.1** away. The provenance criticism stands
> — it was never measured on this benchmark and had no business in that table — but it was
> the *more accurate* of the two numbers on offer, and I should record that I replaced a
> well-attributed guess with a badly-attributed measurement, not the reverse.
>
> **What it licenses.** "Worse than emitting nothing" is now measured rather than asserted:
> abstaining everywhere scores 0, the default scores −8.12, and it is net-negative on 19 of
> 25 repositories. The README's comparison row is now true as written for the first time.
>
> **Worst cases**, all showing 15 papers with 0 actionable: `crypto`, `encryption`,
> `thin-lang` at −30.0 each. `thin-lang` is the thin-documentation cohort behaving exactly
> as §12.1 predicts; `crypto` and `encryption` are the arXiv-coverage gap of §9.4. Neither
> is new — but ungated, each failure is shown to the user at full width.
>
> **A harness defect found while setting this up, before it could bite.** `--rr-window` is
> cut from the ranked candidate list, and with the gate off the candidate depth defaults to
> **10** while `--rr-window` defaults to **15** — so this run would have produced a 10-paper
> digest and recorded `digest_window: 15`, an artifact asserting a width it did not have.
> The harness now refuses that combination. No recorded run is affected: all five runs
> carrying a `digest_window` ran at gate depth 50.
>
> **Fidelity limit, stated in the pre-registration and unchanged by the result.** The
> default template sets `ranking.w_embedding: 1.5`, which the harness cannot reproduce
> (`evaluation.UNMODELLED_KNOBS`), so this measures the default *as installed without the
> `embeddings` extra*.
>
> **Cost** ~$10 of judge verdicts; $0 of Haiku and OpenAI, since nothing gated or rescored.

### Shipping the measured configuration instead of flipping defaults (2026-08-15)

```bash
rr init --measured      # the config behind +5.42, with every value's citation inline
```

The previous entry found that the shipped default enables none of the stages this
benchmark measures, and deliberately changed nothing — flipping an unmeasured default to
another unmeasured value buys nothing. The alternative is to make the measured
configuration a *thing you can ask for*, and to say plainly how weak the default is.

**What ships.** `measured_config_yaml()` is a second template, written by `rr init
--measured`, in which every value carries the measurement that justifies it — `top_k: 50`
cites the +1.00 depth result, `threshold: 2/3` cites the metric's own breakeven, `top_n:
15` cites +1.24, `w_embedding: 0.0` cites the fact that 1.5 is unmeasured in that role.
It states its prerequisites (both keys, `rr sync-index`, ~1.1 GB) and its price
(~$0.01–0.02/repo/run, against ~$0.80 for the agentic baseline) in the file itself.

**The part that makes it more than documentation.** The audit gained a fourth check:
every one of the **39** fields in `BENCHMARK_HEADLINE` must be reproduced by the preset,
**with no exemption mechanism at all**. The default template may differ from the benchmark
for declared reasons; this one may not, because its entire purpose is to be the
configuration behind the published number. A test asserts `DECLARED` is not even
referenced in that function, so the recommendation cannot be excused away from the
measurement it cites. Told a user "this gets you +5.42", we owe them the actual arms of
that run — and now a script checks that we delivered them.

Setting the preset's `finescale.timeout` to the eval's 60 rather than the product's 30 was
the one edit needed to make the check exact. An exemption would have been easier and would
have started the list that ends in a preset nobody has run.

**Where the default is now labelled.** Three places, because a user who never opens the
config should still hear it: the template's own header, `rr init`'s output (`-11` against
`+5.42`, with the pointer), and `rr update` at the point the gate would have run. The
last one also names the two-field trap — `triage.enabled: true` with `suggestions.provider`
left at `template` gates nothing — which previously printed a mild "skipping".

**What this is not.** It is not a measurement. The out-of-the-box arm is still unrun, so
"−11" remains the pre-gate figure measured during development rather than a fresh draw of
today's keyword-only path, and it is labelled that way in the README. No default changed.

**Cost** $0.

### The audit was asking the wrong object: `rr init` writes a config nobody measured (2026-08-15)

```bash
uv run python evals/audit_product_divergence.py     # $0, offline, exits non-zero on a finding
```

The product/benchmark audit's configuration pass compared **12 hand-listed fields** and
reported "every compared field agrees". Both halves of that sentence were wrong.

**Wrong scope.** The config tree has **79 leaves**. Twelve were compared; the other
sixty-seven were neither compared nor excused, so the pass reported clean about fields it
had never heard of. This is C-14b one level up — a guard scoped to where a bug was last
found — and the fix is the same: every leaf must now be in `BENCHMARK_HEADLINE` (the
benchmark reads it) or `NOT_UNDER_TEST` (with a written reason). A new config field fails
the audit until somebody classifies it.

**Wrong object.** "The shipped default" is not one thing. There are two surfaces:

| | dataclass default | `default_config_yaml()` — what `rr init` writes |
|---|---|---|
| `ranking.w_embedding` | 0.0 | **1.5** |

Where the template sets a value, that value is what a user runs and the dataclass default
is dead text. The old pass compared the dataclass column, so it was clean about a field on
which the product and every published number differ by **the largest ranking weight in the
file** — 1.5 against `w_keyword`'s 1.0. The pass now compares the *effective* value
(template first) and reports template-vs-dataclass disagreement separately.

**What the exhaustive pass found: every stage this project is about is off by default.**

| field | a user runs | measured | why it is declared |
|---|---|---|---|
| `triage.enabled` | `False` | `True` | needs an LLM key |
| `suggestions.provider` | `template` | `claude` | the gate's *second* required field |
| `triage.finescale.enabled` | `False` | `True` | needs a **second** vendor's key (logprobs) |
| `hyde.enabled` | `False` | `True` | needs `rr sync-index`, ~1.1 GB |
| `ranking.hybrid` | `False` | `True` | **no cost excuse** — see below |
| `ranking.w_embedding` | `1.5` | `0.0` | unmeasured in this role, both ways |
| `triage.finescale.timeout` | `30` | `60` | bounded by `enough_scored()` on both sides |

`rr init && rr update` is therefore the ungated heuristic digest — **the configuration
§6.1 measures at mean net@2 −11** — while the paper says "the shipped system" reaches
+5.42. Each opt-in is individually defensible (a default that fails without a credential
is worse than one that under-delivers) and the README documents all of them. What was not
defensible is that the audit built to catch exactly this shape reported clean.

Two details worth their own line. `triage.enabled: true` **alone is a no-op**: `cli.update`
requires `cfg.triage.enabled and cfg.suggestions.provider in ("ollama", "claude")`, so
enabling the gate takes two fields and the config alone does not say so. And
`ranking.w_embedding` is the only field whose behaviour depends on the *install* — it does
nothing without the `embeddings` extra, so one config ranks two ways.

**No default was flipped, and that is the finding's other half.** The temptation was
`ranking.hybrid`: dependency-free, and in every headline. But NR-11 measured it as better
nDCG everywhere and a *lower* headline, with the headline cost recovered only once the
rescore ordered what the gate admits — a stage the shipped default does not run. Turning it
on for a product that gates nothing would ship the NR-11 loss. Same for `w_embedding`: the
only arm that ever touched that channel measured README embeddings as a *query* (7/48 at
top-100, median rank 46,656), which says nothing about weighting it against keywords at
digest time. **Swapping one unmeasured default for another buys nothing and destroys the
record of which one shipped.** Both are declared with their reasons, and the declared
*values* are pinned by a test, so editing 1.5 to 2.5 fails rather than inheriting the
exemption.

**Blast radius, and a defect in the checker that found it.** The same run surfaced 8
duplicate groups in a recorded top-10 — the C-12 shape at a *third* merge site, the HyDE
`collect_by_ids` path. Dated rather than counted, it is history: all 8 sit in one run
(2026-08-13T17:13Z), which predates the fix that gave that merge the shared id rule
(`cae8c88`, 2026-08-14T03:35Z) by ten hours. Collapsing the duplicates moves four cases
(−3.0, −1.0, +2.0, +2.0) and the mean by **+0.00** — the second time this defect's
corrections have cancelled exactly at the mean, and the run was never published in any case.
All three frozen pools are clean, including the two collected before the fix, which matters
more than a single run: a duplicate in a frozen pool is inherited by every arm sharing it.

Except the pool scanner reported `pool-floor` as **0 papers, 0 duplicates** — and it holds
1,250. v1 froze `[paper, score]` *pairs* after ranking; v2 freezes paper dicts before it,
and the scanner knew only v2. It read an unparsed pool as a clean one: **void, not null,
inside the audit's own blast-radius pass**, written the same week that lesson was published.
It now names the format it read and reports `PARTIAL` when the counts disagree, rather than
however many papers happened to fall out.

**Cost** $0 — no network, no LLM, no judge.

### The baseline comparison at the width that ships — and read the precision line first (2026-08-15)

```bash
uv run python evals/run_judge_eval.py --baseline cli --sources arxiv     --rr-pool 50 --rr-rerank --rr-all-time --rr-hybrid --rr-sweep --rr-finescale --rr-hyde     --rr-frozen-pool evals/.work/pool-depth --rr-window 15 < /dev/null
```

Every paired-vs-Opus number in this file was measured at a digest of 10; the product ships
15. This is the comparison at the shipped width, on the frozen pool the window and depth
experiments used. **The Opus responses are served from cache**, so the baseline is the same
agent output already judged in earlier runs — only RepoRadar's side is new.

| | RepoRadar (window 15) | Opus 4.8 baseline |
|---|---|---|
| mean net@2 | **+5.42** | +1.62 |
| shown / actionable | 196 / 174 | 48 / 45 |
| **precision** | 0.888 | **0.938** |
| net-negative repos | 1 | 1 |

**Paired +3.79, 95% CI [+2.17, +5.58], 18 w / 1 l / 5 t, sign p = 0.0001** — five times the
window-15 frozen floor of 0.74.

#### The caveat is larger than the result and goes first

**RepoRadar returns 4× as many papers at five points *lower* precision.** Opus is the more
precise system, 0.938 against 0.888. net@2 rewards each actionable paper linearly and charges
2 per dud, so 8.2 papers per case at 0.89 beats 2.0 at 0.94 by a wide margin — and widening
the digest made that sensitivity *larger*, not smaller.

Whether a maintainer would rather read eight papers at 0.89 or two at 0.94 is a question this
metric does not answer. What defends showing them is that marginal precision sits above the
2/3 breakeven net@2 itself derives; what it does not license is the sentence *"RepoRadar is
better for a reader"*. **"Beats the baseline by +3.79" and "is better to read" are different
claims and only the first is measured.**

#### Two limits, stated

* **24 of 25 cases.** `thin-lang`'s baseline returned `error_max_turns` after 13 turns — the
  thin-docs failure already documented, where the agent's turn limit binds on repositories
  whose code it has to read to learn what they are. Excluded, never scored as a zero.
* **Frozen pool, so this does not replace the live headline** (+3.80 vs +1.57, paired +2.26).
  Different provenance and a different draw; it adds a paired measurement at the shipped
  width rather than restating one.

#### A defect found by the run refusing to start

The first attempt sat at **0.0 seconds of CPU for nine minutes** on a single thread having
produced nothing. `evals/baseline.py:_run_cli` called `subprocess.run` without redirecting
stdin, so `claude -p` inherited the parent's and blocked on it forever.

Every context this benchmark actually runs in is non-interactive — nohup, CI, cron — so the
terminal that hid this in development is the exception rather than the rule. And `timeout=`
does not cover it: **a process blocked on a read it will never satisfy is indistinguishable
from one doing slow work**, which is this project's recurring failure shape in its
process-control form. It was caught by checking CPU time rather than believing "slow".
Now `stdin=subprocess.DEVNULL`, pinned by a test that reads the source.

**Cost** ~$5, 25 cases, Opus served from cache, ~125 judge verdicts already warm from the
window-15 arm.

### The floor is a property of the configuration, and widening the digest loosened it (2026-08-15)

```bash
# two reuse passes over the SAME frozen pool, at the new window
uv run python evals/noise_floor.py <reuse-1> <reuse-2>
```

MRE **0.48** was measured on 2026-08-11 with the returned set cut at 10. The cut is now 15,
and the floor is a property of the measurement configuration rather than a constant — so it
was re-measured before being cited again.

| | window 10 | **window 15** |
|---|---|---|
| residual sd (per case, per draw) | 0.61 | **0.93** |
| whole-run shift | sd 0.10 | sd 0.08 |
| **MRE, paired same session** | 0.48 | **0.74** |
| cases identical across draws | 20 / 25 | 15 / 25 |
| Jaccard, shown | — | 0.877 |

**Widening the digest made the instrument 54% less sensitive**, which inverts the usual
more-data intuition. The mechanism is direct: more papers per case is more chances for
temperature-0 jitter in the gate and the rescore to move one across the display threshold,
and each such paper is worth +1 or −2. Averaging does not help because the variance is in
the count of shown papers, not in a mean over them.

#### A correction to yesterday's verdict

The window result was printed as *"+1.24 is past the **0.48** floor"*. That floor belongs
to window-10 arms, and one of the two arms was window-15. Re-derived against the correct
floor: **+1.24 is past 0.74** — the conclusion is unchanged and the CI still excludes zero,
but the margin is **1.7× the floor, not the 2.6× reported**.

The cause is worth more than the correction. `mre_for` derived the floor from *pool
provenance* and nothing else, so it was confidently returning a value 35% too tight for
every future window-15 experiment. It was built to stop exactly this class of error in the
frozen-versus-live dimension and was silent about a second one — **a guard that is precise
about one axis and silent about another reads as authority on both.** It now keys on
`(provenance, width)`, an unrecognised width falls back to the *widest* floor measured
rather than the nearest (under-reporting a floor turns noise into a finding), and a
mixed-width comparison is read against the wider of its two arms.

#### What this re-sizes

| experiment | measured | 0.48 verdict | **0.74 verdict** |
|---|---|---|---|
| digest window 15 vs 10 | +1.24 | past | **past** |
| gate depth 50 vs 15 | +1.00 | past | **past** |
| stated-intent `blind` | +0.44 | borderline, worth re-running | **unresolvable** |
| absent-category `impute` | +0.04 | inside | inside |
| phrase queries `verified` | +0.04 | inside | inside |

Both shipped decisions survive. The one casualty is the stated-intent re-run this file had
listed as "close to decisive rather than uninformative" at the old floor — it is not, and
that plan is withdrawn rather than left standing on a number that has since moved.

**Cost** ~$3, two reuse passes, 25/25 cases, judge verdicts served from the cache the
window-15 run had already written.

### The digest window: I predicted +0.1 as a ceiling and it came in at +1.24 (2026-08-15)

```bash
uv run python evals/run_judge_eval.py --baseline none --sources arxiv     --rr-pool 50 --rr-rerank --rr-all-time --rr-hybrid --rr-sweep --rr-finescale --rr-hyde     --rr-frozen-pool evals/.work/pool-depth --rr-window 15
uv run python evals/bigram_report.py --label-field digest_window 10=<stored> 15=<new>
```

| arm | net@2 | shown | actionable | precision | net-negative |
|---|---|---|---|---|---|
| **10** (measured) | +3.72 | 141 | 125 | 0.887 | 1 |
| **15** (shipped) | **+4.96** | 196 | 172 | 0.878 | 1 |

Paired: **+1.24/case, 95% CI [+0.48, +2.08], 12+/3−/10=, sign p = 0.0352** — past the 0.48
frozen floor with the interval excluding zero. Arm valid (25/25 changed, Jaccard 0.59).

**The pre-registration below predicted ≈ +0.1 *as a ceiling*. The answer is twelve times
that**, and larger than the gate-depth gain this project shipped a default on yesterday.

#### Why the prediction was wrong

I computed two precision-by-rank curves and extrapolated from the wrong one. The **raw
window** decays 0.68 → 0.48 by rank 10, and I projected ranks 11–15 below that. The **shown**
curve — papers that clear the gate — is 0.75–1.00 and **flat**. It was in the same output,
two commands apart, and it is the one that governs net@2 because net@2 only counts shown
papers.

The mechanism is this project's own central finding turned against my reasoning. **The gate
is near-binary** (§6.5): within the gated set almost everything is a 2, so "ranks 11–15 have
lower `llm_score`" is technically true and nearly vacuous. The thing that actually
discriminates is the fine-scale rescore, and it is applied to every band paper regardless of
its heuristic rank. Rank ordering carries far less information than I assumed, which is the
same conclusion NR-33 reached about the heuristic ranker — I had the finding and did not
apply it.

The marginal papers bear it out: **55 added, 47 actionable — 85.5%**, against a 2/3
breakeven. Each added paper is worth **+0.56** by the metric's own arithmetic, and precision
across the whole digest barely moved (0.887 → 0.878).

#### The control worked, and I over-claimed it

Pre-registered: the 11 cases whose rank-10 paper already fails the gate cannot move, because
`rerank_by_actionability` sorts by `llm_score` and everything below a failing paper also
fails. **Nine of eleven held exactly.** Two moved — `thin-kv` +4.0 → +5.0 and `systems`
+5.0 → +3.0.

The sorting argument is fine; the claim that they "must be **identical**" was not. A frozen
pool freezes retrieval and ranking, **not the LLM stages** — the fine-scale pass re-runs, and
temperature-0 jitter moves a band paper across P = 2/3. §8.8 measured exactly this as the
residual that survives freezing (0.61 sd/case). Both movers are inside it, and `rag` shows
the same signature from the other side: shown 8 → 7 while net@2 rose +5 → +7.

#### What this does to every published number

**All of them understate the shipped system.** Every net@2 in this file was measured at
window 10; the product ships 15. The depth-50 configuration reads +3.72 here and **+4.96**
at the window users actually get. This is the exact mirror of the gate-depth finding —
there the shipped default was *worse* than the measured one, here it is *better* — and both
came from the same audit noticing the two numbers lived in different files.

#### The caveat that does not go away

net@2 has no model of reader attention beyond charging 2 for a dud, so it structurally
rewards precision-preserving expansion. That is the same bias flagged for HyDE in §8.5, and
it is doing real work in a +1.24 that comes entirely from showing more papers. What defends
the result is that the marginal precision (0.855) sits well above the display threshold the
metric *derives* (P ≥ 2/3, from 3p − 2 = 0) — by the system's own decision rule those papers
should be shown. A reader who wants a shorter digest wants a different λ, and nothing here
measures reading time.

#### Decision, per the rule fixed before the run

The rule said: *delta past the floor and positive → keep 15, and move the benchmark to 15.*

* **`output.top_n` stays at 15.** The question that prompted this — should we default to 10
  because 15 is unmeasured and probably marginal — is answered no on the measurement.
* **The benchmark's `--rr-window` default moves 10 → 15**, so future headlines describe what
  ships. This **breaks comparability** with every run recorded before today, exactly as the
  `--rr-bigrams` default change did on 2026-08-12; runs carry `digest_window` so the
  report refuses to compare across it.

Logged as the reversal of my own prediction, and as **NR-35**'s inverse: a shipped default
that was *better* than the measured one, found by the same configuration audit.

**Cost** ~$6, one arm, 25/25 cases, ~125 new judge verdicts (the cache covered ranks 1–10).

---

#### The pre-registration, as committed before the run

### PRE-REGISTERED — should the digest show 15 papers or 10? (2026-08-15)

```bash
uv run python evals/run_judge_eval.py --baseline none --sources arxiv     --rr-pool 50 --rr-rerank --rr-all-time --rr-hybrid --rr-sweep --rr-finescale --rr-hyde     --rr-frozen-pool evals/.work/pool-depth --rr-window 15
uv run python evals/bigram_report.py --label-field digest_window 10=<stored> 15=<new>
```

The last declared product/benchmark divergence: `output.top_n` ships at **15** while every
published number was measured at **10**. Unlike the gate-depth question, this one has an
expected effect that can be computed *before* spending, and it was.

**What the recorded runs already say ($0, from the depth-50 arm).** The digest shows 5.64 of
10 papers per case. A wider window can only help where **all ten window papers clear the
gate** — `rerank_by_actionability` sorts by `llm_score`, so if the rank-10 paper fails, every
paper below it fails too. That holds for **14 of 25 cases**; the other 11 are structurally
incapable of gaining a single paper. Among *shown* papers precision holds 0.75–1.00 at every
rank, but the raw window decays 0.68 → 0.48 and the rank-10 `llm_score` mix is 0:2 / 1:9 /
2:14 / **3:0**. Ranks 11–15 sit below that by construction.

**Prediction: ≈ +0.1 net@2/case, and that is a CEILING** — it assumes the gate-pass rate and
the marginal precision hold past the last observed rank, and both must decay. Against the
frozen floor of **0.48**, the expected effect is a quarter of what the benchmark can resolve.

**So why run it at all.** Because the alternative was an extrapolation past the last judged
paper, and 70-odd new verdicts turn a projection into an observation. The 11 structurally
incapable cases are a built-in control: their net@2 must be **identical**, and if any moves,
the sorting argument above is wrong.

**The decision rule, fixed now so the result cannot pick it.**

* **|delta| inside the floor** (expected): net@2 cannot justify either value. The tie-break
  is then a stated product judgement rather than a measurement — and the judgement is to
  **default to 10**, on the ground that it is the configuration every published number in
  this file describes, not on the ground that it scores better. Recorded as an alignment,
  not a finding.
* **delta past the floor and positive**: keep 15, and move the *benchmark* to 15 — the
  divergence closes the other way, and every future headline is measured on what ships.
* **delta past the floor and negative**: default to 10 on the measurement.

**Estimated cost** ~$4: 1,250 Haiku gate calls, fine-scale on the widened band, and ~70 new
judge verdicts (the cache covers ranks 1–10 already).

### Three rules for "same paper", and a guard that had been looking in five files (2026-08-15)

```bash
uv run python evals/audit_product_divergence.py     # the id survey now covers 102 modules
uv run pytest tests/test_paper_id.py                # and the enforcing guard with it
```

The lesson from the divergence audit — *a guard scoped to the site you found the bug at is
a guard against finding it again* — turned out to describe the guard itself. It listed
**five pipeline modules**, the files C-9 and C-12 happened to live in. A sweep across
everything else found **three competing rules** for one invariant:

| rule | where | what it does to a non-arXiv id |
|---|---|---|
| `dedup_id` | the shared one | leaves it alone (anchored against both arXiv id eras) |
| `arxiv_id.split("v")[0]` | 6 product sites, 14 eval modules | truncates at the **first** lowercase `v`: `solv-int/9801001` → `sol`, `dblp:conf/vldb/X` → `dblp:conf/` |
| `re.sub(r"v\d+$", "", …)` | `signals/hn.py`, `signals/integrity.py` | survives the above, then edits any opaque id that merely *ends* version-shaped |

**All three agree on `2401.12345v2`**, which is why nobody noticed: every rule is correct on
the ids anyone actually looks at. Nothing but a survey was going to find this.

#### It was found while writing something else, again

The OpenAlex probe needed to dedup against the arXiv pool, and `s2_yield.py` — the probe it
was modelled on — turned out to carry its own `_dedup_id` copy. That is three for three:
C-9, C-12 and this were all found sideways, by someone working on a different problem.

#### Consolidated, and the shared rule moved house

Every site now calls `reporadar.paper_id.dedup_id`: **71 calls across 27 files, 0
hand-rolled.** The eight product modules were the ones that mattered — the MCP server
matching a user-typed id against stored ones, the citation graph keying its nodes, two
signal collectors, three source adapters — none of which the old guard had ever opened.

The rule moved out of `collector.py` into its own module, and the reason is measured rather
than aesthetic: **importing `collector` costs ~1.9 s and pulls in 1,250 modules** (it
imports the arXiv client), against **17 ms and 62 modules** for a module that imports only
`re`. Eight callers would have paid the former to normalise a string, and the lazily-imported
ones would have stopped being lazy. *A shared rule nobody can afford to import grows local
copies again* — which is how this started, since `dedup_id` was originally placed beside
`to_plain_keywords` specifically so it could not drift.

#### Two tests changed their fixtures, and that is the finding in miniature

`test_citation_graph.py` seeded ids like `2401.2v1` — one digit after the dot, a shape no
arXiv paper has. They passed under `split("v")[0]`, which will truncate anything, and fail
under an anchored rule. The fixtures were shorthand, and the shorthand was quietly the only
thing keeping the loose rule looking correct. Rewritten to real ids, with the reason noted
in place.

#### The guard now looks everywhere, and says what it deliberately skips

`tests/test_paper_id.py` reads **all 102 modules** in `src/reporadar/` and `evals/` for any
of the three rules, and asserts the twelve modules that need it still call it — because
forbidding copies passes happily on a file that deleted the call, which is exactly how
`to_plain_keywords` sat correct and unused through C-9. One exemption is declared:
`audit_product_divergence.py` keeps the old rule on purpose, since its blast-radius pass
reports the ids on which the two **disagree** and cannot do that with only one of them.
Exemptions are declared by their exact source line rather than by line number, so an edit
elsewhere in the file cannot silently widen the hole, and a stale one fails.

**No behaviour changed for any real arXiv id**, which is the honest summary: the
consolidation removes a class of latent disagreement rather than fixing an observed wrong
number. The old-style ids where the rules genuinely differ are 5 of 1,271 in the recorded
corpus, and none of them has yet duplicated.

Logged as **C-14b**. **Cost** $0 — no run, no LLM. Gate green at **1,687 tests** (was 1,578).

### The fine-scale stage stops paying for papers the digest cannot show (2026-08-14)

```bash
uv run python evals/audit_product_divergence.py     # the wiring pass now covers this too
```

Recorded as a follow-up in the gate-depth entry below and now done. `cli.update` built the
fine-scale band over **every** triaged paper, while `digest.categorize_papers` drops
withdrawn papers and then cuts to `output.top_n`. Anything outside that window reaches no
tier, so the product was buying gpt-4o-mini calls for papers it would never display —
cheap at `triage.top_k` 15, and **tripled** when the depth experiment moved the default
to 50 the same day.

**Outcome-neutral, which is exactly why it survived.** A paper outside the window reached
no tier before and reaches none now; nothing a user sees changes. Only the bill did, and a
bill is not something the benchmark measures.

#### The fix is a shared window, not a second one

The obvious repair — compute the window inside `cli.update` — would have been the fourth
instance of this project's most expensive shape. The rule has a subtlety a re-derivation
gets wrong: **withdrawn papers leave *before* the cut**, so each one pulls the next paper
up into the window. A copy that filtered after the cut would silently shrink the window by
one per withdrawal, and the two implementations would disagree *only* on runs where a
retraction landed in the top slots — the rarest and least testable case there is.

So `digest.digest_window` is now the one home, and both callers use it. `cli.update` feeds
it `store.get_scores_for_run(run_id)` — the exact list `rr digest` reads, same ordering,
same joins — rather than rebuilding the digest's input from the in-memory `scores`, which
would have been the same defect one level up.

The audit's wiring pass covers it: it counts the callers that share the helper and flags
any file that re-derives the drop-withdrawn-then-cut rule, mutation-verified against a
plausible copy planted in `cli.py`. It checks **both** halves — copies forbidden *and*
callers present — because a guard that only forbids copies passes happily on a file that
deleted the call entirely, which is precisely how `to_plain_keywords` sat correct and
unused through C-9.

#### One residual, pinned rather than discovered later

`rr update` has no `--since`, so it scopes the band to the unfiltered top-`output.top_n`. A
later `rr digest --since 7` removes papers from that window, and removal **promotes**
whatever sat below it. A band paper promoted that way has no `finescale_p` and so reaches
Maybe rather than Top Picks, where before it would have carried a score.

The direction is conservative and it is the same rule C-13 established for ungated papers —
unproven is not endorsed — but it is a real reduction in Top Picks for since-filtered
digests, and it is now a test rather than a surprise. Removing it would mean either scoring
the whole triaged set again (the waste this change exists to delete) or inventing a margin
constant with nothing to derive it from.

**Cost** $0 — no run, no LLM. Nineteen tests, three mutation-verified. Gate green at 1,578.

### Gate depth: the shipped default was never measured, and it was costing +1.00 net@2 per case (2026-08-14)

```bash
# one live seeding pass fills the pool; three arms then read it
for DEPTH in 15 25 50; do
  uv run python evals/run_judge_eval.py --baseline none --sources arxiv \
      --rr-rerank --rr-all-time --rr-hybrid --rr-sweep --rr-finescale --rr-hyde \
      --rr-frozen-pool evals/.work/pool-depth --rr-pool $DEPTH
done
uv run python evals/bigram_report.py --label-field gate_depth 15=<A> 25=<B> 50=<C>
```

**The pre-registration below was committed before the run** (`742834e`), including the
alarm that pointed at us. The prediction held.

#### The result

| arm | net@2 | shown | actionable | precision | abstained | net-negative |
|---|---|---|---|---|---|---|
| **15** (was shipped) | +2.72 | 101 | 90 | 0.891 | 6 | 2 |
| 25 | +3.16 | 124 | 109 | 0.879 | 6 | 0 |
| **50** (was measured) | **+3.72** | 141 | 125 | 0.887 | 4 | 1 |

| paired vs `15` | n | mean | 95% CI | sign | p | |
|---|---|---|---|---|---|---|
| `25` | 25 | +0.44 | [−0.24, +1.20] | 8+/4−/13= | 0.39 | inside the 0.48 floor |
| **`50`** | 25 | **+1.00** | **[+0.12, +1.92]** | 12+/4−/9= | 0.077 | **past the floor, CI excludes 0** |
| `50`, excluding controls | 22 | **+1.23** | [+0.27, +2.27] | 12+/3−/7= | **0.035** | |
| `50`, controls only | 3 | −0.67 | [−2.00, +0.00] | 0+/1−/2= | 1.00 | |

Arms **VALID**: 21/25 and 23/25 cases changed their returned top-10, mean Jaccard 0.68 and
0.48. Nothing here is the void that has cost this project three findings.

**The gate-free check I pre-committed agrees, and this time it moves.** Actionable papers
reaching the *returned* top-10, before the digest gate: **5.00 → 5.76 → 6.52** per case.
In the last ranking experiment that same measure moved 0.00 and correctly predicted a null
for $0; here it rises 30% monotonically with depth. A cheap check earns its keep by being
right in both directions.

**Precision is flat while the shown set grows 40%** — 0.891 / 0.879 / 0.887 against 101 →
141 papers shown and 90 → 125 actionable. That is the §8.5 signature: the rescore orders
what a wider gate admits, so more candidates convert instead of diluting. It is the same
mechanism that made HyDE worth +1.36 where the identical expansion had been a wash before
the rescore existed, and it is why the pre-rescore depth evidence had to be discarded rather
than trusted.

#### One blemish, named rather than buried

`webdev` (Flask, a negative control) goes **0.0 → 0.0 → −2.0**: at depth 50 the gate admitted
one paper the judge scored 0 on a repository whose correct output is nothing. That is
precisely the alarm the S2 experiment pre-registered, firing here. Three things bound it —
it is one case, in one draw, and C-7 established that *which* repository goes net-negative
is a per-draw property — but the direction is real: a deeper gate on a repository with no
applicable literature has more chances to be wrong, and the controls-only delta is −0.67.
The 22 real cases pay for it three times over (+1.23, p = 0.035), so the trade is worth
making and the cost is stated rather than averaged away.

#### Shipping decision

**`triage.top_k` moves 15 → 50.** This is the first shipped default in the project changed
on a frozen-pool measurement, and the bar it cleared is the strongest available: past the
0.48 floor, CI excluding zero, monotone across three arms, corroborated by a gate-independent
measure, with the mechanism already established elsewhere.

The cost is **~3.3× the gate calls per run** — Haiku, and the fine-scale band grows with the
triaged set — against a system currently measured at roughly $0.01 per repository versus the
agentic baseline's $0.80. Two orders of magnitude of headroom.

**Depth 25 is unresolved, not equal.** +0.44 sits just inside the floor, so the curve between
15 and 50 is not characterised; only the endpoints are.

#### A cost divergence found while reasoning about shipping, not by measurement

The product rescores **every** band paper across the whole triaged set; the benchmark
rescores only the band inside the returned top-10. At depth 15 those are close. At depth 50
the product would pay for fine-scale calls on papers `categorize_papers` discards before
tiering, since it cuts to `output.top_n` after reranking. Outcome-neutral — a paper outside
the window reaches no tier either way — but pure waste, and it grows with the default this
entry just raised.

It is **not** fixed here, and the reason is worth recording: computing the window inside
`cli.update` means duplicating the ordering `categorize_papers` performs, which drops
withdrawn papers *before* the cut. A second implementation of that ordering is the C-9 shape
this project has now paid for four times, so the fix is to share the ordering, not to
re-derive it. Logged for its own change.

Logged as the reversal of **NR-15** — 4× the candidates bought two papers across 12 cases in
2026-08-02; 3.3× the candidates buys +1.00 net@2 per case today, because the thing that
orders them did not exist then.

**Cost** ~$19: one seeding pass (51 min) plus three arms (15, 17, 34 min), 25/25 cases in
every arm, no collection failures, no HyDE degradations, no judge failures.

---

#### The pre-registration, as committed before the run

**Written before spending anything.** `triage.top_k` ships at **15**, and the
[product/benchmark audit](#looking-for-the-c-9-shape-on-purpose-five-divergences-one-of-them-a-live-product-bug-2026-08-14)
found that no experiment has ever included that value. The closest is
[NR-15](#negative-result-5--widening-the-triage-window-from-20-to-50-does-not-pay-2026-08-02),
which compared windows **20 and 50** on 12 cases — so the shipped default is *shallower than
the shallowest arm ever run*. That comparison is also doubly superseded: its treatment arm
carried a prompt change as well (`--rr-readme-context`), and it predates the fine-scale
rescore.

The mechanism matters more than the history. NR-16 closed pool depth with *"the bottleneck
is not how many papers the gate sees, it is that nothing orders what it returns."* The
rescore made that sentence false — HyDE doubled the pool and converted **+1.36** where the
identical expansion had been a measured wash a month before. Every default resting on
pre-rescore depth evidence is unsupported, and this is one of them.

**Design.** One live seeding pass, three arms reading the frozen pool, gate depth the only
variable. `rr_pool` is in `RANKING_FLAGS`, which is exactly why one pool serves all three
and the floor is **MRE 0.48** rather than 1.04. `--sources arxiv`, the shipped default.
Each run now records `gate_depth`, so the report can refuse an arm whose own file
contradicts its label.

**Prediction.** Depth 50 beats depth 15 by **more than 0.48/case**. If the rescore really
converts a wider candidate set, 3.3× the candidates at a fixed digest size of 10 should show.
*(Confirmed: +1.00/case, CI [+0.12, +1.92].)*

**Alarm, and it points at us.** If depth 50 is *worse* than 15 by more than the floor, the
shallow shipped default is vindicated and it is the **benchmark** that should change — every
headline since 2026-08-07 was measured at `--rr-pool 50`, which would then be a depth that
flatters the published numbers. Naming that before the run so it cannot be reframed after.
*(It did not fire: 50 won. The published headlines were measured at the better depth, and
what that indicts is the shipped default, not the benchmark.)*

**Validity checks, pre-committed.** 25/25 cases must change their returned top-10 between
arms — identical output is VOID, not null, which has cost this project three findings. And
the gate-free measure (actionable papers reaching the returned top-10) is reported beside
net@2, because it moved 0.00 in the last ranking experiment and would have predicted that
null for $0.

**Estimated cost** ~$15–20: 2,250 Haiku gate calls, plus judge verdicts on whatever the
deeper arms newly surface (cached across arms), plus the seeding pass. *(Actual: ~$19.)*

### OpenAlex, the last unmeasured channel: it delivers, and it places where placing is worthless (2026-08-14)

```bash
uv run python evals/openalex_yield.py                                    # $0, ~25 min
uv run python evals/openalex_yield.py --from-json evals/.work/openalex_yield.json
```

OpenAlex was the one source of five with **no measurement of any kind** — DBLP was caught
returning nothing, bioRxiv returning everything, IACR measured at n = 2, Semantic Scholar
measured three times, and every statement about OpenAlex was about an adapter. It also
spent six months on the malformed bridge query (C-9), so nothing predating 2026-08-12 says
anything either. This is the $0 stage-1 probe that comes before proposing a ~$25 A/B.

**25 of 25 cases measured, no refusals, and zero arXiv requests** — 174 cache hits, the
whole sweep served from the response cache built after the volume throttle.

#### The channel delivers, generously

| | per case |
|---|---|
| OpenAlex papers arriving | **229.8** |
| new after dedup against the arXiv pool | 229.7 |
| of those, non-arXiv (`oa:` ids) | **229.4** |
| already in the pool under another id (title match) | 1.3 |

Comparable volume to Semantic Scholar (218.6 arriving, 174.5 non-arXiv). And the coverage
is *real*: only 32 of 5,746 papers across all 25 cases turn out to be a paper the arXiv pool
already held, arriving under an `oa:` id nothing could match it against. That check exists
because OpenAlex mints a synthetic id for any work whose DOI is not an arXiv DOI — including
the published version of a preprint — so "new by id" and "new by content" are different
claims. Here they agree.

#### And then it competes badly, in a specific way

**14 papers reach a ranked top-10, across 7 of 25 cases.** Semantic Scholar reached 73
across 16 of 23 on the same measure. But the count is not the finding — *where* they land is:

| bucket | appearances | |
|---|---|---|
| **negative controls** (`webdev` 3, `cli` 2, `http` 1) | **6** | 43% of all appearances, from 3 of 25 cases |
| **thin-pool cases** (`numerics`, arXiv pool 55 against a median of 235) | **5** | half the usual competition |
| **elsewhere, on merit** (`systems`, `linter`, `ann`, 1 each) | **3** | |

Eleven of fourteen slots are won in repositories whose correct output is *nothing*, or in
the one repository whose arXiv pool was too small to defend them. Three are won on merit,
across 25 cases, under settings chosen to flatter the source (no HyDE, ~100 further
candidates it would have to outrank; no triage rerank).

#### The verdict my own script printed was wrong

The first version tested `cases_with >= n/4`. Seven of twenty-five cleared that bar by three
quarters of a case, and it printed **"OpenAlex competes — a judged A/B is justified"** while
the deciding numbers sat unread in the table directly above it. **Counting cases *touched*
is not counting cases where touching is *good*** — the same shape as `source_ab_report.py`
printing "RESOLVED" on magnitude while its interval spanned zero.

Fixed rather than overridden by hand: the control/thin/merit split is now the computation,
the verdict reads it, and `--from-json` re-derives both at $0 from the stored run so a
correction never needs the network. Six tests pin the split, including that a thin-pooled
negative control lands in exactly one bucket — double-counting it drives `on_merit` negative
and makes the verdict read better than the data.

#### Decision

**Do not spend on the judged A/B.** Not because OpenAlex is a bad source — it supplies more
genuinely non-arXiv content per case than anything else measured here — but because a
25-case mean cannot resolve an effect concentrated in three papers, and the one comparable
channel that *did* compete broadly (S2: 73 appearances, 16 cases) was taken all the way to a
judged A/B and did not help. `sources: [arxiv]` stays the default, and OpenAlex joins DBLP
and bioRxiv as **built, wired, never validated** — with the difference that this time we
know what it would deliver.

The $0 probe has now paid for itself three times: it would have caught DBLP before four
attempts to benchmark it, it sized the S2 experiment, and it has just declined a ~$25 one.

#### A defect noted, not fixed

`openalex.search_papers` returns `[]` both when the API refused and when it honestly found
nothing, so the product cannot tell a throttled fetch from an empty one — the
failure-is-absence class that has cost this project two published numbers. The probe works
around it by wrapping the adapter's own request function and counting `None` returns, and
reports any case with a refusal as **UNMEASURED** rather than zero. The adapter itself
should report failure structurally; that is a separate change.

Logged as **NR-34**. **Cost** $0 — 125 OpenAlex requests, 0 arXiv requests, no LLM, ~25 min.

### Looking for the C-9 shape on purpose: five divergences, one of them a live product bug (2026-08-14)

```bash
uv run python evals/audit_product_divergence.py     # $0, no network, no LLM
```

C-9 (the query bridge, hand-rolled at five call sites) and C-12 (the version-strip before a
source merge, fixed in `cli.py` and not in `evals/harness.py`) are the same defect: **one
invariant, two implementations, one of them fixed**. Both were found by accident, months
apart, while looking for something else. This is the pass that looks for the shape on
purpose, across every module in the collect → rank → gate → show pipeline on both sides of
the product/benchmark line. It is free, and it found five.

#### 1. The gate's own docstring described a rule the product does not implement

`triage.triage_papers` omits a paper whose scoring failed, and says why: *"downstream
tiering treats 'couldn't judge' as 'not a confident Top Pick', not as a confident
rejection."* That is what the benchmark does. It is **not** what `digest.categorize_papers`
did — a paper with no `llm_score` fell through to the heuristic `score_total >= 0.5`
threshold and could reach Top Picks on it.

That threshold is the one Feature 6 was built to replace, at mean net@2 **−11** on the
user-facing output. So the fall-back path promoted ungated papers using the single worst
selection rule this project has measured, and it fires in two situations:

* **`output.top_n > triage.top_k`** — the digest window is wider than the gate's. Every
  paper past rank `top_k` arrives ungated and is tiered heuristically, rendered
  indistinguishable from papers an LLM endorsed. Both default to 15, so a stock install is
  safe; raising `output.top_n` alone silently turns the gate off for the tail.
* **a failed gate call** — one paper, same promotion.

Fixed: when the gate ran, an unscored paper reaches **Maybe**, never Top Picks. With
`triage_threshold=None` (the gate never ran) the heuristic remains the only rule, which a
test now pins — demoting everything there would empty Top Picks for every user with triage
off.

**Blast radius on published numbers: zero, and checkably so.** Across **87 recorded runs and
6,420 returned top-10 papers**, `llm_score` is *present-and-null* — the gate ran and failed
— exactly **0 times**. The benchmark has always scored the strict rule and never had cause
to exercise the other one. The divergence is real, the correction is to the product, and no
result moves.

#### 2. A third implementation of "is this the same paper"

`collector.dedup_id` was the shared rule after C-12. It was not the only one: a bare
`arxiv_id.split("v")[0]` was doing the same job at **8 further call sites** — the HyDE merge
in *both* `cli.py` and `run_judge_eval.py`, the benchmark's judge pool, and its verdict
lookups. The two rules agree on modern ids, which is why the split survived C-12 untouched.

They disagree on old-style ids, and the disagreement is not hypothetical: **5 old-style ids
sit in this project's judged pools** (`cs/0602007v4`, `cs/0007008v1`, …). `dedup_id` left
their versions on; `split("v")[0]` stripped them. So arXiv's `cs/0602007v4` beside Semantic
Scholar's `cs/0602007` would have survived a source merge as two papers — C-12 exactly,
still live — while the judge pool five steps later collapsed them into one. No duplicate
pair has actually occurred yet; the audit found the loaded gun, not a wound.

The split rule is also unsafe in a way `dedup_id` is not: it truncates at *any* lowercase
`v`. `solv-int/9801001` becomes `sol`, and `ss:vector-db-7` becomes `ss:`.

Fixed by making the shared helper strictly better than the copy it replaces — `dedup_id` now
handles both id eras, anchored so no synthetic `ss:`/`dblp:`/`iacr:` id can be truncated —
and routing all 8 sites through it. A consolidation only sticks when the survivor is better.

#### 3. C-12, unfixed at a third call site

`evals/run_eval.py` — the Tier A/S runner — still merged OpenAlex and Semantic Scholar on
raw ids. The guard written when C-12 was found reads `evals/harness.py` **by name**, so it
was green while a runner two files away had the identical bug. This is the C-9 pattern
recursing one level: a fix applied to the sites we were looking at, and a guard scoped to
the file we were looking at.

The replacement guard is parameterised over every pipeline module and is mutation-verified
against the exact defective state.

#### 4. Two eval runners, opposite failure policies

`evals/harness.py` raises on `CollectionError`, with a comment naming the reason: scoring a
throttled arXiv fetch as an honest zero once supplied **−17 of a −21 delta** (C-4).
`evals/run_eval.py` printed a warning and carried on. With arXiv alone that yields an empty
pool and the case is skipped — survivable. With a second source enabled it is worse than
C-4: the case runs on the non-arXiv half of its pool and prints a domain-purity number that
looks like every other one. Now it raises, and `run_live` skips the case out loud.

#### 5. Two configuration fields where the benchmark is not measuring the default

| field | shipped | measured |
|---|---|---|
| `triage.top_k` | **15** | **50** (`--rr-pool 50`) |
| `output.top_n` | **15** | **10** (the harness cuts the returned set at 10) |

Neither is wrong and both are on purpose — the benchmark holds digest size fixed to test
selection rather than volume, and depth 50 came out of NR-15/NR-16. But `arxiv.lookback_days`
shipped at 14 days for a month while every headline was measured all-time, and nobody was
lying then either: the two numbers simply lived in different files. So the audit now carries
the benchmark's headline configuration next to the product defaults, with a written reason
required for each difference, and a test fails on any **undeclared** one. Note what this
implies about depth: the shipped gate is *shallower* than anything measured since the
fine-scale rescore made a deeper pool convert ([HyDE end to
end](#hyde-measured-end-to-end-the-first-result-that-clears-p--005-against-the-baseline-2026-08-09))
— worth revisiting on evidence, not here.

#### What the audit deliberately cannot see

Stated because a clean report is the dangerous kind. The benchmark never runs the withdrawal
/ integrity stage, so a retracted paper reaching the top-10 is judged and scored where the
product would mute it before the window. Product-only stages are invisible to it by
construction, and no amount of static analysis changes that.

Logged as **C-13** (the ungated-paper promotion), **C-14** (a third id normaliser, and
`dedup_id` too narrow for old-style ids), **C-12b** (the third raw-id merge), and **C-15**
(the Tier A runner degrading where Tier B raises). Gate green at **1,543 tests**.

**Cost** $0 — no network, no LLM, ~2 seconds.

### The absent-category bias is real, changes retrieval a lot, and changes nothing — and it undermines yesterday's S2 result (2026-08-13)

```bash
# one live seeding pass fills the pool; three arms then read it
for MODE in omit zero impute; do
  uv run python evals/run_judge_eval.py --baseline none --sources arxiv,semantic_scholar \
      --rr-pool 50 --rr-rerank --rr-all-time --rr-hybrid --rr-sweep --rr-finescale --rr-hyde \
      --rr-frozen-pool evals/.work/pool-abscat --rr-absent-category $MODE
done
uv run python evals/bigram_report.py --label-field absent_category omit=<A> zero=<B> impute=<C>
```

`ranker.score_paper` drops `w_category` when a paper has no categories — every paper from
every non-arXiv source. Writing the two totals out, `(kw + w·cat)/(1+w)` against `kw`, the
uncategorised paper wins **iff `kw > cat`**: not the flat advantage first claimed here (a
paper in exactly the right category still wins, 0.893 to 0.840), but the common case in a
real pool, where category matches are partial or absent.

The frozen pool made this cheap and sensitive: **MRE 0.48 instead of 1.04**, one live
seeding pass and three arms that collect nothing. Arms **VALID** — 25/25 cases changed
their returned top-10, mean Jaccard 0.66 and 0.69.

#### The flag does a great deal to what is retrieved

| arm | non-arXiv papers in the ranked top-10 |
|---|---|
| `omit` (shipped) | **32** |
| `zero` | **14** |
| `impute` | **39** |

`zero` more than halves them. This is not a subtle reordering.

#### And nothing at all to what comes out

| arm | net@2 | shown | actionable | precision | net-negative |
|---|---|---|---|---|---|
| `omit` | +4.04 | 149 | 133 | 0.893 | 1 |
| `zero` | +4.04 | 146 | 131 | 0.897 | 0 |
| `impute` | +4.08 | 147 | 132 | 0.898 | 1 |

| paired vs `omit` | mean | 95% CI | sign | p |
|---|---|---|---|---|
| `zero` | **+0.00** | [−0.72, +0.76] | 6+/6−/13= | 1.00 |
| `impute` | **+0.04** | [−0.40, +0.48] | 5+/6−/14= | 1.00 |

A gate-free measure agrees: actionable papers reaching the ranked top-10 are **6.60 / 6.60
/ 6.36** per case. So this is not the triage gate absorbing a ranking change — the change
never mattered.

**My hypothesis is refuted, and the refutation is more informative than the fix would have
been.** The bias is real, it moves 18 of 32 non-arXiv papers out of the top-10, and the
papers that replace them are **neither better nor worse**. Non-arXiv and arXiv candidates
at that rank are interchangeable in actionability.

#### Which undermines yesterday's displacement story

The same `+s2` configuration, two draws:

| | net@2 | precision | net-negative |
|---|---|---|---|
| arXiv only (live, 2026-08-13) | +4.12 | 0.908 | 0 |
| **+s2 (live, 2026-08-13)** | **+3.24** | 0.854 | 2 |
| **+s2 (frozen, same day, later)** | **+4.04** | 0.893 | 1 |

**+0.80/case apart, for the same configuration.** And the second draw lands 0.08 from the
arXiv-only arm — inside any floor this benchmark has.

Two known differences beyond the draw itself: the C-12 dedup fix landed in between (the
naive post-hoc correction cancelled at the mean, but the real fix changes the *pool*, and
that second-order effect could not be computed after the fact), and today's arms are frozen
where yesterday's were live — which changes variance, not expectation, but is not a clean
paired comparison either.

**So NR-32's −1.05 is a single draw, and it does not reproduce.** The displacement
mechanism it proposed is independently undercut by the result above: displacing S2 papers
back out, which `zero` does at scale, buys nothing. Recorded as a correction to NR-32
rather than a replacement — settling it needs paired live arms in one session, which is
what NR-32 was and which one draw cannot establish.

The shipping decision is unchanged and its basis is now weaker still: `sources: [arxiv]`
stays the default because **S2 has not been shown to help**, not because it was shown to
hurt.

#### What this says about ranking work generally

Two ranking policies that produce visibly different top-10s produce statistically identical
output. Anything downstream of the heuristic ranker — the triage gate, the fine-scale
rescore, the 2/3 threshold — is doing the work that decides quality. **A heuristic-ranker
change now has to clear a high bar to be worth measuring**, and the cheap way to check
before spending is the gate-free actionable-in-top-10 count, which moved 0.00 and −0.24
here.

`ranking.absent_category` ships defaulting to `omit`, unchanged: nothing measured justifies
moving it, and `zero`'s marginally better precision (0.897 vs 0.893) and zero net-negative
repos are both inside noise.

Logged as **NR-33** (the absent-category bias is real, large in effect on composition, null
on outcomes) and as a **correction to NR-32**.

**Cost** ~$14, one seeding pass (2 h 10 m, S2-throttle-bound) plus three arms at ~35 min
each, all 25 cases in every arm, no failures.

### S2 measured at last: it does not help, it displaces — and the controls answer a different question (2026-08-13)

```bash
for SOURCES in arxiv arxiv,semantic_scholar; do
  uv run python evals/run_judge_eval.py --baseline none --sources $SOURCES \
      --rr-pool 50 --rr-rerank --rr-all-time --rr-hybrid --rr-sweep --rr-finescale --rr-hyde
done
uv run python evals/source_ab_report.py arxiv=<A> +s2=<B>
```

The experiment finding 3 only appeared to run. Two arms, 25 cases each, same session, arm A
collecting arXiv live and arm B serving it from the cache. **Both arms complete — no case
lost**, unlike the probe that preceded the cache.

> **Correction (later the same day): −1.05 is one draw and it does not reproduce.** A
> second draw of this configuration scored **+4.04**, 0.08 from the arXiv-only arm. The
> displacement mechanism proposed below is separately undercut: halving the non-arXiv
> papers in the top-10 (`--rr-absent-category zero`) changes net@2 by **+0.00**. See *The
> absent-category bias* above. The recommendation survives; the evidence for it does not.

**Arm VALID, emphatically**: 122 papers returned by the treatment that the control never
returned, across **25/25** cases. Nothing here is the void that finding 3 was.

| arm | net@2 | shown | actionable | precision | abstained | net-negative |
|---|---|---|---|---|---|---|
| `arxiv` | **+4.12** | 142 | 129 | **0.908** | 5 | **0** |
| `+s2` | **+3.24** | 144 | 123 | **0.854** | 4 | **2** |

| paired | n | mean | 95% CI | sign | p |
|---|---|---|---|---|---|
| all cases | 25 | **−0.88** | [−1.84, +0.04] | 5+/10−/10= | 0.30 |
| **excluding controls** | 22 | **−1.05** | [−2.14, +0.00] | 4+/10−/8= | 0.18 |
| controls only | 3 | +0.33 | [+0.00, +1.00] | 1+/0−/2= | 1.00 |

**−1.05 on the 22 real cases is past the 1.04 floor, and its interval still touches zero.**
Those are two different claims and the first version of `source_ab_report.py` conflated
them, printing "RESOLVED" on magnitude alone — and calling `[−2.14, +0.00]` an interval that
excludes zero, because its containment test was really a sign-agreement test. The honest
reading: **big enough to see, not yet established**, and pointing down.

#### The mechanism is displacement, not addition

`thin-kv` shows it cleanly: **8 papers shown in both arms**, actionable 8 → 6. S2 did not add
noise beside the good papers, it **pushed two of them out of the top-10**. `llminfer` is the
extreme (−7.0 raw): S2 supplied five quantization papers, four judged **1** — topically
exact, not actionable. The register mismatch again, now with a fuller candidate pool to
express itself through.

#### A dedup bug found by reading the shown lists

The treatment showed **6 duplicate papers across 4 cases**; the control showed **none**. The
ids give it away — `2605.23815v1` beside `2605.23815`. `cli.py` version-strips before merging
a non-arXiv source; **`evals/harness.py` merged on the raw id**, so arXiv's versioned copy and
S2's unversioned copy both survived. Same shape as C-9: one invariant, two implementations,
one of them fixed. `dedup_id` now lives in `collector.py` beside `to_plain_keywords`, and a
test reads `harness.py` to assert no merge is left on raw equality.

**It does not change the headline.** Removing each duplicate's contribution moves four cases
(`storage` 0.0 → −3.0, `compiler` +4.0 → +3.0, `llminfer` −7.0 → −5.0, `vectordb` −3.0 → −1.0)
and the corrections cancel: the mean stays −0.88 and −1.05 exactly, with the interval
tightening to [−2.00, −0.14]. Per-case results were contaminated; the conclusion was not.

#### The negative controls: the premise was half right

The question was whether `gold_n: 0` means "no research could help this repo" or merely "no
gold *arXiv* papers". Tier B never reads the label, so the judge answered it directly, on the
17 papers S2 added to those three cases:

| case | added | judge scores | net@2 |
|---|---|---|---|
| `webdev` | 7 | **1 × 7** | 0.0 → 0.0 |
| `http` | 8 | 0×2, 1×3, **2×3** | **0.0 → +1.0** |
| `cli` | 2 | 1×1, **2×1** | 0.0 → 0.0 |

**4 actionable, 13 loose.** So: *partially vindicated*. `webdev` is a real negative control —
seven papers on-topic enough to retrieve, every one judged "no concrete actionable
improvement". But `http` is not: three papers judged 2, and one reached the digest —
*PyTrim: A Practical Tool for Reducing Python Dependency Bloat*, which is a genuinely
plausible change for `requests`.

The label is not uniformly wrong, and it is not uniformly right. It encodes arXiv coverage,
and for one of the three repos that understates what exists. But the density is the point:
**4 of 17 added papers (24%) against a 0.854 pooled precision** — the controls are where S2's
papers are *least* likely to be useful, not most.

#### The alarm I pre-registered did not fire, and I had picked the wrong indicators

I said: alarm if a negative control goes net-negative, or pooled precision drops below 0.85.
Precision landed at **0.854** and no control went negative — so by the letter, no alarm. Yet
**two ordinary repos went net-negative** (`llminfer`, `numerics`) where the control arm had
none, and the effect on real cases is −1.05. I had aimed the alarm at the controls because
that is where the stage-1 probe found S2 concentrated; the damage landed on the repos that
have genuine literature, where S2's papers compete with *better* papers rather than with
nothing.

#### Shipping decision

**`sources: [arxiv]` stays the default — now on measured grounds rather than a void.** The
recommendation is unchanged from finding 3 and its basis is completely different: not "S2
adds competitive junk" (that arm returned nothing), but "S2 returns 122 real papers and
displaces better ones".

This does not condemn the source. The stage-1 probe showed ~175 non-arXiv papers per case
arriving, and `compiler` (+4.0), `graph` (+2.0) and `rag` (+2.0) gained. What is refuted is
**adding S2 to the pool undifferentiated**. The plausible next move is a ranker change —
uncategorised papers currently escape `w_category` under the absent-signal rule, which is
exactly the advantage that lets a loose S2 paper outrank a good arXiv one — not a bigger
source list.

Logged as **NR-32** (S2 measured, −1.05 on real cases, displacement not addition) and
**C-12** (the harness merged non-arXiv sources on raw ids).

**Cost** ~$18, two arms, ~3 h 15 m, all 25 cases in both.

### The arXiv throttle was volume, not rate — and the negative-control premise needs testing (2026-08-12)

#### Why we were throttled while obeying the rate limit

The S2 yield probe lost its last two cases to an arXiv throttle, which looked like a
politeness failure. It was not one. The rate limiter is correct and was applied throughout:
`_query_with_retry` takes a turn at the process-wide 3-second gate before **every** attempt,
retries included; `_shared_client` is reused for the life of the process with its own
`delay_seconds` set from the same interval; `page_size=100` against `max_results=50` means
one HTTP request per query, with no hidden pagination.

**The evidence that it was volume:** the failures arrived at cases **24 and 25**, after ~162
requests in that run had already succeeded. A rate violation fails early and uniformly; a
volume ceiling fails late. The day's cumulative total from this machine:

| run | arXiv requests |
|---|---|
| three phrase-query arms (3 × 174 queries + ~75 HyDE id lookups) | ~597 |
| S2 yield probe, before it was cut off | ~162 |
| **total** | **~760** |

arXiv's published guidance is a *rate*. `export.arxiv.org` additionally protects against
sustained bulk usage, and a 3-second gate cannot express "and no more than N per day".

#### The waste was total, and it is now fixed

A 25-case sweep is **174 queries**, and they are byte-identical between runs — same repos,
same profiles, same `build_queries` output. The three arms fetched the same pool three
times; the probe fetched it a fourth. **~700 requests where ~174 would do.**

`reporadar/arxiv_cache.py` caches responses keyed on query, `max_results` and `sort_by`.
`lookback_days` is deliberately **not** in the key: it filters results after the fetch
rather than changing the request, so one stored all-time response serves any window.
Measured on `rag`:

| | papers | time | arXiv requests |
|---|---|---|---|
| cold | 150 | 12.2 s | 5 |
| **warm** | **150** | **0.1 s** | **0** |

**Off unless asked.** `evals/harness.py` enables it; the product does not, because serving a
six-hour-old answer to a daily digest is a behaviour change nobody measured. Same reasoning
as `--rr-frozen-pool`: reuse is a deliberate, labelled act.

One design point worth recording, because the first version got it wrong. The cache
initially refused to store *any* empty result, on the rule that "arXiv found nothing" and
"arXiv refused" are the same bytes on disk — the mistake that once cached seven pools of
429-storm zeros as honest measurements. But `_query_with_retry` **raises** rather than
returning `[]` when it exhausts its retries, so an empty list from it is an answer arXiv
actually gave. The blanket rule was re-fetching 2 of `rag`'s 5 queries on every run to guard
against a failure that cannot reach it from there. Now `put(..., empty_is_real=True)` states
the guarantee at the call site that can prove it, and defaults to the safe behaviour for any
caller that cannot.

#### Are the negative controls really supposed to return nothing?

Raised as a challenge to the benchmark's premise, and it is a fair one that changes the S2
A/B. `webdev` (Flask), `cli` (click) and `http` (requests) are labelled negative controls on
the reasoning that they have "almost no arXiv research overlap" — and **`gold_n: 0` encodes
"no gold *arXiv* papers", which is a claim about coverage, not about whether research that
could improve these repos exists.** Plausible literature does exist for at least two of them:
TLS handshake and connection-pool policy, retry/backoff, certificate validation for
`requests`; session security, CSRF, WSGI/ASGI performance for Flask. Most of it lives in
USENIX/CCS/WWW proceedings — which S2 indexes and arXiv largely does not.

**The judged eval does not beg this question.** `negative_control` and `max_score_threshold`
are read only by `run_eval.py` (Tier A, offline fixtures). Tier B — the judged path that
produces net@2 — passes every paper to the judge with no knowledge of the label, and the
rubric asks whether it could *genuinely improve this repository*. So the judge decides on
merit.

That **corrects the prediction pre-registered in the stage-1 entry below**, which treated S2
papers reaching negative-control top-10s as presumptively an alarm:

* If those papers are genuinely useful, the judge scores them 2–3, they count as actionable,
  and net@2 on those cases goes **up**. That is S2 filling a real coverage gap, and it would
  mean the "negative control" label is arXiv-specific rather than true.
* If they are topically loose, the judge scores 0–1 and precision falls.

Both are informative and they are distinguishable — but only by reading the judge's scores
and justifications on those three cases, not by reading the mean. The A/B should report them
separately.

### S2 stage-1: the channel works, and it floods the negative controls (2026-08-12)

```bash
uv run python evals/s2_yield.py     # $0, no LLM
```

With C-9 fixed and a key in place, S2 returns papers again — so the judged A/B that finding
3 only *appeared* to run is now possible. It costs ~$26 and ~4 h. This is the $0 check that
comes first (the P4 protocol: verify every dependency before building), and it changed what
the A/B should be looking for.

**23 of 25 cases measured.** `thin-kv` and `thin-gnn` are **unmeasured, not zero** — arXiv
threw a sustained throttle and the collector reported failure after 930 s of waiting rather
than returning an empty pool. That guard exists because two cases were once scored as honest
zeros after a 429 storm.

#### The channel delivers, generously

| | per case |
|---|---|
| S2 papers arriving | 218.6 |
| new after dedup against the arXiv pool | 211.0 |
| of those, non-arXiv (`ss:` ids) | 174.5 |
| **reaching the ranked top-10** | **73 papers across 16/23 cases** |

Nothing like DBLP (zero) or IACR (six papers, two cases). S2 contributes real volume, most of
it content arXiv does not have, and it competes on rank.

#### Where those top-10 slots land is the finding

| case | arXiv pool it outranked | S2 papers in top-10 |
|---|---|---|
| **`webdev`** (negative control) | 287 | **10 / 10** |
| **`http`** (negative control) | 257 | **9 / 10** |
| `systems` | 249 | 7 |
| `numerics` | 55 | 7 |
| **`cli`** (negative control) | 250 | **3 / 10** |
| … 11 more cases | | 1–5 |
| `cv`, `rl`, `graph`, `crypto`, `db`, `linter`, `encryption` | | 0 |

**22 of the 73 appearances — 30% — come from the three negative controls, which are 3 of 23
cases.** Those are the repos defined by the benchmark as having almost no research overlap,
where the correct output is *nothing*. On `webdev`, S2 papers took **every slot in the
top-10**, outranking 287 arXiv papers on a Flask app.

#### Finding 3's mechanism was right about a world that did not exist

Finding 3 explained its (void) precision drop this way: *S2 papers carry no arXiv categories,
and the absent-category-is-not-a-zero ranker rule (correctly) stops penalising them — which
makes them more competitive and puts more weight on the triage gate to reject the
non-actionable ones.*

That reasoning is sound, and it was applied to data that did not exist, because S2 returned
nothing. **Now the data exists and the mechanism is visible**: `w_category` cannot penalise a
paper with no categories, S2 supplies ~175 uncategorised papers per case, and the queries
reaching S2 carry no category clause to constrain them either (`to_plain_keywords` strips it,
correctly — S2 has no such field). A repo with a thin or generic profile has nothing to push
them back with.

#### What this changes about the A/B — and a pre-registered prediction

This measures the **ranked top-10, before the triage gate**. The gate is precisely what is
supposed to catch this, and today it does: with S2 off, `webdev`, `cli`, `http` and `linter`
all score net@2 **0.0** — correct abstention. So the judged experiment now has a sharp
question rather than a vague one.

Pre-registered, before spending anything:

* **Prediction** — the gate holds. Negative controls stay at 0.0 and mean net@2 moves less
  than the 1.04 floor.
* **Alarm** — any negative control goes net-negative, or pooled precision drops below 0.85.
  That would mean 175 uncategorised papers per case are reaching digests, and the fix is a
  ranker change (penalise absent categories for uncategorised *sources*, rather than treating
  the signal as merely missing) — not a source decision.

The probe is **optimistic by construction**: no HyDE (~100 further candidates S2 would have
to outrank) and no triage rerank, because both cost money. Real top-10 share under the
shipped configuration is lower than these numbers.

Logged as **NR-31**. **Cost** $0, ~35 minutes, two cases lost to an arXiv throttle.

### S2 resolved: the one published number downstream of C-9 was VOID, and four modules had no rate limiter (2026-08-12)

```bash
uv run python evals/audit_query_transform.py --sources s2 \
    --cases rag,cv,rl,webdev,peft,diffusion,graph,speech,crypto,systems,cli,http
```

The C-9 audit left one thing undetermined and said so: Semantic Scholar was the only source
with a **published** number measured through the broken transform — finding 3, "adding
Semantic Scholar did not help". Two attempts to resolve it were refused by rate limiting
(keyless S2 answered 429 to all 20 requests), and the apparent zeros were discarded rather
than reported. With an API key, it resolves.

#### Before spending the key: three of four S2 modules had no rate limiter

S2's documented key limit is **1 request per second across all endpoints**, per key. Four
modules here call it, and an audit found:

| module | endpoint | rate limiting before this |
|---|---|---|
| `sources/semantic_scholar.py` | `/paper/search` | slept *between* queries in one call only |
| `sources/s2_recommendations.py` | `/recommendations/v1/papers` | **none** (retry backoff only) |
| `specter.py` | `/paper/batch` | **none** |
| `citations.py` | `/paper/batch` | **none** (retry backoff only) |

Retry backoff is not rate limiting — it fires *after* the server has been hit too fast. And
the one limiter that existed never spaced a call's first request, so an eval sweeping 25
repos opened each with an unspaced request. This is what `arxiv_rate` exists for, in its own
words: *"Two independent 3-second limiters permit two requests per three seconds."* Four
independent ones permit four per second. `s2_rate` now gates all four (`specter` imports
`citations._s2_batch_post`, so one gate covers both).

**Two defects surfaced only against the live API**, which is the argument for probing before
measuring:

- **The interval was undershot.** `time.sleep(d)` sleeps *at least* d but is not guaranteed
  to; on Windows it returned ~7 ms early, turning a 1.1 s interval into **1.093 s** gaps.
  Still inside S2's real 1.0 s limit, but a limiter that misses its own target by 7 ms would
  miss a tighter one by the same margin. `wait_turn` now sleeps toward a deadline in a loop.
  Re-probed live: **1.109–1.110 s**, no undershoot.
- **`set_min_interval(0)` did not disable the gate** while a throttle hold was pending, so
  "no rate limiting" still waited 30 s — a knob that did not do what it said, and a
  cross-test leak from whichever test mocked a 429 first.

#### The measurement

12 cases, real `build_queries` output, one query each, both transforms. Spaced at **3.0 s**
— deliberately slower than the 1 RPS floor, because S2 throttles beyond its documented limit
under load and every 429 risks a refusal being counted as a zero. **12/12 measured, no
refusals.**

| | OLD transform | NEW transform | shared |
|---|---|---|---|
| `rag` | 1 | 20 | 0 |
| the other 11 cases | **0** | **20** | **0** |

**Semantic Scholar answers the malformed query with nothing** — the same failure as DBLP and
IACR, and the third source to show it. Zero overlap in all 12.

#### What this does to finding 3

**It makes it VOID rather than null.** Finding 3 reported mean net@2 +0.83 → +0.58, precision
0.91 → 0.76, and `rl` −2.0 / `diffusion` −1.0, and explained them: *S2 papers carry no arXiv
categories, the absent-category rule stops penalising them, so they compete harder and one
non-actionable paper got through the gate.* **That mechanism requires S2 papers in the pool.
There were none.** The moves were run-to-run drift — later measured at ±0.6 at the mean and
±6 per case — attributed to a channel that never delivered a paper.

The headline is accidentally right and its reasoning is wrong. The recommendation ("leave
`sources: [arxiv]` as the default for ML repos") survives on different grounds: not *S2
hurts*, but *S2 has never been tested*.

**The limit of this measurement, stated:** S2 is being asked **today**, not in July 2026, and
its query parser may have changed. That cannot be ruled out. What makes the reading solid
anyway is that the finding's own proposed mechanism is falsified independently — it needs
papers that three sources now agree the malformed query does not return.

#### Third time for the same shape

Void-not-null has now cost this project three findings: the first IACR arm (zero papers
reached a top-10), the phrase-query arms (checked and clean, because `bigram_report` was
built to check), and now finding 3. The lesson has been paid for often enough to state
plainly: **an arm is not a measurement until you have counted whether the channel delivered
anything.** Every future source arm gets a divergence check before its delta is read.

Logged as **C-11** (no S2 rate limiter in three of four modules; the interval undershot) and
**NR-30** (finding 3 void).

**Cost** $0 — no LLM, 24 S2 requests, ~2 minutes.

### Phrase queries: the generator was broken, the repair is free, and the benchmark cannot see either (2026-08-12)

```bash
for MODE in adjacent verified none; do
  uv run python evals/run_judge_eval.py --baseline none \
      --rr-pool 50 --rr-rerank --rr-all-time --rr-hybrid \
      --rr-sweep --rr-finescale --rr-hyde --rr-bigrams $MODE
done
uv run python evals/bigram_report.py adjacent=<A> verified=<B> none=<C>
```

`build_queries` pairs each keyword with its **TF-IDF neighbour** and sends the pair as a
quoted phrase. Nothing ever required the two words to belong together — they merely scored
next to each other. It emits `"use page"` and `"page refer"` for duckdb, `"data cd"` for
redis, `"server code"` for ruff. Three of the five queries every source receives are these,
and they come first.

**The defect is not in doubt; it was measured directly.** Asking DBLP the generated queries
and the benchmark's own hand-written `gold_queries`, same repos, same day:

| case | generated query → returned | gold query → returned |
|---|---|---|
| `db` | `use page` → *Use of simulation to estimate Economic performances of two phenotypes of sows* | `vectorized query execution` → *Incremental Fusion: Unifying Compiled and Vectorized Query Execution* |
| `linter` | `server code` → *Operationalization of Machine Learning with Serverless Architecture* | `incremental program analysis` → *An Incremental Algorithm for Algebraic Program Analysis* |
| `systems` | `source data` → *NPBS database: a chemical data resource* | `log-structured storage engine` → *FlatStore: An Efficient Log-Structured Key-Value Storage Engine* |

#### The three-arm result: nothing resolves

25 cases, `--baseline none`, three arms back to back in one session (3 h 52 m, ~$26).
Both treatment arms are **valid, not void**: 25/25 cases changed their returned top-10,
mean Jaccard 0.50 and 0.47.

| arm | net@2 | shown | actionable | precision | abstained | net-negative |
|---|---|---|---|---|---|---|
| `adjacent` (control) | +4.12 | 139 | 127 | 0.914 | 4 | 2 |
| **`verified`** | **+4.16** | 137 | 126 | **0.920** | 5 | **0** |
| `none` | +3.64 | 142 | 125 | 0.880 | 5 | 1 |

| paired vs control | mean | 95% CI | sign | p | verdict |
|---|---|---|---|---|---|
| `verified` | **+0.04** | [−0.64, +0.88] | 4+/7−/14= | 0.55 | inside the 1.04 floor |
| `none` | **−0.48** | [−1.00, +0.04] | 4+/12−/9= | 0.077 | inside the floor |

A gate-independent retrieval measure — actionable papers reaching the ranked top-10 — gives
the same ordering and the same non-significance: `verified` **6.80**/case, `adjacent` 6.56,
`none` 6.28 (`verified` +0.24, p = 0.58; `none` −0.28, p = 0.18).

#### Why the benchmark is blind to a defect this obvious

**arXiv rescues the bad queries and nothing else does.** Every arXiv query carries
`AND (cat:cs.DB)`, which keeps results in the right field however meaningless the phrase, so
`"use page"` still returns database papers. `to_plain_keywords` correctly strips that clause
for keyword sources, which have no equivalent — they receive the bare phrase and answer it
literally. The benchmark runs on arXiv. **It is measuring the one channel where the bug does
not bite**, which is why a defect visible at a glance on DBLP ties three ways here.

#### What is refuted

**Deleting the phrase queries.** `none` is the worst arm on every axis: −0.48 net@2/case,
precision 0.914 → 0.880, fewest actionable papers retrieved. A meaningless phrase is still a
query returning up to 50 candidates, and dropping three of five shrinks the pool. The obvious
fix is not merely unsupported, it is measured backwards.

**My own framing of the experiment.** I proposed this as the highest-value direction
available because it touches 25 cases rather than DBLP's 10 or bioRxiv's 0. That reasoning
was about *coverage* and ignored *effect size*: a defect the category filter neutralises has
no headroom on arXiv no matter how many cases it runs on. The IACR sizing error was
comparing an effect against the ceiling; this one was assuming breadth implies power.

#### An attractive pattern I am not claiming

`verified` took both net-negative repositories to zero (`speech` −1.0 → +6.0, `thin-lang`
−2.0 → 0.0), which would read as the fix repairing the failures rather than the mean. Three
reasons not to believe it. `speech` is documented swinging +8.0 → −2.0 between runs of an
*identical* configuration. The 25-case headline run had exactly one net-negative case and it
was `numerics` — a different repo than either of today's, confirming C-7's finding that
net-negativity is a per-draw property. And `none` did **not** repair `thin-lang` (still
−2.0) despite also removing the offending phrases. Consistent with noise.

#### Shipping decision, and the one judgment call

**The default changes to `verified`, and not because it scores better.** +0.04 at p = 0.55
justifies nothing. The argument is:

1. `adjacent` demonstrably asks for phrases the repository does not contain — not a
   statistical claim, a direct observation.
2. The repair costs nothing measurable on arXiv, with the 95% CI bounding worst-case harm at
   −0.64/case, and it is *directionally* better on all four quality measures.
3. arXiv is the only channel where it costs nothing. Every non-arXiv source — the whole point
   of Feature 10 — receives the bare phrase.
4. `none`, the alternative repair, is measured worse.

This is the one place here where the evidence does not compel the conclusion, so it is
flagged rather than buried: `queries.bigrams: adjacent` restores the old behaviour in one
line, and **every number published before 2026-08-12 was measured under `adjacent`**.

Logged as **C-10** (the phrase generator never checked co-occurrence) and **NR-29** (repairing
it is unresolvable on the arXiv benchmark; deleting phrase queries is worse).

**Cost** ~$26, 75 runs, 3 h 52 m, all three arms clean — no HyDE degradations, no collection
failures, no judge failures, 25/25 cases in every arm.

### C-9 audit: the fix was two-fifths applied, and bioRxiv failed the other way (2026-08-12)

Follow-up to the section above, asking two questions: did the fix change what DBLP and
bioRxiv return, and which past conclusions rest on the broken queries. **Cost $0** — no LLM,
both APIs free. Scripts: `evals/audit_query_transform.py`.

#### 1. The fix reached two of five call sites

The PR that introduced `to_plain_keywords` claimed it was "routed through all three call
sites." There were five, and it routed two — IACR and DBLP. Still hand-rolling the broken
one-liner afterwards:

| call site | source | state after the "fix" |
|---|---|---|
| `cli.py:330` | Semantic Scholar | broken |
| `cli.py:354` | OpenAlex | broken |
| `cli.py:374` | bioRxiv | broken |
| `evals/run_eval.py:229` | Tier A/S runner (OpenAlex, S2) | broken |
| `evals/harness.py:171` | Tier B harness (all five) | fixed |

The function was correct and unused. **A unit test of a translator cannot detect that**, and
the one shipped alongside it passed on this state — so the guard added now asserts the
*wiring*: `TestEveryBridgeUsesTheSharedTranslator` parses `cli.py`, `harness.py` and
`run_eval.py` and fails any comprehension over `queries` that does not call
`to_plain_keywords`. Mutation-verified against the exact shipped state; it names the line.

#### 2. It was never right — no drift, no working era

The earlier account said the transform was written for an older query shape. Git refutes it:

| | commit | date |
|---|---|---|
| `build_queries` emits `(all:"x") AND (cat:y)` | `29ecffa` | 2026-02-22 |
| the one-liner is written | `18dfe51` | **2026-02-23** |

Checked out at `18dfe51`, the builder already produced the parenthesised form. The bridge was
wrong on the day it was written. Every non-arXiv fetch in the product's history is affected —
not a regression window.

#### 3. Did results change? DBLP yes, bioRxiv catastrophically

Real `build_queries` output from real benchmark repos, all-time lookback so DBLP's year
filter is not what is being measured.

**DBLP** — total hits reported, old transform vs new:

| case | query (repaired) | OLD | NEW |
|---|---|---|---|
| `db` | `duckdb sql` | 0 | 1 |
| `columnar` | `arrow file` | 0 | 0 |
| `compiler` | `numba pr` | 0 | **4** |

DBLP returns **zero for every malformed query**, exactly like ePrint. `columnar`'s 0 → 0 is a
genuine empty: DBLP searches titles only and no title carries that phrase.

*A first pass reported 0 vs 0 everywhere; DBLP had refused 12 of its 18 requests and those
zeros were rate-limiting wearing a zero's clothes. Redone one query at a time, 10 s apart,
with refusals retried and reported — a 0 above means DBLP answered and the answer was empty.*

**bioRxiv fails in the opposite direction, and it is worse.** Its filter keeps a paper if any
query word longer than two characters occurs in the title or abstract. Over a 90-paper window:

| case | OLD kept | NEW kept |
|---|---|---|
| `db` | **90 / 90** | 43 / 90 |
| `columnar` | **90 / 90** | 43 / 90 |
| `compiler` | **90 / 90** | 35 / 90 |

Term by term, for `(all:"duckdb sql") AND (cat:cs.LG OR cat:cs.CL)`:

| term | matches |
|---|---|
| `("duckdb` | 0 / 90 |
| `sql")` | 0 / 90 |
| `(cat:cs.lg` | 0 / 90 |
| `cat:cs.cl)` | 0 / 90 |
| **`and`** | **90 / 90** |

Without `and`: **0 / 90**. The boolean operator is three characters, so it cleared the
`len > 2` filter and matched every abstract in English. **Enabling bioRxiv did not add biology
papers to the pool — it turned the topical filter off and merged the entire recent window.**
DBLP contributed nothing; bioRxiv contributed noise at full volume. A source that returns
nothing announces itself. A source that returns everything looks like it is working.

#### 4. Which conclusions this touches

**None of the benchmark numbers, and this is checkable rather than argued.** No `dblp:` or
`biorxiv:` id appears in any of the 78 recorded run files in `evals/results/` — neither source
ever contributed a paper to a scored pool, so no published net@2, precision, or recall figure
was computed from their output.

What the audit does revise:

- **"DBLP: still unmeasured after four attempts — and now we know why"** — there was a
  **fifth** blocker, and it was upstream of the other four. Attempt 4 concluded DBLP is
  *structurally* mismatched to recency windows because it exposes only a year. That reasoning
  about the adapter's filter is still correct on its own terms, but it was **not what made
  DBLP return nothing**: the measurement above ran at all-time lookback, removing the year
  filter entirely, and the malformed query still returned 0. The heading's "now we know why"
  was premature — the year-granularity finding was diagnosed by reading the adapter, never by
  observing DBLP answer a well-formed query.
- **"`collect_papers` now returns papers where it returned 0"** (the 2026-07-29 TLS fix) —
  cannot have been true through `build_queries` output. The TLS fix was real and necessary;
  the verification behind that sentence must have used a hand-written query, which is the same
  drift that hid C-9 for six months.
- **The ±1 noise floor from the two "same effective configuration" runs** — *unaffected, and
  in fact strengthened*. The pairing was justified by the harness silently dropping `dblp`.
  C-9 means that even with the dblp branch present the runs would have been identical, because
  DBLP returns 0 either way.
- **Feature 10's status.** ROADMAP and README describe bioRxiv and DBLP as serving repos whose
  literature is not on arXiv. On the evidence they have never served anyone: one returns
  nothing, the other returns everything. Both are now marked **built, wired, never validated**.

#### 5. Undetermined, and stated as undetermined

**Semantic Scholar's exposure is unresolved.** It is the one source with a *published* number
downstream of the bug — "adding Semantic Scholar did not help" (mean net@2 +0.83 → +0.58,
precision 0.91 → 0.76), and `git show 4d5416e:evals/harness.py` confirms that run used the
broken one-liner. Whether S2 tolerates the malformed query matters: the finding reports that
outcomes *moved*, which is only possible if S2 returned papers, yet DBLP and IACR return zero
for the same input.

Two attempts to measure it were **refused by rate limiting** — keyless S2 returned 429 on all
20 requests across both passes. The fast pass produced an apparent `OLD → 0, NEW → 50`, and
that zero is discarded: it was a 429, not an empty result. **No claim is made about the S2
finding in either direction.** Resolving it needs an S2 key or a patient off-peak retry, and
until then finding 3 stands as *measured through malformed queries, effect unknown*.

**Logged as C-9a** (fix applied to two of five call sites), **C-9b** (the "older query shape"
account was false), and **NR-28** (bioRxiv's boolean-operator pass-through).

### Frozen pools measured: the floor halves to 0.48 — and the guard I shipped to protect it was broken (2026-08-11)

```bash
# seed once (live collection), then two passes that reuse it
for PASS in seed reuse-1 reuse-2; do
  uv run python evals/run_judge_eval.py --baseline none --case <all 25> \
      --rr-pool 50 --rr-rerank --rr-all-time --rr-hybrid --rr-sweep --rr-finescale --rr-hyde \
      --rr-frozen-pool evals/.work/pool-floor
done
uv run python evals/noise_floor.py <reuse-1> <reuse-2>
```

`--rr-frozen-pool` shipped with an **unmeasured claim**: that it takes the resolvable effect
"toward 0.2–0.3". That is exactly the kind of assertion this project does not accept
elsewhere, so it gets measured. Two passes reusing one pool, with the seeding pass excluded
because it collected live — which is what the separate `frozen-seeded` label is for.

| | live (3 draws) | **frozen (2 reuse passes)** |
|---|---|---|
| residual sd (per case, per draw) | 1.23 | **0.61** |
| whole-run shift | sd 0.27 | **sd 0.10** |
| **MRE, paired same session** | 1.04 | **0.48** |
| MRE against a stored run | 1.07 | 0.49 |
| cases identical across draws | 8 / 22 | **20 / 25** |

**The diagnosis was right and my number was not.** Freezing the pool removes just over half
the residual noise, confirming that *which candidates were collected* is the dominant term.
But the pre-registered prediction (sd ≤ 0.5, MRE ≤ 0.42) **missed**, and the "0.2–0.3" in the
`--rr-frozen-pool` help text was **roughly half the truth**. The floor is **0.48**, and the
help text now says so. The alarm (sd ≥ 1.0, meaning freezing bought nothing) did not fire.

What survives freezing is temperature-0 model jitter in the gate and the rescore: `columnar`
(+5/+2), `cv` (+5/+3) and `rag` (+4/+2) carry **89%** of the remaining variance between them,
and 20 of 25 cases are bit-identical across passes.

#### What 0.48 reopens

| experiment | measured | live floor 1.04 | **frozen floor 0.48** |
|---|---|---|---|
| stated-intent `blind` goals | +0.44 | unresolvable | **borderline — worth re-running** |
| register-flip `docs` goals | +0.12 | unresolvable | still unresolvable |

> **Withdrawn 2026-08-15.** This table reads +0.44 against a 0.48 floor. Re-measured at the
> benchmark's new returned-set width the frozen floor is **0.74**, so the `blind` arm is
> plainly unresolvable and the "worth re-running" plan below rests on a number that has since
> moved. See *The floor is a property of the configuration* above.

The `blind` arm sits just under 0.48, so a frozen-pool re-run would be close to decisive
rather than uninformative — **except that it cannot use one.** A goal changes the HyDE
hypotheses, so the pool fingerprint changes and the harness refuses to reuse: stated-intent
is a *retrieval* experiment. The frozen floor applies to gate models, `min_actionable`,
thresholds and rescore variants, and those are now measurable at less than half the effect
size they needed a day ago.

#### The guard was broken, and the tests could not see it

`noise_floor.py` reported **`mixed`** for both frozen runs. The cause: a pool fingerprint
includes its own case name, so a genuine 25-case frozen run carries 25 *different*
fingerprints, and the first version folded mode+fingerprint per case and took the set —
which is `mixed` for any run over more than one case.

That is not cosmetic. Two runs drawn from **different pools** would both have reported
`mixed`, matched, and compared cleanly — **precisely the failure the guard exists to
prevent.** It shipped because every test in `test_eval_frozen_pool.py` used a single-case
run, where the bug is invisible.

Fixed: `provenance` now digests the *whole set* of per-case fingerprints, and a new
`same_pool` check compares runs pairwise over the cases they **share**, so a 25-case and a
22-case run from one pool stay comparable on their 22. `ablation_report.pool_mode` now
delegates to the same helper rather than keeping a second copy that could rot separately.
Six tests added, all over multi-case runs.

**Cost** ~$11: the seed is a full live collection, the two reuse passes skip arXiv and the
index scan entirely and cost about $1.50 each. My earlier "$4 and 40 minutes" estimate
forgot the seed.

### A third draw: the floor holds, and a single run's *p-value* is worth even less than its mean (2026-08-11)

```bash
uv run python evals/noise_floor.py --assume-unlabelled-live A.json B.json C.json
```

The floor below rested on **one** pairwise difference. A third draw of the identical shipped
config doubles the degrees of freedom, and the estimate barely moves:

| | 2 draws | **3 draws** |
|---|---|---|
| residual sd (per case, per draw) | 1.23 (21 df) | **1.23 (42 df)** |
| whole-run shift | sd 0.32 | sd 0.27, spread 0.64 |
| **MRE, paired same session** | 1.03 | **1.04** |
| MRE against a *stored* run | 1.07 | 1.07 |
| Jaccard, ranked top-10 | 0.498 | 0.494 |

**Nothing published needs revising** — 1.03 → 1.04 — but the floor is now a number with
weight behind it rather than a single difference.

#### The p-value is less stable than the mean

| draw | 22-case mean | 25-case paired vs baseline | sign test |
|---|---|---|---|
| A (08-08) | **+4.55** | — | — |
| B (08-10) | +3.91 | +2.26 | 15 w / 5 l / 3 t, **p = 0.0414** |
| C (08-11) | +4.09 | +2.50 | 18 w / 1 l / 5 t, **p = 0.0001** |

The mean of the headline moves 0.64 across draws. **Its significance moves two orders of
magnitude** — 0.0414 to 0.0001 — on the same 25 repositories, same flags, two days apart.
A single run's p-value is not a property of the system, and reporting one as if it were is
a mistake this project made twice (the 22-case p = 0.0075 and the 25-case p = 0.0414 are
both single draws).

The honest headline is the **mean over draws**: **+4.18** on the 22 shared cases (range
+3.91 to +4.55), and **+3.84** on all 25 with a paired advantage of **+2.38** over the
baseline. Three draws roughly halve a headline's standard error (0.38 → 0.22).

#### Where the noise lives

| | |
|---|---|
| cases identical across all three draws | **8 of 22** |
| the three noisiest cases | `graph` (+10/+8/+5), `speech` (+10/+6/+6), `cv` (+5/+8/+4) |
| share of total variance from those three | **47%** |

The instability is concentrated, and — notably — in cases that score *well*. `graph` and
`speech` are the two repositories most often quoted as saturated wins; they are also the two
that move most between draws.

#### A guard that fired on its own author

Draw C is labelled `live` by the new provenance field; draws A and B predate it and read
`unlabelled`, so `noise_floor.py` **refused to compare them**. That is correct behaviour —
inferring "live" from absent data is exactly what the guard exists to prevent — and the fix
was *not* to backfill the field into historical artifacts, which would make them claim
something they never recorded. Instead `--assume-unlabelled-live` makes the assumption an
explicit argument, printed on every run that uses it. Runs written before `--rr-frozen-pool`
existed could not have been frozen, so the assumption is sound; it just has to be visible.

**Cost** ~$13, 25 runs, ~2 h.

### The benchmark's noise floor, and a frozen-pool mode to lower it (2026-08-10)

```bash
uv run python evals/noise_floor.py A.json B.json        # $0, two runs of the SAME config
uv run python evals/run_judge_eval.py ... --rr-frozen-pool evals/.work/pool-<name>
```

The headline run showed that **−0.64 of a 22-case mean was run-to-run drift on identical
inputs**. That raised a question this project had never answered: *what is the smallest
effect this benchmark can resolve at all?* A null result is only informative if the
instrument could have found the effect had it been there, and that was never checked.

Taking two runs of the shipped configuration, 22 shared cases, two days apart, and treating
their per-case difference as pure noise:

| | |
|---|---|
| per-case delta | mean −0.64, **sd 1.73**, range [−4, +3] |
| identical cases | 11 / 22 |
| cases moving ≥ 3 | 4 / 22 |
| SE of the 22-case mean | **0.37** |
| 95% interval on a single mean | **±0.72** |
| **minimum resolvable effect** (80% power, α = 0.05) | **1.03 net@2 per case** |
| Jaccard overlap, shown papers | 0.541 |
| Jaccard overlap, ranked top-10 | **0.498** |

#### What this says about results already published here

| experiment | measured | vs the 1.03 floor |
|---|---|---|
| HyDE, end to end | +1.36 | **just above** — which is why it landed at p = 0.092, not at a clean win |
| stated-intent `blind` goals | +0.44 | **below** — unresolvable at n = 25 |
| register-flip `docs` goals | +0.12 | far below |

**Two of those experiments could not have succeeded.** Their nulls are real but
uninformative: the right response was a more sensitive instrument, not a firmer conclusion.
The bootstrap intervals reported at the time were honest ([−0.20, +1.12] for `blind`), so
nothing published needs retracting — but the experiments should not have been *run* in that
form, and roughly $30 and eight hours went into questions this benchmark cannot answer.
Future experiments get sized against 1.03 first.

#### Where the noise comes from, and how much is removable

The ranked top-10 overlaps only **0.498** between two runs of the same configuration — about
a third of the papers a run shows are different next time. The dominant term is *which
candidates were collected*, not the judge and not the gate. So `--rr-frozen-pool DIR`
collects each case's ranked candidates once and reuses them, which removes that term for
any treatment living **downstream of retrieval**.

**Keeping frozen and live runs distinguishable is the whole design**, because a frozen run
is not a live measurement and a reader who mistakes one for the other is worse off than
before the mode existed. Five mechanisms, none of them optional:

1. **A pool fingerprint** over every setting that can change what gets collected — sources,
   pool depth, `--rr-all-time`, `--rr-hybrid`, prose budget, `--rr-ablate-docs`, all the
   HyDE settings, the hypothesis model, **and the goal** (a stated-intent arm changes the
   hypotheses, so it is a *retrieval* experiment and must collect live). A mismatch is a
   **hard error** naming both fingerprints, never a silent reuse.
2. **`pool_provenance` on every case** in the results file: `live`, `frozen`, or
   `frozen-seeded` (the run that collected it — labelled separately, because that run *was*
   live and calling it frozen would misdate its candidates).
3. **A stdout banner** at run start and a per-case line naming the collection date.
4. **A `-frozenpool-` filename tag**, so the mode is visible in `ls`.
5. **`noise_floor.py` and `ablation_report.py` refuse to compare across modes** — and runs
   predating the flag read as `unlabelled`, not `live`: they *were* live, but inferring that
   from absent data is how a wrong assumption becomes a published number.

24 tests, mutation-verified: dropping `rr_hyde` from the fingerprint fails two of them.

**What this does not do.** Frozen pools are invalid for anything that changes retrieval —
HyDE, all-time discovery, ablated docs, goals — which is exactly the set the fingerprint
refuses. They are for gate models, `min_actionable`, thresholds, and rescore variants. And
the sd itself rests on a **single pair of runs**, so 1.03 is an estimate with its own
uncertainty; a third draw would tighten it.

### The 25-case headline: significance survives, and the +4.55 was a favourable draw (2026-08-10)

```bash
uv run python evals/run_judge_eval.py --baseline cli --case <all 25> \
    --rr-pool 50 --rr-rerank --rr-all-time --rr-hybrid --rr-sweep --rr-finescale --rr-hyde
```

Every published headline was measured on **22** cases; the benchmark has been **25** since the
thin-docs cohort landed, and those three were chosen precisely because RepoRadar handles them
badly. Running them was an obligation, not bookkeeping — the stale number flattered us, and
the honest question was whether **p = 0.0075** survives a harder benchmark.

The baseline was deliberately left at its **default 12 turns**, the configuration behind every
prior headline. Raising it would make this number non-comparable with the ones it replaces,
and would re-derive the gold set (that is how `graph`'s targets moved 3 → 4 last time).

| | 22-case (2026-08-09) | **25-case (2026-08-10)** |
|---|---|---|
| RepoRadar mean net@2 | +4.55 | **+3.80** |
| baseline mean | +1.82 | **+1.57** |
| paired delta | +2.73 | **+2.26** |
| sign test | 15 w / 3 l / 4 t, **p = 0.0075** | 15 w / 5 l / 3 t, **p = 0.0414** |
| digest precision | 0.94 | **0.898** |
| shown / actionable | 121 / 114 | **137 / 123** |
| net-negative repos | 0 | **1** |

**The result holds.** p = 0.0414 on the harder benchmark — weaker than 0.0075, still under
0.05, and the paired advantage is +2.26 with 15 wins to 5.

#### The drop is drift, not difficulty — and that is the finding

The prediction going in was that the three hard cases would pull the mean from +4.55 to
+4.0–4.3. **The number landed at +3.80, and the reason is not the one predicted:**

| | mean net@2 | contribution to the change |
|---|---|---|
| the same 22 cases, today | **+3.91** | **−0.64** vs the published +4.55 |
| the 3 new cases | +3.00 | −0.11 |

The added cases moved the mean by **one tenth of a point**. The rest is **run-to-run drift on
identical repositories under an identical configuration** — which means the published **+4.55
was a favourable draw**, and the same pipeline on the same benchmark scores +3.91 today. That
is consistent with the ±6 per-case swings documented throughout, now visible in the aggregate.
Any future headline should be read as a draw from a distribution roughly ±0.6 wide at the
mean, not as a fixed property of the system.

#### A published claim that does not survive this draw

**"Zero net-negative repositories" is now false.** `numerics` shows **one** paper, the judge
scores it 0, and the case lands at **−2.0** — the minimum possible negative, but negative. The
fine-scale rescore's rescue of every net-negative repository was true of every run it had been
measured on and is **a per-draw property, not a guarantee**: it takes one dud in a one-paper
digest to break it. Corrected in `paper/DRAFT.md`.

#### Method notes

Two cases are excluded from the paired comparison, **for different reasons, and the harness
distinguishes them on purpose**: `speech` is `arxiv_unverified` (the baseline produced
references and arXiv verification failed on three of them — a transient network failure that
must not be scored as a baseline error), and `thin-lang` is `error_max_turns` (genuine budget
exhaustion, the failure mode already recorded for thin-docs repos). The 12-turn limit still
has never bound on a thick case. A baseline that ran out of budget is not the same as one that
abstained, so neither was converted into a 0 — which would have flattered RepoRadar by +1 and
+6 respectively.

`linter` remains the sharpest single case for the abstention stance: the baseline scores
**−6.0**, RepoRadar returns nothing and scores 0.

**Cost** ~$13, 25 runs, ~2 h (22 baselines cache-valid; only the thin trio needed fresh calls).

### Stated intent at 25 cases: the REGISTER FLIP IS REFUTED, and what is left is unestablished (2026-08-10)

```bash
uv run python evals/make_goals.py --arm docs --cases <all 25>   # same bytes the pipeline has
for ARM in control docs blind; do
  uv run python evals/run_judge_eval.py --baseline none --case <all 25> \
      --rr-pool 50 --rr-rerank --rr-all-time --rr-hybrid --rr-sweep --rr-finescale --rr-hyde \
      ${ARM:+--rr-goals evals/goals/$ARM.json}
done
```

The 6-case run above could not be attributed, and the objection that exposed it is worth
stating in full: **the `blind` arm changes two things at once.** It asks a different
*question* (a need, not an identity) *and* it sees strictly more *information* — a sample of
source the profiler never reads (`scan_source` is False) and a README budget of 3,500
characters against the profile's 300. Those imply completely different products: a register
flip is one cheap LLM call, a source-code dependency means turning scanning on. An experiment
that moves both learns which is true about neither.

So a third arm, **`docs`**, holds information fixed at exactly the bytes the pipeline already
consumes — `repo_context_block` and nothing else — and moves only the question. Three arms ×
25 cases, one session, 75 runs.

| arm | all 25 net@2 | shown | precision | thin trio (3) | real cases (20) |
|---|---|---|---|---|---|
| control | +3.88 | 136 | 0.904 | +3.33 | +4.85 |
| docs | +4.00 | 142 | 0.901 | +3.33 | +4.95 |
| blind | **+4.32** | 138 | **0.928** | **+5.00** | +5.35 |

| paired vs control | mean/case | sign test | bootstrap 95% CI | r(log corpus, gain) |
|---|---|---|---|---|
| **docs** | **+0.12** | 10+/7−/8= (p = 0.63) | **[−0.72, +0.96]** | **+0.085** (wrong sign) |
| **blind** | **+0.44** | 8+/6−/11= (p = 0.79) | **[−0.20, +1.12]** | **−0.362** (as predicted) |

#### The register flip alone does nothing

`docs` — the cheap, shippable version — is **+0.12 net@2 per case with an interval from −0.72
to +0.96**, and its correlation with documentation size is **+0.085, the wrong sign**: no
concentration in thin repositories, no concentration anywhere. Asking the same information a
better question buys nothing measurable. That is the arm this experiment existed to isolate,
and it is refuted.

#### What survives is the information, and it is not established

`blind` is +0.44/case with an interval spanning zero and 8 wins to 6 losses — **not
significant, and not close**. Its one encouraging property is the pre-registered one: the gain
correlates with documentation thinness at **r = −0.362**, and that **survives dropping the
three thin cases added for this purpose** (r = −0.390, n = 17), so it is not an artifact of
the cohort. But t = −1.65 against a 2.11 bar; this is a direction, not an effect.

Read together, the attribution is clean and unwelcome: **the benefit tracks the extra
information, not the better question.** If this is worth having, the change is
`scan_source=True` plus goal synthesis — not one cheap call.

#### The 6-case result: the thin trio replicated exactly, the generalisation did not

`blind` on the thin trio is **+1.67/case**, the same number as the 6-case run. What failed to
appear is any broader effect: across all 25 the same arm is +0.44 with the interval spanning
zero. The earlier headline was a real measurement of three repositories being read as a
measurement of the system.

**Pre-registration.** Predicted: positive mean paired delta on both arms *and* r < 0. `docs`
fails both clauses; `blind` meets both directionally, neither significantly. The kill (delta
≤ 0 on both arms overall and on the thin subset) did not fire. So item 0 is **neither
established nor closed** — but its cheapest implementation is closed.

**Caveats.** Eleven of 25 `blind` comparisons are ties, and the five negative controls are
pinned at 0.0 by design, so the effective *n* for a sign test is nearer 20 than 25. `blind`
does not win by shrinking the digest (138 shown against control's 136), which is the one
flattery net@2 usually supplies and does not here. And `--baseline none` means the judged pool
is RepoRadar's own top-10, so these three arms are comparable to each other and to nothing
else.

**Cost** ~$26, 75 runs, ~5 h, all arms clean.

### Stated intent in the query (roadmap item 0): directional, and the ORACLE LOSES TO THE BLIND ARM (2026-08-09)

```bash
uv run python evals/make_goals.py --arm blind    # repo only, no papers ever shown
uv run python evals/make_goals.py --arm oracle   # shown the confirmed papers; leaky by design
for ARM in control blind oracle; do
  uv run python evals/run_judge_eval.py --baseline none --case thin-lang,thin-kv,thin-gnn,compiler,storage,graph \
      --rr-pool 50 --rr-rerank --rr-all-time --rr-hybrid --rr-sweep --rr-finescale --rr-hyde \
      ${ARM:+--rr-goals evals/goals/$ARM.json}
done
```

Two size-based remedies for the thin-docs failure are already refuted — a similarity floor on
the dense search (the papers are *close*, the query is wrong) and a profile-information floor
(across the ablation the spread *between repos at one budget* exceeds the movement between
budgets: at 1500 chars `speech` scores −6.0 while `db` scores +9.0; and in the real cohort
precision *falls* as corpus grows, 108 ch → 1.00, 3,556 ch → 0.75). What remains is not
volume but **register**. A maintainer's stated goal is the one available input already in the
second register: documentation says what a project *is*, a goal says what it should *adopt*.

**Where the goal is allowed to go, and why that is not a preference.** It reaches
`hyde.generate_hypotheses` only, appended *after* `repo_context_block` rather than merged into
it. P8 measured stated wants fed to the **gate** at net@2 +57 against +95 — the worst arm in
the campaign — and concluded they belong in the query; separately, the fine-scale map is a
frozen logistic fitted to that block's exact bytes, so a goal merged in would move where P
crosses 2/3 with nothing failing loudly. `tests/test_goal_injection.py` asserts the isolation
and was mutation-verified: merging the goal into the shared block fails two of its nine tests.

**The oracle is generated, not hand-written.** Whoever ran the earlier experiments has read
these cases' judged papers, so anything they author carries an unmeasurable amount of that.
Instead a model is shown the judge-confirmed-actionable papers and asked to recover the goal
*a maintainer could have written before seeing them* — the leak becomes a documented input
that re-derives identically. The two arms answer different questions: **blind** (repo only,
including source the profiler never reads) asks whether a goal can be had *without the user*;
**oracle** asks whether register-correct intent is a lever *at all*.

| arm | thin net@2 | thin shown | thin precision | thick net@2 | thick precision |
|---|---|---|---|---|---|
| control | +3.33 | 19 | 0.842 | +6.00 | 1.000 |
| **blind** | **+5.00** | 15 | **1.000** | +5.67 | 0.950 |
| oracle | +4.33 | 19 | 0.895 | +6.33 | 0.955 |

| paired vs control | thin | thick |
|---|---|---|
| blind | **+1.67/case, 3+/0−** (p = 0.25) | −0.33/case, 1+/2− |
| oracle | +1.00/case, 1+/1−/1= | +0.33/case, 2+/1− |

#### Pre-registration: the prediction MISSED by one paper; the kill did not fire

Predicted: oracle lifts thin precision to **≥ 0.90** without degrading the thick trio.
Measured **0.895** (17/19) — short by a single paper — and the thick trio slipped 1.000 →
0.955. Kill band was [0.70, 0.85]; 0.895 sits above it, so **item 0 is not closed**.

#### The unpredicted result is the useful one

**The blind arm beat the oracle** — +5.00 against +4.33, precision 1.000 against 0.895, and
3/3 cases improved against the oracle's 1+/1−/1=. The arm with *less* information did better,
which inverts the entire point of running a ceiling.

The decision-relevant consequence is favourable: **if the achievable arm is already at least
as good as the leaky one, there is no gap for user input to close.** A goal written by a model
reading the repository's own source is enough; the feature does not require asking the user.
Speculatively, the oracle may underperform *because* it is derived from papers already found —
its literature-vocabulary phrasing (`storage`'s oracle goal names "write amplification, read
amplification, space amplification, LSM-tree compaction", the RocksDB literature's exact terms
of art) aims HyDE at the neighbourhood already retrieved, while the blind goal, written from
code, names a genuinely different need. That is a hypothesis this run cannot test.

**Thick repos are unaffected to slightly worse** (+6.00 → +5.67 blind, +6.33 oracle), exactly
what the register account predicts: a repository whose documentation already describes it well
gains nothing from a goal and may take on noise.

#### What this does not support

**Nothing here is significant.** Three thin cases give a best-possible sign test of p = 0.25,
and it lands there; Fisher on the precision jump (15/15 vs 16/19) is p = 0.238. This is a
direction, not an effect.

**`thin-lang` was not rescued.** The 108-character repo goes −2.0 (control) → **0.0** (blind,
abstained) → −2.0 (oracle). The blind goal converted junk into an abstention, which is the safe
direction and not a recovery.

**Blind's precision is partly selectivity.** It shows 15 papers against control's 19 — one
fewer actionable and three fewer duds. net@2 rewards exactly that trade, so it flatters this
result as it has flattered every precision-preserving change in this project.

**Pools are not comparable to the runs above.** `--baseline none` (added here, since this is a
RepoRadar-vs-RepoRadar comparison where the baseline only burns money and would re-disturb the
gold set) shrinks the judged pool to RepoRadar's own top-10. Read these three arms against each
other and against nothing else — the control's thin precision reads 0.842 here against 0.778 in
the ground-truth run for that reason alone.

**Next, and it is cheap:** run the blind arm across all 25 cases. Three thin cases are the
binding constraint on everything above, and 25 paired cases is the first design here that
*could* reach significance. ~$15.

**Cost** ~$7, 18 runs, ~70 min, all arms clean.

### Three real thin-docs cases join the benchmark — and the ablation's ceiling holds up (2026-08-09)

```bash
uv run python evals/run_judge_eval.py --baseline cli \
    --case thin-lang,thin-kv,thin-gnn,compiler,storage,graph \
    --rr-pool 50 --rr-rerank --rr-all-time --rr-hybrid --rr-sweep --rr-finescale --rr-hyde
```

The ablation below is a **ceiling**, not an estimate — stripping detectron2's README does
not make the gate, the hypothesis generator or the judge forget detectron2. Only real,
unmemorised repositories can convert it into a number, so three joined the benchmark, each
**paired with a thick-docs case in the same domain** so a thin-vs-thick difference is not
confounded by domain or by whether literature exists at all. All six ran in one session
with a uniform 30-turn baseline budget.

**First, a correction to how "thin" was being measured.** Earlier sections quote the
benchmark's thinnest README as **1,639 characters** (`webdev`). That is the right measure
for the *gate's prose block* — `README[:300]` — but the wrong one for everything else: the
profiler reads README **plus `docs/`**, and `webdev` carries 78 docs files totalling
**384,456** characters. By corpus, the real floor was `db` at **1,857** and the median case
is **194,999**. The benchmark was further from the target regime than reported, not closer.

| case | corpus chars | | paired with | corpus chars |
|---|---|---|---|---|
| `thin-lang` (fireball, Go compiler) | **108** | ↔ | `compiler` (numba) | 1,034,113 |
| `thin-gnn` (distributed_graph_flow) | **1,073** | ↔ | `graph` (pytorch_geometric) | 238,390 |
| `thin-kv` (sekas, distributed KV) | 3,556 | ↔ | `storage` (rocksdb) | 196,067 |

`thin-lang` at 108 characters is **1,800× below the median case** and 17× below the old
floor. `thin-kv` is the weak member and is labelled as such: its 787-char README plus one
`docs/` file totals 3,556, which sits *above* `db` and `numerics`, inside the existing
distribution rather than below it.

#### Result

| | RepoRadar net@2 | baseline net@2 | digest precision | pool actionable share |
|---|---|---|---|---|
| thick trio | **+7.00** | +4.00 | **21/21 = 1.000** | 35/42 = 0.833 |
| thin trio | **+2.00** | +1.67 | **14/18 = 0.778** | 23/37 = 0.622 |
| degradation | **−5.00** | −2.33 | −0.222 | −0.211 |

| case | shown | actionable | net@2 | vs baseline |
|---|---|---|---|---|
| `thin-lang` | 1 | 1 | +1.0 | +0.0 |
| `compiler` | 1 | 1 | +1.0 | −2.0 |
| `thin-kv` | 8 | 6 | +2.0 | +1.0 |
| `storage` | 10 | 10 | +10.0 | +5.0 |
| `thin-gnn` | 9 | 7 | +3.0 | +0.0 |
| `graph` | 10 | 10 | +10.0 | +6.0 |

**Precision is the clean signal, and it lands where the ablation said it would.** The thin
trio shows four non-actionable papers in eighteen (**0.778**); the thick trio shows zero in
twenty-one. Fisher's exact on shown papers gives p = 0.037 — reported with the caveat that
**papers are not independent units**, since they cluster by repository, so that figure
overstates the power. The ablation predicted 0.853 at its 300-character rung and 0.636 at
zero; real thin repositories land at **0.778, inside that band**. That is the pre-registered
check on NR-25, and it passes: had the thin cases scored like their thick partners, the
ceiling would not have been a ceiling. They did not, so it stands — and the ablation looks
like a *reasonable* proxy rather than a wildly optimistic one.

**Both systems degrade; RepoRadar degrades about twice as hard.** The agentic baseline falls
+4.00 → +1.67 while RepoRadar falls +7.00 → +2.00. Thin documentation is a property of the
*task*, not a defect unique to this system — but the margin RepoRadar holds on thick repos
(+3.00 over the baseline) collapses to **+0.33** on thin ones.

**And the retrieval story is confirmed on real repos.** The actionable share of the judged
pool falls 0.833 → 0.622, the same direction and roughly the same size as the ablation's
0.840 → 0.568. This is upstream of selection.

#### Two things that did not replicate

**`thin-lang`'s abstention is not a stable property.** In the first ground-truth draw it
showed **nothing** — the gate admitted 5 papers and the fine-scale rescore cleared 0 of 5,
which read as a clean demonstration that the shipped second stage is what stands between a
108-character repository and a confident wrong digest. In this run it showed one paper,
which was actionable. Both outcomes are safe, but "it abstains" is a claim this data does
not support; "it does not produce junk" is.

**`thin-gnn` moved 3 points between two runs 20 minutes apart** (+3.0 → +0.0 → +3.0 across
three draws, precision 0.78 → 0.67 → 0.78). Per-case variance at this scale is the norm
here, not the exception, and is why all six cases were finally re-run in a single session
rather than assembled from the runs that happened to have valid baselines.

#### What this cost, and what it costs from now on

~$14 across three runs (two discarded for the reasons above), dominated by baseline calls at
$1.50–1.95 each. **The standing cost is the one that matters: 22 → 25 cases is roughly 14%
more per full benchmark run, forever.**

#### Method notes

The **agentic baseline failed on two of three thin cases at the shipped 12-turn limit**
(`error_max_turns`, 13 turns, still calling tools) — something it has never done on the 22
thick cases. Thin documentation is expensive for an agent that has to read code to learn
what a repository is. Those cases were re-run at 30 turns; `evals/baseline.py` hashes the
CLI flags into the cache discriminator, so raising the limit correctly invalidated **all
six** baselines rather than silently mixing budgets, which is why the final run is uniform.

**A near-miss worth recording: re-running a baseline silently redefines the gold set.** The
30-turn retry was needed for two thin cases, but the discriminator correctly invalidated
*all six* baselines — including the three thick ones, which had never hit the limit. The
gold set is derived as `baseline picks ∩ judge ≥ 2`, so `graph`'s targets moved **3 ids →
4**, which would have shifted the denominator of every published recall figure (21/48,
27/48, 36/48) as a side effect of a turn-limit change. `tests/test_eval_hop_pool.py`, which
pins the frozen `TARGETS` literal against the derivation, failed and caught it; the three
caches were restored from the recorded run file and the gold set is byte-identical at 24
targets. The hazard is now documented on `diagnose_pool.actionable_baseline_ids`. Note the
comparison table above is unaffected — it reads the results artifact, not the cache.

Candidate selection is recorded because two of its steps were wrong at first. A keyword
search returned 110 repos whose *minimum* README was 1,261 characters and median 6,173 —
an artifact, not a finding: **GitHub repository search matches the query against the
README**, so keyword queries can only surface repos that describe themselves. Searching by
curated `topic:` tags and `in:name` instead found 542 candidates with a minimum of 108.
And `pushed_at` is not activity: `yetone/mirdb` (117★, LSM store, an otherwise ideal
candidate) reports pushed 2026-05-29 but its last real commit is **2019-05-02**. Checking
`commits` rejected it and `rust-storage-bench` (dead since 2024) — a repository with no
maintainers cannot act on a recommendation, which breaks the task's premise.

### Thin docs: the ALARM fired, the prediction failed, and the danger zone is *some* documentation, not none (2026-08-09)

```bash
# four arms, back to back in one session, docs the only variable
for B in control 1500 300 0; do
  uv run python evals/run_judge_eval.py --baseline cli --case graph,speech,db,cv,rag,webdev \
      --rr-pool 50 --rr-rerank --rr-all-time --rr-hybrid --rr-sweep --rr-finescale --rr-hyde \
      ${B:+--rr-ablate-docs $B}
done
uv run python evals/ablation_report.py control=…052546Z.json 1500=…054713Z.json \
                                       300=…061027Z.json 0=…063210Z.json
```

**Why this is not the ablation the roadmap dropped.** ROADMAP dropped "thin-docs benchmark
ablation" because *"the existing arms already bracket the answer"* — the prose-budget arms
(0/300/2000/6000, keywords, tagline, extractive, paraphrase). Every one of those varied **gate
context**. The profile has since acquired a consumer that did not exist when that was written:
**HyDE hypothesis generation**, which is load-bearing for the p = 0.0075 result. Nothing
brackets it. And the regime itself was never sampled — the thinnest README in the benchmark is
**1,639 characters against a 300-character prose budget; zero cases under 1,000, zero under
300**. Supply exceeds demand 5.5× at the minimum, so every prose measurement ever made here was
truncating an abundance.

`--rr-ablate-docs CHARS` caps the README and withholds `docs/` for **RepoRadar's profile only**
— the judge keeps seeing the real repository, because ablating ground truth alongside the
treatment produces a confused judge agreeing with a confused system. Manifests are copied
verbatim: a repo with no docs still declares its dependencies. At CHARS ≥ 300 the gate's prose
block is *identical* to the control's (both are `README[:300]`), so those arms move only the
derived keywords, the queries, and the hypotheses — which separates **retrieval degradation from
prompt degradation**.

**Pre-registered before the numbers were read:** net@2 decays toward 0 by *abstention*, pooled
precision holds ≥ 0.85 at every budget. **Alarm:** pooled precision < 0.80, or any arm with
negative mean net@2.

| arm | mean net@2 | shown | actionable | precision | abstained | net-negative | pool actionable share |
|---|---|---|---|---|---|---|---|
| control | **+5.17** | 40 | 37 | 0.925 | 1 | 0 | 0.840 |
| 1500 | +3.00 | 45 | 36 | 0.800 | 1 | 1 | 0.712 |
| 300 | +3.17 | 34 | 29 | 0.853 | 2 | 1 | 0.676 |
| **0** | **−0.50** | 33 | 21 | **0.636** | 2 | 1 | 0.568 |

**Both alarm conditions fired at budget 0. The prediction failed.** The actionable share of the
judged pool falls monotonically (0.840 → 0.568), so this is **retrieval degradation**, not a
selection failure — the gate is handed steadily worse pools.

| case | control | 1500 | 300 | 0 |
|---|---|---|---|---|
| `cv` | 4.0 | 2.0 | 5.0 | 2.0 |
| `db` | 10.0 | 9.0 | **0.0** | **0.0** |
| `graph` | 7.0 | 6.0 | 6.0 | 4.0 |
| `rag` | 4.0 | 7.0 | 10.0 | 4.0 |
| `speech` | 6.0 | **−6.0** | **−2.0** | **−13.0** |
| `webdev` (control) | 0.0 | 0.0 | 0.0 | 0.0 |

#### The mechanism: the register mismatch, in its purest form

At budget 0, `speech`'s profile is whatever Whisper's packaging metadata says: *"weak
supervision, robust speech recognition"*. The digest it produces opens with **"Robust Speech
Recognition via Large-Scale Weak Supervision"** — **the Whisper paper itself**, judged 1 — and
continues with seven near-neighbours of Whisper's own method. Strip a repository's prose and its
profile collapses to a bare self-description, so retrieval returns *what the repo already is*.
That is [RESEARCH.md §1](../RESEARCH.md)'s founding diagnosis reproduced under laboratory
conditions: the register mismatch is not a quirk of keyword search, it is what happens whenever
the only thing describing a repository is the repository.

#### The danger zone is a little documentation, not none

`db` is DuckDB — a C++ project whose clone has no manifest the profiler can parse, so at budget
0 its profile is **literally empty: 0 keywords, 0 domains, 0 prose**. The gate scored **all ten
candidates 0** and the case abstained. Zero information produces no question, so nothing is
retrieved and nothing is shown: net@2 0.0, safe.

`speech` at the same budget had a *thin but entirely plausible* profile — and reached **−13.0**.
The failure needs enough information to form a **plausible but wrong** question. That inverts the
intuition the experiment was built on, and it means "how much does the profile contain" is the
wrong safety axis.

#### Nothing inside the system notices

| arm | gate-3 papers | gate-3 precision | gate-2 precision | mean finescale P |
|---|---|---|---|---|
| control | 9 | **1.00** | 0.92 | 0.799 |
| 1500 | 25 | 0.88 | 0.71 | 0.758 |
| 300 | 16 | 0.88 | 0.83 | 0.782 |
| **0** | 17 | **0.53** | 0.70 | **0.709** |

The gate's **top confidence tier degrades to a coin flip** while it issues *more* 3s than the
control, and the calibrated probability barely moves — 0.709 against a gate-3 precision of 0.53.
P5 measured that tier at 76% judge-3 and gate precision at 0.97 on wild pools; both are
irrelevant here. **The gate defends against off-topic junk, not against on-topic uselessness**,
and these papers are maximally on-topic.

The reason nothing notices is structural: **queries, HyDE hypotheses, the gate and the rescore
all consume the same impoverished profile, so they fail coherently.** Internal consistency is
fully preserved. The judge is the only component that sees the real repository, which is the
only reason this is visible at all.

**This refutes the remedy this experiment was opened to test.** The pre-registered alarm text
said a firing precision bar meant `hyde.search_index` needed a similarity floor, since it is
pure top-k by Hamming distance. It does not. The retrieved papers are **not distant** — they are
the correct answer to the wrong question, and no distance threshold separates those.

#### What it does mean

RepoRadar's failure mode on its stated target user is **confident and internally undetectable**.
The remedy is not a better filter but a different information source, and a previous negative
already points at it: P8 killed user-stated wants as *gate* context and concluded **"stated
wants belong in the query, not in the gate."** A thin-docs repository is precisely the case
where the query has nothing else to work from. That is roadmap item 0, which now has a
measured reason to exist. The nearer-term defensive move is a **profile-information floor** —
refuse to run, the way `db` accidentally did, rather than answer a question the repo never
asked.

#### Caveats, and they are load-bearing

**n = 6, and `speech` supplies most of the magnitude.** Excluding it, budget 0 is **+2.00 at
precision 0.800** — degraded, exactly at the alarm boundary, but not negative. **The per-case
sign tests do not clear significance** (1500: p = 0.375; 300: p = 1.000; 0: p = 0.125). The
alarm is a *pre-registered decision rule on pooled precision*, not a p-value, and the
dose-response is a pattern rather than an established effect. `rag` moved the *wrong* way
(4.0 → 10.0 at budget 300), which is the variance the other numbers also sit inside.

**The control drifted −1.33 from the previous day's run of the identical configuration**
(`graph` −3, `speech` −4, same pool sizes) — which is why the control was re-run in-session
rather than reused; that drift alone would have manufactured a first rung of damage before any
documentation was removed.

**And every number here is a ceiling, not an estimate.** Stripping detectron2's README does not
make the hypothesis generator, the gate, or the judge forget detectron2; all of them have
memorised these repositories, and a genuinely obscure private codebase gets none of that help.
Real thin-docs performance sits at or below this by an unknown margin. **Ablation cannot be made
to answer that**, which is exactly why the real thin-docs cases remain worth adding.

**Cost** ~$11, 24 runs, ~2 h. All four arms clean: no HyDE degradations, no collection failures.

### Calibration audit — the map IS decalibrated, and it provably does not matter (2026-08-09)

```bash
uv run python evals/calibrate_finescale.py            # ~$0.30, cached after the first run
uv run python evals/calibrate_finescale.py --analyse  # re-analyse the cache, $0
```

`finescale.SLOPE` and `finescale.INTERCEPT` are the only fitted numbers in the shipped method,
frozen against one stored judge run. The +4.55 headline is downstream of them, and nothing in
the test suite can catch them moving: the tests pin the prompt *bytes*, and this failure mode
is semantic. So the 220 papers in RepoRadar's own top-10 across the 22-case HyDE run — every
one of which the GPT-5.5 judge already scored, with no selection bias inside the top-10 — were
re-gated (Haiku) and re-scored (gpt-4o-mini) and the map's predictions compared against verdicts
already on disk.

**Reproduction first, because nothing else is readable without it.** The rebuilt policy agrees
with the live run on **117 of 121** shown papers (97%); the four misses are one paper each in
`columnar`, `db`, `rl`, `vectordb`, all in the conservative direction. That is the re-run noise
floor of two temperature-0 model calls, and it is why every counterfactual below is computed
*within* the reconstruction (whose own net@2 is +4.36, not +4.55) rather than against the live
figure.

**The map is decalibrated, and the direction is under-confidence.**

| | |
|---|---|
| band papers governed by the map | 126 |
| observed actionable rate | **0.817** |
| mean predicted P | **0.689** |
| calibration gap (paired bootstrap) | **−0.129, 95% CI [−0.187, −0.067]** |
| ECE / Brier | 0.128 / 0.138 |
| band AUC — live vs at fit time | **0.824** vs 0.841 |

| predicted P | n | mean P | actual | gap |
|---|---|---|---|---|
| 0.0–0.2 | 6 | 0.053 | 0.500 | −0.447 |
| 0.2–0.4 | 17 | 0.298 | 0.529 | −0.231 |
| 0.4–0.6 | 10 | 0.517 | 0.600 | −0.083 |
| 0.6–0.8 | 29 | 0.715 | 0.759 | −0.043 |
| 0.8–1.0 | 64 | 0.868 | 0.984 | −0.117 |

The CI excludes zero, so this is not noise. **Every bin under-predicts, and so do 16 of 19
repositories** (residual mean −0.117, sd 0.105) — it is a *global level shift, not per-repository
dispersion*, which was the hypothesis this audit was opened to test. And the **ordering is
intact**: band AUC 0.824 against 0.841 at fit time. Whatever moved, moved the level and left the
rank alone.

**And it is worth nothing.** Two counterfactuals, weakest claim first:

| | mean net@2/case | |
|---|---|---|
| shipped, frozen map | **+4.36** | — |
| LORO refit — fit on 21 repos, score the 22nd | **+4.36** | Δ **+0.00**, 6+/5−/11=, p = 1.00, CI [−0.45, +0.45] |
| *oracle* threshold P ≥ 0.52, chosen on the test set | +4.64 | Δ +0.27 — **an unachievable ceiling** |

The honest counterfactual (LORO, never sees the repository it grades) moves 11 repositories and
helps and harms in equal measure. The oracle — which cheats, and is reported only as a bound —
buys **+0.27 net@2/case**: it flips 12 papers into the digest, 10 of them actionable, +6 net
across all 22 repositories. That is the *entire* recoverable value of fixing a calibration error
whose existence is statistically solid.

**Why a 13-point error costs almost nothing.** The threshold sits in the trough of a bimodal
distribution. Of 126 band papers, **63 sit in [0.8, 0.9) and only 24 are within ±0.1 of 2/3**:

```
  0.0-0.1  #####                                                            5
  0.2-0.3  ##########                                                      10
  0.4-0.5  ####                                                             4
  0.5-0.6  ######                                                           6
  0.6-0.7  ##########                                                      10
  0.7-0.8  ###################                                             19
  0.8-0.9  ############################################################### 63
```

The near-binary shape of the 0–3 gate **reappears one level down**: the rescore de-quantized it
enough to *order* the band (AUC 0.824) but not enough to spread it across the decision boundary.
A calibration error can only cost where papers actually sit, and they do not sit there. The
second reason is structural: P = 2/3 is the *derived* breakeven, so a paper at the boundary is
worth 3p − 2 ≈ 0 by construction — moving a correctly-derived threshold across a well-ordered
set has near-zero expected value almost regardless of where it moves.

**The prediction that opened this audit was wrong, and the way it was wrong is instructive.** It
came from reading `reporadar_top10` against `reporadar_toppicks`: 43 judge-actionable papers sit
in RepoRadar's own top-10 unshown, worth +11 net@2 across the six repositories where showing more
would help. That reading is unpaired. The same threshold move costs −80 across the nine
repositories where the gate is correctly strict (`cli`/`http` −17 each), and showing the full
top-10 everywhere scores **+1.41 against +4.55**. The LORO refit is the paired version of the
same question and it nets to zero. *An accounting of only the upside of a threshold move is not
a counterfactual.*

**What this settles, and what it does not.** It closes recalibration as a direction: the ceiling
is +0.27/case and no out-of-sample method reaches it. It does not identify the *cause* of the
gap. The fit population's band was 74.3% actionable and the live band is 81.7% (Wilson
[0.741, 0.875]) — consistent with HyDE having enriched the band, but the two are statistically
indistinguishable at n = 126, so "shipping HyDE decalibrated the map by improving the pool" is a
hypothesis this data cannot confirm. What it does give the drift monitor is a baseline and a
split of concerns: **AUC is the alarm, the gap is a gauge.** Ordering holding at 0.824 means the
scorer still works; the level drifting 13 points means only that a number nobody's decision
depends on has moved.

**Cost** ~$0.30 (220 Haiku gate calls + 220 gpt-4o-mini logprob calls), cached per case, and the
analysis re-runs free. The harness now persists `llm_score`/`finescale`/`finescale_p` in the
results file, so the next audit costs $0 — the docstring had claimed those fields landed there
since the rescore shipped, and they did not.

### The live run: the offline replay was right (2026-08-08)

Everything above was an offline replay of a stored run. This is the shipped path executed
end to end on all 22 repos — fresh arXiv collection, live ranking, live Haiku gate, live
fine-scale rescore through `reporadar.finescale` itself — added to the harness as
`--rr-finescale`. Same flags as the frozen run (`--rr-pool 50 --rr-rerank --rr-all-time
--rr-hybrid`), 22/22 cases completed, no exclusions.

| | replay (frozen run) | **live run** | Opus baseline |
|---|---|---|---|
| RepoRadar + fine-scale | +3.14 | **+3.18** | — |
| RepoRadar, show-all | +1.91 | +1.86 | — |
| baseline | — | — | +1.82 |
| fine vs show-all | +1.23 | **+1.32** (8+/2−, p = 0.109) | |
| fine vs Opus | +1.32 | **+1.36** (10/6/6, p = 0.45) | |
| precision of the digest | 0.89 | **0.91** | 0.94 |
| papers shown | 102 | 97 | 49 |
| net-negative cases | 0 | **0** | 1 |

Every headline number reproduces within noise, and the win/loss/tie record against Opus
is identical (10/6/6). The sign tests are unchanged in character: **p = 0.109 against
show-all and p = 0.45 against Opus**, so this remains a clear mean improvement that is
*not* established as reliably per-repo at n = 22.

**The per-case numbers moved a lot, and that is the finding.** Collection is stochastic,
and a fresh fetch reshuffled individual repos hard — `ann` show-all went +4 → **−2**,
`compiler` −5 → **−7**, `columnar` +4 → **+7**, `crypto` +3 → **−2**. Show-all had **six**
net-negative repos this run rather than four. The aggregate held anyway, which is the
useful part: the offline replay was measuring something stable, not one lucky draw. It
also means per-case values in any single run should be read as draws, not properties.

**The stage eliminated all six negatives**, including two the replay never had a chance
to fix because that draw did not produce them (`crypto`, `encryption`). Its one real cost
this run was `columnar`, where it dropped two genuinely actionable papers (+7 → +5) — and
`diffusion` again gave up one. That is the shape of the trade: it pays for itself several
times over on bad bands and occasionally clips a good one.

Band behaviour was sensible across the spread: four cases had no band at all, four kept
every band paper (5/5, 8/8, 4/4), and the two worst repos abstained almost completely —
`compiler` kept **0 of 9** and `llminfer` **1 of 7**. Nothing saturated, nothing collapsed.

**Cost of the run: about $0.30 in live calls.** All 22 Opus baselines came from cache
(discriminators verified matching before launch), and 1,325 of the judge verdicts were
already stored, so only **50 new judgements** were needed. The live spend was ~1,100 Haiku
triage calls, ~150 gpt-4o-mini rescores, and those 50 judge calls. A cold run of the same
thing would be roughly $20 in baselines plus judge.

### Correction — the recorded +2.91 evaluated a map that does not ship (2026-08-08)

The figure above came from `exp_features.loro_fit`, which is the E5 feature-combination
harness: it standardises features and picks the L2 strength `C` by an **inner LORO on
AUC**. It chose C = 0.1 on every fold. The map that actually ships is a plain
unstandardised logistic at the default C = 1.0 — so **+2.91 scored a more heavily
shrunken map than the one in `finescale.py`**. Its LORO-honest counterpart:

| LORO variant | policy mean net@2 |
|---|---|
| **plain LR, C = 1.0 — what ships** | **+3.14** |
| StandardScaler + LR, C = 1.0 | +3.14 |
| inner-CV C by AUC (selects C = 0.1) — what was recorded | +2.91 |

Both are leave-one-repo-out and neither is wrong as arithmetic; they estimate different
maps. The mechanism behind the gap is worth keeping, because it is a trap: **AUC is
rank-only and invariant to any monotone transform of the score**, so selecting
regularisation by AUC cannot see where P crosses 2/3 — the one thing a thresholded
policy depends on. Shrinking the slope leaves AUC untouched and moves the decision
boundary. Selecting a calibration by a ranking metric is choosing on a criterion blind
to the quantity being used.

So the honest number for the shipped configuration is **+3.14 vs +1.91 show-all**, and
the earlier +2.91 stands as a conservative estimate of a nearby variant. The correction
happens to be favourable, which is exactly when to state the mechanism rather than just
the new figure.

### Against the Opus baseline, all 22 repos (2026-08-08)

$0 — the baseline's own picks and verdicts are in the same frozen run file, so this is
the head-to-head the [dead-heat section](#the-opus-baseline-on-all-22-repos-a-dead-heat-reached-by-opposite-strategies-2026-08-07)
left open, computed rather than projected.

| | mean net@2 | vs Opus (paired) | precision | papers shown | abstentions | net-negative cases |
|---|---|---|---|---|---|---|
| **RepoRadar + fine-scale** | **+3.14** | **+1.32** | 0.89 | 102 | 7/22 | **0** |
| RepoRadar, show-all | +1.91 | +0.09 | 0.73 | 132 | 5/22 | 4 |
| Opus 4.8 baseline | +1.82 | — | 0.94 | 49 | 4/22 | 1 (`linter` −6) |

Win/loss/tie against Opus goes from **8/8/6 to 10/6/6**. The sign test is **p = 0.45** —
so on 22 paired cases this is *not* a statistically significant win, and should be read
as "clearly ahead on the mean, not established as reliably better per repo".

| case | show-all | +fine-scale | Opus | Δ vs Opus |
|---|---|---|---|---|
| `db` | +10 | **+10** | +3 | **+7** |
| `diffusion` | +10 | +9 | +2 | **+7** |
| `linter` | −2 | **0** | −6 | **+6** |
| `ann` | +4 | **+8** | +3 | **+5** |
| `peft` | +7 | +7 | +2 | **+5** |
| `storage` | +7 | +7 | +2 | **+5** |
| `systems` | +1 | **+5** | +1 | **+4** |
| `crypto`, `cv`, `rl` | | +3 / +4 / **+4** | +2 / +3 / +3 | +1 each |
| `columnar`, `compiler` | +4 / **−5** | +4 / **+2** | +4 / +2 | 0 each |
| `speech` | +2 | +2 | +3 | −1 |
| `graph`, `llminfer`, `numerics` | | +1 / +2 / **0** | +3 / +4 / +2 | −2 each |
| `rag`, `vectordb` | 0 / **−5** | 0 / **+1** | +3 / +4 | −3 each |
| `cli`, `encryption`, `http`, `webdev` | 0 | 0 | 0 (abstained) | 0 |

**The prediction from the dead-heat section was that removing the four false-positive
cases would put RepoRadar near +2.55 without touching retrieval. It removed all four and
landed at +3.14** — the gate now has *zero* net-negative repos, against Opus's one. The
remaining six losses are a different failure and not one this stage can fix: `rag`,
`graph`, `llminfer`, `numerics`, `speech`, `vectordb` are cases where Opus finds two or
three good papers that RepoRadar's admitted set either never contained (`rag` admits
nothing at all) or where the band genuinely ran thin. That is recall, which P5 already
identified as the gate's real weakness (0.60) and which a precision stage cannot help.

**What this comparison is and is not.** It is a fair offline replay: same 22 repos, same
candidate pools, same judge, same frozen file, and the fine-scale scores were computed
from gate-time information only. It is **not** a fresh end-to-end run of the shipped
code path — no live `rr update` with `triage.finescale.enabled` has been benchmarked, so
collection-time variation is not in these numbers. And the caveat on the metric from the
dead-heat section stands unchanged: net@2 charges 2 for a false positive, so it rewards
shyness, and a precision stage is exactly the kind of change that metric flatters.

### Can the same thing be done without OpenAI? No — measured (2026-08-08)

The one uncomfortable part of the winner is the vendor: reading a token distribution
needs logprobs, and Anthropic exposes none. The obvious workaround is Monte Carlo —
send the *same* 0-9 prompt to Haiku N times at the API's default temperature and take
the mean of the sampled digits. Same prompt, same estimand, approximate estimator
instead of an exact one (`exp_finescale.py --arm haiku --samples 10`, 10× the calls):

| arm | band AUC (A) | band AUC (A300) | judge-3 AUC (A) | modal share > 0.9 |
|---|---|---|---|---|
| **gpt-4o-mini logprobs** | **0.841** | **0.761** | 0.879 | 0.233 |
| Haiku, 10 samples | 0.590 | 0.547 | 0.778 | 0.438 |

**The estimator is the problem, not the judge.** Haiku's judge-3 ordering is respectable
(0.778), so it is reading the papers competently. What fails is the sampling: in 44% of
papers, 9 or 10 of the 10 draws returned the *identical* digit — at default temperature
the model is nearly deterministic on this task, so the draws re-read the mode instead of
revealing the distribution around it. Monte Carlo cannot recover a distribution the
sampler will not explore, and at N=10 the resolution is 0.1 against a continuous
reading. Raising N buys resolution but not exploration, so it does not address the cause.

So the OpenAI dependency is load-bearing rather than incidental, and the Anthropic-native
fallback is measured-and-rejected rather than untried.

### What this changes — shipped (2026-08-08)

The ranking diagnostic's conclusion — "nothing measured can order within the band" — is
now false: a $0.01-per-repo rescore can. Shipped as `reporadar/finescale.py`, opt-in via
`triage.finescale.enabled`, off by default because of the vendor requirement above. Each
paper the 0-3 gate scores *exactly at* `min_actionable` is rescored on the 0-9 rubric,
mapped through the frozen logistic, and reaches Top Picks only at P ≥ 2/3; papers scoring
3 are untouched, since only the band was unreliable.

Three guards ride along, each against a failure that would otherwise be silent: the repo
block is now one shared function (`triage.repo_context_block`) that both the product and
`band_testbeds.py` call, so the prompt the map was fitted against cannot drift from the
prompt that ships; a test asserts the product prompt is byte-identical to the benchmark's;
and a run that scores under half its band skips the gate with a warning instead of
demoting everything into an accidental abstention.

Caveats that stay attached: the map was fitted on judge labels (frozen thereafter; drift
unmeasured), n = 22 keeps each arm's sign test at p ≈ 0.09-0.11, and all labels are
GPT-5.5-relative — mitigated but not eliminated by the Sonnet cross-check.

## Adaptive digest size does not survive contact with the gate's score distribution (2026-08-07)

$0 — derived from the sweep already in the 22-case run.

The previous section proposed "show every score-3 paper; if there are none, show at most k
score-2 papers, and abstain below some floor". A reader raised two objections before any of
this was measured, and both are correct.

**1. There is no interior optimum in k.** Expected net@2 per paper shown, at precision p, is

    p·(+1) + (1−p)·(−2)  =  3p − 2

which is positive above p = 0.67 and negative below it. If the papers are **unranked**,
truncating to k scales the total by k without changing the sign: either show everything or
show nothing. A "show at most k" rule is only meaningful if something orders the set, and
nothing does — the gate gives them all a 2.

**2. Choosing k per repo is fitting to the benchmark.** It would use the outcome to pick the
knob that produces the outcome.

### And the signal it would need does not exist

Score distribution of admitted papers, per case, with the precision of each band:

| case | score-3 | score-2 | actionable of the 10 | p@min≥2 | net@2 | net@2 if score-3 only |
|---|---|---|---|---|---|---|
| `diffusion` | **0** | **10** | 10 | **1.00** | **+10.0** | **0.0** (abstains) |
| `vectordb` | **2** | **8** | 5 | **0.50** | **−5.0** | **+2.0** |
| `columnar` | 0 | 10 | 8 | 0.80 | +4.0 | 0.0 |
| `numerics` | 0 | 10 | 6 | 0.60 | −2.0 | 0.0 |
| `db` | 9 | 1 | 10 | 1.00 | +10.0 | +9.0 |
| `linter` | 0 | 1 | 0 | 0.00 | −2.0 | 0.0 |

**`diffusion` and `vectordb` are the whole argument.** Both admit ten papers, both have almost
no score-3s — and one is 100% actionable while the other is 50%. At gate time they are
indistinguishable. Any rule that abstains on `vectordb` abstains on `diffusion` and gives back
+10 to save +7.

The same holds for `columnar` (0.80) against `numerics` (0.60): identical distributions, 10
score-2 papers each, opposite verdicts.

Across the 17 cases that returned anything, the share of admits scoring 3 correlates with
precision at **r = +0.30** — nowhere near enough to gate on.

### What this leaves

Truncation is a dead end and score-3-only is too shy — the sweep already measured it at mean
**+0.50 with 16/22 abstentions**, giving away `diffusion` +10, `columnar` +4 and `crypto` +3 to
avoid `vectordb` −5 and `compiler` −5.

**The requirement is reliable ranking *within* the score-2 band**, which is where the variance
lives: precision inside that band runs from 0.00 (`linter`) to 1.00 (`diffusion`, `crypto`).
That is the same conclusion the ranking diagnostic reached from the other direction, and
nothing measured so far can do it — not the gate's own score, not the heuristic ranker, and
not a threshold on either.

## The Opus baseline on all 22 repos: a dead heat, reached by opposite strategies (2026-08-07)

The headline comparison has always been 12 cases. This is the first time RepoRadar and the
Opus baseline have been scored against each other on the full 22-repo benchmark.

| | mean net@2 |
|---|---|
| **RepoRadar** (Top Picks, min≥2) | **+1.91** |
| **Opus 4.8 baseline** | **+1.82** |
| paired delta | **+0.09** |

8 cases better, 8 worse, 6 tied. A dead heat — and the means conceal two opposite strategies.

| | Opus | RepoRadar |
|---|---|---|
| papers returned | 0–4, usually 2–3 | 10 |
| precision | ~1.00 on all but one case | 0.73 at pool scale |
| abstentions | **4/22** | rare |
| false positives | **1/22** | 4 cases at −2 to −5 |

**Opus is precise and shy; RepoRadar is broad and noisy.** They tie by cancelling out.

| case | RepoRadar | Opus | | case | RepoRadar | Opus |
|---|---|---|---|---|---|---|
| `db` | **+10.0** | +3.0 | | `graph` | +1.0 | **+3.0** |
| `diffusion` | **+10.0** | +2.0 | | `rag` | 0.0 | **+3.0** |
| `peft` | **+7.0** | +2.0 | | `rl` | +2.0 | **+3.0** |
| `storage` | **+7.0** | +2.0 | | `speech` | +2.0 | **+3.0** |
| `ann` | +4.0 | +3.0 | | `llminfer` | +1.0 | **+4.0** |
| `columnar` | +4.0 | +4.0 | | `numerics` | **−2.0** | +2.0 |
| `cv` | +4.0 | +3.0 | | `linter` | −2.0 | **−6.0** |
| `crypto` | +3.0 | +2.0 | | `compiler` | **−5.0** | +2.0 |
| `systems` | +1.0 | +1.0 | | `vectordb` | **−5.0** | +4.0 |
| `cli`/`encryption`/`http`/`webdev` | 0.0 | 0.0 (abstained) | | | | |

**Where RepoRadar wins, it wins by volume** — 10 good papers against Opus's 2 or 3. **Where it
loses, it loses on false positives** on repos with little to find, exactly where Opus returns
two correct papers or abstains.

### The one structural gap

RepoRadar's four losing cases cost **−14 net@2**, or **−0.64 of the mean**. They are not
retrieval failures; they are *output-size* failures — the digest returns 10 papers whether or
not 10 good ones exist. Removing that cost alone would put RepoRadar at ~+2.55 against Opus's
+1.82, without touching retrieval.

The sweep says a flat stricter gate is not the fix: `min≥3` reaches mean precision **0.97 with
0 false positives** but abstains on **16/22** for a mean of +0.50. What is untested is a
*variable* output size keyed on the confidence distribution rather than a fixed cut.

> **A caveat on the metric, not the result.** net@2 charges 2 for a false positive and credits
> 1 for a hit, so it rewards shyness. Opus's strategy is partly an artefact of that: returning
> 2 papers it is sure of is close to optimal *for this scoring function*. P5 measured the
> gate's real failure as **recall** (0.60), and a change that improves net@2 by returning less
> makes that worse. Optimising net@2 and serving a reader who wants to find things are not the
> same objective, and this table is evidence about the first one.

### Cost, which the benchmark does not score

The baseline costs **~$0.80 per repo per run** (Opus 4.8 with web search; the per-case figures
are in the run log). RepoRadar's full path costs **~$0.05**. Parity at 6% of the cost is a
result the mean does not show.

## Gating the whole pool end to end: a wash (2026-08-07)

```bash
uv run python evals/run_judge_eval.py --baseline cli --rr-rerank --rr-all-time --rr-sweep \
    --rr-pool 300 --rr-prose-chars 300      # arm A: gate the whole pool
uv run python evals/run_judge_eval.py --baseline cli --rr-rerank --rr-all-time --rr-sweep \
    --rr-pool 50  --rr-prose-chars 300      # arm B: the shipped depth
```

Both arms, **same 22 cases, same session, same code**, differing only in how many candidates
the gate sees. The derived arithmetic predicted +3 against a measured +2.75 — a gain too
small to see. Measured end to end, it is not there at all.

| all 22 cases | mean net@2 |
|---|---|
| pool 300 (gate everything) | **+1.73** |
| pool 50 (shipped depth) | **+1.91** |
| paired delta | **−0.18** |

Per case: **6 better with pool 300, 6 worse, 10 tied**, against a per-case sd of 1.67.
**A wash.** Not a win, not a loss.

> #### Corrected 2026-08-07 — the first version of this table read −0.95
>
> `db` and `storage` hit arXiv 429s during collection in arm A, judged **0 papers**, and were
> scored as an honest **net@2 = 0.0** that entered the mean — against +10.0 and +7.0 in arm B.
> That single harness failure supplied **−17 of a −21 total delta**, and it removed two of the
> benchmark's *strongest* cases rather than two average ones.
>
> Re-run on both arms after the fix (PR #94: throttles retried against a 15-minute time budget,
> and a failed collection excluded rather than scored), all four collected cleanly with no
> throttling:
>
> | | pool 300 | pool 50 |
> |---|---|---|
> | `db` | **+10.0** | **+10.0** |
> | `storage` | **+7.0** | **+7.0** |
>
> Both are **exact ties** — the same wash the other 20 cases show, now on two cases that had
> never been measured at all. The corrected delta is **−0.18**, and the conclusion is
> unchanged; what changed is that it no longer rests on a number a failed fetch moved by
> nearly a full point.

### Why, and it is the same answer as the ranking diagnostic

The digest window is **10 papers**, applied after gating and reranking. Gating 300 candidates
instead of 50 surfaces more admits, but the reader still sees 10, and the extra admits are
not better *ordered* — the gate's own score is nearly binary (only 4–14% of admits get a 3),
so ordering within the admitted set falls through to the heuristic ranker, which is flat.

Deeper gating changes *which* arbitrary actionable-ish papers fill the window. It does not
change how good the window is. That is exactly what the ranking diagnostic predicted from
$0 of labels, and it is the same constraint from a third direction:

* P5: the gate's failure is recall (0.60), not precision
* the ranking diagnostic: 94–100% of admits are already actionable; the variance is in the
  score-3 band
* here: giving the gate 6× the candidates moves nothing, because nothing orders what it admits

**The bottleneck is not how many papers the gate sees. It is that nothing ranks what it
returns.**

### What the harness could not tell us

The Tier B results JSON records the *judged* pool (10–14 papers per case) and never the
**candidate** count, so "how many papers did the gate actually see" is not answerable from
the artifact — it had to be measured separately (`gate_full_pool`: 56–296 per repo, mean 227).
Worth recording, since the whole comparison is about that number.

### Also settled: nothing was silently capping the pool

`--rr-pool 300` does reach the gate: `candidate_n` flows into `reporadar_ranked(top_n=...)`
and every returned candidate is triaged. The `10/10 scored` line in the log counts the
returned top-10, not the triage pool. I misread it as a silent cap on first inspection; it
is not one.

## Gating the whole pool: affordable, worth doing, and worth less than it sounds (2026-08-07)

```bash
uv run python evals/gate_full_pool.py --build     # 22 real pools from live arXiv, free
uv run python evals/gate_full_pool.py             # ~$0.07 gate + ~$2.34 judge
```

`triage.top_k` is 15, set in the Feature 6 commit with **no recorded rationale**. The obvious
objection to raising it is cost, and cost turns out not to be the reason.

**First correction: the pool is ~227 papers per repo, not ~2,000.** The 2,030 figure quoted
throughout this file is `diagnose_pool`'s *cross-case total* over 12 repos. Measured here
across 22: **56 to 296 papers each, 4,984 total**, with three cases landing exactly on 296 —
the ceiling of `#queries × max_results_per_query` minus duplicates. So `top_k: 15` shows the
gate **6.6% of the pool**, and gating all of it costs **$0.05 per repo per run** and 5 minutes
sequential (~20 s at 16-way concurrency). Cost was never the constraint.

300 papers sampled uniformly, balanced across all 22 repos; every admit plus 40 non-admits
judged.

| | predicted | measured | |
|---|---|---|---|
| admit rate | 10–25% | **11.0%** [7.9%, 15.0%] | MET |
| gate precision | ≥0.80 | **0.73** [0.56, 0.85] | below |
| kill bars | >40% admit or <0.60 precision | neither | survives |

**Precision was substantially a property of the strata.** 0.92 on the labelled set, 0.97 on
P5's stratified wild sample, **0.73** on the real pool. The CI is wide (33 admits) and does
not exclude 0.80, but it comfortably excludes 0.92.

### The funnel, per repo

| | papers |
|---|---|
| pool | **227** |
| gate admits (11%) | 25 |
| — of which actionable (0.73) | **18** |
| gate rejects | 202 |
| — of which actionable (12% leak) | **25** |
| **actionable in the pool** | **~43 (19% base rate)** |
| **gate recall** | **42%** |

The gate finds fewer than half the actionable papers in the pool and pays 27% junk for the
ones it finds. Both figures are consistent with P5's precision 0.97 / recall 0.60 once you
account for P5 measuring on strata this run does not restrict to.

### What raising `top_k` would actually buy

| | net@2 per repo |
|---|---|
| show all 25 admits | **+5** |
| show 15 of them at the same precision | **+3** |
| measured Tier B Top Picks today | **+2.75** |

**A modest gain, not a transformation.** It roughly quadruples the actionable papers reaching
the digest stage — today's top-15 path can contain at most a handful — but 27% of the admits
are junk, so the extra reach is partly spent on false positives. The honest summary: raise
`top_k` because it is nearly free and strictly more informative, but expect single-digit
net@2, and expect the win to come from what happens *after* the gate.

### The ranker is not as bad as I have been saying

`diagnose_ranker` measured ranks 1–10 at 31% actionable and 11–50 at 33%, and I have been
quoting that as "the ranker is at chance". Against this run's **19% pool base rate**, the
ranker's top-50 is about **1.7× denser than the pool**. It concentrates; what it cannot do is
*order within* the band it concentrates. That is a narrower failure than the one I described,
and it means replacing the ranker wholesale is not obviously right — bypassing it with a
whole-pool gate is a different bet from improving it.

### Eight repos admitted nothing

`cli`, `compiler`, `http`, `linter`, `rag`, `speech`, `systems`, `webdev` — 0 admits from ~14
sampled each. For the non-ML ones that is the correct answer and the negative controls working.
`rag` and `speech` at 0/13 are surprising given their labelled-set base rates (26%, 38%) and
are most likely small-sample noise, but they are worth a second look before any conclusion
rests on per-repo admit rates.

### Caveats

- 300 papers, ~14 per repo. Per-repo admit rates carry no weight individually; the pooled
  11% is the number with a usable CI.
- The judged subsample is all 33 admits plus 40 random non-admits, so precision is exact on
  the admits and the 12% leak is estimated from 40 papers.
- All of it is at the judge's ≥2 bar. P7 measured a second judge one notch stricter, so a
  ~0.55 multiplier applies to every absolute rate here; the comparisons are unaffected.

> **Harness failure worth recording.** The first pool build hit arXiv's rate limit after ~15
> cases. `collect_live_papers` swallows a 429 and returns an empty list **without raising**,
> so the builder cached **seven empty pools** — and a cached empty pool is worse than a
> missing one, because the next run skips it as "already built" and those repos vanish from
> every downstream number while the report still prints a confident admit rate. Same class as
> the citation hop's `failed_chunks`. Now refuses to persist an empty fetch, waits between
> cases, and has four tests.

## Ranking diagnostic — the admitted set does not need re-ranking for precision; it needs ordering by the score-3 band (2026-08-07)

$0 — every number here comes from labels already on disk (`diag_triage_*.json`, `label_pool.json`).

P5 left the plan pointing at "ranking within the admitted set": the gate admits ~38 papers per
repo from a 100-candidate band and a digest wants ~10. Before building a re-ranker, the cheap
question is what there would be to rank *with*. The answer moves the target.

### Among admitted papers, precision is already all but perfect

| stratum (admitted only) | n | judge ≥2 | judge =3 |
|---|---|---|---|
| `hop-coupling3+` | 15 | 100% | **53%** |
| `hyde-top100` | 36 | 94% | **22%** |
| `hyde-100-1k` | 7 | 100% | 0% |
| `hop-coupling1` | 6 | 100% | 0% |
| `hyde-1k-10k` | 2 | 100% | 0% |

**94–100% of what the gate admits is actionable.** There is nothing to gain by re-ranking for
"actionable vs not" — the gate has solved that, which is the same thing P5 said from the other
side (precision 0.97, recall 0.60). A re-ranker aimed at precision would be optimising a
metric that is already at ceiling.

**All the remaining variance is in the score-3 band**, and free structural features order it:
53% for high-coupling papers, 22% for the HyDE head, **0% everywhere else**.

### The gate's own score is a real signal that almost never fires

| arm | admits | gate=2 → judge=3 | gate=3 → judge=3 |
|---|---|---|---|
| `prose 300` (602 labels) | 125 | 38% | **76%** (n=17) |
| `wants` (920 labels) | 109 | 51% | **80%** (n=5) |
| P5 wild sample | 66 | 22% | **67%** (n=3) |

A gate verdict of 3 roughly doubles the chance the judge also says 3. But the gate says 3 on
**4–14% of its admits**, so it cannot order 38 papers down to 10 on its own. Note it is *not*
better at the ≥2 bar — 82% vs 94% on prose-300 — so it is specifically a top-band signal,
not a general confidence score.

### What a digest would actually gain

| ordering of ~38 admits | score-3 papers in the top 10 |
|---|---|
| arbitrary | **~2.4** |
| restricted to the two dense strata | **~3.1** |
| high-coupling first (if enough exist per repo) | up to ~5.3 |

### The restated problem

**"Rank the admitted set" is the wrong objective; "surface the score-3 band" is the right
one.** Score 3 is "directly addresses a known limitation" — the bar the 48 gold targets were
drawn at, and the one P5 showed the pre-registered separation lands on. Ordering for it has a
measured ceiling of roughly double the current rate, from features that cost nothing to
compute.

**This is a diagnostic, not a result.** Only **66** admitted papers carry a judge label, and
the per-stratum admit counts are 36 / 15 / 7 / 6 / 2. The 0% cells are three-to-seven papers
deep. It is enough to redirect the work and nowhere near enough to fit a ranker on: a properly
powered version needs judge labels stratified over *admitted* papers specifically, which is
P5's design pointed one stage further down the funnel.

## P8 — killed: telling the gate what users want makes it strictly worse (2026-08-07)

```bash
uv run python evals/fetch_wants.py                                   # free, ~1 request/repo
uv run python evals/diagnose_triage.py --repo-context wants          # ~$0.10
uv run python evals/compare_triage.py \
    evals/.work/diag_triage_{prose300,wants}.json
```

Five purpose-statement arms converge at **+73..+95** net@2, all derived from documents that
describe what a project **is**. The rubric's score-3 band asks for evidence a paper "directly
addresses a known limitation", and no document-derived arm supplies that. An issue tracker
states it outright. P8 appended the **top 15 open issues by reaction count, verbatim**, to the
shipped prose-300 prompt — deliberately not the failed `improvement_areas` arm (+70), which
was LLM-inferred *and paraphrased*.

**It is the worst arm ever measured.**

| paired on 602 papers | precision | recall | net@2 |
|---|---|---|---|
| `prose 300` (baseline) | 0.92 | **0.68** | **+95** |
| `wants` (prose 300 + verbatim issue titles) | 0.92 | **0.41** | **+57** |
| delta | +0.00 | **−0.27** | **−38**, 95% CI **[−55, −21]**, P(Δ≤0) = 1.000 |

7 papers fixed, **49 broke**. Against the `keywords` control it is still negative
(+73 → +57, −16, CI [−40, +8]). All three pre-registered criteria fail, and the kill fires
harder than it was written: P8 said "kill if it lands inside the +85..+95 band"; it landed
**below every arm in the study**, including the one with no purpose statement at all.

### The mechanism, and it is not the one I expected

**Precision is unchanged and recall collapses.** The wants block does not fool the gate into
admitting junk — it makes it reject work it previously accepted. My first guess was that
most-reacted issues are engineering requests no paper addresses ("Python 3.10 support",
"Support MergeOperator in Java", "Make age parallel"). The data refutes that: `peft` and
`diffusion` have the *most* research-flavoured trackers in the set ("Add EWoRA to PEFT",
"APG: Eliminating Oversaturation...") and they are where the damage is worst.

The real pattern is that **damage scales with how much there was to lose**:

| case | actionable base rate | prose300 recall | wants recall | Δ |
|---|---|---|---|---|
| `peft` | 84% | 0.92 | 0.55 | **−0.37** |
| `diffusion` | 64% | 0.79 | 0.36 | **−0.43** |
| `speech` | 38% | 0.60 | 0.55 | −0.05 |
| `cv` | 33% | 0.61 | 0.39 | −0.22 |
| `systems` | 31% | 0.58 | 0.42 | −0.17 |
| `rag` | 26% | 0.75 | 0.50 | −0.25 |
| `rl` | 24% | 0.47 | 0.12 | **−0.35** |
| `graph` | 22% | 0.36 | 0.27 | −0.09 |
| `crypto` | 14% | 0.57 | 0.29 | −0.29 |
| `http` | 2% | 0.00 | 0.00 | 0.00 |

Recall falls in **9 of 10 cases**, and base rate correlates with the loss at **r = −0.61**.

The reading the evidence supports: a list of 15 named wants **replaces the question**. The
gate stops asking "would this improve the project" and starts asking "is this on the list" —
so every actionable paper that does not match one of 15 specific issues gets rejected. Repos
where most papers are genuinely actionable have the most to lose, which is exactly the
ordering above.

**An internal control confirms the block is the cause.** `speech` (whisper) surfaced **zero**
open issues, so its `wants` prompt is byte-identical to its `prose300` prompt. Its recall
moved −0.05 — one paper, noise. The one case with an empty block is the one case that did not
move.

### Pre-registered scorecard

| | predicted | measured | |
|---|---|---|---|
| net@2 | ≥+105, CI excluding 0 | **+57** | FAILED |
| `graph` recall | 0.36 → ≥0.50 | **0.27** | FAILED |
| negative controls stay clean | no new false positives | `http` 0/0, `webdev` 0 fixed 0 broke | held |

**Kill clause fires.** Information *type* is not the gate's constraint — and P8 was the single
gating bet in the plan. Per its own terms this "directly calibrates item 0's triage arm before
item 0 spends on hand-authored goals": a user stating what they want, fed to the **gate**,
should be expected to narrow it the same way. §5.3 already noted that `improvement_areas` had
never been tried on *retrieval*, which P4 and P5 now show is where the reach actually is.
**Stated wants belong in the query, not in the gate.**

### Caveats

- **The negative controls are weaker than P8 assumed.** It predicted their trackers would be
  "full of feature requests, making them a sharp control". `webdev` (flask) surfaced 2 issues
  and `speech` 0. The controls stayed clean, but they were not the sharp test intended.
- Ranking by reactions is one selection rule among several. A tracker filtered to
  algorithmic/long-open issues might behave differently — but that is a *new* arm, and this
  one is dead as specified.
- 602 paired papers, 12 repos. The labelled set has since grown to 920 across 17 cases; the
  comparison is deliberately restricted to the 602 the +95 baseline was measured on.

> **Harness fix on the way in.** `diagnose_triage.py` wrote its per-arm output
> unconditionally, so a 3-paper `--case rag --limit 3` smoke run overwrote the arm's
> 602-paper file — the same "partial run destroys whole-set results" failure that cost data
> twice in the hop-pool scripts, arriving by a different door. It now merges by `(case, id)`,
> which is safe because the arm is already encoded in the filename.

## P7 — the two judges rank the same papers the same way; they sit at different strictness (2026-08-06)

```bash
uv run python evals/second_judge.py --dry-run   # sample + prompt-hash check, $0
uv run python evals/second_judge.py             # ~$2.20 of Sonnet
uv run python evals/second_judge.py --report    # re-derive, $0
```

Every labelled-set decision here rests on **single-sample GPT-5.5 verdicts** deciding
differences of ±10 to ±22 net@2, and nothing bounded their noise. P6 showed the judge rewards
roughly the right thing *on average*; that is validity, and it says nothing about whether an
individual verdict is reproducible — which is what those differences are made of.

200 labels, stratified by case and verdict, re-judged by **Sonnet** with a byte-identical
rubric. **All 12 cases reproduced their stored `_prompt_hash` before a call was made**, so
the two judges answered the same question; a mismatch would have excluded the case.

| statistic | value |
|---|---|
| exact agreement on 0–3 | 50% |
| agreement on the ≥2 cut | 78% |
| **Cohen's kappa (≥2 cut)** | **0.507** — below the ≥0.60 prediction, above the 0.40 kill bar |
| quadratic-weighted kappa (0–3) | **0.727** |
| base rate, GPT-5.5 / Sonnet | **40% / 22%** actionable |

### The disagreement is a strictness offset, not a ranking difference

| GPT-5.5 ↓ / Sonnet → | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| **0** | **58** | 0 | 0 | 0 |
| **1** | 36 | **22** | 1 | 2 |
| **2** | 4 | 36 | **8** | 0 |
| **3** | 0 | 1 | 20 | **12** |

Almost all the mass is on or one step below the diagonal. GPT's 0s are Sonnet's 0s — **58 of
58, no exceptions**. GPT's 2s are mostly Sonnet's 1s; GPT's 3s are mostly Sonnet's 2s. That is
one judge being consistently one notch stricter, which is why the ordinal kappa (0.727) is so
much higher than the binary one.

Moving only the second judge's threshold makes it explicit:

| GPT ≥2 vs Sonnet ≥ | kappa | agreement |
|---|---|---|
| **≥1** | **0.711** | **86%** |
| ≥2 (the shipped cut) | 0.507 | 78% |
| ≥3 | 0.151 | 64% |

**This distinction decides the remedy.** Genuine disagreement would mean the labels need
adjudication before any further arm is run. A calibration offset means the *rankings* are
reproducible and the threshold is a free parameter — and, crucially, that a **paired
difference between two arms scored by the same judge largely cancels the offset**, which is
what almost every conclusion in this file is.

### Does the +22 survive a different judge?

| prose300 − keywords, on these 200 papers | net@2 |
|---|---|
| under GPT-5.5 labels | **+2** |
| under Sonnet labels | **+20** |

Sign preserved, magnitude far above half — the pre-registered test passes. **It passes weakly
and the reason matters**: the GPT-labelled delta on this subset is only +2, so "≥half of +2"
is nearly vacuous. The subset is stratified *by verdict* to estimate kappa well, which
deliberately over-samples 2s and 3s and distorts a net@2 recomputation. The informative
reading is directional and it is a strong one: under a **stricter** judge the prose-300
advantage is *larger*, not smaller, so the +22 is not an artifact of GPT-5.5's particular
noise.

### What this changes about every number in this file

**Relative comparisons hold; absolute levels are judge-specific.** GPT calls 40% of this
sample actionable and Sonnet 22% — a factor of ~0.55. So P5's "58% of the HyDE top-100 is
actionable" would read ~32% under Sonnet, and P6's "61% of verified adoptions" ~34%. The
*separations* — 58% against a 2% floor, adoptions against random papers — survive intact,
because both judges order papers the same way. The levels do not.

**And the pre-registered bar was missed, not met.** kappa 0.507 is "moderate" agreement. The
labelled set is a noisier instrument than P7 predicted, and every ±10-to-±22 conclusion it has
produced should be read with that in mind, offset-cancellation notwithstanding.

### Caveats

- Byte-identical rubric, but **not byte-identical framing**: the first judge sends the rubric
  as an OpenAI system message and this sends one prompt string to Anthropic. That difference
  is not removable while the judges are different vendors.
- 7 of the first 200 responses were **truncated mid-justification at `max_tokens=500`** and
  failed to parse. The visible scores in the fragments were 2, 2, 3 — so the dropout skewed
  *actionable* and would have biased Sonnet's base rate down. Raised to 1,200 and re-run; the
  numbers above are the full 200. A parse failure that correlates with the verdict is not a
  random 3.5% loss.
- Kappa bounds noise, not validity. Two LLMs can share a famous-technique halo and agree
  confidently on the same wrong answer. P6 is the validity test; these compose.

## P6 — ground truth without a model: the judge agrees with what repos actually adopted (2026-08-06)

```bash
uv run python evals/mine_adoptions.py --mine     # $0, blobless clones + two bibliography reads
uv run python evals/mine_adoptions.py --judge    # ~$1, 31 papers vs the T0 repo
uv run python evals/mine_adoptions.py --hop      # $0, retro-recall from the T0 bibliography
uv run python evals/mine_adoptions.py --report   # re-derive, $0
```

Every number in this file is agreement with GPT-5.5, and the 48 recall targets are Opus
picks the judge then scored — circular twice over. P5 made that heavier, not lighter: "58% of
the top band is actionable" is 58% *by the judge's bar*.

This is the first label in the project that no model produced. An arXiv id **in a repo's docs
at HEAD and absent 24 months earlier** is a technique that project demonstrably took up:

    ids(HEAD) - ids(T0)  =  what this repo adopted, as judged by the repo

| case | adopted | self-cited | too new | usable |
|---|---|---|---|---|
| `graph` | 14 | 1 | 0 | **13** |
| `diffusion` | 9 | 0 | 2 | **7** |
| `peft` | 15 | 10 | 1 | **5** |
| `llminfer` | 2 | 0 | 0 | 2 |
| `rag` | 2 | 0 | 0 | 2 |
| `rl` | 2 | 0 | 0 | 2 |

**31 usable adoptions across 6 repos** — the pre-registered yield bar (≥30 across ≥6) is met,
almost exactly on the line. Self-citation fraction 25%, far under the 80% kill bar; `peft` is
where that filter earns its place, since 10 of its 15 new ids are its own papers under a
Citation heading. Sixteen repos yielded nothing, and the split is informative: seven
(`encryption`, `http`, `linter`, `storage`, `systems`, `vectordb`, `webdev`) have **zero
arXiv ids at either end** — the structural zeros P6 pre-registered — while `cv`, `numerics`,
`speech` and `db` have stable bibliographies (18→18, 2→2) that cite without adding.

### The three pre-registered outcomes

| | predicted | measured | |
|---|---|---|---|
| usable adoptions | ≥30 across ≥6 repos | **31 across 6** | MET |
| retro-recall (T0 hop) | ≥60% | **21/31 = 68%** | MET |
| judge scores them actionable | ≥70% | **19/31 = 61%** | below prediction |
| — the bar that matters | ≥40% or the judge is invalid | **61%** | **judge NOT invalidated** |

Scores `0:3 1:9 2:15 3:4`. For scale, P5 measured this same judge calling **2%** of random
arXiv papers actionable, so 61% against papers a repo verifiably adopted is a long way from
chance and a little short of what P6 predicted.

**Retro-recall is the only recall number in this project whose targets were not chosen by a
model**, and at 68% it is *higher* than the 44% the hop achieves against the Opus-derived
gold set. `llminfer` is the exception at 0/2 — llama.cpp had 3 seeds at T0, and P1 already
measured the seed-count cliff (≥7 seeds → 89%, <7 → 33%).

### What the 12 misses actually are

Each was traced back to the file it lives in, which turns "the judge disagreed" into
something checkable:

- **`rl` 0/2 is a labelling error, not a judge error.** Both ids are in
  `docs/misc/projects.md` — stable-baselines3's *"Projects using SB3"* page. Those are
  **downstream users of the library**; the citation direction is reversed and SB3 adopted
  nothing.
- **All 5 `graph` misses are in one file**, `docs/source/tutorial/graph_transformer.rst`:
  BERT, ViT, Attention Is All You Need, Big Bird, Performers, cited as background in a
  tutorial explaining graph transformers. "Outside PyG's graph-neural-network scope" is right.
- **One is a broken link in the diffusers docs.** `2412.11963` is labelled "LongCat-Image
  Technical Report" in `longcat_image.md`; the paper actually at that id is *Approximating the
  Top Eigenvector in Random Order Streams*. The miner extracted it faithfully and the judge
  scored the real paper 0.

That leaves **4 genuine disagreements** — an autoregressive video model in `diffusers`,
Megatron-LM in `peft`, and two similar cases where the judge said "broadly related, not a
component-level improvement".

### The same numbers on the 23 verified adoptions — post-hoc, and labelled as such

| | all 31 mined (pre-registered) | 23 verified |
|---|---|---|
| judge actionable | 19/31 = **61%** | 19/23 = **83%** |
| retro-recall | 21/31 = **68%** | 17/23 = **74%** |

All 8 excluded rows scored below the actionable bar, and `graph` goes from 8/13 to **8/8**.
This is a post-hoc subset chosen after seeing which rows the judge missed — it is reported
because each exclusion is verifiable by opening the file, not because it is the headline.
**The headline is 61%.** Note also that the verified set spans 5 repos, not 6, so on the
stricter labels the yield bar would be missed.

### What this settles, and what it does not

**Settled:** the judge is measuring approximately the right thing. It was the largest
un-tested assumption under every number in this file, and P6's answer is that papers repos
actually adopt score well above the 2% floor and — once background citations are removed —
above the 70% the plan asked for.

**Not settled:** `ids(HEAD) − ids(T0)` has a noise direction nobody predicted. Docs cite
papers as background, and as a list of people using you. The concrete next refinement is a
**reverse-citation filter** on paths matching `projects|showcase|used[-_ ]by`, the same shape
as the existing self-citation filter; it would have removed `rl` entirely. Nothing here
touches P7 — a second judge's kappa is still the test of whether GPT-5.5's *individual*
verdicts are reproducible, and P6 speaks only to what it rewards on average.

> **Harness failures, recorded because they cost five hours.** The retro-hop was launched by
> `Start-Process` after an earlier `cd evals/.work/fullclone/rl` in a *different* tool call —
> the working directory is shared, so it looked for `evals/mine_adoptions.py` inside a cloned
> repo and died in under a second. The waiter watching it polled for `=== P6` in stdout or
> `Traceback` in stderr; a bare "can't open file" is neither, so it looped for five hours
> against a dead process, and a second stale waiter from an earlier killed run did the same.
> Two lessons, both already known and both re-learned: use `git -C` rather than `cd`, and
> **wait on process exit, not on log content** — a filter that only matches the happy path
> and one failure shape reports a crash as silence. Underneath both was a real bug: `hop()`
> takes its direction positionally as `references`/`citations`, not `forward`/`backward`, and
> `retro_hop` omitted the argument entirely. That one now has three tests.

## P5 — the pool is far denser than every recall number implied, and the gate's problem is recall, not precision (2026-08-06)

```bash
uv run python evals/hyde_replication.py --dump-topk 10000   # $0, the candidate lists
uv run python evals/label_pool.py --dry-run                 # sample composition, $0
uv run python evals/label_pool.py                           # ~$0.3 gate + ~$10.2 judge
uv run python evals/label_pool.py --report                  # re-derive, $0
```

Every retrieval number in this project is **recall**. P1 cut a pool, P4 reached 27 of 48
targets, and neither has a precision figure, because all ~800 cached verdicts describe papers
the *ranker or the baseline* surfaced. This is the first time anything in the candidate pool
has been labelled in the wild.

1,200 papers scored by the shipped Haiku gate; a uniform random 320 of them also judged by
GPT-5.5 under the shipped rubric. **Not one of the 320 is a gold target** — these are entirely
fresh labels, so nothing below is circular with the 48.

| stratum | gated | gate admits | judged | judge ≥2 | judge =3 |
|---|---|---|---|---|---|
| `hyde-top100` | 200 | 38.0% [31.6, 44.9] | 100 | **58.0%** [48.2, 67.2] | **9.0%** |
| `hyde-100-1k` | 200 | 24.5% [19.1, 30.9] | 30 | 43.3% [27.4, 60.8] | 0.0% |
| `hyde-1k-10k` | 200 | 13.0% [9.0, 18.4] | 30 | 13.3% [5.3, 29.7] | 0.0% |
| `hop-coupling3+` | 200 | 43.5% [36.8, 50.4] | 30 | **66.7%** [48.8, 80.8] | **26.7%** |
| `hop-coupling1` | 200 | 13.0% [9.0, 18.4] | 30 | 33.3% [19.2, 51.2] | 0.0% |
| `random-arxiv` | 200 | **0.0%** [0.0, 1.9] | 100 | 2.0% [0.6, 7.0] | 0.0% |

### 1. Separation: measured, large, and the prediction lands on the strict bar

58.0% vs 2.0% is a **29× separation at p < 0.001**, against a predicted ≥6×. Two of the three
pre-registered clauses are met outright; the third — floor ≤1% — misses by **one paper**
(2/100), which at n=100 is inside sampling noise of the bar itself.

At the **≥3 bar — the one the 48 gold targets were actually drawn at** — the prediction is met
on all three clauses and lands almost exactly where it was set: **9.0% vs 0.0%, p = 0.0032**
against a predicted ≥8% vs ≤1%. The ≥2 bar is the permissive one (PR #83 measured half a
random *hop-pool* sample clearing it), so ≥3 is the number that compares to prior recall work.

### 2. The density that recalibrates everything before it

The hop's pools were described as **1 good paper per 5,224 candidates**. That figure measured
*distance to a known gold target*. By the judge's own bar, the same pool's high-coupling band
is **67% actionable and 27% score-3**, and HyDE's top-100 is 58%/9%.

Those are not in conflict — they are two different questions — but every "1 in 5,111",
"1 in 5,224" and "44% reach" in this file is the first question, and it has been read as the
second. **The pool is not sparse in useful papers. It is sparse in *these* useful papers.**

### 3. The gate: kill clause fired, premise refuted

Pre-registered kill: wild-admit >20%. Measured on `hyde-100-1k`: **24.5%**. It fires.

But the clause was written on the assumption that a high admit rate means a junk-dominated
pool is fooling the gate. That premise is wrong here:

- **precision 0.97** — 2 false positives in 66 admits across all 320 judged papers
- **recall 0.60** — 43 actionable papers rejected
- the band it fired on is **43.3% actionable, above its 24.5% admit rate**

The gate is not admitting junk; it is **rejecting actionable papers**. And on the floor it is
perfect: **0 of 200 random arXiv papers admitted**. Compare the labelled set, where the same
gate measured precision 0.81 / recall 0.78 — off-distribution it becomes *more* precise and
*less* sensitive. Read P5's kill as aimed at **recall**, and note that tightening the gate,
the obvious move from every previous run, now costs recall for almost no precision left to buy.

### 4. Where the cascade actually breaks

| operating point | candidates/repo | gate admits | Haiku cost/repo/run |
|---|---|---|---|
| HyDE top-100 | 100 | ~38 | ~$0.02 |
| HyDE top-1k | 1,000 | ~245 | ~$0.23 |
| HyDE top-10k | 10,000 | ~1,300 | ~$2.30 |

The gate is **affordable** at every point and **insufficient** at all of them: even the
tightest admits ~38 papers where a digest wants ~10. The missing stage is not a stricter
gate — at 0.97 precision there is nothing left to tighten — it is **ranking within the
admitted set**. That is the same conclusion the 2026-07-06 all-time run reached, now with
numbers: *"discovery is solved; precision on the enriched pool is the remaining gap."*

### Caveats

- Four strata are judged at **n=30 and are descriptive only** — pre-registered as such,
  because nothing separating 8% from 1% can reach significance at that size. Only the
  `hyde-top100` vs `random-arxiv` pair (n=100 each) is a test.
- `hop-coupling3+` looks like the densest stratum, but its CI overlaps `hyde-top100`'s.
  "At least as dense" is what the data supports.
- The draw is **balanced across repos**, so these are per-repo mean densities. `graph` alone
  is 42,112 of the hop pool's 109,704 rows and took **139 of 200** slots in a flat draw.
- The judge remains the arbiter, not ground truth. P6 (git-history adoption) and P7
  (second-judge kappa) are still the tests of that, and nothing here touches them.

## P4 — Design 2 verified, then replicated: HyDE reaches 27 of 48, and 15 of those the citation hop cannot (2026-08-06)

```bash
uv run python evals/verify_hyde_deps.py              # stage 1, $0, ~40 MB
uv run python evals/hyde_replication.py --build      # 432 MB of range requests, $0
uv run python evals/hyde_replication.py              # ~$0.20 of Haiku, ~25 min CPU
uv run python evals/hyde_replication.py --report     # re-derive from saved ranks
```

**This is the first measured technique that beats the bibliography-seeded citation hop.**
Everything tried since P1 — coupling-degree filtering, gap-phrase matching, synthetic hop
seeds — either failed to replicate or failed to beat a control. This one does not.

### Stage 1 — the four dependencies, before building anything

RETRIEVAL_DESIGN Design 2 was the largest single recall candidate in the project and every
dependency it rested on was marked unverified. P4's gate was 4/4 or nothing.

| check | pre-registered bar | measured | |
|---|---|---|---|
| C1 exists + licence | resolves, permissive licence, ≥3.0M rows, binary vector column | `bluuebunny/arxiv_abstract_embedding_mxbai_large_v1_milvus_binary`, **apache-2.0**, **3,106,925 rows**, `id: string` + `vector: binary` | PASS |
| C2 columnar range-fetch | ≤25% of a shard's bytes; vectors decode to 128 B | **15.9%** — 25.1 MB of the 158.1 MB 2021 shard, 2 range requests, all vectors 128 B | PASS |
| C3 query latency | ≤4×1.87 s | **1.21 s**, best of 3 — **0.6× the reported figure** | PASS |
| C4 target coverage | every target older than the snapshot present, ≤4 missing | **48/48**, including two papers from 2026-07 | PASS |

Two things the checks corrected rather than confirmed:

- **"~370 MB one-time index sync" is only true with column pruning.** The dataset is
  **2,542 MB** on disk. Fetching `id` + `vector` and nothing else costs **432 MB** — within
  17% of the reported figure. Without C2 the cost line of Design 2 is off by 5.9×.
- **C2 measured column pruning, not row-group pruning.** Each shard holds exactly one row
  group, so "one row group" was the whole year. That is the right test for this workload —
  we want all rows and 2 of 9 columns — but it is not evidence that a reader can skip *rows*.

### The fifth dependency, which P4 did not think to name

The four checks establish that the index exists, is fetchable, is fast, and contains the
targets. **None of them establishes that a vector we compute is comparable to the vectors
in it.** If the publisher had embedded title+abstract, or normalised differently, every
query would have measured nothing while looking perfectly healthy — and this project has
lost a week to exactly that shape of failure twice.

So stage 2 refuses to run until it reproduces stored vectors bit-for-bit. It does:
mxbai-embed-large-v1 over the **abstract alone**, L2-normalised, binarised at >0,
`np.packbits` — **Hamming 0/1024 on 5 held-out papers**. Exact, not approximate.

### Stage 2 — blind HyDE, 48 targets, 17 cases

Hypotheses come from `assemble_repo_context()` and nothing else; the generator never saw
the targets, the pool, or the judge. Four hypothesis abstracts per repo, Haiku 4.5, cached.
Every arm searches the same 3.1M-vector index with the same encoder — **the query text is
the only variable**.

| arm | top-100 | top-1k | median rank | within 4k |
|---|---|---|---|---|
| **hyde4-union** (best of 4) | 5/48 | **27/48** | **837** | **42/48** |
| hyde4-centroid (1 query) | 4/48 | 17/48 | 2,805 | 27/48 |
| hyde1 (one guess) | 2/48 | 12/48 | 4,317 | 23/48 |
| readme (today's `w_embedding`) | **7/48** | 10/48 | 46,656 | 12/48 |
| keywords (today's arXiv query) | 0/48 | 3/48 | 32,582 | 9/48 |

Pre-registered: ≥16/48 in top-1k, median <5,000, crypto 2/2; kill at ≤10/48.
**Verdict: met on aggregate, crypto sub-claim not met.**

**Read `hyde4-union` at 4× the budget.** Best-of-4 has already spent up to 4,000 candidates
per repo by the time it reports a top-1k hit, so it is not comparable to a single-query arm
at 1,000. The equal-candidate column is the fair one, and it says the same thing:
**42/48 vs 23/48 for one hypothesis and 12/48 for the README.** Four diverse guesses are
worth far more than one, at the same candidate cost.

### The number that matters: the two channels barely overlap

| | targets reached |
|---|---|
| citation hop (P1's pools) | 21/48 |
| HyDE-4 top-1k | 27/48 |
| **union** | **36/48 (75%)** |
| HyDE-only — unreachable by the hop | **15** |

Design 2's REPORTED union was ≥17/24 = 71%. Measured on twice the targets: **75%**. This is
the claim that survives most cleanly, and it is the one that matters, because the hop's
44% ceiling is structural — six benchmark repos have no arXiv-indexed bibliography at all.

**Density, not just reach.** The hop's persisted pools hold **109,704 candidates for 21
targets — 1 per 5,224**. HyDE-centroid finds 17 targets in 17,000 candidates (1,000/repo ×
17 repos): **1 per 1,000, 5.2× denser**. HyDE-union finds 27 in at most 68,000: **1 per
2,519**, 2.1× denser, at 62% of the hop's candidate count.

### What did not replicate

- **crypto 1/2, systems 0/1.** The specific claim that made Design 2 attractive — "it covers
  the repos the citation hop cannot, `crypto` 2/2 and `systems` 1/1" — does **not** hold at
  the repo level. `crypto`'s targets rank 459 and 2,976; `systems`' single target ranks
  1,644. The aggregate replaces it: 15 hop-unreachable targets across the whole benchmark,
  which is a better argument than the two repos it was originally made with. The run script
  reports this separately and refuses to let the aggregate absorb it.
- **`readme` wins top-100 and loses everywhere else** — 7/48 in the first 100, then a median
  of **46,656**. It is bimodal: when a repo's README happens to read like the abstract of its
  own literature it lands at the very top, and otherwise it is nowhere. That is the register
  mismatch, visible as a distribution rather than as an average.

### Caveats

- Ranks are **optimistic**: ties break in the target's favour (`(dists < d).sum() + 1`), so
  a reported median is a lower bound. It matters in the tail, not at the head.
- `readme` and `keywords` are proxies for today's channels **inside this index**, not
  measurements of the shipped pipeline (which uses MiniLM and arXiv's search API). They
  answer "is the query text the problem", not "how good is RepoRadar today".
- The encoder truncates at 512 tokens, which caps the `readme` arm's input. Part of its
  weakness is that budget, not only its register.
- This is **retrieval reach, not precision**. Every candidate still terminates in triage, and
  triage collapses on the ranker's top-10. A better pool raises the ceiling; it does not on
  its own produce a better digest.
- 48 targets are a **top stratum**, not "the actionable papers" — see the section below.

## Is "lacks" losing because the judge only rewards improvement? No — but the gold set is a top stratum (2026-08-05)

```bash
uv run python evals/extend_vs_improve.py     # ~$3 of judge calls
```

Negative result 7 concluded that "lacks" phrases suffer a *different-target failure*. A reader
raised the obvious objection: the judge's rubric says **"genuinely IMPROVE"** and scores 3 for
a paper that "directly addresses a known limitation or core capability". A paper adding
offline RL to a library that has none is not improving a known limitation — it is *extending
the project's scope*, and that rubric would score it low even if a maintainer wanted it.

**"Lacks retrieves badly" and "lacks retrieves extensions the benchmark cannot see" predict
identical numbers in §7.** That conclusion was asserted without separating them. This is the
control.

Same papers, same repo context, same model; the **rubric is the only variable**. `EXTEND`
deliberately caps a merely-refining paper at 1 however good it is, so the two rubrics can
dissociate instead of one being a looser version of the other. A **uniform random sample of
the same hop pool** is the third group, without which none of the numbers are readable.

| group (n) | IMPROVE mean (≥2) | EXTEND mean (≥2) |
|---|---|---|
| `lacks` top-8/case (40) | **1.75** (68%) | 1.38 (40%) |
| `targets` (8) | **2.75** (100%) | 2.00 (88%) |
| **`random` pool sample (40)** | **1.38** (50%) | 1.18 (30%) |

### Finding 1 — no dissociation; the judge-framing hypothesis is refuted

Both groups score *lower* under EXTEND, not higher: `lacks` 1.75 → 1.38, `targets`
2.75 → 2.00. The rubric works — it separates targets (2.00) from random (1.18) — and `lacks`
retrievals still do not rise under it. **They are not extensions the improvement judge was
blind to.** §7's conclusion stands, for the reason §7 gave.

### Finding 2 — "68% of lacks papers are actionable" was mostly the judge being permissive

A first reading of the IMPROVE column looked like evidence that the 21-paper gold list badly
undercounts what is actionable. The random control mostly kills that: **50% of uniformly
sampled hop-pool papers also score ≥2.** Papers one citation hop from a repo's own
bibliography are topically adjacent by construction, and "could plausibly be integrated" is a
low bar for them.

`lacks` does beat random — **+0.38 mean score, permutation p = 0.015** (n=40 each) — so it
carries real signal. But it is a modest lift, against `targets` at +1.38 over random.

### Finding 3 — what survives: the gold set is a TOP STRATUM, not "the actionable papers"

| group | score 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| random | 6 | 14 | 19 | **1** |
| lacks | 1 | 12 | 23 | **4** |
| targets | 0 | 0 | 2 | **6** |

The 24 targets are the papers an agentic Opus baseline surfaced *and* the judge scored ≥2 —
in practice mostly 3s. Merely-plausible papers are everywhere in the pool; strongly-useful
ones are rare, and that is what the gold set indexes.

This does **not** invalidate P1/P2/P3: they compare channels against a *consistent* gold set
and those comparisons hold. It does change two readings:

- **"1 good paper per 5,111 candidates"** (§3.5) is distance to a *known target*, not to
  anything useful. By the judge's own ≥2 bar the hop pool is ~50% plausible.
- **"the hop reaches 44%"** is 44% of the top stratum. It is a fair cross-channel number and
  an understatement of usefulness.

### What this leaves for an extension mode

Not built, and not justified by this evidence — the experiment that would have justified it
came back negative. What now exists is the **instrument**: `EXTEND_RUBRIC` separates targets
(2.00) from random (1.18), so extension-oriented discovery is measurable if it is ever wanted.
It would need its own gold set; the current one was built end-to-end by an improvement judge
and cannot score an extension channel fairly.

> **Two harness failures worth recording, and the second is the worst of the week.**
>
> **1. A total failure manufactured a finding.** The first run judged *nothing* — `judge.py`
> reads `OPENAI_API_KEY` from the environment and the script never loaded `evals/.env` — and
> then printed `">>> NO DISSOCIATION: 'lacks' papers are not extensions either. P2 stands."`
> The verdict averaged empty lists to 0.0, so `extend <= improve` held trivially. The other
> silent failures this week destroyed data; this one produced an *answer*, and one that
> agreed with the conclusion already written. The script now refuses to print any verdict
> while a 2×3 cell is empty, naming which, and exits non-zero.
>
> **2. The experiment corrupted the gold set it was measuring against.** `judge_paper` keys
> its cache file on `(model, repo, paper_id)` — **not on the rubric**, which only lands
> inside the file as `_prompt_hash`. Swapping `judge_mod.RUBRIC` and letting the cache write
> therefore overwrote each target's IMPROVE verdict with its EXTEND score. Nine known targets
> dropped below the ≥2 threshold and **`rag` went from 5 targets to 0**, because
> `diagnose_pool.actionable_baseline_ids` reads `score` without checking the hash.
>
> Caught by `tests/test_eval_hop_pool.py` — the guard added earlier in the week pinning the
> frozen `TARGETS` literal to the derived list — on the very next `pytest` run. Restored by
> re-judging the 9 under the shipped rubric; all came back **1 → 3**, and the gold set is
> back to 48 targets. `extend_vs_improve.py` now passes `use_cache=False`, which is
> load-bearing rather than an optimisation, and two mutation-verified tests enforce it: one
> that `score_group` never lets the cache write, one that the swapped module-global is
> restored even when judging raises.
>
> The general rule: **an experiment that swaps a rubric must not share a cache with the
> benchmark it is measured against.** The cache path scheme is the underlying hazard — it
> was left as-is because re-keying it would invalidate every paid-for verdict in
> `evals/cache/`, so the guard sits at the call site instead.

## Negative result 7 — gap-phrases do not beat pasting the keyword profile (2026-08-05)

```bash
uv run python evals/fill_pool_metadata.py --all    # 103,789/103,793 (100%), free
uv run python evals/gap_match.py --top-k 200       # instant, offline
```

P2 was built on §3.2's reading of its own data: the "lacks" prompt *aims correctly* at the
right research and fails only on phrasing — 45 of 54 phrases matched zero papers, and its
example was `"experience replay prioritization methods"` (0 hits) against `"prioritized
experience replay"` (found it). If that is a string-matching failure, matching in a space
where the two are close should unlock it.

**The string matching was fixed and verified.** Snowball stemming collapses the morphology
(`prioritization`/`prioritized` → `priorit`) and BM25 scores bag-of-terms overlap instead of
phrase containment, so word order and compounding stop mattering. §3.2's own example now
scores **2.79** against the target it missed and **0.00** against distractors.

Four arms rank the same 103k-paper hop pool with the same scorer, differing only in the query:

| cut | `lacks` (what it's missing) | **`uses`** (what it implements) | `gaps` (summariser's improvement_areas) | `profile` (keyword profile — CONTROL) |
|---|---|---|---|---|
| top-50 | 2 (10%) | **3 (14%)** | 1 (5%) | 1 (5%) |
| top-200 | 4 (19%) | **7 (33%)** | 2 (10%) | 6 (29%) |
| top-500 | 10 (48%) | **11 (52%)** | 3 (14%) | 10 (48%) |

Pre-registered: **≥44% of reachable targets AND ≥2× the control**. The best arm reaches 52%
at top-500 but beats the control **1.1×** — 11 targets against 10. At top-200 it is 7 against
6. **A one-target difference on n=21 is noise.** Not killed (`uses` clears `profile` at
top-500, which was the kill condition), but the prediction fails at every depth.

### The premise was inverted, and the ranks showed it before the tally did

The ordering is the finding: **`uses` > `profile` > `lacks` > `gaps`**. Describing what a
repo *has* retrieves its targets better than an LLM's account of what it *lacks*, and the
summariser's improvement areas are worst of all.

Per-target ranks on `rl` (percentile in a 29,479-paper pool; random ≈ 50%):

| query | target percentiles |
|---|---|
| `lacks` | 57%, 80%, **94%** |
| `profile` | **2%**, 70%, 84% |

The "lacks" query ranks the targets *worse than random*. Its phrases for
`stable-baselines3` were distributed policy gradients, model-based RL, transformer
actor-critic, offline RL, multi-agent, imitation learning, hierarchical RL — a coherent and
entirely plausible research agenda. The actual targets were **Double Q-Learning, Prioritized
Experience Replay, Dropout Q-Functions**: refinements of the value-based RL the repo already
implements.

**That is not a phrasing failure. It is a different-target failure**, and no retrieval space
fixes it. §3.2's "aims correctly" reading is withdrawn: on this evidence the "lacks" prompt
aims at a *plausible different* agenda, and the resemblance to the right answer in §3.2's
spot-check did not survive measurement across 8 cases.

One fact now explains three previously separate observations:

* why `uses` beat `lacks` in §3.2 (2/24 vs 0/24),
* why the citation hop works at all — seeding on what a repo *has* lands you among
  refinements of it, which is what the judge rewards,
* why `improvement_areas` hurt the triage gate in §5.3 (+70, below the no-description
  control) and finish last here at 14%.

**The judge's "actionable" skews toward improving existing components, not adding new
capabilities.** Every channel that works is one that starts from what the repo already does.

### Consequences

- **LLM phrase generation is not measurably better than pasting the keyword profile into a
  BM25 query.** The cheapest possible baseline is within noise of the best arm at every depth.
- Retrieval work should stop trying to name what a repo lacks. Three separate attempts have
  now failed on it: direct queries (§3.2, 0/24), gate context (§5.3, +70 vs +73 control),
  and pool ranking (here, 19% vs 29% control).
- `gaps` at 10–14% is the second measured negative for the summariser's `improvement_areas`.
  Treat that field as unvalidated for any purpose.
- P2 does not rescue the 44% reach. Combined with P1 (filter does not replicate) and P3
  (synthetic seeds 4/27), **no measured technique has improved on the bibliography-seeded
  hop**, and P4 is the only untested channel left.

## Negative result 6 — synthetic hop seeds recover 4 of 27; only one domain works (2026-08-05)

```bash
uv run python evals/synth_seeds.py                  # vote-ranked seeds  (~$0.02)
uv run python evals/synth_seeds.py --rank citations # hub-ranked seeds
```

The hop reaches 44% of known-good papers because it seeds from arXiv ids the repo cites, and
most repos cite few or none. P3 asked whether LLM "uses" phrases can manufacture the missing
seeds. §3.2 measured those phrases as *accurate but useless for direct retrieval* (2/24) —
the bet was that the hop needs only an **anchor near** the target, not the target itself.

Pre-registered before running (ROADMAP P3): **≥8 of 27** unreached targets, **kill at ≤2**.

| arm | seeds ranked by | targets | pool |
|---|---|---|---|
| P3 | phrase agreement | **3/27 (11%)** | 10,236 |
| P3b | **S2 citation count** | **4/27 (15%)** | 18,272 |

Both below the prediction, both above the kill line. **All recovered targets come from one
case**: `vectordb` (qdrant), which reached 3/4 then **4/4**. Eleven other cases recovered
nothing.

### The mechanism was diagnosed correctly, and fixing it did not help

Why P3 failed is measurable:

| | synthetic seeds | real bibliography |
|---|---|---|
| median citations | **3** | **1,210** |
| zero-citation seeds | 12 of 39 | 0 |
| neighbourhood | 26 papers/seed | **515 papers/seed** |

A bibliography cites **hubs**; phrase search returns whatever matches the string, because
arXiv relevance carries no impact weighting — §3.0's founding defect, one stage earlier.

P3b ranked the same phrase matches by citation count before seeding. **The predicted
mechanism worked**: pools grew **1.8× overall**, and 2.4–3.6× on the cases that needed it
most (`systems` 425→1,534, `crypto` 276→918, `numerics` 509→1,700, `speech` 1,462→3,448).
**Recall moved 3 → 4.** Bigger neighbourhoods in the wrong region of the graph are still the
wrong region — `diffusion` got a 19,747-paper pool and recovered 0 of 2.

So the constraint is not neighbourhood *size*. Phrase-derived seeds sit somewhere the targets
are not, and hub-ranking moves you to a bigger somewhere-else.

### Why `vectordb` is the exception, and what it implies

Its phrases — "Approximate Nearest Neighbor", "Vector Similarity Search", "Product
Quantization" — are crisp terms of art naming a subfield with a dense, self-citing arXiv
presence. There, "what the repo implements" and "where the useful papers live" coincide.

For `crypto`, `db`, `storage`, `compiler`, `columnar` the phrases were equally accurate about
the repo (8–10 of 10 matched real papers) and still landed nowhere near the targets. Those
fields' literature is centred on IACR/VLDB/PLDI; their arXiv presence is a thin, poorly
connected slice, so a hop through it traverses a sparse graph.

**This is the register mismatch's structural twin.** §1 says a repo's vocabulary describes
what it *has*, not what it should *adopt*. P3 shows that even when you accept that and use
the vocabulary only as an anchor, the anchor lands in the wrong neighbourhood unless the
repo's field is natively arXiv-shaped.

### Consequences

- **The citation hop cannot be extended to bibliography-less repos this way.** 11 of 22
  benchmark cases stay unreachable; 23 of 48 targets stay out of reach of any measured channel.
- **Feature 10's non-arXiv adapters (IACR, DBLP, VLDB) move from "domain coverage" to the
  only remaining route** for `crypto`, `systems`, `storage`, `compiler`, `columnar`.
- **P4 (HyDE against a dense index) is now the highest-value untested retrieval direction**,
  and its claimed crypto 2/2 + systems 1/1 is exactly the cell P3 just failed. Its $0
  dependency verification should run before anything else.
- `vectordb` 4/4 says synthetic seeding is worth keeping **as an opt-in for arXiv-native
  domains**, not as a general mechanism.

### Two harness failures found while running this, both nearly fatal to the result

**1. `--rank citations` was unreachable — P3b silently re-ran its own control.**

```python
for rank, pid in enumerate(ids):   # rebinds the `rank` PARAMETER to an int
    order.setdefault(pid, rank)
...
if rank == "citations":            # always False
```

A loop variable shadowed the function parameter. The first P3b returned exactly 3/27 with
**byte-identical pools on 10 of 11 cases** — including two where the seed cap binds, which is
impossible for two different seed sets. That anomaly was the only signal; a perfect match
between variant and control is a **bug signature, not a confirmation**. Post-fix the two
rankings share 15 of 40 seeds and lead with entirely different papers. Mutation-tested.

**2. A `--case` re-run overwrote the whole-set results with one row.** Retrying the refused
`diffusion` case replaced an 11-case file with `[]` and printed a KILL verdict against bars
scoped to all 27 targets. Recovered from the run log; nothing measured was lost. This is the
**second occurrence today** — `diagnose_triage.py` ignoring `--model` in its output filename
destroyed the per-case Sonnet data the same way — and `build_hop_pool` has had the correct
read-update-write pattern the entire time. Results now merge; verdicts are suppressed on
partial runs; both guarded by mutation-tested tests.

## P1 re-run on 22 cases — the 70% cut does not replicate, and the hop reaches 44% not 75% (2026-08-05)

```bash
uv run python evals/build_hop_pool.py --skip-metadata   # 11 pools now, free
uv run python evals/hop_reach.py                        # instant
uv run python evals/sweep_hop_filter.py                 # instant
```

The jackknife predicted this and it still landed harder than expected. Re-running P1's
identical sweep with four more pools:

| | 7 pools (18 targets) | 11 pools (21 targets) |
|---|---|---|
| targets retained | 16/18 = **89%** | 16/21 = **76%** |
| pool cut | **70%** | **10%** |
| both pre-registered bars | retention met, cut missed | **both missed** |

**The 70% cut was a property of the case set, not of coupling degree.** With `ann` and
`llminfer` in the folds — repos with 3 and 2 seeds whose targets sit at low forward degree —
meeting the 83% retention floor on the training folds forces `fwd>=1`, and `fwd>=1` keeps
98.4% of the pool. The filter degenerates to nearly no filter.

The kill condition is now close rather than comfortable: **5 of 21** targets are reachable
from ≤1 seed in both directions (`ann` ×2, `llminfer`, `speech` ×2), against a kill line of 6.
It was 2 of 18 on the old set.

### The larger finding: 18/24 was measured on a favourable subset

`hop_reach.py` asks the question the published figure did not. That 75% was recall over the
nine cohort-1 cases *that had targets* — a denominator which silently excluded every repo the
hop cannot serve. Across all 17 cases that now have targets:

| case | seeds | pool | reached |
|---|---|---|---|
| graph | 121 | 42,112 | 3/3 |
| rl | 30 | 29,480 | 3/3 |
| cv | 18 | 14,874 | 3/3 |
| peft | 18 | 6,365 | 2/2 |
| diffusion | 10 | 4,083 | 0/2 |
| rag | 7 | 1,869 | 5/5 |
| ann | 3 | 3,152 | 2/3 |
| speech | 2 | 6,085 | 2/3 |
| llminfer | 2 | 1,666 | 1/4 |
| db | 1 | **14** | 0/3 |
| numerics | 1 | **4** | 0/2 |
| crypto, systems, storage, compiler, vectordb, columnar | 0 | — | 0/15 |

| | |
|---|---|
| all cases with targets | **21/48 = 44%** |
| bibliography-seeded cases only | 21/33 = 64% |
| **unreachable by construction** | **15 targets across 6 cases** |
| seeds ≥ 7 (6 cases) | 16/18 = **89%** |
| seeds < 7 (5 cases) | 5/15 = **33%** |

**Seed count is the whole story.** Six repos with a real bibliography give 89% recall; five
with a thin one give 33%; six with none give 0 by construction. `db` (duckdb) has exactly one
arXiv citation, and one seed produces a **14-paper pool** — a nonzero bibliography is not a
usable one.

So the honest statement of the hop is not "the channel that works" but **"the channel that
works for repos with a substantial arXiv bibliography"** — 6 of 22 benchmark cases, and
plausibly a small minority of real repositories. That does not retract the 18/24; it corrects
what it was a measurement *of*.

### Consequences

- **P1 is a negative result on the expanded set.** Coupling degree as a threshold does not
  survive a less concentrated benchmark. The persisted pool and its degree annotations remain
  useful infrastructure for P2; the filter does not.
- **P3 (synthetic seeding) is promoted from nice-to-have to the main event.** It addresses the
  6 structurally-zero cases plus the 5 thin-bibliography ones — 27 of 48 targets, versus the
  21 the hop can reach today.
- **P2 should be measured on the seeded cases and reported as such**, since the pool it
  filters only exists for 11 of 22 cases.
- Anything that quotes "18/24" or "75%" should say **"of the papers the hop can reach at
  all"**, and the figure to quote for the system is **44%**.

### A harness bug this exposed, and how it hid

`build_hop_pool` read `diagnose_citation_hop.TARGETS`, a frozen literal covering the nine
cohort-1 cases. Every new case raised `KeyError` — and all ten failed **silently**, because
the command running them piped output through a `grep` that filtered the traceback away. Two
compounding failures: a hardcoded source of truth that could not see new data, and my own
output filter hiding the crash. Targets are now derived via
`diagnose_pool.actionable_baseline_ids()` for every benchmark case, with a test that every
case resolves and that the frozen literal never contradicts the derived list. `TARGETS` is
kept only as the record of what the published 18/24 was measured against.

## The benchmark was overfitting on 12 repos — jackknife, then +10 cases (2026-08-05)

P1's headline (70% pool cut, 16/18 targets retained) was produced with leave-one-case-out,
which stops a threshold being tuned on the fold it is scored on. It does **not** stop a
7-fold set from resting on one fold. Recomputing the whole LOO result with one repo removed:

| dropped | targets | cut | swing |
|---|---|---|---|
| graph | 13/15 | 77% | +7% |
| rl | 13/15 | 70% | −0% |
| diffusion | 16/18 | 69% | −1% |
| speech | 16/16 | 68% | −2% |
| peft | 14/16 | 67% | −3% |
| cv | 13/15 | 66% | −4% |
| **rag** | 11/13 | **11%** | **−59%** |

**Removing one repo moves the headline from 70% to 11%.** `rag` holds 5 of 18 targets (28%)
in the smallest pool (1,869); its targets are the cheap ones, and without them the 15/18
retention floor forces a far looser threshold on every other fold. Top 2 repos held 44% of
targets; effective repo count (inverse Simpson over target share) was **5.4 of 7**.

Corroborating evidence from the same session: at n=7 targets the forward-degree enrichment
read 0.00× and was reported with a mechanism; at n=18 it was +2.32×. Same fragility, one
scale down.

### Ten cases added against four measured criteria

Not "more data" — four named blind spots (thin docs + real research; no arXiv bibliography;
non-ML; citation-rich). The full measured table is in
[`evals/README.md`](README.md#cohort-2-2026-08-05--added-because-12-was-measurably-too-few).

| | before | after |
|---|---|---|
| cases | 12 | **22** |
| targets | 24 | **48** |
| `rag` share of targets | 21% | **10%** |
| effective repo count | 5.4 of 7 (hop cases) | **15.2 of 17** |
| non-ML cases | 5 | **13** |
| no-arXiv-bibliography cases | 5 | **11** |

Baselines cost **~$14** for ten cases — `llminfer` $3.45 and `numerics` $3.25 alone were
half of it; the other eight ran $0.60–$1.10. Cost scales with repo size, not case count.

**Two of my predictions were refuted by the measurement**, recorded because both were
nearly shipped as asserted labels:

1. I expected `llminfer`, `vectordb`, `linter`, `encryption` to be thin-docs cases. All four
   have substantial READMEs — ruff's is 25,182 characters, 4× peft's. The only genuine
   thin-docs additions are `storage` (1,689) and `compiler` (1,686). `benchmark.yaml` now
   carries the measured numbers inline beside each `criteria:` label.
2. I expected the six no-bibliography cases to yield ~0 targets, since a repo citing no
   arXiv work plausibly has none to recommend. **12 of 22 new targets came from them.** ANN
   indexing, columnar compression and LSM compaction have arXiv literature those repos do
   not cite — making them the sharpest cases in the set, where the citation hop is
   structurally blind but the research exists.

Two cases yield 0 and are kept deliberately: `encryption` (Opus abstained — correct, its
literature is on IACR) and `linter` (Opus made 3 picks, the judge rejected all 3 — an
over-firing case).

> **A third correction.** `numerics` was first written up here as *"the headless baseline
> does not finish on a repo scipy's size — a harness limitation"*. It finishes: it needs
> >10 minutes and $3.25, and the batch loop cut it short. Re-run alone it contributes 2
> targets. "It failed" and "I did not wait" are different claims and I published the wrong
> one. Budget wall-clock, not only dollars, when adding large repos.

**Every number measured before today stands on the 12-case set** and inherits its
concentration. P1's 70% cut in particular should be re-run on the expanded set before it is
built on.

## P1 — coupling degree cuts the hop pool 70% and keeps 16/18 targets (2026-08-05)

```bash
uv run python evals/build_hop_pool.py --skip-metadata   # ~40 min, free, keyless
uv run python evals/sweep_hop_filter.py                 # instant, offline
```

The citation hop reaches 18 of 24 known-good papers and buries them in 92,014 candidates
(§3.5). This is the first measurement of whether the **free structural signal** — how many of
the repo's own seeds co-cite a candidate — can cut that pool without losing the papers.

The pool is now **persisted** (`evals/.work/hop_pool/<case>.jsonl`, 7 cases, 104,868 rows
with per-candidate forward/backward degrees), so every future filter idea is an offline
question rather than a 40-minute network sweep. Rebuilt today it holds **18 targets**,
reproducing the published recall exactly.

### Result against the pre-registered bars

| bar (stated in ROADMAP P1 before running) | outcome | |
|---|---|---|
| kill if ≥5 of 18 targets have fwd ≤1 **and** back ≤1 | **2 of 18** (both `speech`) | not killed |
| retain ≥15/18 targets | **16/18** | **met** |
| cut the pool ≥75% | **70%** | **missed** |

Leave-one-case-out throughout: each case's threshold is chosen on the other six and scored
on the held-out one, so nothing here is a description of the pool that tuned it. The chosen
cut was `fwd≥2 OR back≥3`, with cross-repo document frequency ≤2, on all seven folds.

| held out | pool | after | cut | targets |
|---|---|---|---|---|
| diffusion | 4,083 | 62 | 98% | 0/0 |
| peft | 6,365 | 577 | 91% | 2/2 |
| cv | 14,874 | 2,084 | 86% | 3/3 |
| rag | 1,869 | 408 | 78% | 5/5 |
| rl | 29,480 | 9,968 | 66% | 3/3 |
| graph | 42,112 | 18,215 | 57% | 3/3 |
| **speech** | 6,085 | **13** | 100% | **0/2** |
| **total** | **104,868** | **31,327** | **70%** | **16/18** |

**Read the aggregate cut as two different stories.** On five cases it is 78–98%. The 70%
total is dragged down by `graph` and `rl`, which are **68% of all candidates** and cut worst
(57%, 66%) — and which also hold all 13 un-enumerable hub seeds, so their degrees are the
most undercounted. `speech` is the opposite failure: the filter removes 99.8% of its pool
and both its targets with it. Those two targets are the only ones in the whole set reachable
from ≤1 seed in both directions, i.e. exactly the kill-condition papers. Coupling has nothing
to work with there.

### The enrichment, and a correction I made mid-run

| filter | targets kept | pool kept | enrichment |
|---|---|---|---|
| `fwd ≥ 1` | 83.3% | 98.4% | 0.85× |
| `fwd ≥ 2` | 72.2% | 29.7% | **2.43×** |
| `fwd ≥ 3` | 33.3% | 14.4% | 2.32× |
| `back ≥ 1` | 27.8% | 2.1% | **13.0×** |
| `back ≥ 2` | 27.8% | 0.5% | **60.4×** |

Backward coupling is the sharper signal by an order of magnitude — it keeps 28% of targets
while removing 99.5% of the pool — but it is capped at that 28%. Forward coupling is broader
and weaker. The winning cut uses both, which is why the OR matters: `cv`'s Soft-NMS is
backward-only and a forward-only rule drops it by construction.

> **CORRECTION, same session.** With four of seven pools built (7 targets), this table read
> `fwd≥3` at **0.00×** and I reported forward coupling as *anti-correlated* with being a
> target, with a mechanism attached ("high forward degree means the repo already builds on
> it"). At 18 targets `fwd≥3` is **2.32×** — positive. The n=7 reading was a small-sample
> artifact and is withdrawn. This is §6.1 recurring: the earlier number was not wrong to
> compute, it was wrong to interpret.

### The run that had to be thrown away first

The first rebuild returned **10,374 candidates — 11% of the known pool — with `diffusion`
and `speech` at exactly zero, and reported success.** Keyless S2 was throttling, and `hop()`
dropped any chunk whose retries were exhausted with a bare `return`: rate limiting and "these
seeds cite nothing" were indistinguishable to every caller.

Filter thresholds swept over that pool would have looked excellent and meant nothing. It was
caught only by comparing against the published per-case sizes.

`hop()` now returns `HopResult(reached, truncated_seeds, failed_chunks)` and the builder
**refuses to persist any case with a failed chunk** rather than writing an undercount that a
later sweep would treat as truth. On the guarded re-run, four cases completed and matched the
published sizes to within +11 to +50 papers (citation-graph growth); three were refused and
succeeded on retry. Retries went 3→6, backoff 3s→5s, sleep 2s→3s — §3.4's lesson that request
*rate* is the lever.

Note the shape of this bug: the same function already carried an elaborate guard against
*truncation* (the 9,999 nested cap) and none against *failure*. §6.5 one layer down.

### What P1 leaves for P2

> **SUPERSEDED the same day.** Everything below was measured on 7 pools / 18 targets. Re-run
> with 4 more pools the cut falls from 70% to **10%** and both pre-registered bars are missed —
> see [P1 re-run on 22 cases](#p1-re-run-on-22-cases--the-70-cut-does-not-replicate-and-the-hop-reaches-44-not-75-2026-08-05).
> The 70% figure is retained here as the record of what the 7-case set showed, not as a result.

A 70% cut at 89% target retention is real but not sufficient: `graph` still holds 18,215
candidates and `rl` 9,968. Structure alone does not get to a shortlist. It does hand P2 a
persisted, degree-annotated pool a third the size, and a measured statement of where the
remaining work is — the two hub-seeded repos, and the two `speech` targets that no coupling
signal can reach.

## Negative result 5 — widening the triage window from 20 to 50 does not pay (2026-08-02)

```bash
uv run python evals/run_judge_eval.py --baseline cli --rr-triage --rr-rerank \
    --rr-all-time --rr-sweep --rr-pool 50 --rr-readme-context     # 12 cases, ~$11
```

The ranker diagnosis found that ranks 1–10 and 11–50 hold statistically identical actionable
rates (31% vs 33%), so the top-10 cut looked arbitrary and ~13 actionable papers per case were
being discarded before triage ever saw them. Widening the window to 50 was the obvious
consequence. **It is worth approximately nothing.**

| | control (window 20, keyword context) | this run (window 50, README context) |
|---|---|---|
| mean net@2 | **+1.75** | **+2.42** |
| papers returned, 12 cases | 48 | 47 |
| actionable returned | 39 | **41** |
| judged pool | 142 papers, 70 actionable | 142 papers, 69 actionable |

Delta **+0.67**, paired bootstrap over cases **95% CI [−0.50, +2.00]**, **P(Δ≤0) = 0.153**.
Five cases improved, two worsened, five unchanged. **Half the gain is one case**: `speech`
moved +5.0 on its own, and without it the mean delta is **+0.27**. For scale, nominally
comparable configurations in this file have landed anywhere from +0.58 to +1.75 — the noise
band is wider than the effect.

Across 12 cases, 4× more candidates bought **two** additional actionable papers.

**The experiment did happen.** All 12 cases had a full 50-candidate window (verified by
re-running discovery at `top_n=50`), so this is not a case of the flag silently doing nothing.

### Why the extrapolation was wrong — it measured digest size, not selection quality

The rank-stratified estimate predicted ranks 1–50 would score **+8.33** against +1.88 for
ranks 1–10. That estimate scaled the *admitted* papers up to a 50-paper window and counted
every admitted paper as returned:

| scaled band | returns/case | good | junk | net@2 |
|---|---|---|---|---|
| ranks 1–10 | 2.5 | 2.3 | 0.2 | +1.88 |
| ranks 1–50 | **9.9** | 9.4 | 0.5 | +8.33 |

The +8.33 row assumes a digest of ~10 papers *per case that survive the gate*, i.e. roughly
4× the volume. The shipped system cuts the digest at 10 candidates regardless of window
width, and the real run returned **3.9 papers/case**. So the extrapolation answered "what if
we return far more papers", not "what if we choose 10 from a wider pool" — which is the
question that was actually asked. A flat actionable rate across ranks 1–50 means the deeper
band is *no better* than the top band; it never implied selecting from it would be better.

**Two variables moved at once** in this run (window 20→50 *and* keyword→README context), so
the +0.67 cannot be attributed to either. Given it is not distinguishable from zero, the
decomposition is not worth $11 — but no claim about README context in Tier B should be read
out of this run. The +16 net@2 for README context stands on the 576-paper labelled set only.

### What this leaves standing

RepoRadar scored **+2.42 vs the Opus baseline's +1.83** in the same run — the first 12-case
run where it leads. That lead is **not established**: paired over cases the difference is
+0.58, **95% CI [−1.25, +2.58]**, P(≤0) = 0.291, winning 4 cases, losing 4, tying 4. Read it
as parity, which is what every run since 2026-07-12 has shown.

`min_actionable=2` won this sweep cleanly (mean net@2 **+2.42**, 0 false positives on 12
cases, mean precision 0.85) against **−2.25** at `min>=1` and **+0.42** at `min>=3` — the
first sweep in which the shipped default is not contradicted by a stricter gate. Three
earlier sweeps went the other way, so this is one data point against three, not a resolution.

## Four ways to tell the gate what a repo is — and the ceiling they all hit (2026-08-02)

```bash
uv run python evals/diagnose_triage.py --repo-context {keywords,prose,summary,extractive}
```

The prefix approach (`prose 300`) has an obvious weakness: it bets on the purpose statement
sitting in the first N characters. On `graph` that bet loses outright — its first 300
characters are link badges. So the repo was given to an LLM to read properly. Six arms, all
paired on the same **602** labelled papers:

| arm | how the repo is described | chars | precision | recall | **net@2** |
|---|---|---|---|---|---|
| `keywords` | libraries + domains + topics | 270 | 0.83 | 0.74 | +73 |
| `prose 300` | **+ first 300 chars of README** | 300 | 0.92 | 0.68 | **+95** |
| `extractive` | + 3–6 **verbatim** sentences an LLM picked | ~600 | 0.92 | 0.67 | **+91** |
| `prose 2000` | + first 2,000 chars | 1,828 | 0.89 | 0.68 | +86 |
| `summary_nogaps` | + LLM **paraphrase**, purpose + capabilities | 969 | 0.89 | 0.61 | +76 |
| `summary` | + LLM paraphrase incl. improvement areas | 1,894 | 0.91 | 0.52 | +70 |

### Finding 1 — verbatim beats paraphrase by +21, and paraphrase loses to nothing at all

`extractive` (+91) and `summary` (+70) differ **only** in whether the selected content is
quoted or rewritten. That is a clean **+21**. And the full paraphrase (+70) is *below the
no-description control* (+73): an accurate, well-structured LLM summary is worse input than
sending no description whatsoever.

Diagnosis of the paraphrase failure, in the order the hypotheses were tested and dropped:

| hypothesis | test | result |
|---|---|---|
| the gap list reads as an exhaustive whitelist | relabel it "NOT exhaustive" | **dead** — recall unchanged at 0.52 |
| ...so remove the gaps entirely | `summary_nogaps` | partial — recall 0.52 → 0.61, +8 (P = 0.152), still −19 vs prefix |
| the description is simply too long | compare at matched length | **dead** — `summary` (1,894 ch) is 16 *below* `prose 2000` (1,828 ch) |
| paraphrase discards the paper-matching vocabulary | `extractive`, verbatim | **supported** — recovers the whole gap |

A README states techniques in the words papers use — "contextual late interaction",
"token-level embeddings", "`nn.MessagePassing`". Rewriting them into descriptive prose
removes the very signal the gate matches on.

### Finding 2 — semantic selection does NOT beat positional selection

`extractive` vs `prose 300` is **−4, 95% CI [−16, +8], P = 0.778**. Letting a model choose
*which* sentences to send is worth nothing over taking the first 300 characters.

**A failed prediction, recorded because it was made in advance.** The argument for
extraction was `graph`, whose prefix is badges while extraction correctly pulls "All Graph
Neural Network layers are implemented via the `nn.MessagePassing` interface" from deep in
the README. On that exact case extraction scored **+0 against the prefix's +2**. The
mechanism was real, visible in the text, and did not produce the predicted gain.

### The ceiling

Every arm that supplies *any* purpose statement lands between +85 and +95, and none is
distinguishable from another. Four different extraction strategies — a packaging tagline, a
prefix, LLM paraphrase, LLM verbatim selection — converge on the same place. The limit does
not appear to be how the description is extracted; it is what the documents contain.

Two consequences:

* **Nothing changes.** `profiler.prose_chars: 300` stays the default: statistically tied
  with extraction, no per-repo LLM call, no extra disclosure. `repo_summary.py` ships as a
  measured module that `rr update` does not call.
* The untested direction with the most headroom is **the user stating what they want to
  improve**. It is ground truth rather than inference from documents, and it is the one
  input that is not bounded by what the README happens to say. Separately,
  `improvement_areas` has never been tried on *retrieval*, which is the measured
  bottleneck (18/24 reach) — "adaptive quantization strategies" is a search query, and
  gating was possibly the wrong place to spend it.

**Caveat on all 12 cases:** they are popular OSS projects with well-maintained READMEs, so
the prefix bet pays on 11 of 12. A private codebase — RepoRadar's actual target — is far
more likely to have thin or badly-ordered docs, where a prefix should degrade and extraction
should not. This benchmark cannot see that difference, and `graph` is the only case that
even approximates it.

## How much prose does the gate need? Some beats none; the amount is unresolved (2026-08-02)

```bash
uv run python evals/diagnose_triage.py --repo-context keywords              # control
uv run python evals/diagnose_triage.py --repo-context prose --prose-chars N # the sweep
uv run python evals/compare_triage.py evals/.work/diag_triage_{keywords,prose300}.json
```

Five arms, all paired on the same **602** labelled papers, ~$0.10 each:

| arm | repo half of the prompt | precision | recall | **net@2** | vs control | 95% CI | P(Δ≤0) |
|---|---|---|---|---|---|---|---|
| `keywords` | libraries + domains + topics | 0.83 | 0.74 | +73 | — | — | — |
| `tagline` | libraries + a 23–230 char one-liner | 0.88 | 0.70 | +85 | +12 | [−8, +31] | 0.126 |
| **`prose 300`** | **+ first 300 chars of the README** | **0.92** | 0.68 | **+95** | **+22** | **[+4, +41]** | **0.008** |
| `prose 2000` | + first 2,000 chars | 0.89 | 0.68 | +86 | +13 | [−7, +33] | 0.112 |
| `prose 6000` | + first 6,000 chars | 0.90 | 0.68 | +89 | +16 | [−5, +36] | 0.069 |

> ### Correction (same day) — "the curve turns over" was over-read
>
> This section originally claimed **more prose is worse**, and explained it as "300 is
> where a README stops describing the project and starts on badges". **Both halves were
> wrong**, and the correction came from a reader pushing back rather than from the data
> changing.
>
> **The statistics never supported it.** 300 vs 2000 is **+9, P = 0.108**; 300 vs 6000 is
> **+6, P = 0.193**. Neither is significant. 300 is the **argmax of four noisy arms**, and
> a claim about the *shape* of a curve needs the differences *between* its points to be
> real. They are not.
>
> **The mechanism was false on inspection.** Reading the text the extra budget actually
> buys: `rag` chars 300–2000 hold ColBERT's late-interaction explanation (token-level
> embedding matrices, MaxSim); `cv` holds its capability list (panoptic segmentation,
> Cascade R-CNN, PointRend, ViTDet); and **`graph`'s first 300 characters are link badges
> — its real description begins *after* the cut.** The extra text was not dilution, it was
> the most paper-relevant content in the file, and on one of twelve cases the 300-char
> prefix already fails outright.
>
> **What survives:** *some* purpose statement beats none (+12 to +22, every budget
> positive, Bonferroni-adjusted P = 0.031 for the best arm). **Which budget is best is
> unresolved**, and a prefix is a lottery on document layout that these 12 repos happen
> to mostly win. The default stays at 300 because it is the best *measured* arm, not
> because it is optimal.

Four arms were compared against the control and the best selected, so the figure to act on
is the **Bonferroni-adjusted P = 0.031**, not the raw 0.008.

ColBERT's first 300 characters read "a *fast* and *accurate* retrieval model, enabling
scalable BERT-based search over large text collections" — the one fact its keyword profile
never contained, which instead reports "web APIs" because the project depends on flask.
That is what a purpose statement buys when the prefix happens to contain one.

One hypothesis this **kills**, which an earlier section of this file was built on:

*"The keyword block is actively misleading, so removing it helps."* `tagline` drops it,
`prose` keeps it, and at fixed budget they are statistically indistinguishable.

It also retires a wrong lead from this same session: cases whose README hit the 2,000 cap
averaged +1.71 against +0.20 for those under it, which looked like evidence the cap was
binding. It was an artifact of *which* repos have long READMEs. Twenty cents of sweep
beat the inference.

**Shipped:** `profiler.prose_chars` defaults to **300**. Against the 2,000 first
implemented that is +9 net@2, ~85% fewer prose tokens per triage call, and materially less
of a possibly-proprietary README leaving the machine — better on all three axes.

### Confirmed on Tier B: +1.00 net@2, and the digest's junk nearly halved

```bash
uv run python evals/run_judge_eval.py --baseline cli --rr-triage --rr-rerank \
    --rr-all-time --rr-sweep --rr-pool 50 --rr-prose-chars 300     # 12 cases, ~$11
```

| | previous shipped (window 20, no prose) | now (window 50, prose 300) |
|---|---|---|
| mean net@2 | **+1.75** | **+2.75** |
| papers returned, 12 cases | 48 | 48 |
| actionable | 39 | **43** |
| **junk papers in the digest** | **9** | **5** |
| mean precision at `min>=2` | 0.85 | **0.92** |

Delta **+1.00**, paired bootstrap over cases **95% CI [+0.00, +2.00]**, **P(Δ≤0) = 0.032**.
**Seven cases improved, one worsened**, four unchanged — and unlike the depth-50 run, it does
not rest on one case: dropping the largest mover (`speech`, +4.0) still leaves **+0.73**.

The product statement is the junk row: at an unchanged digest size, **the number of
non-actionable papers a reader has to wade through fell from 9 to 5 across 12 repos.**

**What Tier B cannot settle.** Holding the window at 50, `prose 300` vs `tagline` is
**+0.33, 95% CI [−0.42, +1.50], P = 0.330** — 2 cases better, 3 worse, 7 unchanged. At
n=12 cases Tier B simply cannot resolve a difference the 602-paper instrument puts at +10.
Decide prompt questions on the labelled set; use Tier B to confirm the direction.

**Attribution.** This run changed window *and* prose against the +1.75 control, so +1.00 is
the combined figure. The window is unlikely to be the source: widening it alone moved
returned-actionable from 39 to 41 and was
[not distinguishable from zero](#negative-result-5--widening-the-triage-window-from-20-to-50-does-not-pay-2026-08-02),
while the labelled set puts prose at +22 on its own. Read the gain as mostly prose, but a
window-50/no-prose arm was never run at Tier B.

**Against Opus, still parity.** RepoRadar +2.75 vs the baseline's +1.83 in the same run;
paired over cases that is **+0.92, 95% CI [−0.67, +2.75], P = 0.148**, winning 5, losing 3,
tying 4. The second consecutive run where RepoRadar leads on the mean and neither run
establishes it.

## Correction — what the "README variant" actually sent (2026-08-02)

> **The +16 net@2 "README context" result was not about READMEs.** It read
> `_collect_text_corpus(repo)[0]`, and that corpus puts `_packaging_metadata_text` **first**
> whenever a project declares a description. On **11 of the 12** benchmark repos, element 0
> is the packaging one-liner.

| case | what "README context" actually sent | chars | the real README |
|---|---|---|---|
| `http` | "Python HTTP for Humans." | **23** | 1,893 |
| `cli` | "Composable command line interface toolkit" | **41** | 1,557 |
| `peft` | "Parameter-Efficient Fine-Tuning (PEFT) deep learning" | **52** | 9,802 |
| `rag` | ColBERT's one-line subtitle | **84** | 10,047 |
| `systems` | *(genuinely the README — the only one)* | 41,195 | 41,195 |

It compounds: that variant also **dropped** the `Domains:` and `Key topics:` block. So the
prompt it produced was **smaller than the keyword control on 9 of 12 cases**:

| | keywords | "readme" | delta |
|---|---|---|---|
| mean chars | 270 | 381 | +111 |
| median case | — | — | **negative** |

The entire +111 mean is `systems` alone (+1,708). Strip it and the "more context" variant is
*less* context on nearly every case.

**Two claims in this file were therefore wrong and are withdrawn:**

1. That the variant "closes part of the 17× judge/gate asymmetry." It does not — it slightly
   narrows the prompt on most cases.
2. That the mechanism is "the gate was judging nearly blind." The plausible mechanism is the
   opposite: the keyword block is *actively misleading* (ColBERT profiles as "web APIs"
   because it depends on flask), and **deleting** it is what may have helped.

The measured +16 itself stands as a number — that comparison ran. What it is evidence *for*
was misattributed. The decomposition below separates the two mechanisms.

**Root cause, and the fix.** The eval harness and the diagnostic each rebuilt the triage
prompt instead of calling `build_triage_prompt`, so neither was measuring the shipped gate.
`run_judge_eval.py` now always uses the shipped prompt and controls the variant through
`ProfilerConfig.prose_chars`; `--rr-readme-context` is gone, replaced by `--rr-prose-chars`.
A harness that reimplements the thing under test measures the harness.

## The gate's repo context is 13% of its own prompt (2026-08-02)

Measured on the 12 case repos with `build_triage_prompt` and `assemble_repo_context`:

| | chars (mean) | range |
|---|---|---|
| repo half of the triage prompt | **366** | 244 (`cli`) – 459 (`diffusion`) |
| full triage prompt | 2,802 | — |
| repo context the **judge** sees | **6,375** | 4,668 – 9,015 |
| ratio | **17×** | 14× – 23× |

The repo half is **13% of the triage prompt**; the candidate paper's abstract (`[:1500]`) is
54% of it. The gate is asked to match a rich description of the paper against a term list of
the repo — and the judge that *defines the labels* reads 17× more about the repo than the
gate being graded against those labels.

**This is not a cost decision.** Raising the repo half to judge size costs **+$0.024 per run**
at the shipped `top_k=15` (Haiku list price, ~4 chars/token). Cost is the stated reason triage
is off by default; it never justified the size of the repo half.

The cause is structural. `build_triage_prompt(paper, profile)` takes a `RepoProfile`, which is
the **ranker's** data structure — a bag of extracted terms with no prose field to pass. The
`keywords[:12]` / `domains[:5]` / `anchors[:12]` slices are not a token budget, they are
"don't dump 400 keywords into a prompt". Triage reused the ranker's input type and inherited
its information loss, including the register mismatch: the profile enumerates what the repo
*has* (dependencies, identifiers), never what it is *for*.

Consequences, in decreasing confidence:
- `RepoProfile` should carry the repo's prose, so the gate can be told what the project is
  *for*. **Shipped** as `RepoProfile.prose` / `profiler.prose_chars` (default 2000).
- The +16 attributed to "README context" does **not** close this asymmetry — see
  [the correction](#correction--what-the-readme-variant-actually-sent-2026-08-02); that
  variant sent a 23-230 character packaging tagline on 11 of 12 cases and was *smaller*
  than the keyword prompt it beat.
- `abstract[:1500]` has never been measured. At 54% of the prompt, shrinking it would rebalance
  the repo/paper ratio for free. **Unmeasured** — do not assume it helps.

## Triage measured properly — it is not at chance (2026-08-02)

```bash
uv run python evals/diagnose_triage.py           # ~$0.10, 428 papers
```

> **Correction.** This file previously said the triage gate "carried no discriminative signal"
> and was "at chance", based on the ~10 papers a single Tier B case happened to surface. At
> n=428 that is wrong, and it was repeated across several sections. The claim is withdrawn.

**428 judged papers were already sitting in `cache/judge/`** across 12 repos — every verdict
the benchmark has ever paid for. They are a labelled set, and they make the gate falsifiable
offline for ~$0.10 instead of ~$12 for a Tier B pass.

| | precision | recall | base rate |
|---|---|---|---|
| all 428 labelled papers | **0.81** | 0.78 | 32% |
| excluding the 27 Opus baseline picks | **0.78** | 0.75 | 28% |

A gate with no signal scores precision equal to the base rate. Triage scores **+0.50 above
it**. The earlier "at chance" reading came from n=10 samples of an adversarially selected
subset, not from the gate's actual behaviour.

**Per case, the failure modes are opposite — which a single pooled number hides:**

| case | base | precision | recall | reading |
|---|---|---|---|---|
| diffusion | 77% | 1.00 | 0.91 | excellent |
| systems | 42% | 0.90 | 0.82 | |
| peft | 90% | 0.89 | 0.89 | |
| rag | 30% | 0.81 | 0.81 | |
| cv | 35% | 0.79 | 0.79 | |
| speech | 47% | 0.75 | 0.83 | |
| graph | 26% | 0.75 | 0.33 | **too strict** |
| crypto | 22% | 1.00 | 0.29 | **too strict** — never wrong, misses 5 of 7 |
| rl | 19% | 0.50 | 0.70 | **too loose** |
| cli, http | 0% | — | — | correct total abstention (53 papers, 0 false positives) |
| webdev | 0% | 0.00 | — | 1 false positive in 39 |

`crypto` and `rl` fail in opposite directions. A single global threshold cannot fix both, and
tuning `min_actionable` up — which the sweeps appear to recommend — would make `crypto` and
`graph` worse while helping `rl`.

### Why it looked like a coin flip inside Tier B

Tier B only ever judges the ranker's top 10. On that subset triage really does collapse:

| subset (Opus picks removed from both sides) | n | base | precision |
|---|---|---|---|
| the ranker's current top-10 | 20 | **15%** | **0.33** |
| judged but not in the current top-10 | 63 | 37% | **0.82** |

Triage is good at telling an actionable paper from an unrelated one, and poor at telling them
apart **among papers the heuristic ranker has already selected for surface similarity**. That
is the subset where its judgement is actually load-bearing.

**n=20, so this is suggestive and not established** — only two cases have per-paper records,
since the harness did not record returned ids until 2026-07-31.

### A claim I nearly published and had to withdraw

The 12-case run shows the ranker's top-10 at 40% actionable against a "pool" at 49%, which
looks like the ranker selecting *against* good papers. **That is an artifact.** The Tier B
pool is RepoRadar's top-10 *plus the baseline's picks*, and all 22 baseline-only papers in
that run were actionable. The pool rate is 49% precisely because it mixes RepoRadar's 40%
with Opus's 100%. There are no judge labels for the rest of RepoRadar's candidate pool, so
**whether the ranker beats a random draw from its own pool is unmeasured**, not measured and
failed.


### Negative result — giving triage the README does not help
> **[SUPERSEDED — see the Correction immediately below.](#correction--the-readme-variant-helps-the-null-was-measured-on-the-wrong-metric)**
> This section's conclusion was produced by scoring the variant on *accuracy*. On net@2, the
> metric the digest is graded on, the same two runs are **+16 (+28%)**.

```bash
uv run python evals/diagnose_triage.py --repo-context readme     # ~$0.10
```

The shipped prompt describes ColBERT to the gate as `functions, indexer, searcher functions,
trainer functions, trainer, searcher, colbert, file, torch, transformers, master` in the
domains `NLP, deep learning, scientific computing, web APIs` — 354 characters that never say
*retrieval*, and say *web APIs* because the project depends on flask. Its README's own first
line is "Efficient and Effective Passage Search via Contextualized Late Interaction over
BERT". The hypothesis was that the gate is not failing at judgement but judging nearly blind.

**It is not.** Same 428 papers, same labels, same model, only the repo half of the prompt
changed:

| | precision | recall | accuracy |
|---|---|---|---|
| `keywords` (shipped) | 0.81 | 0.78 | 0.87 |
| `readme` | 0.89 | 0.72 | 0.88 |
| delta | +0.08 | −0.06 | **+0.01** |

Aggregate numbers cannot separate a real effect from Haiku's non-determinism, so this is
paired per paper — for each one, did the variant fix a baseline mistake or break a baseline
success?

**21 fixed, 17 broke.** 38 discordant pairs, two-sided exact binomial **p = 0.63**. The flips
are symmetric: this is run-to-run noise with a precision/recall redistribution on top, not an
improvement. Per case the churn cancels out — `rl` +5/−2, `speech` +3/−5, `cv` +3/−3.

**Why the intuition was wrong.** Triage already sees the candidate paper's title and abstract,
which carry most of the signal for "is this applicable"; the repo description only has to be
good enough to establish the domain, and even a bad keyword list does that. Query building has
no such crutch — it has *only* the profile, which is why the same impoverished representation
is binding there (0 of 24 target papers reachable) and not here.

That distinction is worth keeping: **profile quality is a retrieval problem, not a judgement
problem.** Effort spent enriching the profile should be justified by retrieval gains, not by
expectations about the gate.

One small real effect, in the wrong direction: `cli` is a negative control with a 0% base rate
and the `keywords` gate abstained perfectly on all 23 papers. Adding prose about what the
project is produced one false positive. More context gives a model more surface on which to
find a connection, which is precisely wrong where the correct answer is "nothing here helps".


### Correction — the README variant helps; the null was measured on the wrong metric

> **Read `readme` in this section as `tagline`.** The variant named "README" here did not
> send a README on 11 of 12 cases; it sent the packaging one-liner and dropped the keyword
> block. The net@2 numbers below are real, the label on them was not. See
> [what the "README variant" actually sent](#correction--what-the-readme-variant-actually-sent-2026-08-02).

The section above concluded that giving triage the README "does not help", from 21 fixed /
17 broke at p = 0.63. **That test was run on accuracy, and accuracy is the wrong objective
for this gate.** net@2 charges 2 for returning a junk paper and 1 for missing a good one. A
flip from false-positive to correct-abstention is worth **+2**; a flip from true-positive to
miss costs **−1**. Counting both as "one flip" throws the asymmetry away.

Re-scored on the metric the digest is actually graded on, same two runs, same 428 papers:

| gate | returned | actionable | **net@2** | accuracy |
|---|---|---|---|---|
| Haiku, `keywords` (shipped) | 123 | 99 | **+57** | 0.87 |
| Haiku, `readme` | 103 | 91 | **+73** | 0.88 |
| Sonnet, `keywords` | 73 | 67 | +55 | 0.84 |
| a perfect gate | 129 | 129 | +129 | 1.00 |

Paired bootstrap over papers, README vs shipped: **delta +16 (+28%), 95% CI [−2, +36],
P(delta ≤ 0) = 0.046.** The interval still grazes zero, so this is *suggestive, not
established* — but "does not help" was wrong, and it was wrong because of the measurement,
not the sample.

**Where the gain comes from is the interesting part.** On the ranker's top-10 — the only
subset a Tier B run ever judges, and where the gate collapsed to 0.33 precision:

| gate | top-10 precision | elsewhere |
|---|---|---|
| Haiku, `keywords` | 0.33 | 0.81 |
| Haiku, `readme` | **0.50** | 0.85 |
| Sonnet | **abstained entirely** | 0.83 |

The README context does its work precisely where the gate was weakest. n=20 on that subset,
so treat the size with suspicion and the direction as real.

### Sonnet answers the capability-vs-information question, and the answer is neither

The test was posed as a fork: if a stronger model rescues the top-10, the gate is
capability-limited and deserves a model tier on the ~10 papers that matter; if it collapses
too, the papers are genuinely indistinguishable from title and abstract.

**Sonnet did neither — it abstained on the top-10 entirely**, returning nothing rather than
discriminating. Overall it is far more conservative than Haiku (precision 0.92 vs 0.81,
recall 0.52 vs 0.78), which is a different operating point rather than a better gate: paired
net@2 delta **+4, 95% CI [−16, +25], P(delta ≤ 0) = 0.37**. It also failed 6 of 428 calls.

So a stronger model is **not worth adopting here**, and this is now the second time that
conclusion has been reached — Feature 6 found `claude-sonnet-5` metric-identical in 2026-07.
The lever is the prompt's repo half and the operating point, not the model.

### The lesson worth keeping

`evals/compare_triage.py` now reports net@2 with a paired bootstrap as the primary result and
labels the accuracy view as secondary. **A harness that scores a component on a different
objective than the product will confidently report a null on a real improvement** — which is
exactly what happened, and the tool that produced the mistake is the one that had to change.

## The ranker measured for the first time — it discriminates coarsely, not finely (2026-08-02)

```bash
uv run python evals/diagnose_ranker.py --per-stratum 4      # ~$5 of GPT-5.5
```

Every label the benchmark owned came from the ranker's own top-10 plus the baseline's picks,
so the ranker had never been scored, and an earlier attempt to infer its quality from the
Tier B pool produced an artifact. This judges a **rank-stratified sample** of the real
candidate pool instead, and adds those verdicts to the shared cache — the labelled set went
from 428 to **576 papers**.

| rank band | n | actionable |
|---|---|---|
| 1–10 | 48 | **31%** |
| 11–50 | 48 | **33%** |
| 51–150 | 48 | 15% |
| 151+ | 44 | 7% |

**Two findings, and the second is the useful one.**

The ordering is *not* decoration: 31–33% in the top 50 against 7% past rank 150 is a real,
large separation. The heuristic weights do sort relevant from irrelevant.

But **ranks 1–10 and 11–50 are indistinguishable** — 31% vs 33%. The ranker cannot tell which
of its top 50 are the best 10, which makes the top-10 cut arbitrary. Per case that discards
roughly **13 actionable papers to keep 3**.

That is a concrete, cheap change with a measurable prediction: the gate currently sees ~3.1
actionable papers per case and could see ~16.5 at depth 50, for the cost of more triage calls
(Haiku, pennies). `--rr-rerank` already triages a deeper pool, but at `RERANK_POOL = 20` it
stops well inside the flat region.

It also reframes the retrieval work. The citation hop's 18/24 recall sits in a 92,014-paper
pool; a ranker that cannot order within its own top 50 will not order 92,014. **Depth of the
gate is the near-term lever; ordering quality is what blocks the citation-hop pool from being
usable at all.**

## Two-case re-benchmark after the quoting fix (2026-08-02)

```bash
uv run python evals/run_judge_eval.py --case rag    --baseline cli --rr-triage --rr-rerank --rr-all-time --rr-sweep
uv run python evals/run_judge_eval.py --case speech --baseline cli --rr-triage --rr-rerank --rr-all-time --rr-sweep
```

PR #62 quoted multi-word query terms, which on the arXiv API is the difference between OR and
a phrase — `all:speech recognition` matches 246,802 papers, `all:"speech recognition"` matches
6,845. Diffing query sets across all 12 benchmark repos showed **93% unchanged**; only `rag`
and `speech` moved, so only those two were re-run rather than paying for ten unchanged cases.

**Both went down.**

| case | 07-31 | 08-02 | pool actionable | triage found |
|---|---|---|---|---|
| rag | −1.0 | **−2.0** | 5/13 → **3/13** | 0 of 3 |
| speech | −2.0 | **−4.0** | 7/13 → **6/13** | 2 of 6 |

**The pool changed, and the new papers are worse.** `rag` judged 4 papers it had never seen,
scoring 0, 0, 1, 1 — four junk papers displacing four previously-judged ones, two of which
were actionable. `speech` judged 5 new, scoring 1, 1, 1, 1, 2.

**The two cases changed for opposite reasons, and only one is a real improvement.**

For `speech` the fix is unambiguously right in isolation: the OR-query collapsed to a real
phrase, and the nonsense `"speech speech recognition"` and `"speech recognition recognition"`
disappeared. Narrower, more precise, fewer wasted queries — and net@2 still fell.

For `rag` the fix is correct in form and empty in effect. ColBERT's top TF-IDF terms are
`indexer`, `trainer`, `searcher`, `functions` — its **API surface, not its subject**. Quoting
turned `all:searcher functions` (a broad OR that at least returned IR papers) into
`all:"searcher functions"` (an exact phrase almost no paper contains). The queries got
narrower and the pool got emptier.

That is not a query-syntax problem. It is the profiler again: ColBERT's docs are autodoc API
reference, so TF-IDF extracts method names. `indexer` and `trainer` survive the boilerplate
stoplist because they are ordinary English words — the earlier Sphinx-directive filter cannot
catch them. **A repo whose documentation is an API reference gets profiled by its API.**

**Attribution, honestly.** Three things differ between the runs: the query change, live arXiv
drift, and a non-deterministic judge. n=2. The direction is consistent and the pool-composition
evidence is concrete, but this cannot support "the quoting fix caused a regression" — the
`speech` change in particular is provably a better query that produced a worse score, which is
more consistent with the *pool* being poor either way than with the fix being wrong.

### `min_actionable=2` is now contradicted on every case where it has been swept

| case | `min>=1` | `min>=2` (shipped) | `min>=3` |
|---|---|---|---|
| speech (07-31) | −5.0 | −2.0 | **+0.0** |
| speech (08-02) | −11.0 | −4.0 | **+0.0** |
| rag (08-02) | −6.0 | −2.0 | **+0.0** |

Three sweeps, three times the strictest gate wins — **by abstaining**. RESULTS.md still calls
`min_actionable=2` "the decisively-correct gate" on pre-fix cross-case data. That claim is not
re-established and the evidence now runs against it.

The reason is visible in the same table: **triage found 0 of `rag`'s 3 actionable papers and
2 of `speech`'s 6.** On the top-10 subset — the only one Tier B judges — the gate really does
have little signal, so abstention is most of what it has left there. Note this is *not* true of
the gate in general: at n=428 it scores 0.81 precision against a 32% base rate. That is not an argument for shipping
`min_actionable=3` — it is an argument that the gate is not currently doing the job the
threshold is tuned for.

## Candidate-pool diagnosis — what RepoRadar cannot reach, and why (2026-08-01)

The Tier B headline (+1.75 vs Opus +1.83) says the two systems score alike. They do not
recommend alike. This section records the measurement that established that, and the five
candidate fixes tried against it. **Four are negative results**, preserved so nobody pays to
rediscover them; the fifth — a citation hop from the repo's own bibliography — is the only
one that reaches the papers at all, and its own caveat is as large as its result.

| approach | recovered of 24 |
|---|---|
| current TF-IDF keyword queries | 0/24 |
| LLM phrases, "name what the repo uses" | 2/24 |
| LLM phrases, "name what the repo lacks" | 0/24 |
| citation-count-sorted search | 1/24 |
| fetch deeper (raise `max_results`) | ≤3 |
| **citation hop from the repo's bibliography** | **18/24** |

Every number here is re-derivable rather than asserted:

```bash
uv run python evals/diagnose_pool.py                                  # free, keyless
uv run python evals/diagnose_citation_hop.py                          # free, keyless
uv run python evals/diagnose_query_generation.py --prompt uses        # ~$0.01
uv run python evals/diagnose_query_generation.py --prompt lacks       # ~$0.01
```

### The measurement: 2030 papers fetched, 0 of 24 known-good reached

Take every paper the Opus baseline recommended **and** the GPT-5.5 judge scored ≥2
(genuinely actionable). That is 24 papers across 9 repos. Ask one question: was it in the
pool RepoRadar's own queries fetched?

| | |
|---|---|
| papers RepoRadar fetched across the 9 repos | **2030** |
| known-good papers among them | **0** |

Not a near miss — a disjoint set. The papers are canonical and on-topic: ConvNeXt,
Prioritized Experience Replay, Double Q-learning, Soft-NMS, WhisperX, Distil-Whisper,
Speculative Decoding, Exphormer.

**The obvious explanations were checked and none hold:**

- *Not an id-matching artifact.* Pool ids are `1907.04378v1`; version-stripping matches the
  baseline's `2303.00747` form correctly.
- *Not the category filter.* **23 of the 24 are inside the categories being searched.** Only
  TinyLFU (`cs.OS`) sits outside.
- *Not an arXiv limitation.* With a precise phrase, `all:"prioritized experience replay"`
  returns 1511.05952 at **rank 1**, and `all:"ConvNet for the 2020s"` returns 2201.03545 at
  **rank 1**.

**The mechanism.** TF-IDF produces generic single terms, so RepoRadar sent `all:model`,
`all:image`, `all:torch` — each matching tens of thousands of papers — and kept the first 50
by arXiv's *lexical* relevance, which has no impact weighting. It was sampling near-randomly
from a huge match set, eight times per repo. (The unquoted-space bug that made this
dramatically worse is fixed in PR #62; it was a compounding factor, not the root cause.)

A second-order problem the fix does not touch: even the right phrase can fail.
`all:"speculative decoding"` does **not** return the 2022 original in its top 20, because
hundreds of later papers use the term and nothing ranks by influence.

### Negative result 1 — repos do not cite what would improve them

Harvesting a repository's own bibliography looks like a free, keyless, offline way to seed a
corpus. It is not: **0 of the 24 appear anywhere in the repos' own docs**, including `graph`
(pytorch_geometric), whose README lists 112 arXiv papers.

The reason is structural and worth stating plainly, because it recurs below: **a codebase
cites what it implements. The valuable paper describes what it should adopt next.** Those
are disjoint by construction. A project's bibliography is a well-targeted index of things it
already does.

### Negative result 2 — LLM-generated search phrases, tested two ways, recover 8% and 0%

A dependency verification estimated that LLM-generated technique phrases would recover
**19/24**. That estimate was produced by people who already knew the answers. Both prompts
below were run for real against Haiku, given the repo profile plus its documentation and
never the targets, with identical inputs, phrase counts and arXiv checks — so the only
variable is the question asked.

| prompt | recovered | phrases matching **zero** papers |
|---|---|---|
| current TF-IDF keywords (control) | 0/24 | — |
| *"name the techniques this repo uses"* | **2/24 (8%)** | 19/54 |
| *"name what this repo lacks"* | **0/24 (0%)** | **45/54** |

**They fail for opposite reasons, and that is the finding.**

*Asked what the repo uses*, the model answered accurately and uselessly. For detectron2 it
produced *"Mask R-CNN with feature pyramid networks"*, *"panoptic segmentation"*,
*"Cascade R-CNN"* — a correct description of what detectron2 implements. The targets were
Soft-NMS, Copy-Paste augmentation and ConvNeXt, which it does not. For whisper it emitted
*"multilingual speech recognition"* while the targets were WhisperX, Distil-Whisper and
Speculative Decoding, all inference-acceleration work. **This is the same asymmetry as
negative result 1**: an LLM reading a repository inherits the repository's vocabulary, and
that vocabulary describes what the repo *has*. The two recoveries prove the rule — `rl`
found Prioritized Experience Replay precisely because stable-baselines3 already implements
PER, so "what it has" and "what would improve it" coincided. That is the case where a user
needs the tool least.

*Asked what the repo lacks*, the model aimed correctly — quantization, knowledge
distillation, ViT backbones, cross-encoder reranking, adapter merging are all exactly the
right classes of work — and then phrased them as descriptive compounds that no paper title
contains. **83% of its phrases matched nothing.**

The sharpest single comparison is `rl`. The "lacks" prompt emitted *"experience replay
prioritization methods"* — conceptually the exact target paper — and got **0 hits**. The
"uses" prompt emitted *"prioritized experience replay"* and **found it**. Same concept: one
is the literature's term of art, the other is a description of it.

**Do not read this as a prompt-tuning problem.** The failing prompt already specified "2-5
words", "real terms of art only" and "favour method names over topic names", and explicitly
forbade naming techniques the repo already implements. Both failures are about the gap
between how a codebase describes itself and how the literature names its own methods —
a retrieval-representation problem, not a wording one. Any further attempt here should be
justified by a mechanism that closes *that* gap (e.g. validating emitted phrases against the
index and discarding zero-hit ones, or matching in embedding space rather than by exact
phrase) rather than by another rewording.



### The one thing that worked — a citation hop from the repo's own bibliography (18/24)

Reproduce: `uv run python evals/diagnose_citation_hop.py` (free, keyless).

Negative result 1 established that a repo never cites the papers that would improve it. That
is true and it is not the end of the story: those papers are **one hop away** in the citation
graph.

Seeds are the arXiv ids the repo itself cites (README, `docs/`, `.bib`, `CITATION.cff`) —
the only seed set a cold-start repo has, since a fresh install has no ratings or stars. One
hop in each direction, seeds capped at 60, via the Semantic Scholar batch endpoint the
Feature 8 plumbing already uses.

| case | seeds | candidates | recovered | un-enumerable seeds |
|---|---|---|---|---|
| rag | 7 | 1,856 | **5/5** | 0 |
| cv | 18 | 7,378 | **3/3** | 0 |
| rl | 30 | 29,430 | **3/3** | 5 |
| peft | 18 | 6,342 | **2/2** | 0 |
| graph | 121 | 36,868 | **3/3** | 8 |
| speech | 2 | 6,068 | 2/3 | 0 |
| diffusion | 10 | 4,072 | 0/2 | 0 |
| crypto | 0 | — | 0/2 (no arXiv-indexed bibliography) | — |
| systems | 0 | — | 0/1 (no arXiv-indexed bibliography) | — |
| **total** | | **92,014** | **18/24 (75%)** | 13 |

Against 0/24 for keyword search, 2/24 for LLM phrases and 1/24 for citation-sorted search,
this is the only approach that reaches the papers at all. The targets are genuinely
discovered, not handed over: seeds are subtracted from the candidate set, and 0 of the 24
appear in any repo's own documentation.

**Both hop directions earn their place.** Forward (papers *citing* a seed) does most of the
work — consistent with "what improves a codebase is later work building on what it already
uses" — but the backward direction uniquely contributes several targets, so dropping it would
cost real recall.

> #### Correction: an earlier version of this measurement said 14/24
>
> It was a transport artifact, not a result. The first script sent 100 seeds per request, and
> `/paper/batch` truncates **nested** items at 9,999 across a request, filled greedily in id
> order. Verified directly: 18 seeds in one request returned `[9999, 0, 0, …]` — one seed's
> citations and seventeen empty arrays, HTTP 200, no error. Per-seed requests returned 47,897
> nested items against 9,999.
>
> Two conclusions drawn from the bad number were wrong and are withdrawn:
>
> - **`graph` did not score 0/3 because its seeds were irrelevant. It scored 0/3 because it
>   has the most seeds (121)** — one hub consumed the budget and the other 59 came back
>   blank. Corrected, it is 3/3.
> - **"Seed count does not predict recall" was backwards.** Under the buggy chunking more
>   seeds actively destroyed recall. There is no evidence here for or against a seed-relevance
>   effect; the apparent one was an artifact.
>
> The corrected script chunks at 4 seeds and splits any response that comes back pinned at the
> cap. The same defect existed in shipped code — `citations.fetch_references` chunked at the
> 500-id limit and was silently losing ~29% of `w_citation_proximity`'s edges.

**A hard wall worth knowing about.** 13 seeds across `rl` and `graph` saturate the 9,999 cap
even when requested alone, and the API enforces `offset + limit < 10000` on paging. A paper
with tens of thousands of citers cannot be fully enumerated by anyone, keyed or keyless. The
script counts these rather than silently accepting a truncated answer.

**The caveat grew along with the result: 92,014 candidates for 18 papers, a density of
1 in 5,111** — worse than the 1 in 2,042 the truncated run appeared to show, because the
recovered pool is 3.2x larger while recall rose by less than a third. Recall is transformed
and precision is untouched. But this changes the shape of the problem rather than solving it:
the pool now *contains* the answers, so selection becomes a real and tractable problem
instead of a moot one. At ~10,000 papers per repo it is a heavier load than the embedding
cache and vector index carry today, and the filter that reduces it is unbuilt — see
[`RETRIEVAL_DESIGN.md`](../archive/RETRIEVAL_DESIGN.md).

### Negative result 3 — citation-sorted retrieval is a multiplier, not a fix

Ranking search results by citation count sounds like the obvious answer to "arXiv has no
impact signal". As a drop-in replacement it recovers **1 of 24**. It improves the ordering
of a match set; it cannot repair a match set that never contained the paper. It is worth
having *after* query quality is fixed, not before.

It also hard-errors on RepoRadar's modal queries: the Semantic Scholar bulk endpoint returns
HTTP 400 above 10M hits, and `model` (24.4M) and `data` (24.8M) both exceed that.
`fieldsOfStudy=Computer Science` brings `model` down to 6.46M and works.

### Negative result 4 — fetching deeper is not a fix and carries a real cost

Raising `max_results_per_query` recovers at most 3 of the 24 into the *pool* (not into the
digest — nobody measured whether they survive a Top-10 cut against ~16,000 competitors), and
**9 of the 24 are outside the match set at any depth** — all three `rl` papers are absent
even from a 265,785-result query.

The cost is not theoretical. Sustained polling at ToU-compliant 3.2 s spacing earned this
project's machine an **IP-level block that survived 30 minutes of complete silence and
lasted roughly 70 minutes**. Request *rate* is the lever, not page size.

### A precondition that applies to every option

All 24 targets are **≥11 months old**, and `collector.py` discards anything older than
`lookback_days` (default **14**). Under a default `rr update`, every option above scores 0/24
on merit alone. Any retrieval work must be developed and evaluated in the `--foundational`
path, which already sets relevance sort and a 100-year window.

## Re-benchmark after the query-construction fix (2026-07-31)

```bash
uv run python evals/run_judge_eval.py --baseline cli --rr-triage --rr-rerank --rr-all-time
```

Same 12 cases, same configuration and same judge as the 2026-07-29 run, on `main` @ PR #59.
Baseline spend $9.77 (12 Opus calls) plus judge/triage.

**Why this re-run was warranted, and why it is not a like-for-like comparison.** PR #59 fixed
`profile_repo` — `setup.py` dependencies were never read, and `stop_words="english"` let README and
Sphinx furniture through, so the benchmark had been transmitting queries like `(all:license)` and
`(all:https)`. Diffing the query sets built by the pre- and post-fix profiler over these same 12
repos, **only 32 of 96 queries survive**. The benchmark is fetching substantially different papers,
which is the point of the fix and also the reason the two runs are not measuring the same pool.

| Case | 2026-07-29 | 2026-07-31 | Δ | pool actionable/judged |
|---|---|---|---|---|
| rag | 0.0 | −1.0 | −1.0 | 5/13 |
| cv | +1.0 | +1.0 | — | 9/13 |
| rl | 0.0 | +2.0 | +2.0 | 9/13 |
| webdev (neg. control) | −2.0 | −2.0 | — | 0/10 |
| peft | +4.0 | **+7.0** | +3.0 | 11/12 |
| diffusion | +6.0 | **+10.0** | +4.0 | **12/12** |
| graph | −4.0 | 0.0 | +4.0 | 6/13 |
| speech | +8.0 | **−2.0** | **−10.0** | 7/13 |
| crypto | +3.0 | 0.0 | −3.0 | 2/12 |
| systems | +2.0 | **+6.0** | +4.0 | 9/11 |
| cli | 0.0 | 0.0 | — | 0/10 |
| http | 0.0 | 0.0 | — | 0/10 |
| **mean** | **+1.50** | **+1.75** | **+0.25** | |

**Do not read the mean as the result.** It moved +0.25, which is *smaller* than this harness's own
stated per-case noise band (±1 net@2), and it is the average of movements an order of magnitude
larger in both directions. The informative content is in the per-case column.

**Where the fix landed, it landed hard.** `diffusion` is the case whose queries were most corrupted
— three of its eight were `license`/`https` boilerplate — and it went +6.0 → +10.0 at **precision
1.00 with every one of the 12 judged pool papers actionable**. `systems` (+4.0), `graph` (+4.0),
`peft` (+3.0) and `rl` (+2.0) moved the same direction. That is five of the seven repos that had
zero anchors before the fix, which is the pattern you would predict if the defect was upstream in
query construction.

**`speech` regressed by 10 points and is not explained by the profile.** Its keywords are now
`tiktoken, speech, speech recognition, recognition, model, whisper, audio` and its anchors resolve
correctly (`torch`, `tiktoken`, `numba`) — the profile is *better* than before. The proximate cause
is the gate, not retrieval: triage marked only 4 of the top 10 actionable, and the judge agreed with
2 of those 4, while the surrounding pool held 7 actionable of 13. So the papers were there and the
triage step rejected them. `crypto` shows the same shape more mildly (+3.0 → abstained, pool 2/12).
Whether that is triage non-determinism or a real interaction with the new candidate order is
**unresolved and worth its own investigation** — recorded here rather than averaged away.

**Abstention moved in both directions, which is the honest read of a precision gate.** `graph`
abstained where it previously returned junk (−4.0 → 0.0, a win) and `crypto` abstained where it
previously returned something useful (+3.0 → 0.0, a loss). Both `cli` and `http` abstained
correctly on empty pools, as before.

**Confounds, stated plainly.** Three things differ between the runs besides the fix: the judge is
non-deterministic; the triage gate is a separate non-deterministic LLM call; and the harness fetches
live from arXiv, so the corpus has moved (mitigated but not eliminated by `--rr-all-time`'s
relevance-sorted all-time window). A single 12-case run cannot cleanly attribute the +0.25 mean.
What it *can* support is the narrower claim: the fix changed two thirds of the transmitted queries,
and the resulting per-case movements are concentrated in the repos the bug affected.


### `speech` regression, diagnosed (2026-07-31)

```bash
uv run python evals/run_judge_eval.py --case speech --baseline cli \
    --rr-triage --rr-rerank --rr-all-time --rr-sweep
```

The −10.0 swing reproduced (net@2 −2.0 again). With the harness now recording *which* papers
each system returned, the cause is visible and it is **not** retrieval.

**The Top-10 candidate pool held five genuinely actionable papers. Triage admitted four
papers and got two of them right.**

| rank | admitted | judge | paper |
|---|---|---|---|
| 1 | **yes** | 1 | Improving Children's Speech Recognition by Fine-tuning… |
| 2 | **yes** | **2** | FiLM-Based Speaker Conditioning of a SpeechLLM… |
| 3 | **yes** | **2** | POWSM: A Phonetic Open Whisper-Style Speech Foundation Model |
| 4 | **yes** | 1 | TokenChain: A Discrete Speech Chain via Semantic Token Modeling |
| 5 | no | **2** | A Density Ratio Approach to Language Model Fusion |
| 6 | no | 1 | Low-resource speech recognition and dialect identification |
| 7 | no | **2** | Omni-Router: Sharing Routing Decisions in Sparse MoE |
| 8 | no | 1 | Differentiable Allophone Graphs |
| 9 | no | **2** | Multi-task Language Modeling for Improving Speech Recognition |
| 10 | no | 1 | FlashSpeech: Efficient Zero-Shot Speech Synthesis |

**On this pool the triage gate carried no discriminative signal.** (Superseded in scope by
[Triage measured properly](#triage-measured-properly--it-is-not-at-chance-2026-08-02): the
gate is well above chance overall, and weak specifically on top-ranked papers like these.)
Precision 50% (2 of 4),
recall 40% (2 of 5) — against a pool base rate of 50% actionable. Selecting four papers at
random would have scored the same precision. The two false positives were ranked **1st and
4th**, above three papers the judge scored 2, so the listwise rerank is ordering by an
`llm_score` that does not track the judge either. This is one case at n=10 and cannot support
a general claim about triage, but it does mean `speech`'s −2.0 is a gate failure, not a
consequence of the query-construction fix changing what was fetched.

**The threshold sweep contradicts `min_actionable=2` on this case.**

| gate | returned | net@2 |
|---|---|---|
| `min>=1` | 10 | −5.0 |
| `min>=2` | 4 | **−2.0** |
| `min>=3` | 0 | **+0.0** (abstains) |

Strictest wins, and it wins by returning nothing. Where the gate has no signal, its only
remaining value is abstention — which is exactly what `graph` did in the 12-case run
(−4.0 → 0.0). RESULTS.md calls `min_actionable=2` "decisively correct" on pre-fix cross-case
data; that claim is not re-established post-fix and should not be treated as settled.

**The candidate pool is not stable between runs of the same command.** This run judged
`2606.06211v1` for the first time — a paper absent from the pool eight hours earlier under an
identical invocation, because the harness fetches live from arXiv. The pool went from 7
actionable of 13 to 8 of 13. That the net@2 landed on −2.0 both times is therefore partly
coincidence, not evidence of stability, and it is a reminder that **two Tier B runs are never
strictly comparable** — the confound applies to every historical row in this file, including
the ones showing "no regression".

## Re-benchmark after Features 1–8 and 10 — no regression (2026-07-29)

Nine features shipped after the 12-case benchmark (MCP server, GitHub Action, `rr search` +
sqlite-vec, citation alerts, domain sources, learned recommendations, SPECTER2 — plus a ranker
change that treats a *missing* category as absent rather than zero). None of it had been
re-measured. Four runs, same 12 cases, judge = GPT-5.5, baseline = Opus 4.8 (`--baseline cli`).

### 1. The validated configuration did not regress

Replicating the 2026-07-12 flags exactly (`--rr-triage --rr-min-actionable 2 --rr-rerank
--rr-all-time`):

| Case | 2026-07-12 | 2026-07-29 | Δ |
|---|---|---|---|
| rag | 0.0 | 0.0 | — |
| cv | +1.0 | +1.0 | — |
| rl | 0.0 | 0.0 | — |
| webdev | −2.0 | −2.0 | — |
| peft | +4.0 | +4.0 | — |
| diffusion | +7.0 | +6.0 | −1.0 |
| graph | −4.0 | −4.0 | — |
| speech | +7.0 | **+8.0** | +1.0 |
| crypto | +3.0 | +3.0 | — |
| systems | +1.0 | **+2.0** | +1.0 |
| cli | 0.0 | 0.0 | — |
| http | 0.0 | 0.0 | — |
| **mean** | **+1.42** | **+1.50** | **+0.08** |

**9/12 cases identical; the 3 that moved moved by exactly ±1.0 — inside the measured noise floor
(below).** Against Opus: **+1.50 vs +1.83**, a 0.33 gap — *identical* to the 0.33 gap on 2026-07-12.
All 12 baselines ran this time (crypto's `claude -p` succeeded), and RepoRadar hallucinated **0**
references across all four runs.

### 2. Noise floor, measured rather than assumed

Two runs were executed with the *same effective configuration* (see finding 4 — the second
intended to add DBLP but silently did not). Treating them as a repeat measurement:

- **mean net@2 moved 0.08**; **exactly one case moved, by 1.0**.

This independently confirms the "±1 net@2 is noise" convention this file has used since the start,
and it is the yardstick for reading everything above: a per-case Δ of ±1 means nothing; the `rl`
regression in finding 3 (−2.0) is outside it.

### 3. Adding Semantic Scholar as a source did not help

> **VOID, not null (corrected 2026-08-12).** The S2 arm added **no papers**. Measured with a
> real API key, S2 returns **0 results for 11 of these 12 cases** when given the malformed
> query this run sent it (C-9); the repaired query returns 20 everywhere, with **zero
> overlap**. The mechanism proposed below — S2 papers competing harder because the ranker
> stopped penalising their missing categories — requires S2 papers in the pool, and there
> were none. The per-case moves and the precision drop are run-to-run drift attributed to a
> channel that never delivered. The **recommendation survives, its basis does not**: leave
> `sources: [arxiv]` as the default not because S2 hurt, but because S2 was never tested.
> See *S2 resolved* below.

A clean A/B — same day, identical flags (`--rr-triage --rr-hybrid`), only `--sources` differs:

| | arXiv | arXiv + Semantic Scholar |
|---|---|---|
| mean net@2 | **+0.83** | **+0.58** |
| precision (non-abstained) | **0.91** | **0.76** |
| abstained | 7/12 | 6/12 |

Only two cases moved and **both got worse** (`rl` −2.0, `diffusion` −1.0); **none improved**. The
mean shift (−0.25) is inside the noise floor, but `rl`'s −2.0 is not, and the precision drop is
consistent with it: S2 papers carry no arXiv categories, and the new
*absent-category-is-not-a-zero* ranker rule (correctly) stops penalising them — which makes them
more competitive and puts more weight on the triage gate to reject the non-actionable ones. Here
one got through. **Recommendation: leave `sources: [arxiv]` as the default for ML repos.**

### 4. A benchmark-integrity bug: `--sources arxiv,dblp` was silently a no-op

The third run was intended to measure the Feature 10 DBLP adapter. It measured nothing:
`harness.collect_live_papers` only implemented `openalex` and `semantic_scholar` branches, so
`dblp` was **silently dropped** and the run was arXiv-only — a result that *looked* like a valid
DBLP evaluation. Fixed: `dblp`/`biorxiv` branches added, plus a `ValueError` on any unrecognised
source so a typo or unsupported adapter can never again masquerade as a measurement.

### DBLP: still unmeasured after four attempts — and now we know why (2026-07-29)

> **Correction (2026-08-12): there was a fifth blocker, upstream of all four.** Every query
> DBLP ever received was arXiv boolean syntax (C-9), for which it returns zero — measured, at
> all-time lookback so the year filter below is not involved. The heading's "now we know why"
> was premature. Attempt 4's year-granularity reasoning is still correct *about the adapter*,
> but it was diagnosed by reading the code, never by watching DBLP answer a well-formed query,
> and it was not what made DBLP return nothing. Likewise "`collect_papers` now returns papers
> where it returned 0" at the end of this section cannot have held for `build_queries` output.
> See *C-9 audit* above.

The DBLP arm of this comparison has never produced a number. Each attempt hit a *different*
blocker, and all four were real:

| Attempt | Blocker | Status |
|---|---|---|
| 1 | `harness.collect_live_papers` had no `dblp` branch — the flag was silently dropped | fixed (this PR's harness change) |
| 2 | `dblp.org` failed TLS verification (GEANT CA absent from the Windows trust store) | fixed (PR #50) |
| 3 | **Rate limiting** — DBLP dropped connections under the sweep's ~60 queries (47 failures in one run; measured 1/6 rapid requests succeeding) | mitigated: process-wide request throttle + a real User-Agent |
| 4 | **Year granularity** — DBLP exposes only a publication *year*, so a 90-day lookback filters to `year >= 2026` | *structural, not fixable by retrying* |

Attempt 4 is the interesting one, and it is a **design conclusion rather than a bug**: because DBLP
has no publication date, `collect_papers` can only filter `year >= cutoff_year`. With Tier B's
90-day window that keeps only the current calendar year — a thin slice of DBLP — and the effective
window swings with the calendar (the same filter run in January would also admit the prior year).

**So DBLP is structurally mismatched to recency-windowed discovery.** Its value is the non-arXiv
systems/PL/DB literature, which is mostly *not* from the last 90 days. The honest recommendation:
pair `sources: [arxiv, dblp]` with **`rr update --foundational`** (all-time, relevance-first), where
`cutoff_year` stops filtering anything — and don't expect it to move a recency benchmark. Measuring
it properly needs a Tier B variant run in all-time mode.

Arm 1 of the comparison did complete on current `main` (arXiv only, post-tie-break):
**mean net@2 +0.83, precision 0.91, 0 hallucinations, all 12 baselines valid** — per-case identical
to the pre-tie-break run, confirming that the tie-break leak was specific to the *label-ordered*
Tier A/S fixtures and never touched Tier B's live, fetch-ordered pools.

**Original note (superseded above):** DBLP was unmeasured at the time of this run because
`dblp.org` failed TLS verification.
**Correction (2026-07-29):** that was *not* a local CA quirk as first written here — it reproduced on
a second machine. dblp.org chains to GEANT / Hellenic Academic and Research Institutions CA, which
ships in `certifi` but is absent from the Windows system trust store Python's default SSL context
uses, so **the `dblp` source was non-functional for every Windows user**. Fixed by giving the adapter
a certifi-backed context; `collect_papers` now returns papers where it returned 0. The run is
therefore unblocked:

```bash
uv run python evals/run_judge_eval.py --sources arxiv,dblp --rr-triage --rr-hybrid
```

### 5. Rerank + all-time discovery are worth ~+0.67

Falling out of runs 1 and 4, same day, same 12 cases:

| Configuration | mean net@2 |
|---|---|
| `--rr-triage --rr-hybrid` | +0.83 |
| `--rr-triage --rr-rerank --rr-all-time` | **+1.50** |

The listwise rerank over a deeper candidate pool plus all-time (foundational) discovery are
carrying **~+0.67** of the headline — worth more than any source addition tested here, and a
reminder that `rr update --foundational` is not a nicety.

### Features 1, 9 and 11 shipped since — no number above moves, and the benchmark is blind to one of them (2026-07-30)

PRs #53, #54 and #55 added `ranking.w_community` (HF upvotes), `ranking.w_attention`
(Hacker News), the Feature 9 integrity signal, and `rr eval`. Checked rather than
assumed, so the next audit does not have to redo it:

| Change | Effect on the numbers above |
|---|---|
| `ranking.w_community` | **None.** Defaults to `0.0` and no runner sets it. |
| `ranking.w_attention` | **None.** Defaults to `0.0` and no runner sets it. |
| `score_recency(now=)` | **None.** Optional; the harness does not pass it, so behaviour is unchanged. |
| `signals.integrity` | **None on these numbers — but the default ranking did change.** See below. |

The integrity signal is the one worth writing down. `signals.integrity` defaults to
**True**, so a real `rr update` now applies a hard multiplicative `withdrawn_penalty`
(0.1) and `categorize_papers` pulls withdrawn papers out of the digest tiers. The
recorded numbers do not move because the benchmark cannot see any of that:
`harness.py` calls `rank_papers` without `withdrawn=`, computes tiers from
`score_total` directly, and none of the 185 fixture papers is withdrawn.

**So the shipped default ranking changed while the benchmark stayed structurally blind
to it.** That is not a stale number, it is a coverage gap: measuring the integrity
signal needs a harness change plus a fixture containing a withdrawn paper, not a re-run.
Recorded here so nobody reads the unchanged numbers as evidence the change was neutral.

`rr eval` (Feature 11) is the in-CLI counterpart to this benchmark and shares its
metrics via `src/reporadar/metrics.py`. It scores against a user's own ratings rather
than fixtures, so it produces no number for this file.

### 6. What still cannot be measured

Three shipped features are **structurally unmeasurable** by this harness: **SPECTER2** (F7),
**citation proximity** (F8) and **learned recommendations** (F5) all key off the user's
starred/highly-rated papers, and the harness ranks a pool directly with `rank_papers` — it never
builds a store, so there are no stars or ratings and those components never fire. Measuring them
needs seeded preferences; the clean design (no leakage) is to star a few of the Tier A *labeled*
fixtures per case and score the lift on the held-out relevant ones.

Until then, the honest statement is that F5/F7/F8 are **tested** (unit + integration) but **not
benchmarked**.

**Update (same day):** this gap is now closed for **Feature 7** — `run_seeded_eval.py` implements
exactly that design and SPECTER2 measures **+0.147 mean nDCG@10** (see
[Tier S](#tier-s--specter2-measured-for-the-first-time-2026-07-29) above). Feature 8 remains
unmeasured (fixtures aren't citation-linked) and Feature 5 needs the Tier B judge rather than a
labeled pool.

## 12-case benchmark — RepoRadar is net-positive and competitive with Opus (2026-07-12)

The expanded 12-case set (`--baseline cli --rr-rerank --rr-all-time`, triage = claude-haiku-4-5,
`min_actionable=2`). 11/12 cli baselines ran; `crypto`'s `claude -p` exited 1 (empty stderr, the
CLI flakiness — *not* a safety refusal). Re-ran it with `--baseline api`: Opus completed cleanly
($1.21, one genuinely-actionable recommendation), confirming it was a CLI-mode issue, not a
cybersecurity guardrail declining the crypto topic. Crypto's baseline below is from that api run.

**RepoRadar Top Picks (min>=2) vs Opus baseline, all 12 cases:**

| Case | bucket | RepoRadar Top Picks | Opus baseline | winner |
|---|---|---|---|---|
| **rag** | ML | 3 · 2 act · net 0.0 | 3/3 · net **+3** | baseline |
| **cv** | ML | 4 · 3 act · net +1.0 | 3/3 · net **+3** | baseline |
| **rl** | ML | abstained · net 0.0 | 3/3 · net **+3** | baseline |
| **peft** | ML | 10 · 8 act · net **+4.0** | 2/2 · net +2 | **RepoRadar** |
| **diffusion** | ML | 7 · 7 act · net **+7.0** · prec 1.00 | 2/2 · net +2 | **RepoRadar** |
| **graph** | ML | 2 · 0 act · net −4.0 | 3/3 · net **+3** | baseline |
| **speech** | ML | 7 · 7 act · net **+7.0** · prec 1.00 | 3/3 · net +3 | **RepoRadar** |
| **crypto** | adjacent | 3 · 3 act · net +3.0 · prec 1.00 | 1/1 · net +1 · api | ≈tie † |
| **systems** | adjacent | 1 · 1 act · net +1.0 | 1/1 · net +1 | tie |
| **webdev** | control | 1 · 0 act · net −2.0 | abstained · net 0 | baseline |
| **cli** | control | abstained · net 0.0 | abstained · net 0 | tie ✓ |
| **http** | control | abstained · net 0.0 | abstained · net 0 | tie ✓ |
| **mean** | (all 12, w/ baseline) | **+1.42** | **+1.75** | |

† `crypto`'s baseline is from a separate `--baseline api` fill-run (Opus completed — **not** a
guardrail refusal). That run drew RepoRadar at 4 · 3 act · net +1.0, i.e. **a tie**; the full run's
+3.0 was a lucky triage draw (0 duds). Treat crypto as a tie within ±2 net@2 triage noise.

**Findings:**

1. **RepoRadar is net-positive at scale — the opposite of the 4-case story.** Cross-case Top Picks
   mean net@2 = **+1.42** (was −0.25 on 4 cases). It shines on arXiv-rich ML — `peft` +4 (8/10
   actionable), `diffusion` +7 (7/7, precision 1.00), `speech` +7 (7/7, precision 1.00) — and
   **beats the agentic Opus baseline outright on all three** (diffusion +7 vs +2, speech +7 vs +3,
   peft +4 vs +2). On its home turf it returns *more* genuinely-actionable papers than the
   conservative baseline.

2. **The baseline's residual edge is narrow and structural, not a capability gap.** Across all 12
   cases Opus averages **+1.75** vs RepoRadar's **+1.42** — a **0.33 gap**, down from ~+3 on the
   4-case set (and it keeps narrowing as cases are added). It wins on the three "hard" ML cases
   (`rag`/`cv`/`rl`) plus `graph`; RepoRadar wins on `peft`/`diffusion`/`speech` and ties `crypto`/
   `systems` + the controls. Crucially, **every baseline pick is `recent=0/N`** — seminal/foundational
   work RepoRadar's window structurally can't see. The remaining gap is the foundational-corpus scope
   (now shipped as `rr update --foundational`, PR #36), not a ranking bug.

3. **The negative controls pass cleanly.** `cli` and `http` (pure engineering) both **abstain** —
   RepoRadar returns nothing, matching the baseline. Only **2/12** cases leak a false positive
   (`webdev`, `graph`) — the residual gate-precision issue is now a small tail, not the headline.

4. **`min_actionable=2` is decisively the right default.** The 12-case sweep settles the
   gate-threshold question the underpowered 4-case sweep got wrong (it favored `min>=3`):

   | `min_actionable` | mean net@2 | abstained | false-positive | mean precision |
   |---|---|---|---|---|
   | ≥1 | −2.92 | 2/12 | 2/12 | 0.47 |
   | **≥2** | **+1.42** | 3/12 | 2/12 | 0.69 |
   | ≥3 | +0.17 | 10/12 | 0/12 | 1.00 |

   `min>=3` over-abstains on **10/12** — it throws away every ML win (`peft`/`diffusion`/`speech`/
   `crypto` have no score-3 papers, so they all abstain), collapsing the mean to +0.17. At scale
   `min>=2` wins by a wide margin — vindicating the default, the rubric revert, and the choice not
   to bump to 3.

**Bottom line — the arc's conclusion.** Triage + rerank + all-time discovery turned RepoRadar from
"confidently returns 10 non-actionable ML papers" (pre-triage mean net@2 ≈ **−11**) into a tool that
is **net-positive (+1.42), correctly abstains on out-of-domain repos, and beats a strong agentic
Opus baseline on the ML domains it's built for**. What remains is a narrow, well-understood gap: the
baseline still cites foundational work outside RepoRadar's fetch window (a scope change, not a bug),
and the gate leaks on ~2/12 hard cases. The 4-case benchmark said "the baseline dominates"; the
12-case benchmark says "RepoRadar is competitive and wins on its home turf" — which is why the eval
expansion mattered.
