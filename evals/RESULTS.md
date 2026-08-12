# Tier B benchmark — results

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
[`RETRIEVAL_DESIGN.md`](../RETRIEVAL_DESIGN.md).

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
