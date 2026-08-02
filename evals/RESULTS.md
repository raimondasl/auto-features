# Tier B benchmark — results

> **Headline (2026-07-31, 12 cases):** RepoRadar Top Picks mean net@2 **+1.75** vs the Opus
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
> **`min_actionable=2` is now contradicted on every case where it has been swept** — three
> sweeps, three times the strictest gate wins *by abstaining*, because triage found 0 of `rag`'s
> 3 actionable papers and 2 of `speech`'s 6. Read the shipped default as unvalidated
> post-fix rather than as a settled result. See
> [Two-case re-benchmark](#two-case-re-benchmark-after-the-quoting-fix-2026-08-02).
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
