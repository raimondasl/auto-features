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
