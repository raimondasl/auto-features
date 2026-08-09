# RepoRadar — what we tried, what worked, and what didn't

A consolidated record of the retrieval, ranking and gating experiments run between
2026-07-04 and 2026-08-02. It exists so that nobody — including us — pays twice to
rediscover a negative result, and so that the positive ones are stated with the caveats
that were true when they were measured.

> **Sections 1–6 are the record as of 2026-08-02 and are left as written.** The campaign
> continued for six more days (ROADMAP P1–P9, then the score-2 band work) and **overturned
> two of this document's load-bearing conclusions**: that the gate is at its ceiling, and
> that search is the binding constraint. §7 and §8 have been rewritten to the current state;
> where an earlier section contradicts them, §7 wins. The full later record is
> [`evals/RESULTS.md`](evals/RESULTS.md) and [`paper/DRAFT.md`](paper/DRAFT.md).

**Every number here is reproducible.** The commands are given inline; raw per-run detail
lives in [`evals/RESULTS.md`](evals/RESULTS.md), which is chronological and much longer.
This document is organised by *problem* instead.

Every numeric claim below was fact-checked against the repository by an independent pass
(252 claims; 21 confirmed wrong and corrected before publication, including two Sonnet
claims that were supported by nothing in the repo and one section whose sign was
inverted). The corrections are in the git history of this file.

**Provenance labels** used throughout:

| label | meaning |
|---|---|
| **MEASURED** | run against real data in this repo, numbers below are its output |
| **CORRECTED** | measured, published, then found wrong and re-measured — both states shown |
| **UNTESTED** | plausible, never run. No number attached. |

---

## 1. The problem, and why it is hard

RepoRadar reads a code repository and surfaces arXiv papers that could *improve* it. The
hard part is not finding papers on the same topic. It is that:

> **A codebase's vocabulary describes what it has. A useful paper describes what it should
> adopt. These are disjoint by construction.**

This single sentence explains most of the negative results below. It was arrived at
empirically, from three independent failures (§3.0, §3.1, §3.2), before it was stated as a
principle.

The system is a four-stage pipeline, and the experiments group by stage:

```
   profile the repo  ──▶  SEARCH  ──▶  RANK/POOL  ──▶  GATE (triage)  ──▶  digest
                          §3            §4              §5
```

---

## 2. How everything is measured

Without this section the numbers below are uninterpretable.

### 2.1 The metric: net@2

```
net@2  =  (# actionable papers returned)  −  2 × (# non-actionable papers returned)
```

Errors are **asymmetric on purpose**: shipping a junk paper wastes a reader's attention and
costs 2; missing a good one costs 1. Abstaining scores 0, so a system that returns nothing
beats one that returns noise.

**Accuracy is the wrong objective here and actively misled us once.** A prompt variant
measured on accuracy looked like a null (21 fixed, 17 broke, p = 0.63); the same two runs
scored **+16 net@2**, because most of its flips were false positives becoming correct
abstentions — worth +2 each. See §6.2.

### 2.2 The two instruments

| | Tier B | labelled-set diagnostic |
|---|---|---|
| what it is | end-to-end run: RepoRadar vs an Opus 4.8 baseline, both judged blind by GPT-5.5 | re-score cached judge verdicts with a component under test |
| unit | 12 repositories | 602+ individual papers |
| cost | **~$11** | **~$0.10** |
| resolution | paired CI over cases ≈ **±2 net@2** | paired CI over papers ≈ **±20 net@2** on a base of ~+80 |

**The single most useful methodological lesson in this document:** at n=12 cases Tier B
cannot separate two prompt variants. `prose 300` vs `tagline` — a gap the 602-paper
instrument puts at +10 — comes back **+0.33, 95% CI [−0.42, +1.50], P = 0.330**. It can
still register a *large* prompt change in aggregate: adding 300 characters of README, worth
+22 on the labelled set, moved Tier B **+1.00, 95% CI [+0.00, +2.00], P(Δ≤0) = 0.032**
(§5.6) — though that run widened the candidate window at the same time and the interval
grazes zero, so **no prompt-only arm has ever been cleanly resolved at Tier B**. Prompt and
gating questions must therefore be decided on the labelled set at $0.10 a run, with Tier B
used only to confirm direction. We spent roughly $35 learning this.

The labelled set is mostly a by-product: every judge verdict the benchmark has ever paid for
is cached, so it grows as runs accumulate — **428 → 576 → 602** papers across 12 repos at a
28–32% actionable base rate. One step was not free: 428 → 576 came from the rank-stratified
ranker diagnostic (§4.1), **~$5** of GPT-5.5 bought specifically for labels outside the
top-10 that the benchmark would never produce on its own.

### 2.3 The 12 cases

`rag` (ColBERT), `cv` (detectron2), `rl` (stable-baselines3), `peft`, `diffusion`
(diffusers), `graph` (pytorch_geometric), `speech` (whisper), `crypto` (pyca/cryptography),
`systems` (redis), `cli` (click), `http` (requests), `webdev` (flask).

The last three (`cli`, `http`, `webdev`) are **negative controls** — flagged
`negative_control: true` in `benchmark.yaml`, repos where the correct answer is usually
"abstain". On the 602-paper labelled set `cli` has 0 actionable papers in 39 judged and
`http` has 1 in 46, so a system that returns anything there is almost always wrong. (§5.1's
per-case table predates that growth and reports the pair as 0 in 53; triage's zero false
positives on them holds at 602 too.) `crypto` and `systems` are a separate
"research-adjacent but not on arXiv" pair — **not** controls: `systems` runs at a 31–42%
actionable base rate.

**A limitation that applies to everything below:** all 12 are popular open-source projects
with well-maintained READMEs. RepoRadar's actual target — a private codebase with thin docs
— is not represented, and at least one result (§5.3) is suspected to be an artifact of this.

---

## 3. Paper search — reaching the right papers at all

### 3.0 The measurement that started it: 2,030 papers fetched, 0 useful

**MEASURED** · `uv run python evals/diagnose_pool.py` (free, keyless)

Take every paper the Opus baseline recommended *and* the judge scored ≥2. That is **24
papers across 9 repos**. Ask whether RepoRadar's own queries ever fetched them.

| | |
|---|---|
| papers RepoRadar fetched across those 9 repos | **2,030** |
| known-good papers among them | **0** |

Not a near miss — a disjoint set. The targets are canonical and on-topic: ConvNeXt,
Prioritized Experience Replay, Double Q-learning, Soft-NMS, WhisperX, Distil-Whisper,
Speculative Decoding, Exphormer.

Every obvious excuse was checked and none survived:

- **Not id matching.** Version-stripping (`1907.04378v1` → `1907.04378`) matches correctly.
- **Not the category filter.** **23 of 24 are inside the categories being searched.**
- **Not an arXiv limitation.** `all:"prioritized experience replay"` returns the target at
  **rank 1**. `all:"ConvNet for the 2020s"` returns ConvNeXt at **rank 1**. The papers are
  one good query away.

**Mechanism.** TF-IDF over a README produces generic single terms, so RepoRadar sent
`all:model`, `all:image`, `all:torch` — each matching tens of thousands of papers — and kept
the first 50 by arXiv's *lexical* relevance, which carries no impact weighting. It was
sampling near-randomly from an enormous match set, eight times per repo.

### 3.1 Negative — a repo's own bibliography (0/24)

**MEASURED.** Harvesting arXiv ids from README, `docs/`, `.bib` and `CITATION.cff` looks
like a free, offline, keyless corpus. **0 of the 24 targets appear anywhere in the repos'
own documentation** — including `graph`, whose README lists **112 arXiv papers**.

*Why:* a codebase cites what it implements. A project's bibliography is a well-targeted
index of things it already does.

### 3.2 Negative — LLM-generated search phrases, two prompts (2/24 and 0/24)

**MEASURED** · `evals/diagnose_query_generation.py --prompt {uses,lacks}` (~$0.01 each)

A dependency review predicted this would recover 19/24. That estimate was made by people who
already knew the answers.

| prompt | recovered | phrases matching **zero** papers |
|---|---|---|
| TF-IDF keywords (control) | 0/24 | — |
| *"name the techniques this repo uses"* | **2/24 (8%)** | 19/54 |
| *"name what this repo lacks"* | **0/24 (0%)** | **45/54 (83%)** |

**They fail for opposite reasons, and that is the finding.**

*Asked what the repo uses*, the model was accurate and useless. For detectron2 it produced
"Mask R-CNN with feature pyramid networks", "panoptic segmentation", "Cascade R-CNN" — a
correct description of what detectron2 *implements*. The targets were Soft-NMS, Copy-Paste
augmentation and ConvNeXt, which it does not.

*Asked what the repo lacks*, the model **aimed correctly** — quantization, knowledge
distillation, ViT backbones, cross-encoder reranking — and then phrased them as descriptive
compounds no paper title contains. 83% of its phrases matched nothing.

**The sharpest single comparison, `rl`:**

| prompt | phrase emitted | hits |
|---|---|---|
| "lacks" | `experience replay prioritization methods` | **0** |
| "uses" | `prioritized experience replay` | **found the target** |

Same concept. One is the literature's term of art; the other is a description of it.

**This is not a prompt-tuning problem.** The failing prompt already specified "2–5 words",
"real terms of art only", "favour method names over topic names", and explicitly forbade
naming techniques the repo already implements. Any further attempt needs a *mechanism* that
closes the vocabulary gap — validating emitted phrases against the index and discarding
zero-hit ones, or matching in embedding space — not another rewording.

### 3.3 Negative — citation-count-sorted retrieval (1/24)

**MEASURED.** Sorting results by citation count sounds like the answer to "arXiv has no
impact signal". As a drop-in replacement it recovers **1 of 24**. It improves the *ordering
of a match set*; it cannot repair a match set that never contained the paper. Worth having
**after** query quality is fixed, not instead of it.

It also hard-errors on RepoRadar's modal queries: the Semantic Scholar bulk endpoint returns
HTTP 400 above 10M hits, and `model` (24.4M) and `data` (24.8M) both exceed it.

### 3.4 Negative — fetching deeper (≤3/24, and it got the machine IP-blocked)

**MEASURED.** Raising `max_results_per_query` recovers **at most 3 of 24** into the pool,
and **9 of the 24 are outside the match set at any depth** — all three `rl` targets are
absent even from a 265,785-result query.

The cost is not theoretical. Sustained polling at ToU-compliant 3.2 s spacing earned this
project's machine an **IP-level block that survived 30 minutes of silence and lasted roughly
70 minutes**. Request *rate* is the lever, not page size.

### 3.5 **Positive** — a citation hop from the repo's bibliography (18/24)

**MEASURED (CORRECTED)** · `uv run python evals/diagnose_citation_hop.py` (free, keyless)

§3.1 established that a repo never cites what would improve it. That is true, and those
papers are **one hop away in the citation graph**. Seeds are the arXiv ids the repo itself
cites — the only seed set a cold-start repo has. One hop in each direction, seeds capped
at 60.

| case | seeds | candidates | recovered |
|---|---|---|---|
| rag | 7 | 1,856 | **5/5** |
| cv | 18 | 7,378 | **3/3** |
| rl | 30 | 29,430 | **3/3** |
| peft | 18 | 6,342 | **2/2** |
| graph | 121 | 36,868 | **3/3** |
| speech | 2 | 6,068 | 2/3 |
| diffusion | 10 | 4,072 | 0/2 |
| crypto | 0 | — | 0/2 (no arXiv-indexed bibliography) |
| systems | 0 | — | 0/1 (same) |
| **total** | | **92,014** | **18/24 (75%)** |

Against 0/24, 2/24 and 1/24 for everything else, this is **the only approach that reaches
the papers at all**. The targets are genuinely discovered: seeds are subtracted from the
candidate set, and none of the 24 appear in any repo's documentation.

Both hop directions earn their place — forward (papers *citing* a seed) does most of the
work, but backward uniquely contributes several targets.

> **CORRECTION — this measured 14/24 first, and the error inverted a conclusion.**
> Semantic Scholar's `/paper/batch` truncates **nested** items at 9,999 *per request*,
> filled greedily in id order. 18 seeds in one request returned `[9999, 0, 0, …]` — one
> seed's citations and seventeen empty arrays, HTTP 200, no error.
>
> Two published conclusions were withdrawn:
> - *"`graph` scored 0/3 because its seeds are irrelevant"* — no: it scored 0/3 **because it
>   has the most seeds (121)**; one hub consumed the budget. Corrected: **3/3**.
> - *"Seed count does not predict recall"* was **backwards** — under the bug, more seeds
>   actively destroyed recall.
>
> **The same defect was in shipping code**: `citations.fetch_references` chunked at the
> 500-id limit and was silently losing ~29% of `w_citation_proximity`'s edges.

**The caveat is as large as the result.** 92,014 candidates for 18 papers is a density of
**1 in 5,111** — *worse* than the truncated run appeared to show, because the recovered pool
is 3.2× larger while recall rose by less than a third. **Recall is transformed; precision is
untouched.** It changes the shape of the problem rather than solving it: the pool now
contains the answers, so selection becomes tractable instead of moot. The filter that would
make ~10,000 papers/repo usable is unbuilt.

**A hard wall:** 13 seeds across `rl` and `graph` saturate the 9,999 cap even when requested
alone, and the API enforces `offset + limit < 10000`. A paper with tens of thousands of
citers cannot be fully enumerated by anyone, keyed or keyless.

### 3.6 Fixed defects that were suppressing search quality

| defect | impact | status |
|---|---|---|
| **Unquoted phrases were OR, not AND.** arXiv treats a space after a field prefix as OR: `all:speech recognition` matches **246,802**; `all:"speech recognition"` matches **6,845**. | every multi-word query was an OR union rather than a phrase — **36× too broad on the one query measured**, though across the 12 repos **93% of queries were unchanged**; only `rag` and `speech` moved | fixed, PR #62 |
| **`setup.py` dependencies never parsed**, and `stop_words="english"` let README furniture through — the benchmark was literally sending `all:license` and `all:https`. | **only 32 of 96 queries survived** the fix | fixed, PR #59 |
| **`profiler_cfg` omitted at 4 call sites** (`rr workspace update`, its scoring half, `rr watch`, and `rr audit`). Omitting it forces `scan_source` to its `False` default, so those paths built **docs-only** queries where source-derived was configured — and `rr audit` reported that docs-only set as what `update` sends | `scan_source: true` silently a no-op; audit understated | fixed, PRs #57, #58 |

The `diffusion` case — three of whose eight queries were `license`/`https` boilerplate —
went **+6.0 → +10.0 net@2** after the query fix, with all 12 judged pool papers actionable.

### 3.7 A precondition that defeats every option above

**All 24 targets are ≥11 months old**, and `collector.py` discards anything older than
`lookback_days` (default **14**). Under a default `rr update`, every approach in this
section scores 0/24 on merit alone. Retrieval work must be developed in the `--foundational`
path, which sets relevance sort and a 100-year window.

---

## 4. Pool selection and ranking

### 4.1 The ranker discriminates coarsely, not finely

**MEASURED** · `uv run python evals/diagnose_ranker.py --per-stratum 4` (~$5)

Every label the benchmark owned came from the ranker's own top-10 plus the baseline's
picks, so the ranker had never been scored. This judges a **rank-stratified sample** of the real pool.

| rank band | n | actionable |
|---|---|---|
| 1–10 | 48 | **31%** |
| 11–50 | 48 | **33%** |
| 51–150 | 48 | 15% |
| 151+ | 44 | 7% |

**The ordering is real** — 31–33% in the top 50 against 7% past rank 150 is a large
separation. The heuristic weights do sort relevant from irrelevant.

**But ranks 1–10 and 11–50 are indistinguishable.** The ranker cannot tell which of its top
50 are the best 10, so the top-10 cut is arbitrary — discarding roughly **13 actionable
papers to keep 3** per case.

This also bounds the retrieval work: a ranker that cannot order within its own top 50 will
not order the citation hop's 92,014.

### 4.2 Negative — widening the gate's window 20 → 50

**MEASURED (falsified a prediction made in advance)**

§4.1 produced an explicit prediction: at depth 50 the gate should see ~16.5 actionable
papers per case instead of ~3.1, worth **+8.33 net@2**. Tested on a full 12-case run:

| | window 20 | window 50 |
|---|---|---|
| mean net@2 | +1.75 | +2.42 |
| papers returned (12 cases) | 48 | 47 |
| actionable returned | 39 | **41** |

**+0.67, 95% CI [−0.50, +2.00], P(Δ≤0) = 0.153.** Half the gain came from one case. **4× the
candidates bought two more actionable papers across twelve repos.**

Verified the experiment actually ran: all 12 cases had a full 50-candidate window.

**Two variables moved at once.** The run passed `--rr-readme-context` alongside
`--rr-pool 50`, so the arms differ in repo context as well as window width — and per §6.3
that flag sent a 23–230 character packaging tagline while dropping the keyword block, not
the README it claimed. +0.67 is a *combined* figure; a window-50 arm with matched context
was never run at Tier B. Since the delta is indistinguishable from zero either way, the
decomposition was judged not worth $11.

**Why the extrapolation was wrong — it measured digest *size*, not selection *quality*.**
The +8.33 estimate scaled *admitted* papers to a 50-paper window and counted every admitted
paper as returned — 9.9 returns/case. The shipped system cuts the digest at 10 regardless,
and the real run returned 3.9/case. It answered *"what if we return 4× more papers"*, not
*"what if we choose 10 from a wider pool"*. A flat actionable rate across ranks 1–50 means
the deeper band is no **worse**; it never implied selecting from it would be **better**.

### 4.3 Mixed — hybrid BM25 + RRF fusion

**MEASURED.** Fusing the heuristic ranking with a BM25 lexical ranking via reciprocal rank
fusion, aimed at papers buried by vocabulary mismatch:

| case | metric | without | with hybrid |
|---|---|---|---|
| rag | Top-10 | 4 act · net −8 | **6 act · net −2** |
| webdev (control) | Top Picks | 1 · 0 act · net −2 (leak) | **abstains · net 0** |
| mean (4 cases) | Top Picks | −0.25 | **−0.5** |

It helps the diagnostic Top-10 and closes a negative-control leak, but does not improve the
returned set. Shipped as opt-in (`ranking.hybrid`), not defaulted.

### 4.4 **Positive** — SPECTER2 and citation proximity both help

**MEASURED** · `evals/run_seeded_eval.py`. 5 seeds drawn round-robin across the fixture's
strata, removed from the candidate pool, each component scored on how well it ranks the
*held-out* gold:

| case | baseline nDCG@10 | SPECTER2 `w=0.25` | SPECTER2 `w≥0.5` | proximity `w≥0.5` |
|---|---|---|---|---|
| rag | **1.000** (ceiling) | +0.000 | +0.000 | inactive |
| cv | 0.915 | **+0.085** | +0.085 | inactive |
| rl | 0.599 | +0.000 | **+0.401** | **+0.128** |

**Report the sweep, never a point.** SPECTER2's response is step-shaped — mean ΔnDCG@10 is
**+0.028 at `w=0.25`** and **+0.162 at `w≥0.5`**, saturating after — so a single weight
would have been an arbitrary pick on a plateau.

**Read the mean as two measurements, not three.** `rag` sits at the 1.000 ceiling, so its
delta is bounded at ≤ 0 by construction; of the remaining two, **`rl` supplies 82%** of the
mean.

**Citation proximity was called unmeasurable, and that was a seed-selection bug.** With
seeds taken as a fixture-order prefix, no candidate cited a seed and the component looked
structurally untestable. Stratifying the seeds pulled in *Soft Actor-Critic* — heavily cited
— and it immediately scored **+0.128 nDCG@10** on `rl`.

Coverage is 3 of 12 cases (only those have Tier A fixtures), and Tier A draws distractors
from clearly different fields — so this measures **topical discrimination, not
actionability**. A real positive result on a narrow base.

> **A ranker bug found while building this changed every prior number.** `rank_papers`
> sorted stably on `score_total` alone, and the fixture builder writes all gold papers before
> all distractors — so **every score tie silently resolved in gold's favour, making every
> Tier A/S baseline a best-case ordering** (`cv` nDCG@10: 0.931 in fixture order vs 0.842
> reversed). Now tie-broken on `arxiv_id`. This was a **shipping-code** bug: production
> ranking was fetch-order-dependent.

---

## 5. Triage gating

The gate scores each candidate 0–3 for whether it could genuinely improve *this* repo, and
admits ≥2. Instructions live in [`src/reporadar/triage.py`](src/reporadar/triage.py)
(`_RUBRIC`).

### 5.1 The gate works — and a wrong claim that survived three sessions

**MEASURED (CORRECTED)** · `uv run python evals/diagnose_triage.py` (~$0.10)

> **CORRECTION.** RESULTS.md previously said triage "carried no discriminative signal" and
> was "at chance". That came from the ~10 papers a single Tier B case happened to surface,
> was repeated across several sections, and is **wrong**.

| | precision | recall | base rate |
|---|---|---|---|
| all 428 labelled papers | **0.81** | 0.78 | 32% |
| excluding the 27 Opus picks | **0.78** | 0.75 | 28% |

A gate with no signal scores precision equal to the base rate. Triage scores **+0.50 above
it**.

**Per case the failure modes are opposite, which a pooled number hides:**

| case | base | precision | recall | reading |
|---|---|---|---|---|
| diffusion | 77% | 1.00 | 0.91 | excellent |
| crypto | 22% | **1.00** | **0.29** | **too strict** — never wrong, misses 5 of 7 |
| graph | 26% | 0.75 | **0.33** | too strict |
| rl | 19% | **0.50** | 0.70 | **too loose** |
| cli, http | 0% | — | — | correct total abstention: 53 papers, **0 false positives** |

**The narrower, sharper survivor of the withdrawn claim:** triage collapses *specifically on
the ranker's top-10* — 0.33 precision there (n=20) against 0.82 elsewhere (n=63) — and that
is the only subset Tier B ever judges. That is how n=10 produced "at chance". **This one is
n=20 across two cases, so it is suggestive and not established**: the harness did not record
returned ids until 2026-07-31.

### 5.2 Negative — a more capable model (Sonnet) is not the lever

**MEASURED**, 428 labelled papers, paired. Sonnet gets **byte-identical** instructions and
the same `keywords` repo context; only the model id differs.

| gate | precision | recall | **net@2** |
|---|---|---|---|
| Haiku 4.5 (shipped) | 0.81 | **0.78** | **+57** |
| Sonnet 5 | **0.92** | 0.52 | +55 |

Sonnet is not worse — it is a **different operating point**, far more conservative, and on
net@2 indistinguishable from the shipped gate: paired delta **+4, 95% CI [−16, +25],
P(Δ≤0) = 0.37** (it also failed 6 of 428 calls). It returned **73** papers to Haiku's 123.

The test was posed as a fork — *if a stronger model rescues the ranker's top-10, the gate is
capability-limited* — and it answered neither way: on that top-10 Sonnet **abstained
entirely** rather than discriminating. This is the second time a stronger model has been
rejected here; Feature 6 found `claude-sonnet-5` metric-identical in 2026-07.

> **Per-case detail for this arm is not recoverable.** `evals/diagnose_triage.py` names its
> output by repo-context and ignores `--model`, so the Sonnet run was written to
> `diag_triage_keywords.json` and later overwritten by the 602-paper Haiku run. Only the
> aggregates survive; any per-repo Sonnet claim needs a re-run. **A previous draft of this
> document asserted "on `crypto` Sonnet missed all 7" and that its rejections rested on
> false factual claims about the repo — neither is supported by anything in this
> repository, and both are withdrawn.**

### 5.3 What repo context to give the gate — eight arms, one ceiling

**MEASURED**, 602 labelled papers, all paired.

Starting observation: the gate saw **366 characters** of repo description on average, while
the judge defining its labels reads **6,375** — a **17× asymmetry**. ColBERT's keyword
profile says *"web APIs"* (it depends on flask) and never says retrieval.

| variant | description chars | precision | recall | **net@2** |
|---|---|---|---|---|
| keywords only (old default) | 270 | 0.83 | 0.74 | +73 |
| packaging tagline (libraries + one-liner; **drops** domains/topics) | ~380 | 0.88 | 0.70 | +85 |
| **+ first 300 chars of README** | 300 | 0.92 | 0.68 | **+95** |
| + LLM-selected **verbatim** sentences | ~600 | 0.92 | 0.67 | +91 |
| + first 2,000 chars | 1,828 | 0.89 | 0.68 | +86 |
| + first 6,000 chars | 4,161 | 0.90 | 0.68 | +89 |
| + LLM **paraphrase** (purpose+capabilities) | 969 | 0.89 | 0.61 | +76 |
| + LLM paraphrase incl. improvement areas | 1,894 | 0.91 | **0.52** | **+70** |

**Finding 1 — verbatim beats paraphrase by +21, and paraphrase loses to sending nothing.**
The verbatim and paraphrase arms differ *only* in whether the selected content is quoted or
rewritten. The full paraphrase (+70) scores **below the no-description control** (+73).
Three competing explanations were tested and dropped first:

| hypothesis | test | outcome |
|---|---|---|
| the gap list reads as an exhaustive whitelist | relabel it "NOT exhaustive" | **dead** — recall unchanged at 0.52 |
| …so remove the gaps entirely | drop the section | partial — +8, P = 0.152; still −19 vs prefix |
| the description is simply too long | compare at matched length | **dead** — paraphrase at 1,894 ch is 16 *below* raw text at 1,828 ch |
| paraphrase discards the paper-matching vocabulary | send verbatim sentences | **supported** — recovers the whole gap |

A README states techniques in the words papers use — "contextual late interaction",
"`nn.MessagePassing`". Rewriting them removes the signal the gate matches on. **This is a
general warning for feeding model-written descriptions to another model.**

**Finding 2 — semantic selection does not beat positional selection.** Verbatim-selected
sentences vs a plain 300-char prefix: **−4, 95% CI [−16, +8], P = 0.778**.

> **A failed prediction, recorded because it was made in advance.** The argument for
> extraction was `graph`, whose first 300 characters are link badges while extraction
> correctly pulls *"All Graph Neural Network layers are implemented via the
> `nn.MessagePassing` interface"* from deep in the README. On that exact case extraction
> scored **+0 against the prefix's +2**. The mechanism was real, visible in the text, and
> produced no gain.

**The ceiling.** Every arm supplying *any* purpose statement lands between +85 and +95, and
none is distinguishable from another. Four extraction strategies converge. The limit appears
to be **what the documents contain**, not how they are read.

> **CORRECTION — "more prose is worse" was over-read.** An earlier version of this claimed
> the curve turns over and explained it as "300 chars is where a README stops describing the
> project". Both halves were wrong. 300 vs 2,000 is **+9 at P = 0.108**; 300 vs 6,000 is +6
> at P = 0.193 — 300 is the **argmax of noisy arms**. And chars 300–2,000 actually contain
> ColBERT's late-interaction explanation and detectron2's capability list — the *most*
> paper-relevant text in the file. Only "some purpose statement beats none" is supported;
> the amount is unresolved.

**Shipped:** `profiler.prose_chars = 300` — best-measured, no per-repo LLM call, least
disclosure. Not a demonstrated optimum.

### 5.4 Negative — rewriting the rubric to encode known failure modes

**MEASURED.** The rubric was rewritten to name the exact failure modes that fool a lenient
"2" (measurement-not-method, wrong-layer, application-level, wholesale-replacement) plus a
grounding test: to score ≥2, name the concrete component of *this* repo the method changes.

Outcome: the target leak was unchanged, and **`cv` lost 2 genuine actionable papers**.
Making the rubric stricter traded recall for precision the metric did not want.

### 5.5 The gate threshold is contested, not settled

**MEASURED**, and the two runs disagree:

| sweep | `min≥1` | `min≥2` | `min≥3` |
|---|---|---|---|
| 4-case (2026-07-12) | −8.75 | −0.25 | **+0.25** |
| 12-case (2026-08-02) | −2.17 | **+2.75** | +0.33 |

At 4 cases the strictest gate won *by abstaining*; at 12 cases `min≥2` wins cleanly (0 false
positives, mean precision 0.92). The 12-case evidence is stronger, but three earlier
two-case sweeps went the other way. Shipped default is `2`.

### 5.6 Where the gate ended up

| | before | after |
|---|---|---|
| mean net@2, 12 cases | +1.75 | **+2.75** |
| papers returned | 48 | 48 |
| actionable | 39 | **43** |
| **junk in the digest** | **9** | **5** |
| precision at `min≥2` | 0.85 | **0.92** |

**+1.00, 95% CI [+0.00, +2.00], P(Δ≤0) = 0.032**; 7 cases improved, 1 worsened. At an
unchanged digest size, non-actionable papers fell from 9 to 5 across 12 repos.

**Against the Opus baseline this is parity, not a win:** +2.75 vs +1.83, but paired over
cases **+0.92, 95% CI [−0.67, +2.75], P = 0.148** — 5 wins, 3 losses, 4 ties.

---

## 6. Cross-cutting lessons

These cost more than any single experiment and generalise beyond this project.

### 6.1 Small-n conclusions are worse than no conclusions

"Triage is at chance" came from **n=10** and was repeated across three sessions and several
document sections. At n=428 it is precision 0.81 against a 32% base rate. The cost of
checking was **$0.10**.

### 6.2 Measure on the metric the product is scored on

The README variant was published as a null on **accuracy** (p = 0.63) and was **+16 net@2**.
Accuracy treats a false positive and a false negative as equally bad; the product does not.

### 6.3 A harness that reimplements the thing under test measures the harness

Both the eval harness and the diagnostic rebuilt the triage prompt instead of calling
`build_triage_prompt`. Consequence: a result was published as "README context" when it
actually sent `_collect_text_corpus(repo)[0]` — the **packaging one-liner on 11 of 12
repos**, 23–230 characters ("Python HTTP for Humans."). It also silently dropped the keyword
block, so the "more context" arm was *less* context on 9 of 12 cases. Both consumers now
call the shipped code path.

### 6.4 Read the raw numbers, never a script's auto-verdict

Two scratch scripts printed conclusions inverted from their own data — an OR-vs-AND
threshold that printed "AND" when the numbers said OR, and a bug check that printed "NOT
confirmed" at 1.4× when the threshold was 1.5×. Both were caught only by reading the
numbers.

### 6.5 Silent API truncation looks exactly like a real result

S2's `/paper/batch` returned HTTP 200 with `[9999, 0, 0, …]` — one seed's data and
seventeen empty arrays. It produced a plausible 14/24 and a confidently-stated,
**backwards** conclusion about seed count. Detect-and-split now guards it.

### 6.6 Argmax of noisy arms is not a curve

Four budget arms, differences at P = 0.108 and P = 0.193, and the winner was shipped with a
mechanism story attached. Selecting the best of k arms inflates significance; report the
Bonferroni-adjusted figure and do not narrate a shape the intervals do not support.

### 6.7 Record predictions before testing them

Two predictions made in advance were falsified by their own test cases (§4.2 depth-50,
§5.3 `graph`). Both are more informative than the results that confirmed expectations,
and neither would have been legible if the prediction hadn't been written down first.

---

## 7. Where things stand (rewritten 2026-08-09)

| stage | status |
|---|---|
| **Search** | **Two channels reach the papers; neither ships yet.** Default keyword retrieval still reaches 0/24 — that never improved and is not expected to. The citation hop reaches **21/48 = 44%** across the expanded benchmark (18/24 was a favourable subset; seeds ≥7 → 89%, no bibliography → 0%). HyDE against a 3.1M-vector dense index reaches **27/48**, of which **15 are unreachable by the hop**; **union 36/48 = 75%**. Both live in `evals/` only. |
| **Pool quality** | **Far better than every recall number implied.** The HyDE top-100 band is 58% actionable against a 2% random-arXiv floor (29× separation); the hop's high-coupling band is 67%. The "1 good paper per 5,111" figures measured distance to a *known gold target* — a top stratum — not to useful papers. |
| **Ranking** | Coarse ordering works and is better than this document said: the heuristic ranker's top-50 is **1.7× denser** than the pool. Fine ordering within a band it cannot do, and that turned out to be the whole problem. |
| **Gating** | **Not at its ceiling — the ceiling was an artifact of scale quantization.** Precision 0.97 / recall 0.60 on wild pools: the gate rejects actionable papers, it does not admit junk. Its 0–3 scores are near-binary, and within the modal band per-repo precision ran 0.00–1.00 invisibly. Re-asking the same question on a 0–9 scale and reading the answer token's *distribution* orders that band at AUC 0.84 and eliminates every net-negative repository. **Shipped** as `reporadar/finescale.py`. |
| **Overall** | **+3.18 mean net@2 against the Opus-4.8 baseline's +1.82** on a live 22-repo run (paired +1.36; 10 win / 6 lose / 6 tie; sign p = 0.45, so ahead on the mean and not established per repo). Digest precision 0.91. Zero net-negative repositories, against the baseline's one. |

**What overturned this document.** Two conclusions above are direct reversals of §5 and §6:

- *"Gating is bounded by repo description, which has hit a ceiling"* — the four extraction
  strategies really do converge (§5.3 stands), but the gate was never description-bound. It
  was **resolution-bound**: the information was present and the 0–3 scale threw it away.
  Three exhausted levers (threshold, bigger model, rubric) all operated on the wrong axis.
- *"Search is the binding constraint"* — true when written, false now. Retrieval has two
  measured channels at 75% union; the binding constraint moved to selection, and then to
  **recall inside the gate**, which is where it sits today: every remaining loss to the
  baseline is a case whose admitted set never contained what the baseline found.

---

## 8. Open directions (rewritten 2026-08-09)

Four of the six directions listed here on 2026-08-02 were subsequently measured. They are
kept with their answers, because a direction that came back negative is worth more than a
blank line.

**1. User-stated improvement goals — HALF ANSWERED, and the answered half is negative.**
Fed to the **gate**, verbatim issue-tracker wants are the worst arm ever measured: −38 net@2,
recall −0.27, damage proportional to how much a repo had to lose (P8). A want-list *replaces*
the question — the gate stops asking "would this improve the project" and starts asking "is
this on the list". The **retrieval** half remains untested and is now the more interesting
one, since stated wants are shaped like queries.

**2. Improvement areas as *search* queries — still untested, and now lower value.** The
mechanism was "search is the binding constraint"; it no longer is. Note also that
`improvement_areas` has now failed twice as gate context (§5.3, and P8's harsher relative),
and gap-phrase matching finished last of four arms as a *pool ranking* signal (P2). Three
independent negatives on the same field argue against a fourth attempt without a new
mechanism.

**3. A filter for the citation-hop pool — ANSWERED, negatively, then made moot.** Coupling
degree cut 70% of the pool at 89% retention on 7 cases and **did not replicate** on 22
(cut → 10%): it was a property of the case set. The larger point is that the filter question
dissolved — P5 showed the pool is dense and the gate's problem is recall, and gating the
*entire* pool end to end is a **wash** (−0.18 paired), because nothing orders what the gate
admits. Filtering was never the missing stage; ordering was.

**4. Validate generated phrases against the index — ANSWERED, negatively, at the mechanism
level.** P2 did exactly this: stemming plus BM25 closed the morphological gap completely
(the canonical failing example now scores 2.79 against the paper it missed and 0.00 against
distractors), and the "lacks" arm *still* ranked targets worse than random. It is not a
phrasing failure but a **different-target failure**: the prompt names a plausible different
research agenda. No retrieval space fixes that.

**5. Shrink the paper side of the triage prompt — deliberately dropped.** `abstract[:1500]`
is still 54% of the prompt and still unmeasured. It was reclassified as third-order once the
gate's real defect turned out to be scale resolution rather than prompt balance.

**6. A benchmark case with bad documentation — still open, and still the sharpest gap.**
All 22 cases are popular OSS with maintained READMEs; the prose-300 prefix bet pays on 11 of
12, and RepoRadar's actual target user — a private codebase with thin docs — is
unrepresented. This is named as a limitation in [`paper/DRAFT.md`](paper/DRAFT.md) §11 and
nothing in the benchmark can currently see it.

### What is actually open now

1. ~~**Ship the two retrieval channels.**~~ **HyDE shipped** (PRs #105/#106) and is measured
   end to end: +4.55 vs the baseline's +1.82, paired +2.73, 15 w / 3 l, **p = 0.0075** — the
   first result in the project's history to clear p < 0.05 against the baseline. The
   bibliography-seeded hop remains in `evals/`, and the case for shipping it **weakened when
   HyDE landed**: its 9 targets that HyDE misses sit in `cv`/`graph`/`llminfer`/`peft`/`rag`/
   `rl`/`speech`, six of which now win and two of which (`graph`, `speech`) return ten
   actionable papers out of ten, where nothing can be added. Of the three repos still losing,
   it reaches **one** target in `llminfer`, zero in `compiler`, and zero in `numerics` — whose
   bibliography yields a hop pool of four candidates. Worth shipping for repos unlike the
   benchmark; **not** the highest-value next move, and the benchmark cannot currently measure
   it either way.
2. **Non-arXiv sources for the repos that need them.** P3 concluded the IACR/DBLP/VLDB
   adapters are "the only remaining route" for `crypto`, `systems`, `storage`, `compiler`,
   `columnar` — and post-HyDE the loss column is `llminfer`, `compiler`, `numerics`, of which
   `compiler` is squarely in that set. 5 of the 6 targets those three repos miss are reached
   by **neither** shipped channel.
3. ~~**Calibration drift.**~~ **Measured 2026-08-09** and closed as a direction. The frozen map
   *is* decalibrated — under-confident by −0.129, 95% CI [−0.187, −0.067], on the 126 band
   papers of the live run — and recalibrating it is worth **+0.00 net@2 under LORO** against an
   oracle ceiling of +0.27, because the threshold sits in the trough of a bimodal score
   distribution. `evals/calibrate_finescale.py` is the standing instrument; what a monitor
   should watch is the **AUC** (0.824 live vs 0.841 at fit time — the alarm), not the gap
   (a gauge). See [`evals/RESULTS.md`](evals/RESULTS.md).
4. **Thin docs — MEASURED 2026-08-09, and it is the worst result in the project.** Ablating
   the profile's sources across four budgets on six repos (docs the only variable, judge on
   the real repo): mean net@2 **+5.17 → −0.50**, pooled precision **0.925 → 0.636**. Both
   pre-registered alarm conditions fired; the prediction of graceful abstention failed. The
   mechanism is §1's register mismatch in pure form — strip a repo's prose and its profile
   collapses to a self-description, so `speech` is served **the Whisper paper** for the
   Whisper repo. **Nothing inside the system notices**: gate-3 precision 1.00 → 0.53 while
   the gate issues *more* 3s, and the calibrated probability moves 0.799 → 0.709. Every
   stage eats the same impoverished profile and they fail *coherently*. Two consequences:
   a similarity floor on HyDE — the remedy this was built to test — is **refuted**, since
   the papers are close to a query that is simply wrong; and the danger zone is *a little*
   documentation, not none (`db`, whose profile ablates to literally nothing, correctly
   abstains). See [`evals/RESULTS.md`](evals/RESULTS.md).

   **Still open, and now with a measured reason:** (a) real obscure thin-docs cases —
   ablation is a *ceiling*, since these models have memorised the benchmark repos; (b) a
   **profile-information floor** so the system refuses rather than answers a question the
   repo never asked; (c) roadmap item 0, user-stated goals, which P8 concluded belong in
   the **query** — exactly where a thin-docs repo has nothing else to offer.
