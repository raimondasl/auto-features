# PLANS — what RepoRadar does next

The single forward-looking document. If an idea is not on this list, it is either shipped,
closed with a measurement, or nobody has proposed it yet. History is deliberately **not**
repeated here — it lives in:

| document | role |
|---|---|
| [`RESEARCH.md`](RESEARCH.md) | the experiment record, organised by problem (§9 is current) |
| [`ROADMAP.md`](ROADMAP.md) | the feature/probe ledger, item by item, with verdicts |
| [`evals/RESULTS.md`](evals/RESULTS.md) | the chronological raw record (P1–P18, NR series, C-1–30; a few NR ids are assigned in paper/DRAFT.md's appendix) |
| [`archive/`](archive/) | superseded plans, kept verbatim (MVP plan, original sketch, retrieval designs) |

## Where the system stands (2026-08-25)

The measured configuration (`rr init --measured`) ships HyDE dense discovery, hybrid
fusion, the Haiku actionability gate, and the fine-scale logprob rescore. Published
headline: **mean net@2 +5.72** on the 25-repo benchmark against the agentic baseline's
+1.56 (paired +4.16, sign p = 0.0004), precision 0.892 — the comparator is understated
by 0.28/case and corrects to **+1.84 / paired +3.88**, CI [+2.24, +5.60], p = 0.0007
(C-25; `evals/restate_c25.py`); independent draws of the same
configuration land +5.7 to +6.2, one-flag control variants around +4.8–+5.2 (C-7: a
single draw's level is not a property of the method). The keyword-only default remains **−8.12** — worse than emitting nothing.

**The margin is comparator-sensitive, and there are now three comparators [P26].** Same 25
repositories, same RepoRadar run:

| comparator | its net@2 | /case | shipped arXiv arm | arXiv+EPMC arm |
|---|---|---|---|---|
| Opus 4.8, v1, 12 turns — **published** | +1.84 | 2.2 | **+3.88** *p*=0.0007 | +4.32 *p*<0.001 |
| Opus 4.8, v2, 30 turns | +2.16 | 4.2 | +3.56 *p*=0.019 | +4.00 *p*=0.007 |
| **Opus 5**, v2, 30 turns | **+4.20** | 9.1 | **+1.52** 12w/**13l** *p*=1.00 | +1.96 *p*=0.54 |

Every cell against an Opus 4.8 comparator clears *p* < 0.05; **not one cell against Opus 5
does.** The comparator's +2.36 of strengthening decomposes as **+0.32 harness, +2.04 model** —
so the published figure is not an under-resourced baseline, it is Opus 4.8 being Opus 4.8.
Over all 37 repositories the shipped arm is 18w/18l/1t against Opus 5. Both source arms are
carried side by side because arXiv+EPMC scores higher everywhere and **is not shipped**;
it flips no significance verdict. Derived by `evals/restate_comparator.py`, pinned by
`tests/test_comparator_ladder.py`.

## The selection rule for new work

Two facts, four measurements each, decide what is worth proposing:

- **Discovery-channel changes have converted.** HyDE end to end is +1.36 and produced the
  project's first p < 0.05 result against the baseline.
- **Profile-side changes have not, four times in a row:** NR-33 +0.00, NR-35 +0.00,
  NR-36 −0.52, P11 −0.32. The gate and rescore downstream already extract what the profile
  knows. Each of those four won a cheap probe first — stage-1 correlation has a 0-for-4
  conversion record on this side of the pipeline.
- **The remaining losses are coverage failures** (RESEARCH.md §7): every case still lost
  to the baseline is one whose admitted set never contained what the baseline found —
  concentrated in `llminfer`, `compiler`, `numerics`, where 5 of 6 missed gold targets are
  reached by *neither* shipped channel.

So: prefer new discovery channels and work downstream of the gate. A profile-side proposal
needs a genuinely new mechanism, not a better variant of a measured one.

**The discipline, unchanged:** check the ledger first (nobody pays twice) → $0 stage-1
probe with a pre-registered prediction and kill condition → labelled-set diagnostic
(~$0.10) where applicable → live Tier B (~$25) only if stage-1 survives and the change
touches the pool. Pool-affecting changes cannot reuse frozen pools (`POOL_FLAGS`).

---

## Open items

**Numbers are stable ids, not ranks.** They were ranks once, and stopped being so as items
closed: 1-4 are answered, parked or built and are kept for their reasoning rather than deleted.
The list is ordered by current priority, so a high id can sit above a low one. Referring to an
item by number therefore stays valid across re-orderings, which is the point -- `litsearch_recall.py`
and `tests/test_litsearch_recall.py` both cite "PLANS item 4" and should not have to be edited
when something overtakes it.

**Currently first: item 14** (RepoRadar *and* the agent — two arms over 12 of 37 cases,
neither separated, with a monotone mechanism; core 25 outstanding), then item 11 (MCP distribution, which item 14 is the evidence for), then
item 7 (product work). Items 1-4 are answered or built; 6, 9, 10, 12 and 13 closed
negative; item 5's remainder is conditional on a proposal that has not appeared.

**Read NR-52 before spending anything on the net@2 ladder.** `evals/RESEARCH-net2-directions.md`
sequenced ten rungs behind a second-judge validity gate. That gate ran (rung 1) and returned a
result the ladder's own framing does not handle cleanly: it **passes as written** -- consensus
margin +0.57 against GPT's +0.32, inside the +-0.5 bar with the sign preserved -- and the bar was
**near-unfalsifiable**, demoting 0.7% of shipped and 1.7% of Opus 5 GPT-actionable papers. The
reading the pre-registration named as informative, the Sonnet-only sign, **flips**: our digest
runs 0.585 precision under Sonnet, below net@2's 2/3 break-even, while Opus 5's runs 0.714 above
it. And the GPT margin itself moves +0.54 -> +0.32 across two of our own draws.

**NR-56 is the first VALIDITY result, and it is the uncomfortable one.** Against adoption --
`ids(HEAD) - ids(T0)` over a repo's own docs, the only label here no model produced -- the
primary judge's discrimination gap is **0.153, CI [-0.040, +0.339], spanning zero**, and it
calls **49.2% of matched controls actionable**. `claude-sonnet-5`'s gap (0.282) excludes zero,
but the difference between them (0.129) misses the registered 0.15 bar, so **no better
instrument is named**.

Read it precisely: **absence of evidence, not evidence of error.** n = 31 across 6 cases with
`graph` contributing 13, noisy negatives biasing both gaps down, and adoption measuring what a
repo did rather than what it should have done. What is established is that the judge every
number in this project is scored against has **unestablished validity** on the one anchor
available. That belongs in the paper's limitations regardless of what else happens.

**NR-57 tried the obvious remedy and it does not work.** Enlarging the adoption set was called
the highest-value work available; mining all 37 cases (from 22) moved usable adoptions **31 ->
35**, because several cases carry no arXiv ids in their docs at all and others have no history
before the 24-month T0 cutoff. The result **replicates on an independently drawn control
sample** -- gaps 0.143 and 0.243, GPT still spanning zero, still not separated -- so NR-56 was
not an artifact of one draw. But **55 adoptions would be needed to settle it and the channel
tops out at 35**, so this benchmark cannot answer the question. Reaching 55 needs a longer T0
window or cases chosen for citation-rich documentation: a differently-constructed case set, not
more effort on this one. **Do not re-run the mining expecting a different answer.**

**A regression NR-56 caught, shipped by NR-54/55:** `temperature=0` breaks the Claude 5 family
(`sonnet-5`, `opus-5` answer 400 "deprecated for this model"); Claude 4.x accepts it. The gate
runs `haiku-4-5` so it was fine, which is exactly why the change looked verified. Fixed with a
narrow retry. **Consequence: the judge cannot be made deterministic this way**, so NR-53's 8.4%
self-disagreement is a standing property of the instrument and NR-55's gain applies to the gate
alone.

**NR-54 sized the measurement question and it is smaller than it looked.** Holding the pool
byte-identical and re-running, the gate/downstream sd is **1.44** per case against an implied
pool-collection component of **1.71** -- so retrieval drift, not the gate, is the larger source,
and the 2.23 quoted after NR-52 confounded the two (those runs used pools sharing a median
Jaccard of 0.365). The gate is 35% of paired variance in a frozen-pool arm, so making it
deterministic would tighten +-0.78 to **+-0.63**, a fifth. **Worth doing and not transformative:
ladder rungs run +0.20 to +0.45 and stay unresolvable, so the bundle-only rule stands.** **Done 2026-09-01**: `_call_claude` now sends
`temperature=0`, which covers the gate, HyDE's hypotheses, the repo summary, typed anchors and
the second judge in one place, guarded by a wire-level test in `tests/test_llm_client.py`.
`_call_ollama` is deliberately untouched -- different field, no measured arm uses it.

**One consequence to carry forward:** runs before and after differ by construction. Frozen pools
and cached judge verdicts are unaffected (the fingerprint does not cover temperature, and the
judge cache is keyed by prompt and model), but any *gate* comparison spanning 2026-09-01 is
confounded, and the shipped arm's +5.51 was measured pre-fix.

**NR-53 retired the one live objection to NR-52.** The Claude path sends no temperature, so
Sonnet's verdicts are sampled while GPT's are greedy, and the sign flip might have been one judge
disagreeing with itself. Measured: self-agreement kappa **0.798** against a cross-judge **0.199**,
label flip rate 8.4%. The gap is a property of the judges, not the sampler. NR-52 stands, and
fixing the temperature is hygiene rather than a correction.

Nothing is auto-killed by that -- the registered bar passed and is not being rewritten after the
fact. But the ladder's premise was "chase the margin to significance", and the margin is now
measured as judge-dependent in direction, not just in size. **A decision to spend down rungs 2-8
should be taken with that in hand, and the paper's comparator claim should be judge-relative
either way.**

**Item 12 is closed for good [NR-51].** It closed on reach (NR-49), reopened the same day on rank
(NR-50), and its paid arm returned **-0.19, CI [-0.84, +0.43], 9w/8l/20t** against a bar of 0.78
registered before the run. The whole cycle is worth reading as one thing: a licence bar set with
an effect size did its job in both directions -- it licensed an arm the reach null would have
skipped, and it killed the arm cleanly when the arm came back a wash.

**There is again no open evidence-led lead, and now it is a stronger statement.** Retrieval width
(NR-47), gate depth (NR-48) and iterative retrieval (NR-51) have each been measured end to end
and none pays. That is not three separate failures; it is one finding about a frontier -- feeding
this gate more, deeper, or differently-aimed candidates does not convert into net@2. Anything
proposed next has to change what the gate *does* with a candidate.

### 1. A wider dense corpus — probed and parked; the requirement is freshness [P12]

**Answered 2026-08-17, $0, `evals/openscholar_yield.py`.** Both the original item and its
rewrite are resolved, and neither the way they were posed.

The first version's stage-1 check was mis-specified: it asked whether the never-reached gold
targets are in OpenScholar's datastore, and they are already in **ours** — all 56 gold
targets sit in the shipped arXiv index, the unreached ones at ranks up to 223,245. That is a
**ranking** failure; more corpus cannot fix it. The rewrite then asked the right question
against the right ground truth (the 79 judged-actionable non-arXiv papers, which the
arXiv-only gold set is structurally blind to) and got: **43/79 = 54.4%, MARGINAL** — with
**zero** papers excluded for access and **38% excluded purely by date**, 28 of them from
2025–2026.

**peS2o v3 is not too narrow; it is frozen at October 2024.** For a freshness product that
is disqualifying, and it worsens with time. Adoption would also have meant ~378 GB (≈875× the
arXiv index) and a full re-embed, since its vectors come from OpenScholar's retriever rather
than ours.

**What survives as a requirement for any future proposal:** a second dense corpus must be
**re-syncable** the way `rr sync-index` is. The off-arXiv value is real — 79 actionable
papers, ~50% actionable rate, concentrated in `bio-*`/`mat-*` — and no shipped dense channel
can reach it. But a static snapshot is not the way, and that is now measured rather than
assumed. Revisit only with a corpus that updates.

### 2. The gold set is a sample — measured, and it changes how every recall figure is read [P15, P16, P17]

The 30-turn question is **answered and parked**: on six cases probed with a paired fresh
12-turn control, the cap bound on none of them (all controls returned `ok`; reaching
`--max-turns` fails loudly), and the turn effect on picks is **−0.13 Jaccard, CI [−0.38,
+0.11], n = 5** — inside the noise of simply re-running. Raising the cap is not the priority
it looked like.

**And two of the four "turn-limited" cases were never turn-limited [C-28].** Re-run at the
identical flags, `mat-mlip`/MACE and `mat-phonon`/phonopy **succeed at 12 turns**; only
`bio-scvi` and `mat-toolkit` reproduced their failure, and both complete at 30. P14's reading
— that the budget "does not transfer to large scientific codebases" — attributed to domain
what belongs to nondeterminism.

**Done, and it half worked [C-30].** Re-run at unchanged flags, `mat-phonon` succeeded
(3 picks, **+2 gold targets**) but **`mat-mlip` failed at 12 turns again** — one success in
three draws. "We already know they succeed" over-read a single draw: C-28's point is that
the *failures* are draws, and the successes are draws in the same way. Cohort **8/12 → 9/12**;
gold set 73 → 75; `benchmark25` untouched at 56.

The run also stranded a cache — `mat-phonon`'s baseline was cached, then arXiv threw HTTP 429
and none of its picks were judged, leaving a case that looked finished and contributed
nothing, indistinguishable from one the judge had rejected. Repaired at the root
(`incomplete_cases` / `judge_only` / an `incomplete` field in the frozen artifact); one pick
remains unjudged until the throttle clears.

**Still open on this cohort:** `bio-scvi`, `mat-mlip`, `mat-toolkit`. Given the draw
behaviour, retrying them at unchanged flags is cheap and may simply work — but note the
selection effect it feeds: every cached answer is conditioned on a run that *completed*, so
cases whose agent tends to run long are represented only by their short draws. That bias is
benchmark-wide, unmeasured, and gets slightly worse each time we retry until success.

**What the control arm found instead is the real item.** A re-run of the *identical*
configuration disagrees with the stored answer on **~59% of picks** (mean J = 0.41). Pick
counts swing 3→5, 4→2, 2→1→3, and `bio-singlecell` **abstained in the cache and returned two
papers on re-run** — abstention is not stable either.

Every published recall denominator (21/56, 34/56, 43/56) divides by a gold set derived from
**one draw** of that process. It has been treated as ground truth; it is a sample. This also
reframes 2026-08-09: that was read as "a flag change invalidated the caches", when re-running
at all would have moved the set nearly as much.

**What to do about it, cheapest first.** All of these are now subscription-billed rather than
dollar-billed, which is what makes them thinkable:

| option | cost | what it buys |
|---|---|---|
| **measure the spread** — re-run the baseline on all 25 benchmark cases k times, report how many gold targets are stable across draws | k × 25 agentic runs | the number that should accompany every `/56`: how much of the denominator is draw-dependent |
| **union gold set** — take the union of picks across k draws as the target set | same runs | a denominator that stops moving, at the cost of being larger and looser |
| **stability-weighted recall** — weight each target by the fraction of draws it appears in | same runs | keeps a single number, prices its own uncertainty |
| leave it, document it | $0 | every recall figure carries "single draw" as a caveat |

**The witness set v2 is built [P16]** (`evals/witness_set.py`, $0). Coverage is restated as
per-source reach probabilities with CIs (LOSO — reporadar grades nothing it found itself),
digest-level coverage is replaced by regret@15, and the union/denominator question dissolves:
growing the set tightens the intervals instead of degrading any number. Two findings that
outrank the bookkeeping: the shipped pool contains only **~14–23% of non-self witnesses**
(channels reach 77% at depth 1000; the pool cuts far shallower — the gap is the rank story of
§5.5 made concrete), and **1 of 19 adoption-mined papers** — the model-free source — is in
the pool at all.

**The P17 redraws are now pooled into it, and the set grew by a fifth.** 319 → **385
witnesses** over 31 cases: cli 75 (the gold set exactly, untouched), cli-redraw 92,
cli-redraw@30 10, api 50, reporadar 189, adoption 19. Still nearly disjoint — 349 of 385
single-source. Each draw is labelled by the *configuration* that produced it rather than by
the file it came from, because the prompt and the turn cap both change what gets found; `cli`
stays exactly the frozen gold-set derivation and its reach is unmoved at 8/56. Two things
follow:

- **Digest regret is +4.80 net@2/case** on top of +5.72, against **+3.48 measured over the
  319-witness set**. Regret is a function of the set's size by construction — more
  certificates reveal more headroom, bounded by the digest window — so both are correct at
  their own size and the pair is quoted rather than the older figure overwritten (C-17).
  It remains almost entirely a discovery deficit rather than a selection one.
- **Chao1 at witness level over three cli-redraws: ≥ 236.5 against 92 observed**, with 68 of
  the 92 seen in exactly one draw. Kept apart from the pick-level P15 estimate (≥ 34.3 over
  24) because the units differ. Redrawing one searcher three times found under 40% of what
  that *single configuration* can find — which is the quantitative case for pooling more
  searchers, i.e. for the v2 prompt, rather than for more draws of this one.

**The k = 3 spread is run [P17]** — 75 draws over the 25 cases, judged, caches untouched.
It answers the item and closes it:

- **The pre-registered prediction failed.** Target-level reproducibility is **0.39** against
  the pick-level 0.41: the judge absorbs *none* of the churn. Membership of the denominator
  is not stable, and the pre-registered rule (< 2/3) fires.
- **But the estimate is stable.** Four independent gold sets give pool-reach estimates of
  0.143 / 0.154 / 0.205 / 0.261 — all overlapping, pooled 0.19 [0.14, 0.25]. Membership churn
  is **variance, not bias**, so the published figures survive as estimates with intervals:
  hop **0.38** [0.26, 0.51], HyDE **0.61** [0.48, 0.72], union **0.77** [0.64, 0.86].
- **No saturation, and the population is large.** Union over the 11 always-present cases runs
  33 → 50 → 63 → **81** with no flattening; Chao1 ≥ **262** (70% singletons, so a loose lower
  bound). A union converges to "everything this one searcher ever finds", not to ground truth.
- **~19% of baseline runs hit `error_max_turns`** benchmark-wide, and `thin-lang` and
  `vectordb` failed all three draws while sitting in the gold set via cached lucky draws —
  the selection effect, now measured rather than suspected.

**What follows, and it is a documentation change rather than an experiment:** stop treating
the gold set as a set. Report reach as a probability with an interval (P16's design, now
validated by four draws agreeing) and spend further draws on tightening intervals, never on
chasing a denominator that does not converge. **Done** — restated in `paper/DRAFT.md` (§5.5,
abstract, §12, lesson 15); `README.md` carries no recall fractions.

**The generator is now cheap enough to use [P18].** `gold_spread.py` splits into phase A
(agentic runs, concurrent — touches nothing rate-limited) and phase B (verify + judge,
serial). A 4-case trial compressed 711 s of work into 275 s at concurrency 4, and **30 turns
rescued both chronic failures** (`thin-lang`, `vectordb`, which failed all three 12-turn
draws) with 0/4 failures. Safe because P15 measured the turn effect on *successful* runs as
inside noise: raising the cap converts failures into successes without mixing populations.
Nothing shared moved — `use_cache=False`, discriminator still `da766b38114e`.

### 3. The baseline's next configurations, in the order the evidence supports

Three changes are queued for the baseline — the **v2 prompt** (allow non-arXiv papers),
**Opus 5**, and the **30-turn cap**. P13–P18 split the baseline's two roles, and the ordering
falls out of the split:

- **As a witness generator, no validation is owed.** Any searcher that yields
  judged-actionable papers produces valid witnesses; there is nothing to attribute, because
  the witness set is a pool rather than a comparison. The three-arm validation once planned
  for v2/Opus-5 was a *comparator*-role requirement and does not apply here.
- **As the published comparator, all three change the rival's strength**, so `+1.84 / paired
  +3.88` would have to be re-measured. That is a separate, deliberate decision — and
  `gold_spread.py` cannot touch it, since it never writes `cache/baseline/cli/`.

Recommended order: **(1) widen `resolve_references`** — no LLM spend, unblocks everything;
**(2) v2-prompt witness draws** at 30 turns with concurrency, the highest-value searcher
change because P16 measured the shipped pool reaching *1 of 19* adoption-mined papers and P12
found 79 judged-actionable non-arXiv papers the arXiv-only set is structurally blind to;
**(3) Opus 5** as a further witness source, lower marginal value since it searches the same
arXiv-shaped space; **(4) the comparator re-measurement**, deliberately and separately.

**(1) is done** (PR #204). `verify.resolve_references` now classifies four outcomes across
arXiv, the DOI Handle API, Semantic Scholar and Europe PMC — and widening it exposed C-31, a
second "is this an arXiv id" rule answering *True* for the one id the project knows is bogus.

**(2) is drafted and unrun.** `BASELINE_PROMPT_V2` allows journal, conference and
bioRxiv/medRxiv papers and asks for `{"id", "title"}` where `id` is an arXiv id **or** a bare
DOI. Three things it had to get right before a single call:

* **The prompt is versioned, not edited.** `_cache_path` had no discriminator in it, so
  editing `BASELINE_PROMPT` in place would not have invalidated the 34 stored answers — it
  would have overwritten them, which is how `compiler`, `graph` and `storage` came to hold a
  restoration note instead of a transcript on 2026-08-09. v1 keeps its exact path and its
  pinned discriminator `da766b38114e`; v2 writes to `cache/baseline/cli-v2/`.
* **The parser was widened over live data.** `run_baseline` re-derives ids from cached `raw`
  on every hit, so accepting `id`/`doi` re-parses the gold set. Safe only because all 130
  stored recommendation items carry exactly `arxiv_id` and `title` — surveyed, then pinned as
  a test rather than left as a claim.
* **One id per reference.** `paper_id.canonical_ref` collapses every DOI spelling to
  `doi:…`, because a pick is stored as the model wrote it while the resolver returns the
  prefixed form — which would have broken `gold_spread`'s own `targets ⊆ picks` invariant for
  every non-arXiv paper.

To run it: `uv run python evals/gold_spread.py --prompt-version v2 --max-turns 30
--concurrency 4`. It writes `gold_spread_v2.json`, never `gold_spread.json`, and `report`
refuses to call its overlap with the frozen set "reproducibility" or apply the P17 decision
rule to it — a different prompt is a different searcher, not a redraw.

**The draws now reach the witness set.** `witness_set.gather_witnesses` reads every judged
`gold_spread` row across every prompt version, labelling each by its configuration
(`cli-redraw`, `cli-redraw@30`, `cli-v2@30`, …). Wiring it against the *existing* v1 draws
first was the point: it grew the set 319 → 385 and moved regret +3.48 → +4.80 **before** any
v2 call was billed, so what the v2 draws then moved is attributable to the prompt rather than
to the plumbing arriving at the same time. Source labels are discovered from the data, so a
v2 draw appears in the reach table on its own; `tests/test_witness_set.py` pins the full label
set, so a new one still has to be classified self or grading by a human.

### 3a. The v2 sweep is run — 75 draws, 2026-08-26

`gold_spread_v2.json`, 25 cases × 3 draws at 30 turns, **0 failed and 0 partial** (v1's
12-turn draws failed 6/5/3). Against v1's three 12-turn draws:

| | picks | DOI picks | targets | DOI targets | precision |
|---|---|---|---|---|---|
| v1 @12 (23 cases) | 140 | 0 | 124 | 0 | 0.912 |
| **v2 @30 (25 cases)** | **270** | **97** | **196** | **36** | **0.867** |

**The prompt reaches what it was written to reach.** 36% of picks and 18% of targets are
non-arXiv — papers v1 had no field to put in an answer. Precision falls 0.912 → 0.867, but
the caps differ so that is not cleanly the prompt's doing; the only clean control is the four
cases with a v1@30 draw, where v2 returned 67 picks to v1's 13 and 30 targets to 10.

**The instrument was the binding constraint, and it has been fixed.** 44 of the 270
references could not be scored, every one a DOI, most of them ACM (`10.1145/…`): POPL, PLDI,
OOPSLA, CACM, precisely the literature a code benchmark should be reading.

Which source to add was settled by measurement, not preference — all 28 distinct unscoreable
papers probed against both candidates: **OpenAlex had abstracts for 20, Crossref for 14, and
Crossref added exactly zero OpenAlex did not already have.** One tier, not two.

**Shipped, and it cashed in: 44 unscoreable references → 13.** Non-arXiv targets 36 → **61**,
sweep targets 196 → **221**, and **no ACM paper is unscoreable any more**. The searcher did
not change: those 25 new witnesses are papers v2 had already found and named, which nothing
in the pipeline could previously read. The residual 8 is now a named floor rather than an
impression — 5 Springer book chapters, 1 Elsevier paper, and 2 fabricated DOIs.

**`unjudgeable` is permanent only relative to the sources we asked**, which is why adding a
tier needed machinery and not just a function. `verify.TIER_SET` names the tiers, every
judged row records it, and `retryable` reopens a row when the current set is a strict
superset of the recorded one — reordering is not growth, and losing a tier can only find
less. Without it the 31 recoverable references would have sat behind a predicate correctly
refusing to re-ask a settled question. It is `prompt_version`'s lesson one module over: a
cached verdict has to carry the configuration that produced it, or it outlives its evidence.

**3 invented DOIs in 97** (3.1%), caught by the DOI Handle API rather than by the prompt —
v2 was deliberately given no anti-fabrication coaching v1 lacks, so this is the unassisted
rate and the number a comparator re-measurement would need.

**Pooled into the witness set: 385 → 462 → 482 witnesses**, regret **+4.80 → +5.56 →
+6.24** net@2/case. Reach into `pool-wemb`: `cli` unmoved at 8/56, `cli-redraw` at 19/92, and
`cli-v2@30` at **19/155 = 0.123** — the lowest of the family, and *falling as it grows*,
because the papers the tier unlocked are ones the shipped pool holds even less often than the
arXiv ones. Pooled non-self reach fell 0.174 → 0.149 → **0.138**, which is the measure working
rather than a regression. Chao1 for `cli-v2@30` is **≥ 305.2** from 155 observed with 104
singletons, so this searcher is no closer to exhausted than the last one.

**Opus 5, draw 1, is run — as a comparator candidate, with witnesses as the side benefit.**
25/25 cases at the v2 prompt and a 30-turn cap, 2026-08-27. As a comparator it is stronger
than Opus 4.8 and the reason is volume, not accuracy:

| | mean net@2 | precision | papers returned |
|---|---|---|---|
| **Opus 5** | **+4.20** | 0.820 | ~9.2/case |
| Opus 4.8 (draw 1) | +2.16 | 0.838 | ~4.0/case |

19W/5L/1T paired on case. Precision is a hair *lower*; both sit well above net@2's
break-even of p = 2/3, so returning 2.3x more papers is strictly rewarded. Opus 5 abstains
less, which wins big on arXiv-rich cases (`llminfer` +12, `numerics` +11) and loses badly
where near-abstention is right (`systems` -12, `http` -11).

**It is not established yet, and the control says why.** Opus 4.8's own three draws give the
noise floor — same model, prompt and cap on both sides — at 0.64 / -0.12 / -0.76. The Opus 5
effect is +1.92 to +2.68 against each 4.8 draw, roughly 3x that floor, but **two of the three
paired intervals cross zero** and all three rest on one Opus 5 draw. P17 already showed a
single draw reproduces 39% of its own targets. **Two more draws (~$390 notional) are what a
comparator claim would need.**

Witness side benefit: 187 witnesses from 25 runs against Opus 4.8's 155 from 75. Set
482 -> 592, regret +6.24 -> +7.52 net@2/case.

**Cost, measured not guessed.** The CLI's `total_cost_usd` under subscription auth is
notional API dollars, which is exactly this question. Opus 5 ran $163.91 for 21 runs
(~$7.81/run) against Opus 4.8's $103.86 for 75 (~$1.38) — **5.6x per run**, decomposing into
~4.8x per turn and ~30% more turns. One full draw over 25 cases is ~$195; over all 37, ~$290.

### 3c. Non-arXiv sources — CLOSED 2026-08-28 [NR-42], and re-closed on a second instrument [NR-58]

C-9 recorded that every non-arXiv source had been sent arXiv boolean syntax as a keyword query
for the product's whole history, and C-9b that the repair was published as routing "all three
call sites" when there were five and it routed two. **Verified against the live code
2026-08-27: all six `KEYWORD_SOURCES` now receive the translated queries** from a single
`to_plain_keywords` call. `(all:"vectorized execution") AND (cat:cs.DB)` arrives as
`vectorized execution columnar storage`.

So the objection to multi-source is no longer "it does not work". It is the measured evidence:
the one channel tested properly delivered ~175 uncategorised papers per repository at **+0.00
net@2**, and NR-11 recorded a wider pool making the headline *worse* when nothing ranked it.
**That argues for one targeted probe rather than switching everything on.**

**Answered for Europe PMC, and the answer is no [P22].** Sending each core-25 repository's
own queries to Europe PMC returns hits for **all 25, never zero** — 1,721 hits, **68% indexed
into MeSH**, every repository between 57% and 87%. `lint code` returns *"postoperative
pneumoencephalus in posterior fossa surgery"*; `arrow file` returns zebrafish telomerase.

That is the expensive outcome. A source that goes quiet outside its domain is free to enable
everywhere; one that answers confidently off-domain feeds every repository's pool, where net@2
charges 2 per false positive and non-arXiv papers escape the ranker's category component.

**And the collision is a property of the QUERY, not the repository**, which rules out
per-domain routing too: `crypto` gets real post-quantum cryptography from `cryptography` and
biology from `key`. Any domain classifier would route `crypto` "on" and still admit the noise.

**So: no multi-source default, and no per-domain routing on this evidence.** What the data
supports is a *relevance condition* on non-arXiv results — which is what the ranker's category
component would have supplied if uncategorised papers did not escape it. That is the item
worth opening, and it is a ranking change rather than a retrieval one.

**Probed, and it collides too [P23]: 48% off-domain against Europe PMC's 68%.** Same
queries, same 25 repositories, classified by OpenAlex's own `primary_topic.field` taxonomy.
Computer Science is the largest field at 34% and still a minority; Biochemistry (15%) and
Medicine (8%) follow. Zero repositories get silence.

The spread is wider and that is the finding: **24% (`speech`) to 84% (`webdev`)**, against
Europe PMC's flatter 57–87%. Distinctive technical vocabularies retrieve cleanly; generic
English does not. **That is P22's conclusion arriving by a second route — the collision is a
property of the QUERY**, so no domain classifier can fix `webdev`, an ordinary software
project whose queries happen to be common words.

**Measured against a matched control [P24, NR-41]: +0.32 on the core 25, CI [-0.24, +0.88].**
Two fresh same-day frozen pools, 37 repositories, everything fixed but `sources`. The control
reproduces the published headline to **+0.12**, so the delta is an effect and not a redraw.

The mechanism is not what the collision probes implied. Europe PMC supplies **2 of 205 shown
papers** on the core 25 — the gate rejects nearly all of it rather than admitting biomedical
noise. Across all 37 cases **29 of 30** non-arXiv papers that reach a digest are actionable.
**The "relevance condition" item below is therefore retired**: there is nothing for a filter to
remove, and it was proposed from the collision measurement without first checking the digests.

**P21's bio +4.00 does not survive the control** — matched, it is +1.00 over twelve scientific
cases with an interval crossing zero. Most of it was the collection, not the source. That is a
caution about every uncontrolled source comparison in this repository's history.

**OpenAlex measured too, and it goes the other way [P25]: -0.76 over 37 cases**, negative on
every cohort (core 25 -0.44, scientific 12 -1.42), 12W/17L. It reaches the digest more than
twice as often as Europe PMC (20% vs 9%) at **0.75 precision against 0.97**, admitting **17**
non-actionable papers where Europe PMC admitted 1. Reach was never the constraint.

**C-33: "the gate handles the collision" was generalised from one source and is false.** P24
retired the relevance-filter item on Europe PMC evidence alone. Corrected: the gate rejects
*obviously* off-domain material and admits *near-domain* material — biology beside a linter is
easy, Engineering beside a compiler is not. **The item was reopened in that narrower form**,
and was **closed again eight paragraphs below by NR-42** — see "CLOSED 2026-08-28". The
sentence that stood here said it was "the best-supported open item on this list", which was
true when written and false three days later; on 2026-09-02 it sent a reader at a closed item.
**A restatement that outlives its measurement is the C-17 shape**, and the fix is to say so
here rather than to delete it.

**Stacking is measured, not cautioned against:** +0.54 and -0.76 on the same 37 cases. A
three-source arm would most likely net negative. **No further source arms without a mechanism
change first.**

**Where multi-source is worth having: the scientific cohort** (non-arXiv 23% of the digest at
0.96 precision), and nowhere else on present evidence.

**C-34 [P26]: the -0.76 was attributed to the wrong term, and the remedy followed the error.**
Decomposing each arm's delta exactly — every non-arXiv paper is one the source supplied,
everything else is arXiv churn — gives **+EPMC: +0.54 = +0.73 own papers − 0.19 displacement**
and **+OpenAlex: −0.76 = +0.46 own papers − 1.22 displacement**. OpenAlex's own papers are
**net positive**; the 17 misses arrive alongside 51 actionable ones. The loss is 142 arXiv
papers leaving the digest and 100 different ones arriving. The materials six settle it: Europe
PMC contributes *zero* papers there and the arm still moves +0.50/case, 16 arXiv out and 16 in
— a source can move the score without appearing in the digest at all.

**Both probes together: no multi-source default, no per-domain routing.** The obvious next item
was a **relevance condition on non-arXiv results**. C-34 demotes it: that filter addresses the
term which is already positive in both arms. **The term that costs is displacement**, it scales
with how much a source is admitted, and both sources pay it — Europe PMC's own papers merely
cover it. That is also the real argument against stacking a third source.

**CLOSED 2026-08-28 [NR-42].** The narrowed form C-33 left open — a filter for the *near-domain*
admissions — was priced at $0 over artifacts already on disk, and it does not survive:

- **Neither instrument can discriminate.** The gate and the fine-scale rescore are the system's
  only pre-judge signals, and the two stages that solved this exact problem for arXiv. On
  OpenAlex's non-arXiv papers the gate-3 rate is **0.588 among actionable and 0.588 among
  non-actionable**; the rescore reaches **28 of 28** band papers and scores them **0.842 vs
  0.850 — the wrong way round**.
- **The best filter buildable today makes both sources worse:** restricting non-arXiv to gate-3
  takes Europe PMC +0.73 → +0.38 and OpenAlex +0.46 → +0.27.
- **The cost is upstream of the digest.** Only 3–5 of 37 cases reach the 15-paper window; the
  slot loss happens in the gate's `gate_depth: 50` input, shared across sources.
- **Nothing realistic clears the MRE of 1.04.** Fixing Europe PMC's slot term is worth +0.28;
  its entire oracle ceiling is +0.78. Unmeasurable on this benchmark even if built.

**What the probe found instead is a scoring defect, not a relevance one.** A quarter of OpenAlex
candidates arrive with **no abstract** (Europe PMC: none of 17,511), and both the gate and
`finescale.build_prompt` read `paper["abstract"]` with no guard — so they are scored on their
titles. Among shown papers, 4 of 17 non-actionable have none against 1 of 51 actionable
(intervals barely disjoint at n=17; recorded as a defect, not as an effect size). A paper with
no abstract is not irrelevant, it is **unmeasured**, and the product already takes the opposite
stance one stage over: *"a paper whose rescore call fails is omitted, never scored."*

**SHIPPED 2026-08-28** — `src/reporadar/evidence.py`, used by both LLM stages. Neither the gate
nor the fine-scale rescore will score a paper with no abstract; both report what they skipped.
Four boundaries, each pinned: **absence not brevity** (no character threshold — that would be
tuning against net@2 through a back door), **no backfill** (a skip shortens the batch rather
than pulling the next paper up, keeping the change a pure removal), **a skip is not a failed
call** (`enough_scored` counts attempts, so the pipeline partitions the band before passing a
denominator — otherwise an abstract-poor band reads as an outage and abandons the rescore), and
**not configurable** (a flag whose off-position restores "score what you cannot read" is a
footgun). No published number moves: every run to date is at ~100% coverage, so the guard is a
no-op on all of them. It shipped on the argument, not on a benchmark win.

**What would reopen the filter item:** a genuinely better discriminator. The oracle ceiling —
perfect discrimination, zero displacement — is **+1.38 over the control on OpenAlex**, which
does clear the MRE. The item is closed on the absence of an instrument, not the absence of
value.

**The third instrument is measured and it does not reopen it [NR-58, 2026-09-02].** NR-42
tested the gate and the fine-scale rescore, both LLM stages. The **dense embedding** — the one
scoring component non-arXiv papers do *not* escape, and the reason the filter was proposed in
the first place was that they escape the **category** component — was never asked. Registered
at 0.65 with an interval excluding 0.5, measured **AUC 0.578, CI [0.415, 0.673]** over 779
within-case pairs, 100 non-actionable papers against NR-42's 17 and not conditioned on having
been shown. **The item stays closed, now on two instruments.**

The arXiv control is **0.586, CI [0.531, 0.643]** — the same magnitude. So this is not "the
embedding covers arXiv and fails off it": it is a *weak actionability signal wherever it is
pointed*, which is not a defect, because it is weighted 1.5 in the shipped ranker for
**relevance** and relevance is not what a filter would be asking of it.

**NR-42's evidence was selection-conditioned, and that is the finding [C-36].** The same
signal reads **0.096** on NR-42's shown-only panel and **0.612** on the papers the pipeline
passed over, **intervals disjoint**. An instrument evaluated on the set it helped select looks
*worse than it is*: a paper admitted despite a low score on that instrument got in on
something else, and that something else correlates with being actionable. The arXiv panel
moves the same way at far higher n (0.485 shown vs 0.554 not-shown) with **overlapping**
intervals, so that half is directional only and is recorded as such.

What it overturns is narrow and worth stating exactly: **NR-42's conclusion survives** — the
wider panel independently says no filter is buildable — **but the argument it published does
not.** "The rescore scores them the wrong way round" was read off a panel where a genuinely
discriminating instrument would also read low, and NR-42's own artifact carried the truncation
caveat it then read past. `evals/embedding_discriminator.py`, pinned by
`tests/test_embedding_discriminator.py`.

**If a source is switched on regardless, OpenAlex is the one:** a third less noise, real
ACM/IEEE/VLDB coverage (599 CS works), and a field label already on every result that a filter
could read with no new machinery.

**Superseded — the original note, kept for its reasoning:** Run the same $0 collision check
before any judge call: OpenAlex is general rather than biomedical, so it may not collide at
all — but "may not" is what P22 was written to stop us assuming. It reaches the 43 ACM/IEEE/VLDB
targets Europe PMC structurally cannot, and `pool-oa-treat`/`pool-oa-control` exist (on the
`mat-` cases; a core-25 probe needs no pool at all).

**OpenAlex as a retrieval source is the probe.** Europe PMC is biomedical and reaches the
Nature/NAR/BMC end of Opus 5's non-arXiv targets; it cannot reach the **43 ACM/IEEE/VLDB**
targets, which are the largest slice. P20 established that OpenAlex carries exactly those
abstracts — that is what the new verifier tier does. `pool-oa-treat` and `pool-oa-control` are
already on disk, and `--sources arxiv,openalex` is one flag. Note `openalex` throttles keyless
callers, so the probe needs `OPENALEX_API_KEY` (present in this environment).

**Not "all seven sources".** `biorxiv` is documented as not a keyword search at all — its API
is a date-interval listing, so under a long lookback it returns the oldest postings in the
window rather than anything about the repository, and `validate_config` warns about precisely
that.

**The 6 bio cases are run, and they cost about twice what was projected.** $91.68 for six,
**$15.28/run against the core cases' $8.06** — scientific repositories are larger and the
agent works longer on them. The earlier ~$94 estimate for all 12 scientific cases used the
core-case mean and was out by 2x; measured, the 12 are ~$183 and a full 37-case draw is
**~$385**, not ~$290. `bio-mdtraj` is the line worth remembering: **$20.22 and 22 minutes to
return nothing** — an abstention costs what an answer costs, and net@2 scores it 0 either way.

**The prediction that the scientific cohort would shrink Opus 5's advantage was wrong on the
bio half.** Against the stored comparator (v1, 12 turns) Opus 5 is **+4.60 vs +1.60**, paired
**+3.00** over the 5 bio cases with a stored run — a *larger* gap than the core 25's +2.04,
because the v1/12-turn configuration is at its weakest here (2 picks per case, 0 for
`bio-singlecell`). The 6 `mat-` cases are unrun and may still differ.

### 3b. RepoRadar vs Opus 5 — the comparator question, and it is close

**Completed to all 37 cases 2026-08-28 [P26].** Six materials-science runs at the settings
every other Opus 5 row used finish draw 1 ($58.34; arm to date $351.40). Against RepoRadar's
arXiv+EPMC arm:

| cohort | RepoRadar | Opus 5 | paired | |
|---|---|---|---|---|
| core 25 | +6.16 | +4.20 | +1.96 | 14W/10L |
| bio 6 | +7.50 | +5.83 | +1.67 | 4W/1L |
| **matsci 6** | +5.50 | **+8.67** | **−3.17** | 2W/4L |
| **all 37** | +6.27 | +5.19 | **+1.08** CI [−0.97, +3.16] | 20W/15L |

**The margin was +1.90 over 31 cases and is +1.08 over 37, interval now crossing zero.** The
materials reversal is not volume: Opus 5 wins on precision there too (0.895 vs 0.841 —
RepoRadar's worst cohort, Opus 5's best), and no source arm rescues it (+arXiv 5.00, +EPMC
5.50, +OpenAlex 3.83). **It is not non-arXiv reach either:** matsci is where Opus 5 reaches
outside arXiv *least* (6.6% of picks, vs 34% on core 25 and 70% on bio), worth +0.33 of the
+8.67. It picks better arXiv papers than our ranker does, from a literature the pool holds.

**The split below replicates at n=37 and is the durable result:** on the 32 cases where Opus 5
does not over-answer the two are level (−0.06); all of the +1.08 comes from 5 cases where it
does, 4 of which RepoRadar abstains on outright (70% of the margin). Pinned in
`evals/opus5_arm.json` / `tests/test_opus5_arm.py`.

Paired over the 25 core cases, RepoRadar's headline run against Opus 5 draw 1:

| | mean net@2 |
|---|---|
| RepoRadar (shipped) | +5.72 |
| Opus 5 (v2, 30 turns) | +4.20 |
| **paired** | **+1.52**, 95% CI **[-1.04, +4.16]**, 12W/13L, sign *p* = 1.00 |

Against the published v1/12-turn comparator the same run is paired **+3.88**, CI [+2.24,
+5.60], p = 0.0007. **So the margin falls to +1.52 and loses significance against a stronger
baseline.** That does not make the published figure wrong — it is correct for the comparator
it names — but the headline is sensitive to comparator strength, and Opus 5 at the v2 prompt
and 30 turns is stronger on three axes at once.

**Where the +1.52 comes from is the finding.** Split by whether Opus 5 over-answered:

| | cases | RepoRadar | Opus 5 | delta |
|---|---|---|---|---|
| Opus 5 net@2 < 0 | 4 (`http`, `linter`, `systems`, `webdev`) | +0.75 | -10.25 | **+11.00** |
| Opus 5 net@2 >= 0 | 21 | +6.67 | +6.95 | **-0.29** |

On the 21 cases where Opus 5 does not over-answer, **RepoRadar is 0.29 behind**. The whole
advantage is four cases where Opus 5 answers and is punished 2 per miss; RepoRadar abstains
on 4 cases, Opus 5 on none. Change the penalty from 2 to 1 and the gap narrows to +0.80.
**RepoRadar's edge over Opus 5 is abstention discipline, not discovery** — which is what
DRAFT.md's own limitation ("the metric rewards shyness") predicts.

**Superseded 2026-08-27 by the matched arm [P21].** The comparison below was between an
arXiv-only RepoRadar and a baseline drawing **68% of its bio targets from outside arXiv** —
not a like-for-like contest. Run at window 15 with `arxiv,europepmc` and `w_embedding` 1.5,
**RepoRadar is +8.17 against Opus 5's +5.83, paired +2.33, 4W/1L.** The 5.5-point spread
between the two old runs decomposes as window -1.50, Europe PMC **+4.00**, `w_embedding`
**-1.33** [NR-40]. What follows is kept because it is what the arXiv-only data said.

**On bio, quote the window-15 run and not the window-30 one.** Two RepoRadar runs cover the
bio cases and differ by 5.5 points, entirely because of configuration:

| run | window | sources | bio net@2 | vs Opus 5 (+5.83) |
|---|---|---|---|---|
| 2026-08-20 (12 cases) | 15 | arxiv | **+5.50** | **-0.33** |
| 2026-08-21 (6 cases) | **30** | arxiv + europepmc | +11.00 | +5.17 |

net@2 sums over what is returned, so doubling the digest window mechanically raises it while
precision stays above break-even — the same effect the 10 -> 15 change was measured at
(+1.24/case). Comparing a window-30 RepoRadar against Opus 5 would be a rigged comparison.
At the comparable window, **Opus 5 is slightly ahead of RepoRadar on bio**. Neither bio run
uses the shipped `w_embedding` 1.5, so neither is the shipped configuration; the honest
statement is that RepoRadar has no bio measurement at the shipped settings.


### 4. LitSearch as a recall regression gauge for the dense index — BUILT 2026-08-29

597 gold-labelled queries (arXiv:2407.18940) over recent ML/NLP. The index had bit-identical
encoder verification but **no recall-fidelity gauge**: verified for *identity*, unmeasured for
*usefulness*. `evals/litsearch_recall.py` closes that, $0, no LLM or judge calls.

| arm | R@5 | R@20 | R@100 | median rank when found |
|---|---|---|---|---|
| **bare** (the shipped form) | 0.247 | 0.376 | **0.560** | 8 |
| prefixed | 0.259 | 0.396 | 0.530 | 5 |

**The coverage number is what makes it a fidelity gauge:** 458 of 574 gold papers carry an
arXiv id and **456 of 456 distinct ones are already in our shards**, so a query that fails
fails at *retrieval*. The 99 queries with no gold paper in the index are excluded and counted,
never scored — counting them would measure LitSearch's overlap with arXiv and drift every time
arXiv grows.

**The query prefix was decided by the paired test, and it reverses the aggregate reading.**
`mxbai-embed-large-v1` is asymmetric, so the prefix belongs on the query side or nowhere. The
aggregates say it wins at the top (+0.012 at k=5, +0.020 at k=20); McNemar over the same 498
questions puts both at p = 0.34 and p = 0.12, and the **one resolved difference is at k=100,
p = 0.020, favouring bare**. Two aggregate rates are not a comparison — which is why the arms
run together and their per-query ranks are stored.

**Where it sits:** a post-`rr sync-index` gate (`--check`, ~40 min of CPU), **not** one of the
six per-commit gates and **not** in the product CLI. `rr sync-index --verify` already answers
the identity question; this answers the usefulness one and needs a dataset the product does
not ship. `tests/test_litsearch_recall.py` reads the frozen artifact, which is the part CI can
afford. Explicitly **not** a net@2 claim — researcher questions are a different register from
repo→paper, which is §5's own finding.

### 9. The freshest slice — CLOSED NEGATIVE 2026-08-29, same day it was filed [C-35]

Filed by NR-43 with a mechanism and a $0 confirmation step. The step ran and killed both.

**`w_recency` is 0.0** — shipped default, `evals/harness.py`, and every benchmark run since
2026-07-06, because `--rr-all-time` *is* `w_recency 0`. The proposed mechanism does not exist,
and one minute in `config.py` would have established that before the item was written.

**And the effect is not in the product.** Of the 159 judged papers from the newest month,
**11 were ever shown by any run — 6.9% against 36–44% for every other period — and all 11 are
actionable**. Promotion is flat in age (146.1 per 10k pooled pre-2026 against 183.2 newest).
The collapse is entirely in papers the gate declined; the 148 unshown score 0.176.

**So this was the gate working, not failing** — and NR-43 mistook the judge cache for a sample
of the product. It is the union of every experiment ever run, including deliberately deep pool
draws. `evals/fresh_slice_probe.py`, pinned by `tests/test_fresh_slice_probe.py`.

**Do not reopen without a sample that is actually the product's output.** The pool-side fact
survives — the freshest slice is the hardest, 0.176 among never-shown against 0.38–0.50 older
— but it is a statement about candidates, and the gate already answers it.

### 5. $0 hygiene, run when convenient

- **Judge-contamination re-analysis** — **DONE 2026-08-29 [NR-43], and it found something
  else.** `evals/judge_date_stratify.py`, $0, 4,019 cached verdicts. The hypothesis is
  **refuted**: the actionability trend rises 0.31 (2013) → 0.64 (2025) with no step, and where
  it does collapse — **2026-07, 0.233 over 159 papers** — a *second* model with a different
  cutoff collapses with it (Sonnet 0.105 vs GPT 0.237) and the two **agree more there than
  anywhere else** (0.868). One model's training data cannot do that. Not case mix (10 of 11
  cases fall within-case, mean −0.221); not the index boundary (the in-index half still falls
  to 0.333 against June's 0.510).

  Residual recorded, not buried: a cutoff shared by *both* judges would predict the same thing
  and is not excluded (`shared_cutoff_excluded: false`).

  **It also filed a retrieval claim that did not survive the day [C-35].** July papers draw a
  *0* 52% of the time against 10%, and NR-43 read that as "the freshest slice is where
  off-topic material enters the pool, and RepoRadar is a freshness product whose freshest
  slice is its worst". Item 9 tested it: `w_recency` is **0.0** so the named mechanism does not
  exist, and **11 of those 159 papers were ever shown, all 11 actionable**. The collapse is in
  papers the gate declined. The judge cache is the union of every experiment ever run, not a
  sample of the product — stratifying it by date measured the sampling. See item 9.
- **P6 adoption cross-check** — score repo-side channels against the 31 git-history-mined
  adoptions, the project's only model-free ground truth. Its consumer weakened when item
  16 closed; run it only if a new repo-side proposal appears, and run it *before* that
  proposal's Tier B.

### 6. OpenAlex-Topic community match — CLOSED NEGATIVE 2026-08-29 [NR-44]

$0, no LLM, no judge. Pre-registered bar AUC >= 0.65 at the judge==3 target on testbed B.
**Missed by a wide margin in both arms, with both intervals excluding the bar:**

| arm | AUC | 95% CI |
|---|---|---|
| `repo_text` — OpenAlex classifies the repository's own description | **0.453** | [0.343, 0.571] |
| `own_papers` — community from the case's own non-band papers | **0.409** | [0.297, 0.529] |

Two independent failure modes, either one fatal. **The taxonomy is too coarse**:
`subfield:Artificial Intelligence` is the modal community for six of nine scored cases, and a
band already filtered to one topic cannot be ordered by a constant. **And the classifier cannot
read software prose**: `diffusion` classifies as NMR spectroscopy (Physics), `cv` as brain-tumour
detection (Neuroscience), `crypto` and `graph` as Computational Physics — §5's register mismatch
reaching a new instrument. Real README prose fixes `cv` and leaves the others wrong, so it is the
vocabulary, not the truncation.

**Two corrections the probe forced.** The incumbent's 0.841 is pooled across three testbeds; on
testbed B at judge==3 it is **0.760**. And "the 602-paper labelled set" is the whole labelled
set — the band that can actually be scored is **108 papers with 41 positives**, giving an
interval of about ±0.11. A marginal pass was never available here; the bar was set without
reference to the resolution. It separates 0.65 from 0.45 easily, which is what was needed.

**Do not reopen with a finer taxonomy alone** — both failures would have to be fixed, and the
second is the project's oldest known obstacle. `evals/topic_community_probe.py`, pinned by
`tests/test_topic_community_probe.py`.

### 7. Product work, judged on demand rather than evidence

- **`rr ask`** (ROADMAP 15) — citation-grounded Q&A, "a product bet, not a research one",
  sequenced v2.0. OpenScholar's cite-or-abstain recipe and PaperQA2 are the named
  precedents when it happens.
- **MCP distribution** — promoted to its own entry, **item 11**, now that literature exists
  for it.
- **Zotero/BibTeX bridge** (ROADMAP 17) — pure integration, unaffected by any measurement.
- **Digest theme headers** (from Eliot, 2605.27610) — cluster the ~15 shown papers under
  labelled headers. Cosmetic, cannot touch net@2 by construction; if done, assert
  digest-set equality in tests.

### 10. Widen the HyDE union — CLOSED NEGATIVE 2026-08-30 [NR-47]

**Measured, not guessed.** Of Opus 5's 302 judged-actionable picks, **164 (54.3%) sit in our
own 3.1M dense index and never reach our candidate pool**, against **one** paper outside the
index entirely. The lever is pool assembly. P12's "more corpus cannot fix a ranking failure"
now holds against a frontier model's picks, not just against the gold set.

**And the fork is already resolved.** `hyde.top_k = 100` per hypothesis, four hypotheses, union
feeds the ranker. Those papers' ranks under *our own* hypotheses:

| cut | recovered | |
|---|---|---|
| **100** (shipped) | 12 / 104 | **11.5%** |
| 1,000 | 51 | 49.0% |
| 2,000 | 65 | 62.5% |
| 5,000 | 82 | 78.8% |
| 10,000 | 93 | 89.4% |

Median rank **1,087**, p25 323, p75 3,562. **The union is too narrow; the hypotheses are in the
right register.** That is the cheap branch — a config integer, not a new mechanism.

**Why this is still a measurement and not a patch.** Reach is not net@2, and this project has
the scar: **NR-11** recorded a wider pool meeting a near-binary gate and making the headline
*worse*, and P4's pool expansion measured as a wash until the fine-scale rescore ranked what the
gate admitted (§8.2's composition finding). Widening `top_k` puts 4,000 candidates where 400
were, and `gate_depth` still shows the gate only 50 of them — so more reach can arrive as more
dilution.

**STAGE 1 PASSED 2026-08-29 [NR-46].** Simulated without re-collecting any pool — a witness is
reached at cut *K* if already pooled or its HyDE rank is below K. Pre-registered bar was ≥ 0.25
at K = 1000, kill below 0.20:

| | reach |
|---|---|
| baseline, simulated at the shipped cut of 100 | 0.2231 |
| **cut 1,000** | **0.4481** (+0.225, **+101%**) |
| cut 5,000 | 0.6077 |
| ceiling (121 of 122 unreachable witnesses are non-arXiv) | 0.7654 |

`cli-v2-opus5@30` — the comparator source NR-45 traced — goes **0.202 → 0.424**, so the widening
reaches the population the item exists for. All 37 cases now have hypotheses, including the six
materials ones generated for this.

**Two things stage 1 changed about stage 2.** The baseline had to become the simulation at the
shipped cut rather than the collected pool, because the pool used a different hypothesis draw
and that alone is worth **+0.058** — so **both arms of the paid run must share one pinned
hypothesis set** (`rr_hyde_hypotheses`), or the draw swamps the effect. And the ceiling says
the remaining gap is the non-arXiv population, which P24/P25 already closed.

**STAGE 2 RAN, AND THE KILL CONDITION FIRED [NR-47].** ~$25, two same-day arms over 37 cases,
`hyde.top_k` 100 against 1000, one pinned hypothesis set shared by both.

| cohort | control | treatment | paired |
|---|---|---|---|
| **all 37** | +5.51 | +4.73 | **−0.78** CI [−1.59, −0.03], 13w/17l/7t |
| core 25 | +5.92 | +5.40 | −0.52 |
| bio 6 | +4.67 | +4.17 | −0.50 |
| matsci 6 | +4.67 | +2.50 | −2.17 |

Reach doubled exactly as stage 1 simulated, and bought nothing. **The direction is closed: do
not widen the retrieval cut.**

**But the diagnostic did its job, and it points somewhere specific.** The papers the wider cut
adds are *as good as the ones already there* — precision **0.882 against 0.878**. What went
wrong is that a **5.9× larger pool produced a SMALLER digest**, 8.3 → 7.4 papers per case.
Splitting the −0.78 exactly: **−0.609 from showing 32 fewer papers (78%)**, −0.175 from the ones
shown being slightly worse.

A candidate set six times larger meets a gate that still reads `gate_depth` **50** of it. That is
the *"the gate never saw them"* branch, and it opens **item 13** rather than closing the
direction outright.

### 13. Widen the gate's input — CLOSED NEGATIVE 2026-08-30, same day it opened [NR-48]

NR-47 pointed here: the wider cut's papers were fine (0.882 against 0.878) and the digest shrank
because a 5.9x pool met a window still reading `gate_depth` 50. Item 13 moved the window --
50 -> 150, `hyde.top_k` held at 1000, same pinned hypotheses, **reusing NR-47's pools** because
`rr_pool` is not a POOL_FLAG.

| arm | `top_k` | `gate_depth` | net@2 | digest/case | precision |
|---|---|---|---|---|---|
| **A ships** | 100 | 50 | **+5.51** | 8.3 | 0.889 |
| B wide | 1000 | 50 | +4.73 | 7.4 | 0.880 |
| **C deep** | 1000 | **150** | +5.32 | **9.0** | 0.864 |

**Pre-registered: digest ≥ 8.3/case AND net@2 ≥ +5.51.** The digest recovered (9.0). **net@2 did
not (+5.32).** That was the stated kill, word for word — and the mechanism is visible: the deeper
window admits **99 papers at 0.859 precision, displacing 41 at 0.951**. Ranks 50–150 are thinner
material; above break-even, so depth recovers **+0.59** of NR-47's −0.78, but not all of it.

**C vs what ships: −0.19, CI [−1.14, +0.78], 14w/15l/8t** — a wash, at 5.9x the pool and 3x the
gate calls. **No gain, real cost.**

**Both pool-volume levers are now spent**, which is the finding worth carrying: retrieval width
and gate depth were the two ways to feed this pipeline more candidates, and neither pays. NR-11
said a wider pool met a near-binary gate; the sharper version is that **more candidates do not
help this system at any depth we can afford to judge**. Four nulls (NR-11, P4, NR-47, NR-48) are
one statement about a frontier rather than four separate attempts.

**Do not reopen with a different depth or a different cut.** A proposal here needs to change what
the gate *does* with a candidate, not how many it sees.

### 11. MCP distribution — product work, and the self-run just supplied its reading list

ROADMAP 2's remaining half: the server ships (`rr mcp` exposes `get_repo_profile`,
`get_ranked_papers`, `explain_relevance`, `rate_paper`); what is missing is **registry publish
plus a Claude Code plugin**. Unchanged in kind — a distribution bet, judged on demand rather
than on evidence, and it cannot move net@2 by construction.

**What is new is that RepoRadar's run on itself (2026-08-29, run #2) returned three MCP papers
as its entire Top Picks tier**, all through the `all:mcp` query:

| paper | gate | what it offers |
|---|---|---|
| [2608.23992](http://arxiv.org/abs/2608.23992) Hybrid Semantic Tool Discovery for Enterprise MCP Gateway | 3/3 | semantic ranking over large tool catalogues — the direct answer to the tool-count constraint below |
| [2606.30317](http://arxiv.org/abs/2606.30317) MCP Server Architecture Patterns for LLM-Integrated Applications | 3/3 | a pattern survey; the first systematic one we have seen |
| [2603.17339](http://arxiv.org/abs/2603.17339) citecheck: an MCP Server for Bibliographic Verification and Repair | 2/3 | adjacent to `verify.py`'s tier set, as a service rather than a library |

**The constraint that still governs the design:** keep the server under ~10 tools — Haiku-class
tool-selection accuracy degrades at 10-15 — and prefer parameterized tools over more of them.
2608.23992 is interesting precisely because it attacks that ceiling from the other side, with
retrieval over the catalogue instead of a smaller catalogue.

**Sequencing:** unchanged. This waits for a decision that product work is on, not for another
measurement. Nothing here is blocked.

### 12. Iterative retrieval (PRF-HyDE) — CLOSED NEGATIVE 2026-08-31 [NR-49, NR-50, NR-51]

**The paid arm ran and the item is closed [NR-51].** ~$15, 37 cases, treatment differing from
control only in a pinned hypothesis file carrying round 1 UNION round 2 (8 abstracts/case through
the shipped `hyde.discover`).

| | paired | | |
|---|---|---|---|
| **all 37** | **-0.19** | CI [-0.84, +0.43] | 9w/8l/20t, *p* = 1.0000 |
| the 33 with a real round 2 | -0.21 | CI [-0.97, +0.52] | 9w/8l/16t |

**|-0.19| < 0.78, the bar registered before the run. Null, and item 12 does not reopen again.**
The digest grows 8.3 -> 8.5 while precision falls 0.889 -> 0.876: PRF shows slightly more and
slightly worse.

**And NR-50's free prior had the sign wrong.** It read the judge cache over window papers, got
dp = +0.054 favouring round 2, and flagged it weak (61%/73% void, selected by prior exposure).
Measured in the digest: **85 added at 0.882 displacing 77 at 0.935, dp = -0.053**. Same
magnitude, opposite sign. The caveat was right and the number was not evidence about direction
at all -- worth carrying forward as a rule about judge-cache priors, not just a note on this arm.

**The four no-round-2 cases came back as exact ties with byte-identical picks**, which is what
confirms the two arms differed in one thing only. Two cases (`compiler`, `numerics`) lost HyDE
to an arXiv 429 and 503 and were re-collected at identical flags rather than dropped -- NR-47's
confound, twice.

---

**The reopen that licensed it, kept because the reasoning stands:**

**Reopened the same day by `evals/prf_rank_probe.py` [NR-50], and the paid arm is licensed.**
NR-49's own caveat was that reach is counted over 520 witnesses while the pools differ across
their whole contents. NR-47 and NR-48 had named the binding constraint -- the gate reads a fixed
`gate_depth` of a *ranked* pool -- so the follow-up was: do round 2's papers rank into the top 50?

**They take 20.61% of it. 340 of 1650 slots, 10.3 per case, displacing 349**, spread from 2/50 to
21/50 across every cohort. The bar was pre-registered *with an effect size* (kill <5%, licence
>=16%, both derived from the window-to-digest-to-3dp chain against NR-47's +-0.78 bootstrap), and
20.61% clears the licence.

**What that does and does not mean.** The share is a **magnitude, not a direction**: if round 2's
papers are worse than the ones they displace, the same arithmetic gives an equal loss. It says
the effect would be *resolvable*, which is exactly what the reach null could not say. And the
free quality prior cuts against the licence -- entering papers judge at 0.714 against 0.660
displaced, dp = +0.054, implying **+0.28 net@2/case, inside the noise**. Both stand; the
pre-registration is not revoked after seeing the prior, because that is the failure NR-49 records.

**And the depth grid weakens it further, on a reading fixed in advance.** `gate_depth` 100 had
never been run at the shipped `top_k`; taken to depth 300 in one pass, round 2's share **rises**
-- 17.6% at 25, 20.6% at 50, 22.3% at 100, 26.6% at 300. The probe committed beforehand to
reading a rising share as *marginal*, and it is: **15.2% of ranks 1-10** against 26.6% by 300.
Round 2 is weakest exactly where the digest is drawn. Using its **16.77%** share of the top-15
digest window rather than 20.61% of the top-50 cuts the implied effect to **+0.84** at a generous
dp and **+0.23** at the dp measured. The licence stands because it was registered at depth 50 and
is not re-decided; the estimate it licensed has shrunk toward the noise floor every time it has
been refined.

**The paid arm, when it runs:** treatment = shipped pool + round 2 at `top_k` 100, control =
the shipped arm already collected, one pinned hypothesis set per round, 37 cases, ~$15. The
pre-registered expectation is |net@2| >= 0.78 to resolve; anything smaller is a null and closes
the item for good. **On current evidence that is the likely outcome** -- which is a reason to
decide deliberately, not a reason to skip a licensed arm.

---

**The earlier close, kept because the reach result stands unchanged:**

**Closed 2026-08-30, ~$0.05, `evals/prf_hyde_reach.py`.** Stage 1 ran budget-matched -- round 1
at 100 against round 1 at 50 unioned with round 2 at 50 -- because NR-47 and NR-48 had just spent
both volume levers and an unmatched test would have re-run them under a new name.

**It cleared the pre-registered bar (0.2288 against 0.2231) and the pass was refused.** That is
+0.0057, three witnesses of 520, McNemar *p* = 0.68. NR-46 measured a plain hypothesis *redraw* at
+0.0577 -- the effect is **one tenth of the noise floor of the procedure it modifies**. The bar
named a threshold and no minimum effect size, so a null cleared it; that is a defect in the
pre-registration, recorded rather than repaired, and the lesson for the next one is to state an
effect size before the data exists.

**The honest risk written below was the right risk, and it is not what happened.** Round two did
*not* merely re-search the neighbourhood: five witnesses sit past round-1 rank 1000 -- up to 4187
-- inside round 2's top 100, and NR-47's widest measured cut was 1000, so width cannot buy them at
any cut this project has run. Seven of the thirteen gains have ever been judged and all seven are
actionable. **This is post hoc and does not reopen the item**; it is the one thing here worth a
future pre-registered test, and the test is not a reach test -- NR-47 and NR-48 between them show
the binding constraint is what the *ranker and gate* do with a candidate, so the question is
whether PRF's unique finds rank highly enough to reach the gate at all.

**What also emerged, and constrains any retry:** PRF is structurally blind on the cases where
retrieval most visibly failed. Four of 37 -- `cli`, `http`, `linter`, `webdev` -- produced no
round 2, because the shipped arm showed nothing to feed on; they hold 34 witnesses, **none reached
by any route**. The abstention discipline that makes RepoRadar competitive with Opus 5 is the same
thing that starves feedback. Any iterative design inherits this.

**The original reasoning, kept because the item's premise still stands:**

**The one thing Opus 5 does that our pipeline does not: it iterates.** Thirty turns of search,
read, refine, search again. Our discovery is one-shot — four hypotheses generated from the repo
profile, one search each, union, done. Pseudo-relevance feedback closes that gap in its cheapest
form: seed a *second* round of hypotheses from round one's **gate-admitted abstracts**, search
again, union the two rounds.

**Why it belongs on the list.** Discovery-channel changes are the only family with a converting
record here — HyDE end to end is +1.36 and produced the project's first p < 0.05. Profile-side
is 0-for-4 (NR-33 +0.00, NR-35 +0.00, NR-36 −0.52, P11 −0.32). This is paper-side and it is a
new channel rather than a better variant of a measured one, which is what the selection rule
asks for.

**Item 10 ran first and closed negative, which sharpens this rather than blocking it.** NR-47 and
NR-48 spent both pool-volume levers: widening retrieval cost −0.78, widening the gate's window
recovered the digest but not the score, and the pair is a wash against what ships. So **more
candidates is a closed direction**, and item 12's value is now specifically that it does not ask
for more — it asks a *different question* on the second round, seeded by what the gate admitted
on the first. If PRF is worth anything it will be because round two's hypotheses are better
aimed, not because they retrieve more.

**Free stage 1, same as item 10 — but read NR-46's lesson before trusting it.** Witness reach over
the 520 non-self witnesses costs nothing and is the right gate on spending. It is also exactly
what stage 1 of item 10 passed, at +101%, before the paid arm lost 0.78 net@2. **Reach is
necessary and demonstrably not sufficient**, so a reach win here buys a *cheaper* paid arm, not a
likely one.

**Prior art the self-run surfaced** (2026-08-29 digest, Maybe tier): *Novelty-Aware Agentic
Retrieval* ([2606.22151](http://arxiv.org/abs/2606.22151)) — multi-step agentic retrieval for
literature search, structured comparison of contributions; and *Discovering seminal works with
marker papers* ([1901.07352](http://arxiv.org/abs/1901.07352)) — bibliometric expansion seeded
from known-relevant markers, which is the citation hop's logic generalised and a plausible
cheaper variant of the same idea. Read before designing the arm.

**The honest risk.** Round two inherits round one's register. If the first round's admits are
already the papers we would have found anyway, PRF re-searches the neighbourhood we have and
adds nothing — and NR-11's warning applies here too, since a second round widens the pool
against the same near-binary gate.

### 14. The more RepoRadar you give an agent, the less it looks elsewhere [P27]

Every comparator figure this project had was **either/or**. Two arms now measure **both**:
Opus 5 in agentic mode with RepoRadar's MCP server attached, over the 12 scientific cases,
pre-registered in `evals/PREREG-mcp-arm.md` before each arm existed. Scored by
`evals/mcp_arm_report.py` into `evals/mcp_arm.json`.

| arm | what RepoRadar it got | net@2 | shown/case | precision |
|---|---|---|---|---|
| **B** Opus 5 alone | none | **+7.25** | 10.8 | 0.891 |
| **C** + MCP, digest picks only | ~12 papers | **+5.83** | 8.1 | 0.907 |
| **C-wide** + MCP, whole pool | 712-1252 papers | **+4.75** | 6.2 | 0.920 |

C - B = **-1.42**, CI [-4.75, +1.42]. C-wide - C = **-1.08**, CI [-2.67, +0.50], 3W/6L/3T,
sign *p* = 0.51. **Neither separates.** Under the registered rule the wide arm reads
*anchoring*: the corpus was not what made the agent narrow.

**But the secondary makes the mechanism much sharper than that word.** Where each arm's
picks came from -- RepoRadar's digest, its wider pool, or the agent's own search:

| arm | digest | pool only | **off-pool (its own finds)** | search_papers calls |
|---|---|---|---|---|
| B | 4 | 21 | **104** | -- |
| C | 19 | 13 | **65** | 48 |
| C-wide | 15 | **37** | **23** | **109** |

**The treatment was consumed enthusiastically and that is why it lost.** Widening the
corpus more than doubled `search_papers` use and took pool-only picks from 13 to 37 (13% ->
49%) -- so the agent was never starved, and a null result here is not "the tool went
unexercised". What it did instead was **substitute**: its own off-pool finds collapsed 104
-> 65 -> 23, monotonically, as it was given more of RepoRadar. Precision rose at every step
(0.891 -> 0.907 -> 0.920) and volume fell (10.8 -> 8.1 -> 6.2), and under net@2 above the
2/3 break-even, fewer loses.

**The product reading:** RepoRadar makes an agent's recommendations better and fewer. On a
metric that rewards volume that is a loss; for a maintainer with limited attention it may be
exactly the trade they want. The benchmark cannot distinguish those, and saying so is the
honest end of this item rather than a number.

**A prediction that was right for the wrong reason.** I registered "C-wide ~ C" and argued
that starvation would require the agent to substitute RepoRadar's index for web search,
"a strong assumption about a tool it had just met". The interval came out as predicted and
**the argument was wrong**: it substituted heavily. It just did not help.

**Two defects an adversarial audit found before the $86 ran**, neither reachable by
`--dry-run`, which returns before the code that differs: `gold_spread` accepted `--tools
web+rrwide` and served the **narrow** store (my plumbing patch had silently not landed and
I had "verified" it with a dry run); and the end-of-run message announced the control arm's
filename after writing the treatment's. A third -- the call log pooling a retried run's
calls with the dead attempt's -- was real and did not fire. All three are now pinned, and
the toolset -> corpus mapping lives beside `TOOLSETS` as `baseline.wide_corpus`.

**Remaining: core 25**, ~$180 notional per arm. The registered rule is over 37 cases and
cannot be called from 12; core 25 is where A and B are furthest apart on their own
(+6.16 vs +4.20), so it is the half that decides. Whether it is worth two more arms is a
budget question, not an evidence one -- the mechanism above is already legible and
monotone.

### 8. Held — real gaps with no affordable next step

- **Thin docs** — still the sharpest gap (RESEARCH.md §8.6; measured in paper/DRAFT.md
  §12.1–12.2). What remains unestablished is the expensive implication: `scan_source`
  plus goal synthesis (the `blind` arm's information-tracking correlation, r = −0.39, not
  significant). No cheap probe is currently designed; proposals welcome, with the
  four-null profile-side record as the prior to beat.
- **Bibliography-seeded hop** — a verified channel (21/48, 89% at ≥7 seeds) living in
  `evals/` only; its shipping case weakened when HyDE landed (RESEARCH.md §8.1). Revisit
  if a user repo class with rich bibliographies and poor HyDE coverage shows up.
- **`rr apply`** (ROADMAP 20) — moonshot; input quality is finally met, everything else
  is unbuilt.
- **CPU cross-encoder rerank** (ROADMAP 7 remainder) — predates E1–E5, which the
  pointwise logprob read won; re-derive against that record before any work, or drop.

## Recently closed — do not reopen without a new mechanism

| what | verdict | record |
|---|---|---|
| Typed README anchors (NERdME) | real, discriminates, does not reach the digest: −0.32/case, rescued cohort −1.00 | P9–P11 |
| Roadmap 16 (technique fingerprinting) | closed — grounding vocabulary obtainable and inert | P11 |
| ForeCite alias table | dead with item 16; its consumer no longer exists | papers-vs-the-ledger + P11 |
| NR-39's "only anchors discriminate" | case-mix artifact (−0.6pt per case) | C-21 |
| NR-39's "keywords are noise at +0.2pt" | saturation artifact; size-matched they score +3.3pt | C-22 |
| `w_embedding: 1.5` | resolved **positive**, +1.00/case, keep | NR-38 |
| PRF-HyDE (iterative retrieval) | closed on reach, reopened on rank, **killed by its paid arm at −0.19 against a 0.78 bar**. The third volume lever to be measured and fail | NR-49, NR-50, NR-51 |
| Research-gap radar, query rewriting, gap-phrase search | four independent negatives on the same mechanism | ROADMAP 14/19 |
| Multi-source keyword adapters (S2/OpenAlex/IACR/bioRxiv) | built, wired, measured null-to-negative | NR-27..34, C-9 |
| OpenScholar/peS2o as a second dense corpus | literature is there, snapshot frozen at Oct 2024; 38% of known off-arXiv value postdates it | P12 |
| "gold targets the channels cannot reach" | mis-framed: all 56 are in the index already; it is a ranking failure, not coverage | P12 |
