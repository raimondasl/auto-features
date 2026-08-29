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

## Open items, ranked

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

### 3c. The non-arXiv sources are testable again — OpenAlex is the next probe

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
easy, Engineering beside a compiler is not. **The item is reopened in that narrower form**, and
it is now the best-supported open item on this list.

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
cover it. That is also the real argument against stacking a third source. A relevance condition
may still be worth having for the near-domain admissions C-33 identified, but it is no longer
the item with the best evidence behind it, and it would not have recovered the -0.76.

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
| **paired** | **+1.52**, 95% CI **[-1.08, +4.16]**, 12W/13L |

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


### 4. LitSearch as a recall regression gauge for the dense index — NEXT UP, one afternoon

597 gold-labelled queries (arXiv:2407.18940) over recent ML/NLP — squarely inside the
index's coverage. The binary-quantized index has bit-identical encoder verification but
**no recall-fidelity gauge**: nothing today would notice if binarisation, column pruning,
or one bad yearly shard silently cost 15 points of recall. Embed the queries, freeze
recall@5/@20, wire as an exit-nonzero gate after `rr sync-index`. Explicitly **not** a
net@2 claim — researcher questions are a different register from repo→paper.

### 5. $0 hygiene, run when convenient

- **Judge-contamination re-analysis** (from LitLLMs): stratify the already-cached judge
  verdicts by paper publication date vs judge-model cutoff; test whether actionability
  rate or GPT-5.5/Sonnet agreement (P7 data) shifts post-cutoff. Re-analysis of stored
  data, no new protocol.
- **P6 adoption cross-check** — score repo-side channels against the 31 git-history-mined
  adoptions, the project's only model-free ground truth. Its consumer weakened when item
  16 closed; run it only if a new repo-side proposal appears, and run it *before* that
  proposal's Tier B.

### 6. OpenAlex-Topic community match for ordering the gate-admitted band — $0 probe, weak prior

The one open idea downstream of the gate, which is where the four-null record says any
remaining headroom must be. From "Topic Is Not Agenda" (arXiv:2605.07158), reduced to its
cheapest testable form: OpenAlex Topic IDs as a free community proxy on the 602-paper
labelled set. **Bar: AUC ≥ 0.65** within the score-2 band (the NR-21 metadata family sits
at 0.585; the finescale incumbent at 0.841). No graph build under any circumstance — the
S2 truncation wall (§3.5 correction) and six no-bibliography repos price it out. Pairs
with ROADMAP item 12's unshipped Topics work if it ever passes.

### 7. Product work, judged on demand rather than evidence

- **`rr ask`** (ROADMAP 15) — citation-grounded Q&A, "a product bet, not a research one",
  sequenced v2.0. OpenScholar's cite-or-abstain recipe and PaperQA2 are the named
  precedents when it happens.
- **MCP distribution** (ROADMAP 2 remainder) — registry publish + Claude Code plugin. One
  constraint recorded from the digest exercise: keep the server under ~10 tools
  (Haiku-class tool-selection accuracy degrades at 10–15); prefer parameterized tools.
- **Zotero/BibTeX bridge** (ROADMAP 17) — pure integration, unaffected by any measurement.
- **Digest theme headers** (from Eliot, 2605.27610) — cluster the ~15 shown papers under
  labelled headers. Cosmetic, cannot touch net@2 by construction; if done, assert
  digest-set equality in tests.

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
| Research-gap radar, query rewriting, gap-phrase search | four independent negatives on the same mechanism | ROADMAP 14/19 |
| Multi-source keyword adapters (S2/OpenAlex/IACR/bioRxiv) | built, wired, measured null-to-negative | NR-27..34, C-9 |
| OpenScholar/peS2o as a second dense corpus | literature is there, snapshot frozen at Oct 2024; 38% of known off-arXiv value postdates it | P12 |
| "gold targets the channels cannot reach" | mis-framed: all 56 are in the index already; it is a ranking failure, not coverage | P12 |
