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

**The instrument, not the model, is now the binding constraint.** 44 of the 270 references
could not be scored, and **every one of them is a DOI**: 41 `unjudgeable` — real papers,
proven to exist, that neither Semantic Scholar nor Europe PMC carries an abstract for — and
3 `hallucinated`. Most are ACM (`10.1145/…`): POPL, PLDI, OOPSLA, CACM, precisely the
literature a code benchmark should be reading. **Closing that gap is now the highest-value
verification work**, and a Crossref or OpenAlex abstract tier is the obvious next tier.

**3 invented DOIs in 97** (3.1%), caught by the DOI Handle API rather than by the prompt —
v2 was deliberately given no anti-fabrication coaching v1 lacks, so this is the unassisted
rate and the number a comparator re-measurement would need.

**Pooled into the witness set: 385 → 462 witnesses**, regret **+4.80 → +5.56** net@2/case.
Reach into `pool-wemb`: `cli` unmoved at 8/56, `cli-redraw` at 19/92, and `cli-v2@30` at
**19/135 = 0.141** — the lowest of the cli family. Pooled non-self reach consequently *fell*,
0.174 → 0.149, which is the measure working rather than a regression: v2 found papers the
shipped collection step is even less likely to fetch. Chao1 for `cli-v2@30` is **≥ 252.3**
from 135 observed with 88 singletons, so this searcher is no closer to exhausted than the
last one.


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
