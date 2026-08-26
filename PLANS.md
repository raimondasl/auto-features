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

**The witness set v2 is built [P16]** (`evals/witness_set.py`, $0): 319 witnesses across
four sources (cli 75 = the gold set exactly, api 50, reporadar 189, adoption 19), nearly
disjoint (306 of 319 single-source). Coverage is restated as per-source reach probabilities
with CIs (LOSO — reporadar grades nothing it found itself), digest-level coverage is replaced
by regret@15, and the union/denominator question dissolves: growing the set tightens the
intervals instead of degrading any number. Two findings that outrank the bookkeeping: the
shipped pool contains only **~14–22% of non-self witnesses** (channels reach 77% at depth
1000; the pool cuts far shallower — the gap is the rank story of §5.5 made concrete), and
**1 of 19 adoption-mined papers** — the model-free source — is in the pool at all. Digest
regret vs known witnesses is **+3.48 net@2/case** on top of +5.72, nearly all of it a
discovery deficit rather than a selection one.

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

**The v2 prompt is blocked on a prerequisite nobody has done.** `verify.resolve_references`
resolves arXiv ids only, and `BASELINE_PROMPT` demands `{"arxiv_id": ...}`. A v2 baseline that
recommends a *Nature* paper today produces a pick that cannot be verified, cannot be judged,
and therefore cannot become a witness — the non-arXiv region would stay invisible while
looking like it had been searched. Widen verification first: reuse `citations._s2_batch_post`,
EuropePMC as tier 2 (it recovers 5 of the 77 bioRxiv DOIs S2 returns null for), and the DOI
Handle API as the authoritative existence test.

Recommended order: **(1) widen `resolve_references`** — no LLM spend, unblocks everything;
**(2) v2-prompt witness draws** at 30 turns with concurrency, the highest-value searcher
change because P16 measured the shipped pool reaching *1 of 19* adoption-mined papers and P12
found 79 judged-actionable non-arXiv papers the arXiv-only set is structurally blind to;
**(3) Opus 5** as a further witness source, lower marginal value since it searches the same
arXiv-shaped space; **(4) the comparator re-measurement**, deliberately and separately.


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
