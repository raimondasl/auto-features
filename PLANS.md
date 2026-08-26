# PLANS — what RepoRadar does next

The single forward-looking document. If an idea is not on this list, it is either shipped,
closed with a measurement, or nobody has proposed it yet. History is deliberately **not**
repeated here — it lives in:

| document | role |
|---|---|
| [`RESEARCH.md`](RESEARCH.md) | the experiment record, organised by problem (§9 is current) |
| [`ROADMAP.md`](ROADMAP.md) | the feature/probe ledger, item by item, with verdicts |
| [`evals/RESULTS.md`](evals/RESULTS.md) | the chronological raw record (P1–P15, NR series, C-1–30; a few NR ids are assigned in paper/DRAFT.md's appendix) |
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

### 2. The gold set is a sample, not ground truth — and that outranks the turn budget [P15]

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

Recommendation: **measure the spread first (k = 3 on the 25)** and decide the denominator
question with the number in hand. Do not adopt a union or a weighting before knowing whether
the instability is 10% or 60% of targets — the probe says ~59% of *picks*, but picks are not
targets, and the judge filter may absorb much of it.

Until that is known, no re-run of the baseline should be undertaken for any *other* reason,
because it would move the gold set as a side effect and confound this measurement.


### 3. LitSearch as a recall regression gauge for the dense index — NEXT UP, one afternoon

597 gold-labelled queries (arXiv:2407.18940) over recent ML/NLP — squarely inside the
index's coverage. The binary-quantized index has bit-identical encoder verification but
**no recall-fidelity gauge**: nothing today would notice if binarisation, column pruning,
or one bad yearly shard silently cost 15 points of recall. Embed the queries, freeze
recall@5/@20, wire as an exit-nonzero gate after `rr sync-index`. Explicitly **not** a
net@2 claim — researcher questions are a different register from repo→paper.

### 4. $0 hygiene, run when convenient

- **Judge-contamination re-analysis** (from LitLLMs): stratify the already-cached judge
  verdicts by paper publication date vs judge-model cutoff; test whether actionability
  rate or GPT-5.5/Sonnet agreement (P7 data) shifts post-cutoff. Re-analysis of stored
  data, no new protocol.
- **P6 adoption cross-check** — score repo-side channels against the 31 git-history-mined
  adoptions, the project's only model-free ground truth. Its consumer weakened when item
  16 closed; run it only if a new repo-side proposal appears, and run it *before* that
  proposal's Tier B.

### 5. OpenAlex-Topic community match for ordering the gate-admitted band — $0 probe, weak prior

The one open idea downstream of the gate, which is where the four-null record says any
remaining headroom must be. From "Topic Is Not Agenda" (arXiv:2605.07158), reduced to its
cheapest testable form: OpenAlex Topic IDs as a free community proxy on the 602-paper
labelled set. **Bar: AUC ≥ 0.65** within the score-2 band (the NR-21 metadata family sits
at 0.585; the finescale incumbent at 0.841). No graph build under any circumstance — the
S2 truncation wall (§3.5 correction) and six no-bibliography repos price it out. Pairs
with ROADMAP item 12's unshipped Topics work if it ever passes.

### 6. Product work, judged on demand rather than evidence

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

### 7. Held — real gaps with no affordable next step

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
