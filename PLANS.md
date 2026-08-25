# PLANS — what RepoRadar does next

The single forward-looking document. If an idea is not on this list, it is either shipped,
closed with a measurement, or nobody has proposed it yet. History is deliberately **not**
repeated here — it lives in:

| document | role |
|---|---|
| [`RESEARCH.md`](RESEARCH.md) | the experiment record, organised by problem (§9 is current) |
| [`ROADMAP.md`](ROADMAP.md) | the feature/probe ledger, item by item, with verdicts |
| [`evals/RESULTS.md`](evals/RESULTS.md) | the chronological raw record (P1–P11, NR series, C-1–22; a few NR ids are assigned in paper/DRAFT.md's appendix) |
| [`archive/`](archive/) | superseded plans, kept verbatim (MVP plan, original sketch, retrieval designs) |

## Where the system stands (2026-08-17)

The measured configuration (`rr init --measured`) ships HyDE dense discovery, hybrid
fusion, the Haiku actionability gate, and the fine-scale logprob rescore. Published
headline: **mean net@2 +5.72** on the 25-repo benchmark against the agentic baseline's
+1.56 (paired +4.16, sign p = 0.0004), precision 0.892; independent draws of the same
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

### 1. OpenScholar datastore as a second HyDE corpus — NEXT UP

The one surviving build candidate from the self-scan digest review
(`reports/papers-vs-the-ledger.html`), and the only open item that is a discovery-channel
change aimed at the named coverage hole. Every prior non-arXiv attempt failed through
**keyword APIs** (NR-27, NR-31/32, NR-34, C-9 — the refutations hit the query channel,
not the corpus); dense HyDE over OpenScholar's ~45M open-access papers is the untried
variant that uses the mechanism that worked. Absorbs ROADMAP item 10's goal for the
off-arXiv repos.

- **Stage-1 probe, $0:** (a) membership — are the 5–6 never-reached gold targets in the
  released datastore at all (plus, secondarily, the one IACR paper NR-27 got judged
  actionable)?
  (b) reproducibility — can a locally computed vector reproduce a stored one
  bit-identically, the Design 2 fifth-dependency check the shipped index had to pass?
- **Kill:** fewer than 4/6 targets present, or the encoder is not reproducible.
- **Then:** rank probe on the 48-target gold set; live Tier B only after both.
- **Honest prior:** stage-1 wins are 0-for-4 on profile changes but 1-for-1 on discovery
  channels; this is the reference class that has paid before. Build cost if it survives is
  large (index scale, encoder verification), which is what the probe is for.

### 2. LitSearch as a recall regression gauge for the dense index — instrument, one afternoon

597 gold-labelled queries (arXiv:2407.18940) over recent ML/NLP — squarely inside the
index's coverage. The binary-quantized index has bit-identical encoder verification but
**no recall-fidelity gauge**: nothing today would notice if binarisation, column pruning,
or one bad yearly shard silently cost 15 points of recall. Embed the queries, freeze
recall@5/@20, wire as an exit-nonzero gate after `rr sync-index`. Explicitly **not** a
net@2 claim — researcher questions are a different register from repo→paper.

### 3. $0 hygiene, run when convenient

- **Judge-contamination re-analysis** (from LitLLMs): stratify the already-cached judge
  verdicts by paper publication date vs judge-model cutoff; test whether actionability
  rate or GPT-5.5/Sonnet agreement (P7 data) shifts post-cutoff. Re-analysis of stored
  data, no new protocol.
- **P6 adoption cross-check** — score repo-side channels against the 31 git-history-mined
  adoptions, the project's only model-free ground truth. Its consumer weakened when item
  16 closed; run it only if a new repo-side proposal appears, and run it *before* that
  proposal's Tier B.

### 4. OpenAlex-Topic community match for ordering the gate-admitted band — $0 probe, weak prior

The one open idea downstream of the gate, which is where the four-null record says any
remaining headroom must be. From "Topic Is Not Agenda" (arXiv:2605.07158), reduced to its
cheapest testable form: OpenAlex Topic IDs as a free community proxy on the 602-paper
labelled set. **Bar: AUC ≥ 0.65** within the score-2 band (the NR-21 metadata family sits
at 0.585; the finescale incumbent at 0.841). No graph build under any circumstance — the
S2 truncation wall (§3.5 correction) and six no-bibliography repos price it out. Pairs
with ROADMAP item 12's unshipped Topics work if it ever passes.

### 5. Product work, judged on demand rather than evidence

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

### 6. Held — real gaps with no affordable next step

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
