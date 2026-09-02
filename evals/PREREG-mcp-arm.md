# Pre-registration — Opus 5 with RepoRadar attached [P27]

Written and committed **before any augmented run is billed**, for the same reason NR-56's
was: the interesting outcomes here are all defensible after the fact, and a bar chosen
after seeing the number is not a bar. Nothing below may be edited once a `web+rr` row
exists; a change of mind gets a new section with a date.

## The question

PLANS 3b measured RepoRadar **or** Opus 5 (paired +1.08 over 37, CI [−0.97, +3.16]).
Nobody has measured **both** — the configuration a user would actually run, with
RepoRadar's MCP server attached to an agent that keeps its own web search.

## The three arms

| arm | model | prompt | turns | tools | source |
|---|---|---|---|---|---|
| **A** RepoRadar | — | — | — | — | frozen `…234024Z.json` (arXiv+EPMC) |
| **A′** RepoRadar as shipped | — | — | — | — | A minus the papers the product mutes |
| **B** Opus 5 | `claude-opus-5` | v2 | 30 | WebSearch, WebFetch | `gold_spread_v2_opus5.json` draw 1 |
| **C** Opus 5 + RepoRadar | `claude-opus-5` | v2 | 30 | + 4 read-only `rr mcp` tools | this arm |

**C is B with one thing changed.** Same model, same byte-identical v2 prompt, same turn
cap, same auth (`RR_EVAL_CLI_AUTH=subscription`), same web tools, same judge, same cohort.
The MCP server is the only difference, and the prompt does not mention it.

**A′ is what C's tool serves.** The product mutes papers the repository's own README,
CITATION file or bibliography already cites; the benchmark harness has no such rule. That
is 11 of A's 325 picks (3.4%) across 8 cases — 8 judged actionable, 3 not, so the eight
`+1`s and the three `−2`s very nearly cancel: **A − A′ = +0.05/case, CI [−0.14, +0.24]**
over the 37 (`evals/mcp_arm.json`). Small, and worth naming rather than assuming, because
it was assumed first — an estimate of +0.22 from counting only the actionable side was
wrong by a factor of four. A is reported too, because A is the published number.

Sanity check on the reader, run before any of this was written: `mcp_arm_report.py`
reproduces P26's published A − B as **+1.08, CI [−0.97, +3.22]** from an independent code
path. A′ − B is +1.03, CI [−1.05, +3.19] — the correction flips nothing.

## Predictions, in the order I would bet on them

1. **C ≥ B.** The agent is given a gate-passed shortlist and loses nothing. If C < B by
   more than noise, the tool is actively misleading the agent, which is the most
   informative outcome available and the one I would least expect.
2. **C's gain over B is concentrated on the over-answering cases.** P26 found the whole of
   A's margin on 5 of 37 cases where Opus 5 answers into a repository it should have
   abstained on; on the other 32 the two are level (−0.06). RepoRadar's edge is
   **abstention discipline, not discovery**. So the mechanism available to C is *checking
   its own picks against a shortlist*, and if C helps anywhere it should help there.
3. **C ≤ A′ + B's unique finds.** C cannot exceed the union of what both arms can see.

## Decision rule — registered before the first row

Paired over the 37 cases, C − B, with `bigram_report.paired_bootstrap` and
`band_testbeds.sign_test` — the project's own estimators, not new ones.

| outcome | reading |
|---|---|
| C − B ≥ **+1.50** and the CI excludes zero | **the augmented arm is the product claim.** Attaching RepoRadar to an agent beats the agent. |
| CI includes zero | **not separated.** Reported as such, and not resolved toward whichever arm looks better. |
| C − B ≤ **−1.50** and the CI excludes zero | **the shortlist hurts.** Reported as prominently as a win, with the transcripts. |

**+1.50 is not arbitrary:** A − B is +1.08 with a half-width of ~2.10 over these 37 cases,
so a bar below the existing arm's own effect size would let C "win" at a level A already
failed to reach. The interval, not the point, decides.

## Kill conditions

- **The treatment was absent.** If more than 3 of 37 `ok` rows record **zero** MCP calls,
  the sweep is not a measurement of RepoRadar-plus-agent and is reported as a
  discoverability result instead. Tool use is read off the server's own call log
  (`RR_MCP_CALL_LOG`), per run, never inferred from the answer.
- **More than 3 rows fail to produce an `ok` baseline** — `gold_spread`'s existing rule.
  A throttled row is *unasked*, not a failure, and is retried.
- **Any judge field reaches the store.** `judge_score` and `judge_justification` are the
  answer key. If `rr_mcp_arm.verify_case` reports a leak on any case, no row from that
  sweep is usable. Checked before the first run and pinned by
  `tests/test_rr_mcp_arm.py`.

## What this cannot show

- **One draw.** C-7: a single draw's level is not a property of the method, and B is
  itself draw 1 of three. C is compared against the same draw B is published at.
- **The judge is the same judge NR-56 could not show discriminates adoption.** Every
  figure here inherits that limitation; the arm does not escape it by being new.
- **A is frozen and C is live.** A's gate verdicts are from 2026-08-27. That is what makes
  C's input identical to A's output, and it also means A cannot benefit from anything
  learned since.

## Cost

Notional, on the subscription: B's 37-case draw recorded **$351.40** (`cost_usd`, which
under subscription auth is what the tokens would have cost on the API, not money spent).
C should land in the same range plus the tool calls. The 2026-08-27 sweep exhausted the
subscription 21 runs in, so this is expected to need more than one sitting;
`gold_spread.py` is resumable and records `throttled` as unasked.

---

# Addendum, 2026-09-01 — the wide-corpus arm (C-wide)

Written and committed **after C ran on the 12 scientific cases and before any C-wide row is
billed.** Nothing above this line is edited; that is the rule the original section set for
itself, and this is the "change of mind gets a new section with a date" branch.

## What C measured, and the reason to doubt it

C − B = **−1.42** over the scientific 12, CI [−4.75, +1.42], 5W/5L/2T. C was **more
precise** (0.907 vs 0.891) and returned **a quarter fewer papers** (8.1 vs 10.8 per case).

The doubt is in the call log. **48 of C's 87 MCP calls were `search_papers`** — more than
any other tool — against a store holding only that case's digest picks: 9 papers on
`mat-featurize`, 12 on `mat-chgpot`, 14 on `mat-mlip`. The product's `search_papers` covers
everything RepoRadar ever fetched, which on these cases is **724, 718 and 1252**. So the
agent reached for breadth 48 times and got back, each time, a handful of papers it had
already been handed.

Two readings fit that equally well and they call for opposite conclusions:

- **anchoring** — the shortlist made the agent answer at digest size, and the search tool is
  incidental. Then C's −1.42 is a real measurement of what attaching RepoRadar does.
- **starvation** — the search tool kept coming up empty, RepoRadar looked exhausted, and the
  agent stopped. Then C's −1.42 is an artifact of how I seeded the store, and the arm is a
  floor rather than a measurement.

## The arm

**C-wide is C with one thing changed: the corpus.** The whole frozen pool goes into the
`papers` table; `paper_scores` and `paper_llm_scores` keep **exactly the picks they had in
C**. `get_ranked_papers` reads the scored run and `search_papers` reads the corpus, so:

- `get_ranked_papers` is **byte-identical** between the two stores — not argued, *proved*
  per case by `rr_mcp_arm.compare_stores`, which serialises both payloads and compares
  bytes. If it ever returns `identical: False` the arm is measuring two changes and its
  number means nothing.
- `search_papers` goes from ~12 papers to ~700–1250. Measured ratio on `mat-featurize`:
  **80×**.

Same 12 cases, same model, prompt, turn cap, auth and judge. It also stays a real product
state rather than a contrivance: in a live install `get_all_papers` spans every run ever
made while `get_scores_for_run` covers only the latest.

## Prediction, before the first row

**I expect C-wide ≈ C** — that the narrowing is anchoring, not starvation. The reason is
that the agent never lost WebSearch or WebFetch: an empty RepoRadar search costs it nothing
it could not get from the open web three seconds later, so starvation would require it to
have been *substituting* RepoRadar's index for web search. That is a strong assumption
about a tool it had just met.

I would rather be wrong here, because the other branch is the one where C's number was my
own artifact.

## Decision rule — registered before the first row

Paired over the same 12 cases, **C-wide − C**, with `bigram_report.paired_bootstrap` and
`band_testbeds.sign_test`.

| outcome | reading |
|---|---|
| C-wide − C ≥ **+1.42** and the CI excludes zero | **starvation.** The corpus explains the narrowing; C was handicapped by my seeding and C-wide supersedes it as the measurement of the product. |
| CI includes zero | **anchoring.** C's −1.42 stands as a measurement of the shortlist, and the corpus was not what made the agent narrow. |
| C-wide − C ≤ **−1.42** and the CI excludes zero | **a wider corpus makes it worse.** Reported as prominently, and the transcripts go in the artifact. |

**+1.42 is not arbitrary:** it is exactly the gap C-wide would have to close to make C − B
vanish. A bar below it would let "the corpus mattered" be declared on a difference too
small to explain the thing it was invoked for.

## Secondary, computed at $0 from the rows

**How many of C-wide's picks are in the frozen pool but outside the digest picks?** That is
the direct evidence of whether the wider corpus fed the agent anything it could not have
had in C — a number, not an inference from the headline. If it is ~0, "anchoring" is not
just the surviving hypothesis but the demonstrated one.

## Kill conditions

- **`compare_stores` reports any case not identical** — the arm is void, no row is usable.
- **More than 3 of 12 `ok` rows record zero MCP calls** — as before.
- **`search_papers` call count collapses to near zero** — then the treatment was never
  exercised, and the comparison says nothing about corpora. Recorded per run either way.

## Cost

12 cases at C's measured rate (~$7.2/case) ≈ **$86 notional**, one sitting.
