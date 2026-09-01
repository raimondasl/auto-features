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
