# Pre-registration — rung 1, the second-judge validity gate [NR-52]

**Written and committed 2026-08-31, BEFORE any margin was computed under any label.** The
whole point of this rung is that it decides whether to keep spending; a bar chosen after
seeing the answer would decide nothing.

## Why this runs before anything else

Every direction on the ladder in `evals/RESEARCH-net2-directions.md` operates in the judge's
least reliable region. GPT–Sonnet kappa on the n = 324 band is **0.199**. The 80 gate-withheld
band papers are **74% actionable per GPT** (+0.21/paper expected value, above net@2's 2/3
break-even) and **26% under a consensus label** (−1.21/paper, far below it). The *sign* of
every rescue idea's expected value flips depending on which judge you believe, and 33 of the
34 false positives we would try to remove are judge-score-1 boundary papers.

So before buying any product change, ask whether the published **+0.54/case** margin over
Opus 5 is a property of RepoRadar or a property of GPT-5.5.

## Arms and instrument

* **Control**: shipped RepoRadar arm, `judge-gpt-5.5-frozenpool-bigrams_verified-wemb1.5-`
  `20260830T034455Z.json`, with `bio-mdtraj` from its `20260830T075622Z` repair. 306 shown
  papers over 37 cases.
* **Comparator**: Opus 5 draw 1, `evals/gold_spread_v2_opus5.json`, status `ok`, picks
  restricted to those carrying a verdict (hallucinated and unjudgeable picks are **absent**,
  not scored negative — the void-not-null rule). 357 picks over 37 cases.
* **Second judge**: `claude-sonnet-5` through `evals/second_judge.py::second_verdict` —
  byte-identical rubric to the first judge, cached under `.work/second_judge/`, **never** in
  the gold cache.

**Coverage as measured today**, and it corrects the figure the research doc was committed
with. The doc said 557 verdicts from the summary JSONs; the on-disk cache actually holds
**853**, so the doc undercounted what is already bought. Recomputed against the cache:

| arm | papers | already cached | fresh needed |
|---|---|---|---|
| shipped | 306 | 191 (62.4%) | **115** |
| Opus 5 | 357 | 12 (3.4%) | **345** |
| | | | **460 total, ~$5–10** |

The prior cross-judge work was aimed at our own band and digest, which is why the comparator
is almost uncovered.

## The three labels — all registered now, all reported afterwards

| label | rule | question |
|---|---|---|
| **GPT** | GPT ≥ 2 | the published margin, +0.54/case |
| **consensus** | GPT ≥ 2 **and** Sonnet ≥ 1 | does it survive when both judges must agree? |
| **Sonnet-only** | Sonnet ≥ 2 | what is it if we simply swap the judge? |

All three are computed from the same verdict set and **all three are reported regardless of
which flatters the result**. Choosing among them afterwards is the label-shopping this gate
exists to prevent.

## The bar

**On the consensus margin only:**

* **PASS** — within **±0.5 net@2/case** of the GPT margin **with the sign preserved**, AND at
  least **4 of the 6 big science losses** (mat-mlip −14, mat-chgpot −12, cv −9, llminfer −9,
  mat-toolkit −7, numerics −6) persisting at **≥ 50% of their GPT magnitude**.
* **KILL** — a sign flip, or |Δ| > 0.5/case. That kills **all** margin-chasing spend below:
  the bundled paid arm, comparator draw purchases, and the n = 60 expansion. The honest
  terminus is then the judge-relative, cohort-decomposed claim the paper already supports,
  with this cross-check shown.

**The Sonnet-only margin carries no kill condition.** Its absolute level is dominated by one
judge's severity — Sonnet reads 0.537 precision where GPT reads 0.852 on the same 65 shown
papers, so under net@2's −2 penalty **both arms are expected to go negative**. A bar on that
level would be measuring the judge, not the system. It is reported and interpreted as "who
loses less", and its *sign* is the informative part.

**Scope limits, stated in advance.** The ±0.5 read is descriptive gating, not a hypothesis
test: the consensus margin's own standard error is ~1.0/case at n = 37, so this cannot and
does not claim the two margins are statistically equivalent. A pass means **robust among the
judges we have**, never "objective" — a cutoff both models share (NR-43) is undetectable by
any two-judge construction.

## Integrity guards, inherited and non-negotiable

* **Prompt-hash drift check.** `verify_contexts` rebuilds each case's repo context and
  compares `sha256(RUBRIC \0 context)[:12]` against the hash stored with the GPT verdicts. A
  mismatch means the clone moved under the cache, so the stored label answers a question we
  can no longer reconstruct. Such cases are **excluded and named**, not silently compared.
* **Void is not null.** A paper without a Sonnet verdict is excluded from that label's
  arithmetic; it is never scored as non-actionable. Any exclusion is reported per arm.
* **Framing limitation, not removable.** The first judge sends the rubric as an OpenAI system
  message; the second sends one prompt string. That is a real difference between conditions
  and it is a limitation of the comparison, reported as one.
* The second judge's verdicts never enter `evals/cache/judge/`.

## Expected outcome, recorded so the result cannot be re-read

Genuinely uncertain, and that is why it is worth $5–10. Two considerations point opposite ways.

*Toward a pass*: the margin's mechanism is **abstention** — 5 over-answer cases supply 105% of
it, and 4 of those are cases where RepoRadar returns nothing at all. Returning nothing scores
0 under **every** label, so the largest component of the margin is judge-invariant by
construction.

*Toward a kill*: the 32 non-over-answer cases are level at −0.06 under GPT, and Sonnet's much
lower base rate will drive both arms deeply negative there. Opus 5 shows ~9.7 papers/case
against our 8.3, so under a harsh judge the arm that shows **more** loses more — which would
*widen* our margin rather than shrink it. If that happens, the pass is real but for a reason
the GPT margin does not name, and the write-up must say so.
