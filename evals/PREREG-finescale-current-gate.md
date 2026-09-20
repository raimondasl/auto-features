# Does the fine-scale rescore still order the gate's modal band?

Committed before any score is computed. Registered 2026-09-20.

## Why ask

The band ROC-AUC of **0.841** was measured on Testbed A, a 22-case frozen run from 2026-08-07.
Every fine-scale artifact in the project predates 2026-08-09. Since then the gate's score
distribution has moved a long way: the share of admitted papers scoring 3 was 4.5% to 13.6% on
the August diagnostics and is **32.1%** on the 2026-09-08 sweep (149 of 464 admits). The modal
band is still the bulk of the digest at 67.9%, but the population inside it is not the
population the 0.841 describes.

So the ordering result is currently asserted on a configuration the product no longer runs. That
is a gap in the record, and it is cheap to close.

## Question

Does the same scorer, unchanged, order the score-2 band of the current configuration as well as
it ordered Testbed A's?

## Design

One arm. No comparison, no new method. The 2026-09-08 sweep
(`judge-gpt-5.5-frozenpool-bigrams_verified-wemb1.5-20260908T163150Z.json`, 37 cases, frozen
pool `pool-cut100`) supplies the population: **315 papers with a recorded gate score of exactly
2**, each already carrying a GPT-5.5 judge verdict, all 315 joining to the frozen pool for their
abstracts. Band base rate is 232 of 315 actionable, **73.7%**, against Testbed A's 75.0%, so the
discrimination task is similarly balanced.

Scoring reuses `evals/exp_finescale.py` unchanged: same `SCALE_PROMPT`, same `gpt-4o-mini`, same
temperature 0, same top-20 logprobs, same `digit_expectation`. The repository side comes from
`band_testbeds.repo_block`, which delegates to the shipped profile builder. Reusing the original
path is the point. A new prompt would measure a different thing and would not be comparable to
0.841.

Primary endpoint: ROC-AUC of `exp09` against judge >= 2 over the 315 band papers.

## Prediction, recorded before running

**Point estimate 0.78, plausible range 0.70 to 0.85.** Lower than 0.841.

The reasoning: the gate now routes far more of the clearly-actionable papers to a 3, so the
papers left in the band should be more homogeneous and therefore harder to separate. Working
against that, the band's label balance is almost unchanged, which is why the predicted range
still sits well above chance.

## Bars, fixed now

| outcome | reading |
|---|---|
| AUC >= 0.75 | replicates. The ordering claim holds on the current configuration. |
| 0.65 to 0.75 | transfers, weakened. Report both numbers and the drop. |
| AUC < 0.65 | does not transfer. The ordering claim must be scoped to the August configuration, and that scoping is the finding. |

A result above 0.841 would be a surprise and should be treated as one, not as good news. The
first thing to check would be leakage between the scorer and the judge verdicts, since both are
OpenAI models and the judge labels sit in the same file as the gate scores.

## Secondary, reported whatever the primary says

Split by cohort: the 25 legacy cases against the 12 scientific ones. The fine-scale stage's
end-to-end net@2 value already flips sign between cohorts and judges, so its ordering may differ
too. No bar is set on this. It is descriptive at n = 12.

## Cost

630 `gpt-4o-mini` calls, two per paper, well under one dollar. Results cached per paper, so a
re-run is free.
