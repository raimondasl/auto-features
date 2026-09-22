# Where does a third vendor's judge fall between GPT-5.5 and Sonnet?

Committed before any verdict from this judge exists. Registered 2026-09-23.

## Why ask

Two LLM judges score this benchmark. GPT-5.5 is the primary judge and Claude Sonnet 5 the second.
They order papers alike and disagree about level. On the 324-paper score-2 band of the
2026-08-20 session, the fine-scale expectation orders the band at AUC 0.729 against GPT-5.5 and
0.702 against Sonnet, while GPT-5.5 calls 0.873 of the band actionable and Sonnet 0.494 (NR-59,
NR-66). The system comparison inherits that level: NR-52's margin is +0.32 under GPT-5.5 and -3.41
under Sonnet.

Two judges from two vendors cannot show whether that spread is typical of LLM judges or an
accident of these two models. A judge from a third vendor can place itself inside the spread or
outside it. This registration asks where Google's Gemini falls.

## What this can settle, and what it cannot

It can settle where one Gemini model's level and ordering fall relative to the two existing
judges on the same papers and prompts.

It cannot settle which judge is right. It does not pick a judge and does not combine judges. A
majority vote or an average of judges is one more cut-off, not a measurement of the true level.
Adoption (NR-61, NR-62) confirms positives only: a repository not citing a paper does not make the
paper useless to it. So no population here measures how many papers are truly actionable.

One comparison is weaker than the others. 120 of the baseline's 357 picks are DOI picks, and the
prompt text NR-52 sent Sonnet for them was never stored. Their prompts are rebuilt from a fresh
resolution, so they match Sonnet's in construction only, and nothing here can show they match it
byte for byte. E4 reads the baseline arm, so E4 inherits that. `--plan` reports the count per
population.

## Model

`gemini-3.8-flash`, the current stable Gemini model, called through the `generateContent` REST
endpoint (`v1beta`) by `evals/gemini_client.py`, a standard-library client inside `evals/`. The
only current-generation Pro model is a preview, which can be withdrawn at short notice and would
leave the result unrepeatable.

The request sets `thinkingConfig.thinkingLevel = "MEDIUM"`, the model's documented default, since
thinking cannot be disabled on current Gemini models. It sends no temperature, topP, topK or seed:
Google's guidance for this generation is to leave them at their defaults. `maxOutputTokens` is
16,384, which includes thinking tokens. The four adjustable safety categories are set to
`BLOCK_NONE`, because academic abstracts about attacks or security should not be blocked before
they are judged. Each row records the response's `modelVersion`, `finishReason` and token usage.

## The prompt

One user message: `judge.RUBRIC`, a blank line, then `judge._build_user_prompt(context, paper)`,
the same string `second_judge.second_verdict` sends Sonnet. The verdict is parsed by the same rule:
the text from the first `{` to the last `}` is read as JSON, and its `score` must be an integer
from 0 to 3. A boolean or a non-integer score is a parse failure.

- Band and comparison papers use the repository context `assemble_repo_context` of each case clone
  at HEAD, checked against the stored prompt hash by `second_judge.verify_contexts`. `--judge`
  refuses if any case has drifted.
- The baseline's arXiv picks carry the versioned ids GPT-5.5 saw (NR-67), with the stored
  resolutions of `.work/sonnet_id_probe/papers.json`. Its DOI picks are resolved once through
  `verify.resolve_references`, stored, and reused.
- Adoption papers use each repository's T0 context and the item construction of
  `judge_validity_pool.judgeable_item`, exactly as NR-61 and NR-62 asked the other two judges.

Every paper of every population must carry a prompt. A paper with none, because its DOI did not
resolve, because it has no abstract, or because its context drifted, is a hole in a registered
population rather than a void, since no retry rule ever reached it. `--judge` and `--report` both
refuse while any population holds one, naming the population, the count and the causes.

Verdicts are cached under `.work/third_judge/gemini-3.8-flash/`, keyed by the sha256 of the full
prompt, so an identical prompt is asked once.

## Populations

| population | papers | reference judges |
|---|---|---|
| band: the 2026-08-20 score-2 band (`.work/second_judge_band.json`) | 324 | GPT-5.5 0.873, Sonnet 0.494 actionable |
| comparison: NR-52's shown papers, ours 306 and the baseline's 357 | 663 | margin +0.32 (GPT-5.5), -3.41 (Sonnet) |
| adopted: NR-61's adoptions | 188 | GPT-5.5 0.819, Sonnet 0.644 actionable |
| cross-repository controls: NR-62's distinct repository-paper pairs | 502 | GPT-5.5 0.255, Sonnet 0.068 |

NR-61's 752 arXiv-window controls are left out. Both judges call almost none of them actionable
(0.089 and 0.007), so they cannot separate levels, and they would add a third of the cost.

## Endpoints

Actionable means a score of 2 or more. Intervals are percentile intervals from 10,000 bootstrap
draws with seed 20260923 that resample repositories, not papers, taken at the 250th and 9,750th of
the sorted draws counting from 0. The comparison margin is the one exception: it uses NR-52's own
`bigram_report.paired_bootstrap`, which resamples cases under its own seed 20260812, not 20260923.

- E1, level on the band. Gemini's actionable share of the 324, with its interval.
- E2, ordering on the band. The AUC of the fine-scale expectation stored with each band paper
  against Gemini's labels, with its interval. The scorer is gpt-4o-mini, which is none of the judges.
- E3, adoption. Gemini's actionable rate on the 188 adopted papers and on the 502 cross-repository
  controls, each with its interval, and the AUC of Gemini's score for adopted against control.
- E4, the comparison. NR-52's margin, ours minus the baseline's net@2 per repository over 37 cases,
  under a Gemini-at-2 label, with its interval, and each arm's precision under that label.

## Reading, fixed now

For E1 and for each E3 rate, the reference pair is the GPT-5.5 and Sonnet values in the population
table, and the reading comes from Gemini's interval, first match wins:

1. `overlaps both`: the interval contains both reference values.
2. `overlaps GPT-5.5` or `overlaps Sonnet`: the interval contains one reference value.
3. `between`: the interval lies strictly between the two.
4. `above both` or `below both`: the interval lies strictly outside the pair.

E2 reads `orders the band` if its interval lies above 0.5, and `does not order the band` if it does
not. An endpoint whose interval cannot be formed reads `no interval`, which is not a finding either
way. E4 reads `between` if its point estimate lies between -3.41 and +0.32, and `above both` or
`below both` otherwise; its interval is reported beside it and decides nothing. No other reading is
available after the data.

A population with more than 5% void prompts is `Unreadable`, and its endpoints are not read. A void
is a prompt with no verdict after the retries below, including a response truncated at the token
limit, a blocked response and a parse failure. Voids are reported by population and by cause, and
never scored. An endpoint whose population is `Unreadable` is also not scored against the
prediction below: its result is null, and so is the adopted-against-control AUC when either
adoption population is `Unreadable`.

The comparison's rule is per arm, because E4 is a difference of two arms and a void simply leaves
the arm it falls on, moving the margin mechanically. The comparison is `Unreadable` when our arm
or the baseline arm is over 5% void on its own, and also when the two arms' void rates are more
than 3 percentage points apart, however low both are.

Beside the voids, and reading nothing, each population reports the share of its void members the
two existing judges called actionable next to the same share among its scored members.

Before any Gemini verdict is read, `--report` recomputes from the existing labels NR-66's band
figures (0.873 and 0.494, AUC 0.729 and 0.702), NR-52's margins (+0.32 and -3.41) and NR-61/NR-62's
rates for the two existing judges, and refuses unless they reproduce to the precision quoted here.

## Prediction, recorded before any call

- E1: Gemini's band share is about 0.70, within 0.55 to 0.85. Reading: `between`, at about 0.5;
  `overlaps GPT-5.5` at about 0.25; any other reading at about 0.25.
- E2: AUC about 0.70, within 0.62 to 0.78. Reading: `orders the band`, at about 0.9.
- E3: adopted rate about 0.75, within 0.60 to 0.88; cross-repository rate about 0.15, within 0.07
  to 0.28. Reading `between` on each at about 0.45. The adopted-against-control AUC is at least
  0.75.
- E4: margin about -1.5, within -3.41 to +0.32, so the reading is `between`, at about 0.6.
- Voids: under 2% in every population.

The reasoning: a judge from a third vendor has no reason to sit at either extreme, and the band's
level gap is wide enough to hold one. Against it: Gemini thinks by default, which may make it
stricter than both, and its Flash tier may make it more lenient. Either would put it outside.

## Operation

- The key is `GEMINI_API_KEY` in `evals/.env`, sent only in the `x-goog-api-key` header. The run
  checks that a key is present before its first call. The project behind the key must be on a paid
  tier, which the author confirms before the first call and records with the result.
- This registration, `evals/third_judge.py` and `evals/gemini_client.py` are committed together
  before the first call, and `--judge` and `--report` both refuse if any of the three differs from
  HEAD. `--report` records the HEAD sha and each file's blob sha beside the readings.
- Retries: an HTTP 429 waits for the delay the error carries and is retried without limit on count,
  and it is not charged to the paper. A transport error or a 5xx is retried up to three times. A
  parse failure, a truncated answer or a blocked answer is retried once; a 200 whose body carries
  neither a candidate nor a block reason is a parse failure, not a transport one. A prompt still
  without a verdict is void.
- A key or authentication refusal stops the run and is charged to no paper. Twenty consecutive
  failures stop the run as an outage, and those failures are refunded to the prompts' retries.
- A rate-limited call bought nothing, so it does not count against the 3,000-call cap; 429s are
  counted in their own ledger field. Twenty rate-limited calls in a row with no verdict between
  them are a tier or a quota the run cannot wait out, so it stops, resumably, saying so.
- Spend: the run computes spend from each response's `usageMetadata` at the registered prices,
  $0.75 per million input tokens and $3.75 per million output tokens including thinking. It stops
  before a call that would take the total past $40, and it refuses a 3,001st call.
- `--plan` makes no paid call and reports counts, drift, the reproductions and a cost estimate.
  `--judge` buys the verdicts, resumably from the cache. `--report` writes
  `evals/third_judge.json` with every row, pinned by `tests/test_third_judge.py`. `--report`
  refuses for every reason `--judge` refuses except the key, which it does not need.

## What each reading changes

`between` on E1 and on both E3 rates: the two judges bracket a third vendor's judge, and the spread
is recorded as the range to report. `above both` or `below both` on any of them: the spread
understates how far LLM judges differ in level, and the range to report widens to include Gemini.
Any `overlaps` reading: Gemini sits at one judge's level on that population, which is recorded as
such. E2 and E4 are recorded beside them. In every case the result enters RESULTS.md as a numbered
entry, and no product setting changes.
