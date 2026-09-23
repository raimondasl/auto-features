# Does the fine-scale rescore behave the same on Azure's gpt-4.1-mini?

Committed before any gpt-4.1-mini score or new second-judge verdict exists. Registered 2026-09-21.

## Why ask

On Azure OpenAI the fine-scale rescore runs on whatever deployment the user names. It cannot run on
the model it was built for. The probability map in `src/anonymous/finescale.py` (`SLOPE`,
`INTERCEPT`) was fitted on gpt-4o-mini's outputs. Azure refused a new gpt-4o-mini deployment on the
test subscription, because the model is being retired. So an Azure user runs the stage on a
different model, through a map fitted to another one. `rr doctor`, the `setup_repo` result and the
plugin README all call it uncalibrated on Azure. Nobody has measured what that means.

gpt-4.1-mini is the logprob-capable non-reasoning model verified on the test resource (PLANS item
17). It returns 20 logprob alternatives there and accepts the rescore's request once
`reasoning_effort` is dropped. It retires on Azure on 2027-04-14, so this result is useful for about
seven months.

This registration went through four adversarial review rounds before commit. They changed which band
is measured, which comparators are used, how calibration is tested, what can be read about the
stage's value, and how every failure and stop is handled.

## What this can settle, and what it cannot

It can settle whether gpt-4.1-mini on Azure ranks and admits band papers like gpt-4o-mini on OpenAI.
That is a contrast between two scorers on the same papers and prompt.

It cannot settle whether the stage is worth running. Computed at registration from files on disk,
with band net counting the admitted papers, +1 for each actionable one and -2 for each that is not:

| band | judge | papers counted | stage (gpt-4o-mini, frozen map) | show every counted paper | show none |
|---|---|---|---|---|---|
| L, Luna gate | GPT-5.5 | all 315 | +83 | +66 | 0 |
| L, Luna gate | GPT-5.5 | the 212 with a Sonnet verdict | +82 | +86 | 0 |
| L, Luna gate | Sonnet | the same 212 | -134 | -190 | 0 |
| H, Haiku gate | GPT-5.5 | all 328 | +162 | +163 | 0 |
| H, Haiku gate | GPT-5.5 | the 284 with a Sonnet verdict | +161 | +158 | 0 |
| H, Haiku gate | Sonnet | the same 284 | -97 | -163 | 0 |

Compare cells only within one set of papers. The papers with a Sonnet verdict are not a random
subset. Earlier experiments bought verdicts mostly for papers the stage admits, so the uncovered
papers are mostly ones it rejects: 93 of band L's 103 and 43 of band H's 44. That choice of subset
alone moves band L's GPT-5.5 comparison from the stage ahead by 17 to show-all ahead by 4.

Under GPT-5.5 the stage and show-all are close, and both beat showing none. Under Sonnet, showing
none beats both, and that holds whatever the missing verdicts turn out to be. Even if every missing
paper were actionable, band L's Sonnet cells would stay at or below -124 (stage) and -87 (show-all),
and band H's at or below -96 and -119. So against showing nothing, the sign of the stage's value is
set by how strict the judge is. Value is therefore reported under both judges, with no bar and no
product change. That question belongs to the benchmark, not to an Azure registration.

## Population

Two score-2 bands, because the band is a property of the gate model (C-36). They share 165 papers,
so they are not independent replications and are never pooled.

Band L is primary. A keyless Azure install gates papers on the user's own deployment, and band L is
the closest measured stand-in. It is not that install: it was gated on OpenAI, where gpt-5.6-luna
refuses temperature 0 and Azure accepts it, and a user's gate deployment may be another model.

- Run `judge-gpt-5.5-frozenpool-bigrams_verified-wemb1.5-20260908T163150Z.json`. Its
  `ranking_config` names `rr_gate_provider` openai, `rr_gate_model` gpt-5.6-luna, `rr_gate_effort`
  none.
- 315 papers, 35 cases, 232 actionable under GPT-5.5 (base rate 0.737).
- Control `evals/finescale_current_gate_luna.json`. It was written at `withheld` as
  `evals/finescale_current_gate.json` and moved unchanged at `withheld` (C-36). All 315 are scored.
  Control AUC 0.6749, 95% interval [0.594, 0.747] under the recipe below. NR-64 reported [0.596,
  0.748] with a different seed; the bars use this one. The frozen map admits 173.

Band H is secondary and carries no bars. It is the band of the shipped two-key configuration.

- Run `judge-gpt-5.5-frozenpool-bigrams_verified-wemb1.5-20260908T063132Z.json`. Its
  `ranking_config` names `rr_gate_provider` claude and leaves `rr_gate_model` empty. The benchmark
  then gates with `pool_config.rr_triage_model`, which is claude-haiku-4-5 (`run_judge_eval.py`:
  `args.rr_gate_model or args.rr_triage_model`).
- 328 papers, 34 cases, 273 actionable (base rate 0.832).
- Control `evals/finescale_current_gate.json` (C-36). Control AUC 0.7257. The frozen map admits 231.

Fingerprints, each sha256 over UTF-8 bytes. Lines are sorted by (case, id), id is the unversioned
id, and every line ends in `\n`. Each is computed over the band as loaded by
`finescale_current_gate.load_band` from its run file and `evals/.work/pool-cut100`.

| fingerprint | recipe | band L | band H |
|---|---|---|---|
| population | `case/id` | `973a571d23708b631069264bb29636adb3e78ba5c68a25eebd121d8b4f9d03c7` | `698bd2caeee0a5803d00b63c9b76ab183f1fba1b872cae9cccff9163ca531222` |
| repository side | `case\n` + `band_testbeds.repo_block(case)` + `\n`, per case in sorted order | `798213b74f23f22d24f27d37202ab0cfa770e4ad6f440ed513e17e5cb2026209` | `d393f0e37c37b6083302af88e492af309121b4f64efb57dc844ccae4d9c02483` |
| prompt | `case/id\n` + the scale prompt + `\n`, where the scale prompt is `exp_finescale.SCALE_PROMPT` filled with `repo_block(case)`, the run file's title and the pool abstract cut at 1500 characters | `72336bb88a75c678dff5566c55bdc3011a385bb8df50da9b413b3e5c7a3d49fe` | `ea26abef5a86b2355f0e2b99257128e99ef53f0e2bcc828f8439b1f3f53eb84d` |
| GPT-5.5 label | `case/id/judge`, judge being the run file's score | `70ad98d8656e8c02d5b3b78ea2c8d72ca9b87353941d5e8100a03fa6d4968898` | `233a34d2b26e14f14b2e415f22248a90b15a5e309a86b9f7b8b8cdf3eee88509` |
| existing Sonnet verdicts | `case/id/score`, for each paper with a cached verdict under its versioned id | `5f5b198048fabe3159ff60cafc71ed54c8bf0048293ca3d9ed3bf2f7c2a49316` (212) | `03f6130115fffd311fdd6907e86ad0adca08c48192bebf8a88e217a60b7ab0d2` (284) |

The run files, the pool, the repository clones and the Sonnet cache are gitignored. From
registration on, these fingerprints fix them. They cannot show that the controls saw the same text,
because the control caches kept scores and not prompts. That rests on file times: every file the
prompts and labels read predates both controls, which were scored on 2026-09-20. No file on the
prompt path has changed since either control was written: `evals/band_testbeds.py`,
`evals/exp_finescale.py`, `src/anonymous/profiler.py`, `src/anonymous/triage.py`,
`src/anonymous/finescale.py`, and `load_band`, whose only change since was which run it reads.

## Labels

GPT-5.5 gives the primary labels, read from the run file: actionable means score >= 2.

Sonnet (`claude-sonnet-5`) is the second judge: actionable means Sonnet score >= 2. Verdicts come
from `evals/second_judge.py`'s `second_verdict`, unchanged, cached under `.work/second_judge/`. They
are looked up by the run file's versioned arXiv id only, which is the id each GPT-5.5 verdict was
cached under. Twenty-two band papers also have a verdict under the unversioned id, drawn
independently. Those are not used.

Before any gpt-4.1-mini call:

0. On the first pass, before any Sonnet purchase, the existing-verdict fingerprint is recomputed
   with the recipe in the fingerprint table, over the cache as it then stands. It must equal the
   registered hash and count: 212 for band L and 284 for band H. The artifact records that list of
   (case, versioned id). A later pass recomputes the recipe over the recorded list only, and it must
   again equal the registered hash.
1. `second_judge.verify_contexts` runs over every case in both bands. A drifted case is left out of
   every Sonnet analysis, and the count is reported. None had drifted at registration.
2. Each band paper without a verdict under its versioned id is judged with `second_verdict(case,
   contexts[case], {**pool_record, "arxiv_id": versioned_id}, "claude-sonnet-5")`. Here
   `pool_record` is the `pool-cut100` candidate for that paper, as `gate_swap_second_judge.py` does
   it. That is 124 distinct papers: 103 in band L, 44 in band H. The labels do not depend on any
   arm, so this spend sees no treatment.
3. A purchase that raises is retried up to 3 more times. A paper still without a verdict is void
   under Sonnet. It is left out of both arms of every Sonnet figure and is never counted as
   non-actionable. Void counts and ids are reported per band. Voids would not be random, because the
   missing papers are mostly ones the control rejected. So if more than 5 band L papers are void,
   E1's Sonnet reading is unresolved whatever its interval says. Whatever the count, the Sonnet
   Delta AUC is also reported with every void paper set actionable and then non-actionable, with no
   bar.
4. After step 3, each band's full Sonnet label set is fingerprinted and written into the artifact
   before the first gpt-4.1-mini call. Steps 2 and 3 run on the first pass only. A later pass buys
   no verdicts and must reproduce that recorded fingerprint.

In steps 0 and 4, the first pass is every start of the script until step 4's label-set fingerprint
has been written to the artifact. Every start after that is a later pass, including a resume after a
resumable stop. Step 0's list is written to the artifact before the first Sonnet purchase. A
first-pass start that finds that list already recorded computes step 0 over the recorded list only,
and in step 2 it buys only the verdicts still missing. These terms are unrelated to the scoring pass
of the Retest paragraph and to the later and resume passes under Failures.

The consensus label (GPT-5.5 >= 2 and Sonnet >= 1) is not used. On every band paper that has both
verdicts it equals the GPT label, so it would check nothing.

## Treatment and control

The control is what an OpenAI user runs: gpt-4o-mini on OpenAI's API, already scored for every paper
in both bands and not re-run.

The treatment is what an Azure user runs: the `gpt-4.1-mini` deployment on the test resource,
through the shipped transport. The provider is part of the treatment on purpose. The question is
whether an Azure user's rescore behaves like an OpenAI user's. Read with `az` at registration
(2026-09-21), the deployment runs model gpt-4.1-mini, version 2025-04-14, GlobalStandard, at 50K
tokens and 50 requests a minute. These are read again and recorded at run start.

The request. The prompt is the control's: the same code over the same inputs, fixed by the prompt
fingerprint. The script builds each prompt string once, checks that list against the fingerprint,
and sends that same list. It calls `llm_client.top_logprobs(prompt, cfg, top_k=20)` with a config
that sets `provider: azure_openai`, `reasoning_effort: "none"` as the `setup_repo` Azure template
writes, and an empty `redact` list. The script wraps `llm_client._post_adaptive`, which
`top_logprobs` calls, so no second implementation of the request exists. The script sends one
request at a time: at 50 requests a minute a second thread adds contention and no speed. For each
call the wrapper stores the request body as last sent, and the raw response when there is one. It
reads the body in a `finally` clause, which catches it after the transport has edited it in place,
whether `_post_adaptive` returns or raises. The capture is held until the row is classified,
including when `top_logprobs` raised after `_post_adaptive` returned. A 400 content filter leaves no
response, only the exception message, which the row stores beside the body. A scored, no-digit or
content-filtered row then keeps its capture. An error row discards it, even when `_post_adaptive`
returned an HTTP 200. The request-validity rule applies to every scored, no-digit and
content-filtered row. The model-identity rule applies to every row that kept a captured HTTP 200
response.

A request body is valid only if all of these hold:

- `messages` is `[{"role": "user", "content": P}]`, where P is that paper's registered prompt;
- `temperature` is 0, `max_tokens` is 4, `logprobs` is true and `top_logprobs` is 20;
- no key is present apart from those, `model`, and optionally `reasoning_effort` "none".

The only permitted adaptation is dropping `reasoning_effort`, recorded per row. An invalid body
stops the run, because a different request was sent.

Each row stores the full returned token sequence, every generated token's 20 alternatives, the
response's `model` field, the number of alternatives on the first token, and two readings:

- the control's parser, `exp_finescale._digit_expectation`, which reads the first token that is a
  digit, anywhere in the output;
- the product's parser, `finescale.digit_expectation`, applied to the first token's alternatives.

Retest. Band L is scored a second time on the same deployment, into a separate cache that the first
pass never reads. It measures the treatment's own noise. Mean absolute exp09 difference, Spearman
correlation and frozen-map admission flips are reported. The reference is gpt-4o-mini's own
replicate on the 165 papers both controls scored with identical prompts: mean absolute difference
0.071, Spearman 0.993, and 4 of 165 admissions flipped (2.4%).

Failures. Each paper ends in one of four states:

- Scored. The parser in question reads a digit.
- No digit. `_post_adaptive` returned an HTTP 200 answer, and the product's parser finds no digit
  among its first-token alternatives. An empty answer counts here too: a choice whose finish_reason
  is not `content_filter` and whose message content is empty, so that `top_logprobs` raised
  "returned no logprobs". An empty answer is identified from the capture, and neither parser scores
  it. Every no-digit row is cached, counts toward the no-digit stop, and is counted as attempted in
  `enough_scored`. The product-parser reading leaves it unscored and does not admit it. The
  control-parser reading scores it when `exp_finescale._digit_expectation` returns a value over the
  captured tokens, and otherwise leaves it unscored and does not admit it.
- Content-filtered. The transport reports Azure's content filter, either as a 400 `content_filter`
  or as an HTTP 200 with finish_reason `content_filter`. The row keeps the recorded body and is
  cached, never asked again. It counts toward neither stop rule. Every endpoint treats it as the
  product does: unscored, not admitted, and counted as attempted in `enough_scored`.
- Error. Any other exception that `top_logprobs` raises, whether or not it is an `LLMError`: for
  example `http.client.IncompleteRead` from a truncated body, `UnicodeDecodeError`, or an
  `AttributeError` or `TypeError` from a malformed response, which the transport passes through
  without retrying. A 200 with no choices, or with a non-empty answer and no logprobs, is also an
  error: the latter is the configuration failure the transport names, not the model answering. The
  type and message are recorded. The control's scorer caught every exception the same way and cached
  none of them. An error is retried by the script up to 5 times with backoff, on top of the
  transport's own attempts. An error row is not cached, so a later pass asks for it again. Up to
  three resume passes run before analysis. A paper still in error after them is dropped from both
  arms of every endpoint and left out of its case's attempted count. The control never cached a
  failed call, so charging an error to the treatment alone would be noise the control never paid
  for.

Rate limits are not a state. `LLMRateLimited` is caught before any other error. The script waits for
the time the refusal names, or 30 seconds if none, at most 10 minutes per wait, and asks again. A
paper refused 6 times in a row stops the run.

`LLMUnavailable` stops the run at once, as it stops the stage in the product. If its cause is an
HTTP response (a 401, 403 or 404 from Azure, or a refused `reasoning_effort` value), the stop is
final. Otherwise it came from getting a token through `az`, which says nothing about the model.

Two stops are resumable: the rate-limit stop, and an `LLMUnavailable` with no HTTP cause. After
either, the run must be resumed from its caches, no sooner than 10 minutes later, up to 3 times. A
resume continues the pass that stopped and adds no resume passes. Cached rows, and with them the
no-digit counts, carry over. A fourth resumable stop is final. Every stop and resume is reported
with its time and cause.

New script `evals/finescale_model_transfer.py`. Caches under
`evals/.work/exp/finescale_model_transfer/`. Artifact `evals/finescale_model_transfer.json`, with
per-paper rows for both bands and the retest.

## Blocking checks

Each must pass before any gpt-4.1-mini call. If one fails, the run stops and the failure is the
report.

1. Each band, as loaded, has its registered size and matches its population, prompt and GPT-5.5
   label fingerprints. Each loaded label equals the `judge` of the same row in the tracked control
   artifact.
2. Each band's repository blocks match their fingerprint.
3. Each control artifact's `summary.run` equals its band's registered run. The gate is resolved from
   every entry of that run the way `run_judge_eval.py` resolved it: the provider is
   `ranking_config.rr_gate_provider`, and the model is `ranking_config.rr_gate_model`, or
   `pool_config.rr_triage_model` when that is empty. Band L must resolve to openai and gpt-5.6-luna.
   Band H must resolve to claude and claude-haiku-4-5, and must equal that artifact's
   `summary.gate`. Reading `pool_config` alone does not satisfy this, because it names
   claude-haiku-4-5 in both runs (C-36).
4. The control AUCs recompute to 0.6749 (band L) and 0.7257 (band H) from the artifact's `exp09` and
   the loaded labels.
5. The Sonnet steps above have run, with each band's void count recorded. A step 0 mismatch, or a
   later pass that does not reproduce the recorded label-set fingerprint, fails this check. A void
   count alone never stops the run.
6. The treatment config's `redact` list is empty.

## Stop rules during the run

Band L is scored first, and its resume passes run before anything else is sent. Then comes the
retest, then the 163 band H papers not in band L. No-digit counts only rise and are checked as the
run goes, so a no-digit stop that fires early gives the verdict the full band would. A transport
count is taken after that band's resume passes.

- No digit. Band L stops at 16 of its 315 papers, band H at 17 of its 328.
- Transport. Counted after the resume passes. Band L stops at 16 papers still in error, band H at
  17.
- Request. Any invalid request body stops the run.
- Model identity. Every HTTP 200 response kept by a scored, no-digit or content-filtered row is
  checked, in band L, the retest and band H. A 200 on a row that ends as an error is not checked.
  Its `model` field, lower-cased, must equal `gpt-4.1-mini-2025-04-14` or `gpt-4.1-mini`. Any other
  value stops the run, and so does an empty or missing field. The run-start `az` reading must also
  show version 2025-04-14, or the run stops.

Every stop, here and under Failures, is scoped by when it happens, with one exception. A first,
second or third resumable stop is not scoped and is not a band L stop. Wherever it happens, the run
is resumed as described under Failures. A fourth resumable stop is final and is scoped like every
other stop. Up to and including band L's transport count, which is taken after its resume passes, a
stop is a band L stop. After that count, the run ends there: the unfinished retest or band H is
reported as incomplete, no product text changes because of it, and band L's outcome stands. The
retest's counts are reported and never fire a stop.

## Statistics

Every interval is a paired case-clustered bootstrap. Each band gets its own
`random.Random(20260921)` and draws its 4,000 case lists once. A draw picks as many cases as the
band has, with replacement, by `rng.choice` over the sorted case list, and pools the papers of the
picked cases, duplicates kept. Every endpoint, both arms, every comparator and both judges use the
same draws. Inside a draw, papers void under a judge are dropped for that judge, and so are drifted
cases under Sonnet. The interval is the two-sided 95% percentile interval: with k kept draws, the
sorted values at 0-based indices int(0.025k) and int(0.975k) - 1. With k = 4,000 these are the 101st
and 3,900th. AUC is `band_testbeds.auc`, with average ranks for ties. A draw where an AUC is
undefined is skipped for that statistic and counted.

Unscored papers follow the product. A band paper the parser does not score is not admitted. In a
case where `finescale.enough_scored(scored, attempted, 0.5)` is false, the stage does not run for
that case, and all of its band papers are admitted (`pipeline.py`). AUC is computed over the papers
the parser scored. The control scored every paper, so this is the set both arms scored.

Two readings. The control kept only its parser's `exp09`, not its tokens. So every endpoint is
computed twice, once for each reading of the treatment, both against the control's stored values.
The control-parser reading is like for like and is reported first. The product-parser reading is
what an Azure user gets, and mixes parsers across arms. Each reading applies its own set of unscored
papers to AUC, admissions and `enough_scored`. The outcome below is worked out in full under each
reading. A product change is made only if both readings give the same outcome. For this test, O with
over-admission and O with under-admission are different outcomes, and so are W1 and W2. If they
differ, the outcome is U, and both sets of numbers are reported. Numbers quoted in product text come
from the product-parser reading, with the control-parser numbers beside them in the artifact. Rows
where the readings differ by more than 0.01, or where one scores a paper and the other does not, are
counted per band. That includes a digit the control's parser accepts and `int()` rejects, such as a
subscript digit; the control run left such a paper unscored, and so does this one.

## Endpoints and bars

The bars apply to band L. Band H gets the same numbers with no bars. The between-arm Spearman of
exp09 is reported for each band, with no bar.

E1, ordering. Delta AUC = AUC(treatment) - AUC(control). Under each judge separately, a lower bound
of -0.05 or more reads non-inferior, an upper bound below -0.05 reads worse, and anything else reads
unresolved. Then:

| per-judge readings | E1 reads |
|---|---|
| worse under either judge | worse |
| non-inferior under both | non-inferior |
| non-inferior under one, unresolved under the other | split |
| unresolved under both | unresolved |

A split is not a disagreement between the judges. One interval cleared the margin, and the other did
not rule out a larger drop.

The floor. If the treatment's own AUC interval under GPT-5.5 includes 0.5, that reads as "no
demonstrated ordering", whatever E1 says. The control's interval, [0.594, 0.747], excludes 0.5.

E2, threshold location. It needs no labels. In each draw, A = (treatment admissions - control
admissions) / number of drawn papers, under the frozen map.

A is also split into two parts on the same draws. Each part is (treatment admissions - control
admissions) over its papers, divided by the number of drawn papers, so the two parts sum to A. The
threshold part covers the papers the reading's parser scored, in cases where the stage ran. The
failure part covers the rest: the unscored papers, which the treatment does not admit, and every
paper of a case the `enough_scored` fallback admitted whole. E2 takes a reading from the table below
only if the threshold part's interval gives the same reading. Otherwise E2 is unresolved. For each
reading, the counts of no-digit papers, content-filtered papers and fallback cases are reported per
band.

| outcome | reading |
|---|---|
| interval inside [-0.08, +0.08] | the threshold holds |
| lower bound > +0.08 | the frozen map over-admits on gpt-4.1-mini |
| upper bound < -0.08 | the frozen map under-admits on gpt-4.1-mini |
| otherwise | unresolved |

The threshold holds means the frozen map admits within 8% of the band of what it admits on
gpt-4o-mini. For orientation, a uniform shift of +0.48 or -0.34 scale points reaches the margin on
band L. It is tighter below because more papers sit just under the threshold. Extra spread with no
shift also lowers A. The margin is reachable: an identical scorer with gpt-4o-mini's own replicate
noise gives A within [-0.013, +0.006]. The paired mean, median and quartile exp09 shifts are
reported beside A.

E3, band net against the control. Descriptive only, under both judges. Delta = mean over cases of
[band net(treatment) - band net(control)]. It is split into the papers only the treatment admits and
the papers only the control admits, each with its count and actionable rate under both judges. There
is no bar, because at a fixed threshold the sign of a band-net difference from admitting more or
fewer papers is set by the judge's strictness. On band L, shifting the control's scores down
1.25 points reads as a loss under GPT-5.5 (-1.54 per case, [-2.23, -0.83]) and a gain under Sonnet
(+3.11, [+1.94, +4.40]).

E4, value, descriptive only. Band net of the treatment, the control, show-all and show-none, under
both judges, with paired intervals for treatment minus show-all and treatment minus show-none. No
bar and no product change, for the reason at the top.

## What each outcome changes

Readings are applied in this order. First the blocking checks and the band L stops, once. They give
X or S, and if both hold the outcome is X. Then, under each reading, rows W1, W2, U, P and O are
tested in that order, and the first match is that reading's outcome. So the floor overrides a split
or unresolved E1. The two readings are then combined as described under Statistics.

| # | outcome | product change |
|---|---|---|
| X | a blocking check fails; or, up to and including band L's transport count, the transport, request or model-identity stop fires, a final `LLMUnavailable` is raised, or a fourth resumable stop fires | Nothing. The failure is reported, and no endpoint is read from a partial run. |
| S | the band L no-digit stop fires | The Azure caveat gets stronger: gpt-4.1-mini gave no digit, or an empty answer, for N of the M band L papers it answered, so the shipped stage could not score them, and a case with fewer than half its band scored skips the stage. The docs recommend running the rescore on OpenAI's gpt-4o-mini where possible. Whether `setup_repo` should stop enabling the rescore on Azure is proposed in its own PR. |
| W1 | E1 worse | The Azure caveat gets stronger: "Measured on DATE: under <each judge whose own reading is worse under the product-parser reading, a Sonnet reading made unresolved by the void rule not counting>, gpt-4.1-mini ordered band papers more than 0.05 AUC worse than gpt-4o-mini. Delta AUC X [CI] under GPT-5.5 and X' [CI] under Sonnet." The same recommendation and the same separate PR as S. |
| W2 | the floor fires, the control's GPT-5.5 AUC interval on the papers that reading scored excludes 0.5, and E1 is not worse | The Azure caveat gets stronger: "Measured on DATE: under GPT-5.5, gpt-4.1-mini's ordering of band papers was not shown to beat chance (AUC T [CI]), while gpt-4o-mini's was (AUC C [CI], on the same papers). Delta AUC X [CI] under GPT-5.5 and X' [CI] under Sonnet." It never says the ordering was measured to differ from gpt-4o-mini's. The same recommendation and the same separate PR as S. |
| P | E1 non-inferior, and E2 the threshold holds | The Azure caveat becomes conditional, measured wording. It never says "calibrated". |
| O | E1 non-inferior, and E2 over- or under-admits | The Azure note says gpt-4.1-mini ranks band papers like gpt-4o-mini, but it admits N more (or fewer) band papers than gpt-4o-mini: K through the fixed threshold, and F because a paper or a whole case could not be scored. N, K and F are given with the gpt-4o-mini count. No new map. |
| U | E1 split or unresolved, or E2 unresolved with E1 non-inferior, or the two readings disagree, or the floor fires while the control's GPT-5.5 AUC interval on the papers that reading scored includes 0.5 | Nothing changes. The caveat stays, and the numbers are reported. |

The measured wording for P. Anonymous cannot see which model an Azure deployment runs, because the
user names it. So the note is conditional: "If your fine-scale deployment runs gpt-4.1-mini (version
V): measured on DATE, behind a gpt-5.6-luna gate, it ranks and admits band papers like gpt-4o-mini
on OpenAI. Delta AUC X [CI] under GPT-5.5 and X' [CI] under Sonnet; admissions Y [CI], with N band
papers admitted by one arm only. No measurement describes other models. Whether the stage adds value
depends on the judge and is unresolved for every model." V is the version suffix of the response's
`model` field. If no row carries a suffix, V is the version from the run-start `az` reading. A
GPT-5.5 Delta AUC above +0.08 that fails the P7 rule below is quoted with the words "not confirmed
by the second judge". The same wording replaces the caveat everywhere it appears: `rr doctor`, the
`setup_repo` notes and template comments, `config.py`, both READMEs and the paper-discovery skill.
The assertions in `tests/test_azure_openai.py` that check for the caveat change in the same PR.

## Prediction, recorded before running

E1: Delta AUC +0.01 under each judge, range -0.06 to +0.08. Non-inferior under both judges needs a
high between-arm agreement. The only cross-model precedent in the repo is a Spearman of 0.54, and I
expect 0.75 to 0.85 here. Rough odds: non-inferior 0.3, split 0.3, unresolved 0.3, worse 0.1. So
outcome U is the single most likely result.

E2: gpt-4.1-mini's scores sit higher on the scale, mean exp09 up by about 0.5. The frozen map admits
190 to 215 against the control's 173, reading "over-admits" or unresolved.

E3: under GPT-5.5, near zero, because band L's marginal papers are about break-even. Under Sonnet,
negative, because gpt-4.1-mini is predicted to admit more.

Retest: mean absolute exp09 difference at or below 0.1, and at most 4% of admissions flipped,
against gpt-4o-mini's 0.071 and 2.4%.

E4: under GPT-5.5 the treatment scores near show-all. Under Sonnet it scores below show-none, as the
control does.

The P7 rule. A GPT-5.5 Delta AUC above +0.08 would be a surprise, since gpt-4.1-mini and GPT-5.5 are
both OpenAI models. It is believed only if, under Sonnet, it keeps its sign and at least half its
size, the rule P7 used in `second_judge.py`.

## Cost

Azure: 478 distinct band papers plus the 315-paper retest, 793 calls of about 700 input tokens and
`max_tokens` 4. Well under $1 at list prices, from the test subscription's credit, and about 16
minutes at 50 requests a minute. Sonnet: 124 verdicts, about $2 at NR-63's measured rate.
gpt-4.1-mini scores, no-digit responses and content-filter refusals are cached per paper, and so are
Sonnet verdicts. Rate limits and transport errors are not cached, so a re-run asks only for what did
not complete.

## Not in scope

- A refit of the map for gpt-4.1-mini. A map fitted on one of these bands cannot be judged on it. On
  band H, a leave-one-case-out refit of the control's own gpt-4o-mini scores gains 15 band net with
  no change of model. On band L it loses 3. No held-out band exists to check a refit on. If E2 finds
  the threshold moved, the finding is its direction and size, not a new map.
- Whether the stage is worth running. See the table at the top.
- gpt-5.6-luna as the rescore. Azure caps its `top_logprobs` at 5, so it needs its own design.
- The gate. Only the rescore model changes.
- The end-to-end digest net@2. Band net is the stage's direct contribution. The digest also depends
  on the window, which this does not re-run.
