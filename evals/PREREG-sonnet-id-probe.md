# Did the identifier line move the second judge's verdicts on the baseline's picks?

Committed before any verdict in this probe exists. Registered 2026-09-22.

## Why ask

NR-52 compared the shipped arm with Opus 5 under two judges. Under GPT-5.5 the margin was +0.32
net@2 per case. Under Sonnet it was -3.41. The two judges were asked about the same papers with a
byte-identical rubric, but not with byte-identical papers.

The prompt carries an identifier line, `arXiv: <id>`. Our arm's papers come from the frozen pool
with versioned ids (`2406.17968v1`) under both judges. The baseline's picks were resolved through
`verify.resolve_references`, which returns versioned ids. `gold_spread.judge_row` passed that
record straight to GPT-5.5, so GPT-5.5 saw versioned ids for both arms. `rung1_second_judge.judge`
rebuilt the record as `{**resolved, "arxiv_id": <pick id>}`, and the pick id is unversioned
(`2404.14989`). So under Sonnet, and only under Sonnet, the identifier line differs by arm.
`judgeable_item`'s own docstring calls a version suffix a perfect, deterministic arm marker.

Post hoc, from files on disk, Sonnet calls 187 of the baseline's 237 arXiv picks actionable
(0.789), 68 of its 120 DOI picks (0.567), and 179 of our 306 (0.585). The arXiv picks, the only
papers whose id line differed, stand out. That is also a comparison between kinds of paper, so it
does not isolate the id line. Among papers GPT-5.5 scored exactly 2 the rates are 0.731, 0.615
and 0.449, so there the DOI picks keep a gap over ours with no id line in play. Whether the id
line moves verdicts is measurable for about $8, and the Sonnet reading of NR-52 should not be
relied on until it is.

## What this can settle, and what it cannot

It can settle whether the version suffix changes Sonnet's verdicts on these papers, and by how
much that moves NR-52's Sonnet-only margin.

It cannot settle which judge is right, and it says nothing about GPT-5.5's verdicts. It does not
test the DOI picks, whose identifier line is the same under both recipes.

## Population

Opus 5 draw 1, as `rung1_second_judge.opus5_arm()` loads it: 357 judged picks over 37 cases. The
237 of them with an arXiv id are the population. Every one of the 237 has a Sonnet verdict in
`.work/second_judge/claude-sonnet-5` from NR-52 and a versioned-id verdict file in the GPT-5.5
gold cache (`cache/judge/v1/gpt-5.5/<case>/<versioned id>.json`).

The versioned id each paper is shown with is the one in the gold cache, which is the id GPT-5.5
saw. One pick has two versioned files, `db/2607.11271` as v2 and v3. The newer file, v3, is used.

## The instrument

Two arms, same model, same everything except the identifier line:

- Arm V: `arXiv: <versioned id>`, the recipe GPT-5.5 received.
- Arm U: `arXiv: <unversioned pick id>`, the recipe NR-52 gave Sonnet. This is a fresh draw, not
  the NR-52 verdict, so V and U differ only in the id line and not in the draw.

Model `claude-sonnet-5` through `second_judge.second_verdict`, unchanged: the same rubric bytes,
the same prompt assembly, and no temperature field in the request, as in NR-52. The client would
otherwise send temperature 0 first and drop it only after the API refuses it, so the script
registers the model as one that takes no temperature before its first call. The
repository context is `assemble_repo_context` of each case clone, checked against the stored
prompt hash by `second_judge.verify_contexts`. If any case's clone has drifted, `--judge` refuses
to start. Excluding a case would also break the reproduction of NR-52's margins required below.

Title and abstract come from one resolution per paper through `verify.resolve_references`, done
once, stored, and reused byte-identically by both arms. Only the `arxiv_id` field differs. A paper
whose stored resolution has no abstract is void in both arms and is not called, and `--judge`
refuses while any paper is unresolved.

Two differences from NR-52's requests are the same in both arms, so they cannot move E1. The
request splits the prompt into two text blocks at the candidate-paper marker for prompt caching,
where NR-52 sent one string. And 11 of the 237 NR-52 verdicts were drawn on 2026-08-06 by an
earlier probe and reused from cache by NR-52, one of them with an HTML-escaped abstract. Both
matter only to E3, which compares against NR-52's draws.

Verdicts are cached outside the gold cache and outside NR-52's namespace:
`.work/second_judge/claude-sonnet-5#id-versioned` and `.work/second_judge/claude-sonnet-5#id-unversioned`.
The NR-52 verdicts are read, never rewritten.

The two arms' calls for one paper are made back to back, with the order alternating by paper, so
time of day and any drift in the served model fall evenly on both arms.

## Endpoints

Actionable means score 2 or more, as in NR-52's `sonnet_only` label.

Every endpoint uses only papers scored in both arms. A paper void in either arm leaves V and U
alike, so the two arms are always compared on the same papers.

- E1, primary. The difference in actionable rate, V minus U. Its interval is a paired case
  bootstrap: 10,000 draws, seed 20260922, resampling all 37 cases with replacement (the three
  with no arXiv pick included), both arms on each draw, and the 95% percentile interval taken as
  the 250th and 9,750th of the sorted draws, counting from 0. The discordant counts (actionable
  in V only, in U only) are reported beside it.
- E2. NR-52's Sonnet-only margin recomputed twice, with the baseline's arXiv verdicts taken from V
  and then from U. The DOI picks and our arm keep their NR-52 verdicts. Each margin carries its
  interval from NR-52's own paired bootstrap, `bigram_report.paired_bootstrap`, so a replacement
  figure arrives with an interval fixed now. The shift, margin(V) minus margin(U), carries the E1
  bootstrap's interval over the 37 cases. Before either arm is read, the script recomputes NR-52's
  margins from NR-52's verdicts and refuses to report unless they give -3.41 and +0.57.
- E3, descriptive, and it does not enter the reading. U against the NR-52 verdicts on the same
  papers: the flip rate across the actionable threshold (score 2), on all 237 and on the 226 that
  NR-52 drew itself. NR-53 measured 8.4% on a different sample.
- E4, descriptive. The consensus label (GPT-5.5 >= 2 and Sonnet >= 1) recomputed the same way as
  E2.

## Reading, fixed now

- Unreadable: more than 5% of either arm is void, or the two arms' void rates differ by more than
  3 points. A void is a paper with no verdict after the retries below. Voids are reported by arm
  and never scored.
- Material: the E1 interval excludes zero. The id line moved Sonnet's verdicts. NR-52's
  Sonnet-only and consensus margins are then re-reported under arm V, the recipe that matches
  GPT-5.5's, with E2's intervals, in a correction entry, and that figure replaces -3.41 wherever
  it is quoted.
- Immaterial: the E1 interval includes zero. NR-52's Sonnet reading stands, the asymmetry is
  recorded as tested, and E2 is reported beside it as the size of any residual effect.

A shift in E2 alone does not make the reading Material. Checked before commit by simulation on
these 237 papers: with no id effect and each arm flipping at NR-53's replicate rate, the E2 shift
has a standard deviation of about 0.36 net@2 per case and reaches 0.5 in about 14% of runs, while
the E1 interval excludes zero in about 4%. A threshold on E2 would mostly measure sampling. No
other reading is available after the data.

## Prediction, recorded before any call

E1: V minus U is -0.04, with a range of -0.10 to +0.02, and its interval includes zero, so the
reading is Immaterial. I put that reading at about 0.6, not higher. E2: the shift is between -0.5
and +0.5. E3: a flip rate between 5% and 12%.

The reasoning: a version suffix carries almost no information about whether a method is worth
adopting, and among GPT-5.5's 2s the DOI picks keep a gap over our arm with no id line in play.
Against it: over all picks the arXiv picks are the outlier, and they are exactly the papers whose
id line differed. If the reading is Material, the first argument was wrong and the entry will say
so.

## Operation

- Retries: transport errors are first retried inside `llm_client.complete`, as in NR-52. A call
  to `second_verdict` that still fails is retried up to twice for that paper and arm. A paper
  still without a verdict is void in that arm only, and for every endpoint it is dropped from
  both arms.
- Spend: 474 verdicts are planned, about $8 at NR-52's cost of about $8 for 460 Sonnet verdicts.
  The run refuses to make a 601st call to `second_verdict`, retries included.
- Outages: the run checks for an API key before its first call and makes no call without one. A
  failure that is not about the paper (a missing key, an authentication or credit refusal) stops
  the run and is not charged to the paper's retries. If 20 consecutive calls fail, the run stops;
  those 20 failures are treated as an outage, count toward the 600 calls, and are refunded to the
  papers' retries. The run then resumes later from its caches.
- The case list is recorded at the start of `--judge`, which refuses on any drift. `--report`
  uses the recorded list and refuses if it has changed.
- The script is `evals/sonnet_id_probe.py`. `--plan` makes no paid call and prints the population,
  the ids for each arm, the drift check and the resolution status. `--judge` buys the verdicts.
  `--report` writes `evals/sonnet_id_probe.json` with every row and is pinned by
  `tests/test_sonnet_id_probe.py`.

## What each reading changes

Immaterial: nothing in the product, and NR-52 gains a pointer to this entry. Material: a
correction entry, a pointer above NR-52, and every later use of the Sonnet margin takes arm V's.
Unreadable: reported as such, NR-52's Sonnet reading stays flagged as untested, and no voided
verdict is bought again under this registration.
Any third judge added later judges the baseline's arXiv picks with versioned ids, the recipe both
earlier judges would then share.
