# Artifact: code, benchmark, scripts and registrations

This is the anonymised code artifact for the paper. It holds the pipeline's source, the benchmark
definition, the evaluation scripts behind every figure in Sections 3 to 7, the tracked results
those scripts write, the registrations of the studies the paper reports, and their tests. The run
files, frozen candidate pools, judge verdicts and adoption labels that are too large or too
third-party to keep in git are in the separate data bundle, laid out so that its paths drop into
this tree unchanged.

It is a curated snapshot of a development repository, not the whole of it. Development notes,
plans and the running lab record are left out, as are the scripts and data of studies the paper
does not report. Some docstrings still point to those notes; the pointers are historical.

The pipeline's name is replaced by `anonymous` everywhere, in file names, code and data alike.
The package therefore imports as `anonymous`, and the data bundle's field names match the
code's. A checksum recorded inside a file was computed before the replacement, so it may not
match the file as shipped.

## Layout

- `src/anonymous/`: the pipeline. `finescale.py` holds the two-parameter calibration map, the only
  fitted numbers in the rescore stage.
- `evals/benchmark.yaml`, `evals/gold_targets.json`, `evals/witness_set.json`: the benchmark.
- `evals/*.py`, `evals/frame/*.py`: evaluation scripts. Each says in its docstring what it
  computes and how to run it.
- `evals/*.json`, `evals/frame/pool/`: tracked results and the registered adoption-walk record.
- `evals/PREREG-*.md`, `evals/registrations/`: the registrations. TIMELINE.md gives their times.
- `tests/`: the test suite. `uv run pytest tests -q` runs offline; `tests/conftest.py` blocks the
  network and any real API key.

## Where each part of the paper comes from

| paper | script | tracked result | data bundle |
|---|---|---|---|
| Section 3: benchmark, Table 1 | `freeze_gold_targets.py`, `witness_set.py` | `benchmark.yaml`, `gold_targets.json`, `witness_set.json` | `cache/baseline/`, `cache/judge/`, `.work/pool-cohort3/`, `.work/pool-wemb/`, `.work/adoptions.json` |
| Section 3: labels, identifier re-judging, self-agreement | `rung1_second_judge.py`, `sonnet_id_probe.py`, `sonnet_self_agreement.py` | `rung1_second_judge.json`, `sonnet_id_probe.json`, `sonnet_self_agreement.json` | `.work/second_judge/`, `cache/judge/`, `.work/sonnet_id_probe/` |
| Section 3: kappa and its ceiling | `comparison_sensitivity.py` | `comparison_sensitivity.json` | Testbed A's run file |
| Section 3: adoption check, Table 2 | `frame/walk_pool.py`, `judge_validity_pool.py`, `judge_validity_adoption.py` | `frame/pool/`, `judge_validity_adoption.json` | adoption payloads and verdicts, `.work/crossrepo_analysis.json`, T0 contexts |
| Section 3 and 7: third judge | `third_judge.py`, `third_judge_followups.py` | `third_judge.json`, `third_judge_followups.json` | `.work/third_judge/`, `.work/second_judge/`, `cache/judge/` |
| Section 4: channels and Table 3 | `diagnose_query_generation.py`, `hyde_replication.py`, `diagnose_citation_hop.py`, `hop_reach.py` | none | `.work/hyde_*`, `.work/hop_pool/` |
| Section 4: widening the cut | `freeze_hyde_cut_arm.py`, `hyde_cut_reach.py` | `hyde_cut_arm.json`, `hyde_cut_reach.json` | `.work/pool-cut1000/`, run files of 2026-08-30 |
| Section 4: random-pool base rate | `label_pool.py` | none | `.work/label_pool.json` |
| Section 5: gate, ungated run | `run_judge_eval.py`, `band_testbeds.py` | none | run files of 2026-08-07, 08-14 and 09-08 |
| Section 6: Table 4, E1 to E5 | `exp_select.py`, `exp_finescale.py`, `exp_ensemble.py`, `exp_pairwise.py`, `exp_features.py`, `compare_finescale_baseline.py` | none | `.work/exp/`, run files of 2026-08-07 |
| Section 6: 37-repository bands, second scorer | `judge_dependence.py`, `finescale_current_gate.py`, `finescale_model_transfer.py` | `judge_dependence.json`, `finescale_current_gate.json`, `finescale_model_transfer.json` | `.work/second_judge/`, `.work/exp/`, run files of 2026-08-20 and 09-08 |
| Section 6.1: the map and its audit | `calibrate_finescale.py` | none | `.work/calibration*` |
| Section 7: Table 5, penalty sweep | `rung1_second_judge.py`, `comparison_sensitivity.py`, `third_judge_followups.py` | the three JSON files | `.work/second_judge/`, `cache/judge/` |
| Section 7: Fig. 1 | `judge_dependence.py` | `judge_dependence.json` | `.work/second_judge/`, `.work/second_judge_band.json` |
| Section 7: cost | `measure_cost.py` | `cost_measured.json`, `gold_spread_v2_opus5.json` (draw 1 `cost_usd`) | none |

Table 1's domains: machine learning 10 (rag, cv, rl, peft, diffusion, graph, speech, llminfer,
ann, thin-gnn), other software 15 (the three negative controls cli, http and webdev, and columnar,
compiler, crypto, db, encryption, linter, numerics, storage, systems, vectordb, thin-kv, thin-lang),
biology 6 and materials science 6 (the `bio-*` and `mat-*` cases). thin-gnn carries a non-ML flag
in `benchmark.yaml`, which its own description contradicts; the paper counts it as ML.

## Kept only because reported code needs it

These files belong to studies the paper does not report. They are here, unedited, because a
reported script imports them or a kept test exercises them.

- `rr_mcp_arm.py`: imported at module level by `gold_spread.py`.
- `ablation_report.py`, `noise_floor.py`: imported by `bigram_report.py`, which is kept as the
  home of `paired_bootstrap`, the interval every margin in the paper uses.
- `why_case.py`, `finescale_domains.py`: supply constants to `judge_dependence.py`.
- `gate_swap_second_judge.py`: its `pool_meta` feeds the second-judge stage of the second-scorer
  study.
- `fetch_wants.py`: imported, inside a function, by `diagnose_triage.py`.
- `gap_match.py`: imported by `extend_vs_improve.py`, below.
- `run_eval.py`, `audit_product_divergence.py`, `openalex_yield.py`, `s2_yield.py`,
  `synth_seeds.py`, `extend_vs_improve.py`, `thin_docs_detector.py`: exercised by guard tests of
  reported code.
- `gold_spread_v2.json`, `gold_spread_v2_opus5_web_rr.json`, `gold_spread_v2_opus5_web_rrwide.json`,
  `turn_budget_probe.json`: inputs `witness_set.py` reads. The witness set's 785 includes papers
  from these draws; without them it rebuilds to 744.

## Caveats a careful reader will meet

- The `baseline` field in the run files is an earlier comparator, Claude Opus 4.8. The paper's
  comparator is Claude Opus 5, in `gold_spread_v2_opus5.json` draw 1. Some run files show the
  pipeline far ahead of Opus 4.8; the paper does not report that comparator.
- `gold_spread_v2_opus5.json` also holds draws 2 and 3 as 50 stub rows that never ran (HTTP 429).
  Only draw 1 is reported. Its header still says 25 cases; draw 1 covers 37.
- On 13 of 37 baseline rows `num_turns` exceeds `max_turns`, because it counts something the cap
  does not bound.
- `.work/exp/finescale_haiku10_a.json` labels its model gpt-4o-mini in `summary.model`; its
  `arm: haiku` field is right.
- In the reported E1 Sonnet run, peft has 6 of 15 valid shuffles. Counting every failed shuffle as
  selecting everything bounds the result at AUC 0.651 and +1.32 net@2, still a fail.
- `cost_measured.json`'s `published_claim` field records an older cost claim that the measurement
  refuted. Its measured figure is the one the paper uses. The HyDE line is measured separately.
- `hyde_cut_reach.json` was frozen before `witness_set.json` was last regenerated. A rerun moves its
  pooled reach from 0.1654 to 0.1648 and its reach at 1,000 from 0.4481 to 0.4464. The paper quotes
  neither.
- `frame/pool/datasheet.json` records file hashes of CRLF checkouts; the files in git are LF.
- `benchmark.yaml` says in a comment that 8 of the original 12 cases are ML. The domain table
  above is the one the paper uses.

## Running

`uv sync --all-extras` installs the pipeline with the evaluation and HyDE dependencies. Unzip the
data bundle at the root of this tree with a tool that keeps file times, such as `unzip` or 7-Zip,
because one check reads them. The bundle's `DATA-README.md` describes its contents. Every script
that buys verdicts or model calls refuses to run without its API key.

The following were rerun from this tree and the data bundle alone, with the network blocked and no
key set, and each reproduced its tracked result or its recorded figures exactly:

- `freeze_gold_targets.py` and `--check`, `witness_set.py` and `--check`
- `comparison_sensitivity.py`, `sonnet_self_agreement.py`, `third_judge_followups.py`
- `judge_validity_adoption.py`, and `frame/walk_pool.py` resumed with its recorded arguments and
  `--no-verify-pulse` (its `curve` list is per invocation)
- `freeze_hyde_cut_arm.py`, `hyde_replication.py --report`, `label_pool.py --report`
- `compare_finescale_baseline.py`, `exp_finescale.py --arm haiku --samples 10`,
  `exp_select.py --model claude-haiku-4-5`, `exp_ensemble.py`, `exp_pairwise.py --testbed b
  --no-anchors`, `exp_features.py` and `--combined`
- `judge_dependence.py`, `calibrate_finescale.py --analyse` (pass `--out`, or it overwrites the
  bundle's copy), `opus5_funnel.py`, `freeze_opus5_arm.py`

These also need the 37 benchmark repositories cloned under `evals/.work/<case>` at the commits in
the bundle's `REPOSITORIES.csv`: `rung1_second_judge.py`, `sonnet_id_probe.py --report`,
`hop_reach.py`, `exp_select.py --model claude-sonnet-5`, `exp_pairwise.py --testbed a` and
`finescale_model_transfer.py --check` and `--report`. With the original clones each reproduced
exactly. Their context check rebuilds each repository's context and compares it with the hash
stored in the verdicts. The rebuild depends on the platform and on the clone's `.git` contents, so a
fresh clone may be reported as drifted. The bundle's `evals/.work/repo_contexts/` holds each context
as the judge last saw it, and its `index.json` gives the rule for checking the hashes directly.

These cannot be rerun offline:

- `third_judge.py --report` needs the abstracts of the baseline's 120 DOI picks, which are
  publisher text and not in the bundle.
- `exp_finescale.py`'s default OpenAI arm and `finescale_current_gate.py` create their API client
  before they read their caches, so they stop without an OpenAI key. Their caches hold every
  result they report.
- `judge_validity_pool.py --datasheet` and `--xrepo-analyse` verify their seeds against the NIST
  randomness beacon, and the second also queries arXiv. The cross-repository figures of Section 3
  are in the bundle's `evals/.work/crossrepo_analysis.json`.
- `measure_cost.py` measures live token use. `diagnose_query_generation.py` and
  `diagnose_citation_hop.py` query arXiv and Semantic Scholar, and their docstrings record their
  figures.
- `compare_finescale_baseline.py --testbed a300` stops on a failed baseline row. The paper does not
  report that comparison.
