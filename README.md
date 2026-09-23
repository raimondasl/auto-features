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

## Layout

- `src/reporadar/`: the pipeline. `finescale.py` holds the two-parameter calibration map, the only
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
| Section 3: benchmark, Table 1 | `freeze_gold_targets.py`, `witness_set.py` | `benchmark.yaml`, `gold_targets.json`, `witness_set.json` | none |
| Section 3: labels, identifier re-judging, self-agreement | `rung1_second_judge.py`, `sonnet_id_probe.py`, `sonnet_self_agreement.py` | `rung1_second_judge.json`, `sonnet_id_probe.json`, `sonnet_self_agreement.json` | `.work/second_judge/` |
| Section 3: kappa and its ceiling | `comparison_sensitivity.py` | `comparison_sensitivity.json` | none |
| Section 3: adoption check, Table 2 | `frame/walk_pool.py`, `judge_validity_pool.py`, `judge_validity_adoption.py` | `frame/pool/`, `judge_validity_adoption.json` | adoption payloads, T0 contexts |
| Section 3 and 7: third judge | `third_judge.py`, `third_judge_followups.py` | `third_judge.json`, `third_judge_followups.json` | `.work/third_judge/` |
| Section 4: channels and Table 3 | `diagnose_query_generation.py`, `hyde_replication.py`, `diagnose_citation_hop.py`, `hop_reach.py` | none | `.work/hyde_*`, `.work/hop_pool/` |
| Section 4: widening the cut | `freeze_hyde_cut_arm.py`, `hyde_cut_reach.py` | `hyde_cut_arm.json`, `hyde_cut_reach.json` | run files of 2026-08-30 |
| Section 4: random-pool base rate | `label_pool.py` | none | `.work/label_pool.json` |
| Section 5: gate, ungated run | `run_judge_eval.py`, `band_testbeds.py` | none | run files of 2026-08-07, 08-14 and 09-08 |
| Section 6: Table 4, E1 to E5 | `exp_select.py`, `exp_finescale.py`, `exp_ensemble.py`, `exp_pairwise.py`, `exp_features.py`, `compare_finescale_baseline.py` | none | `.work/exp/`, run files of 2026-08-07 |
| Section 6: 37-repository bands, second scorer | `judge_dependence.py`, `finescale_current_gate.py`, `finescale_model_transfer.py` | `judge_dependence.json`, `finescale_current_gate.json`, `finescale_model_transfer.json` | none |
| Section 6.1: the map and its audit | `calibrate_finescale.py` | none | `.work/calibration*` |
| Section 7: Table 5, penalty sweep | `rung1_second_judge.py`, `comparison_sensitivity.py`, `third_judge_followups.py` | the three JSON files | none |
| Section 7: Fig. 1 | `judge_dependence.py` | `judge_dependence.json` | none |
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
- `opus5_funnel.py` and `freeze_opus5_arm.py` read run files and a pool that are not shipped. Their
  frozen outputs, `opus5_funnel.json` and `opus5_arm.json`, carry the figures.
- `frame/pool/datasheet.json` records file hashes of CRLF checkouts; the files in git are LF.
- `benchmark.yaml` says in a comment that 8 of the original 12 cases are ML. The domain table
  above is the one the paper uses.

## Running

`uv sync --all-extras` installs the pipeline with the evaluation and HyDE dependencies. Every
script that buys verdicts or model calls refuses to run without its API key. Apart from the
exceptions listed above, every figure in the paper can be recomputed offline from this tree and
the data bundle.
