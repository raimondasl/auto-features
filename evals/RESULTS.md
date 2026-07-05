# Tier B benchmark — results

**Baseline run:** 2026-07-04, `main` @ PR #16 · judge = **GPT-5.5** · baseline = **Opus 4.8**
(`--baseline cli`). **Feature 6 run:** 2026-07-05, `main` @ PR #18 (`--rr-triage`, triage
model **claude-haiku-4-5**). Commands:

```bash
uv run python evals/run_judge_eval.py --baseline cli               # baseline (heuristic 0.5 gate)
uv run python evals/run_judge_eval.py --baseline cli --rr-triage   # Feature 6 (LLM triage gate)
```

Re-run and compare after a ranking change. (LLM-judged runs are non-deterministic; treat
±1 net@2 as noise. Judge verdicts and baselines are cached, so a re-run mostly re-uses them.)

## Results

Metrics defined in [`metrics.py`](metrics.py) / explained in [`README.md`](README.md#tier-b--actionable-improvement-llm-judged).
`net@2 = (#actionable) − 2·(#non-actionable)` over a system's returned papers; precision
is `n/a` when a system abstains. RepoRadar is shown two ways: its **Top Picks** tier
(score ≥ 0.5, the abstention-respecting output) and a **Top-10** diagnostic.

| Case (repo) | RepoRadar Top Picks | RepoRadar Top-10 | Opus baseline |
|---|---|---|---|
| **rag** (stanford-futuredata/ColBERT) | abstained | 2/10 · prec 0.20 · net **−14** | 5/5 · prec 1.00 · net **+5** |
| **cv** (facebookresearch/detectron2) | 0/2 · net −4 | 1/10 · prec 0.10 · net **−17** | 3/3 · prec 1.00 · net **+3** |
| **rl** (DLR-RM/stable-baselines3) | 0/10 · net **−20** | 0/10 · net −20 | 3/3 · prec 1.00 · net **+3** |
| **webdev** (pallets/flask) — negative control | 0/10 · net −20 | 0/10 · net −20 | *arxiv_unverified* (1 arXiv lookup failed) |
| **mean** (rag/cv/rl, verifiable) | — | precision **0.10** · net **−17** | precision **1.00** · net **+3.7** |

## Findings

1. **Opus wins decisively on every verifiable case** (mean net@2 **+3.7 vs −17**, precision **1.00 vs 0.10**). RepoRadar's keyword+recency ranking surfaces topic-adjacent but non-*actionable* papers.

2. **The decisive nuance — `recent-only net@2 = 0.0` for the baseline on all three cases.** Every Opus pick was older/seminal (`recent=0/N`). Restricted to RepoRadar's 90-day fetch window, Opus recommends nothing actionable either. So:
   - **RepoRadar's fixable problem is precision, not discovery.** It confidently ranks 8–10 non-actionable papers into its top tier. Filtering those (LLM triage → correct abstention) targets **net@2 from −17 → ~0**.
   - **Opus's headline win is largely a paper-age artifact** — it cites timeless foundational work RepoRadar structurally can't see. Closing *that* gap needs a scope change (a seed/foundational-corpus mode), not a ranking fix.

3. **The 0.5 "Top Picks" threshold is miscalibrated.** On `rl` and `webdev`, RepoRadar put **all 10** candidates above 0.5 with **0 actionable** — overconfident, not conservative.

4. **The negative control fails in the actionable sense.** For a web framework, RepoRadar confidently returns 10 ML papers, none useful. (Tier A domain-sanity "PASS" doesn't survive an actionability judge under the live category fallback.)

5. **Honesty guards fired in production.** On `webdev`, one arXiv lookup failed and the harness printed `arxiv_unverified` and emitted **no baseline metrics** — no fabricated result. (PR #16.)

## Feature 6 result — LLM triage gating Top Picks (2026-07-05)

`--rr-triage` replaces the miscalibrated 0.5 "Top Picks" threshold with an LLM
actionability gate (triage model **claude-haiku-4-5**): a paper is a Top Pick only if
the LLM judges it genuinely applicable (score ≥ 2). This changes RepoRadar's
**user-facing returned set**; the raw **Top-10** diagnostic (ungated ranking) is
unchanged by design — triage gates, it does not rerank.

**RepoRadar Top Picks — net@2, before vs after:**

| Case | Before (0.5 gate) | After (LLM triage) |
|---|---|---|
| **rag** | abstained · net 0 | **1/1 actionable · precision 1.00 · net +1.0** |
| **cv** | 0/2 · net **−4** | **abstained · net 0.0** |
| **rl** | 0/10 · net **−20** | **abstained · net 0.0** |
| **webdev** (negative control) | 0/10 · net **−20** | **abstained · net 0.0** |
| **mean** | **−11.0** | **+0.25** |

**What moved:**

1. **User-facing Top Picks: mean net@2 −11.0 → +0.25.** The gate eliminated the
   catastrophic false-positive dumps — `rl` and `webdev` went from **−20** (all 10
   confidently returned, 0 actionable) to a **correct abstention**.
2. **Calibration fixed.** The three repos with nothing genuinely applicable in-window now
   **abstain** instead of returning junk; precision on what RepoRadar *does* return is **1.00**.
3. **The negative control passes.** `webdev` (Flask) now returns nothing — as it should.
4. **Cost:** the judge pool is unchanged, so GPT-5.5 verdicts and the Opus baselines are all
   cache hits — the only new spend is ~10 Haiku triage calls per case (pennies).

**Caveats / next levers:**

- **Recall cost (Haiku is conservative).** On `rag` the judge found 2 actionable papers in
  the Top-10 but triage surfaced 1; on `cv` the judge found 1 but triage abstained. The gate
  never emits a false positive, but it leaves some defensible papers in "Maybe". Levers: a
  stronger triage model (`--rr-triage-model claude-sonnet-5`) or `triage.min_actionable: 1`.
- **Top-10 is unchanged** (−14/−17/−20) because triage only gates the returned set. Reranking
  the Top-10 by `llm_score` (the deferred half of Feature 6) would also lift the diagnostic.
- **The baseline still returns more actionable papers**, but all older/seminal (`recent=0/N`) —
  the foundational-corpus gap, which is a scope change, not a ranking fix.
