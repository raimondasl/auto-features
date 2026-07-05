# Tier B benchmark — baseline snapshot

**Run:** 2026-07-04, on `main` @ PR #16 · judge = **GPT-5.5** · baseline = **Opus 4.8**
(`--baseline cli`, Claude Code headless + web search). Command:

```bash
uv run python evals/run_judge_eval.py --baseline cli
```

This is the **pre-Feature-6 baseline** — the number to beat. Re-run the same command
after a ranking change and compare. (LLM-judged runs are non-deterministic; treat ±1
net@2 as noise. Judge verdicts are cached, so a re-run only re-judges changed pools.)

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

## Target for Feature 6 (repo-aware LLM triage + threshold recalibration)

- RepoRadar Top-10 net@2: **−17 → ~0** (filter non-actionable papers).
- Top Picks: abstain unless genuinely actionable (fix calibration); `webdev` should return nothing.
- Full parity with the baseline is **not** a ranking goal — it requires a foundational-corpus capability beyond the recency-windowed monitor.
