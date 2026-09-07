# Pre-registration — does the judge condition on the repository?

**Registered 2026-09-06, before any verdict against this negative class is bought and before its seed exists.** A companion to `PREREG-judge-validity-pool.md` [NR-61], not an amendment to it: that study is complete, its result stands whatever this one shows, and nothing here re-analyses it.

## 1. The question, and why NR-61 cannot answer it

NR-61 measured the repository-clustered AUC of both judges' ordinal score over 188 real adoptions against 752 controls matched on arXiv primary category, half-year, and never-cited-at-HEAD. GPT-5.5 reached **0.9215** [0.8945, 0.9429] and Sonnet 5 **0.9424** [0.9160, 0.9641]; both intervals excluded 0.5.

Both overshot their registered brackets of 0.60–0.70 and 0.62–0.72 by a wide margin, and NR-61 reports the likeliest reason as a property of the control class rather than of the judges: inside one category and half-year a random contemporaneous paper is unlikely to be relevant to one *specific* repository, so the judges may be separating **"related to this project at all"** from **"unrelated"** — a far easier question than "worth adopting". P7 corroborated: predicted control base rates of ≥ 0.70 and ≤ 0.55, observed **0.089** and **0.007**, which is what obviously-irrelevant papers look like rather than near-misses.

That leaves the product's actual question untested. RepoRadar's gate never sees random papers; it sees papers **retrieved for this repository**, all topically plausible. So the operative property is not "can the judge tell a relevant paper from an irrelevant one" but **does the judge condition on the repository at all, or does it recognise generally good papers?** A judge that scores every well-written, on-topic paper highly would produce NR-61's 0.92 and be useless in the pipeline.

This study replaces the negative class with papers that are demonstrably relevant to *some* repository, and holds everything else fixed.

## 2. Population and the negative class

**Positives are unchanged.** The same adoptions NR-61 analysed, at the same T0 contexts, judged under the same byte-identical rubric. Their verdicts are already bought and are **reused, not re-purchased** — the prompt is identical because the context digest is identical, which is the condition `buy_verdicts` already enforces before reusing any stored verdict. Nothing about the positive arm is re-drawn, re-mined or re-judged.

**A control for a positive in repository A is an arXiv identifier that:**

1. appears in the HEAD identifier set of some repository **B ≠ A** in the same 60-repository pool;
2. shares the positive's arXiv **primary category** and **half-year window**, the same two keys §4 of the companion registration matches on;
3. is **not** in A's own HEAD identifier set, and is not the positive itself.

Clause 1 is what makes the negative class hard: every control is a paper a real maintainer of a real project chose to reference. Clause 2 keeps the arm marker closed — the arXiv identifier encodes its submission month, so a control drawn from a different window would be distinguishable from a positive by its id alone, which is the defect `assert_arm_neutral` exists to catch. Clause 3 keeps a paper A actually cites out of A's negative class.

**Stated as a limitation, not discovered later: these are papers cited at HEAD, not papers adopted.** A HEAD citation may be background or a tutorial reference rather than an adoption. The stricter class — *adopted* by another repository — was measured first and is **infeasible**: only 45 of 188 positives have four such controls and 77 have none. The claim this study can support is therefore "relevant to some repository", which is weaker than "adopted by some repository" and stronger than "same category and half-year".

**Feasibility, measured 2026-09-06 from artefacts already committed, before this file was written.** No verdict was read to obtain these counts; they are properties of the identifier sets alone.

| | |
|---|---|
| positives with ≥ 1 eligible control | **147** of 188 |
| clusters retained | **50** of 60 |
| controls drawn, cap 4 per positive | **538** (mean 3.66) |
| `controls_per_positive` | `{4: 123, 2: 10, 1: 8, 3: 6}` |
| strata | 121 pool, 26 legacy |
| new judge calls | **1,076** (positives reuse existing verdicts) |

**The 41 positives with no eligible control are excluded, and the exclusion is registered here rather than decided later.** They are dropped because the negative class is empty for them, not for anything about their scores. The analysis set is therefore **147 positives over 50 clusters**, and every endpoint below is computed on exactly that set — including the companion arm it is compared against, so the contrast is paired on identical positives.

**No relaxation to reach a larger n.** Widening to adjacent half-years would recover most of the 41 and open the identifier-date leak clause 2 closes. A smaller n with the arm marker shut is the better trade, and choosing it before seeing any result is what makes it a design rather than an excuse.

## 3. The draw

Four controls per positive where four exist, otherwise all that exist, ordered by `sha256(SEED_XREPO ‖ case ‖ ":" ‖ dedup_id(positive_id) ‖ ":" ‖ dedup_id(control_id))` and taken from the front. Deterministic given the seed, and a function of nothing else.

**`SEED_XREPO` is the NIST Randomness Beacon pulse for `2026-09-07T00:00:00Z`** — a value that does not exist when this file is committed and cannot be chosen by anyone. It is fetched only after this registration is merged, written to `evals/frame/pool/SEED_XREPO`, and verified against the beacon at that timestamp by the same check that guards `SEED_POOL`. A run whose seed does not match the named pulse refuses to start.

A control paper may serve more than one positive, and papers shared across clusters are a correlation the repository-clustered bootstrap does not capture, so the interval is very slightly too narrow. This is reported in the artefact as a count, exactly as the companion study reports it, rather than repaired by dropping rows after the draw.

## 4. Endpoints

**Primary, per judge.** Repository-clustered AUC of the raw 0–3 ordinal score, adopted against cross-repository control, bootstrapped over repositories with 5,000 draws, 95 % percentile interval. Ordinal and level-free by construction, for the reason NR-59 established: the two judges order alike and threshold two orders of magnitude apart, so any thresholded statistic measures the threshold.

**Co-primary, and the number that actually answers §1.** The paired contrast

> **Δ = AUC(category-matched controls) − AUC(cross-repository controls)**

computed for each judge **on the same 147 positives and the same 50 clusters**, resampling repositories jointly so the two arms share their bootstrap draw. Both arms use identical positive verdicts, so Δ isolates the negative class and nothing else. The category-matched arm is recomputed on the 147-positive subset for this purpose and is **not** the 188-positive figure published in NR-61; the two are not interchangeable and the subset figure is reported alongside.

**Secondary, descriptive, no consequence attached.** `P(actionable | cross-repo control)` at each judge's own bar, and the per-stratum split (pool vs legacy).

## 5. Pre-committed consequences

Evaluated per judge from that judge's own intervals. Written before the seed exists.

1. **The cross-repo interval excludes 0.5 and Δ's interval excludes 0.** The judge conditions on the repository, *and* NR-61's 0.92 overstated how strongly. Both are reported: §3.4's caveat becomes a measured quantity rather than an argument, and the paper reports the cross-repository AUC as the figure relevant to the pipeline.
2. **The cross-repo interval includes 0.5.** The judge does not discriminate among papers that are relevant to some repository. NR-61's discrimination is then attributable to topic matching, the gate is not shown to be repository-conditioned, and **that is the headline** — it would mean the product's central selection stage is not doing the thing its design assumes. This branch is written first and plainly because it is the one that costs us something.
3. **The cross-repo interval excludes 0.5 and Δ's interval includes 0.** The negative class's difficulty did not matter. NR-61's explanation for its own bracket miss — the one this study exists to test — is then wrong, and §3.4 is corrected to say so.
4. **Fewer than 10 clusters contribute, or either judge's coverage falls below 1.0.** No endpoint is reported; the shortfall is published against this section. A refusal is not a null, and neither is an incomplete purchase.

No outcome switches the primary judge label, which §6.4 of the companion registration fixed as GPT-5.5 before any of this data existed.

## 6. Predictions

Registered with the track record attached: **the companion study met one of seven**, and every miss ran the same way — it under-predicted discrimination and over-predicted how hard the negative class would be. These brackets are adjusted upward for that bias and are still expected to be wide of the mark.

| id | prediction |
|---|---|
| **X1** | AUC(cross-repo) ∈ [0.72, 0.88] for both judges, point 0.80 |
| **X2** | Δ ∈ [0.04, 0.20] for both judges, point 0.11 — i.e. the harder class costs real discrimination but does not erase it |
| **X3** | Both cross-repo intervals exclude 0.5 |
| **X4** | `P(actionable \| cross-repo control)` rises above the category-matched 0.089 / 0.007 for both judges, and the two judges still differ by at least a factor of 5 |
| **X5** | The pool and legacy strata do not separate: their AUCs differ by < 0.10 for both judges |

## 7. What this cannot show

It does not measure ranking **within** the adopted class, which is what the shipped rescore does; the endpoint remains adopted-vs-control.

It does not use RepoRadar's own candidate pool, and deliberately: §4 of the companion registration rejected that class because a judge harsher on RepoRadar-shaped papers would be credited with validity for a property of the control set. The cross-repository class is produced by other projects' maintainers, not by the system under test.

It cannot separate **"this paper is wrong for this repository"** from **"this paper is a weaker paper"**. Cross-repository controls are relevant somewhere, but they are not matched to the positive on quality, and no available label would let them be.

A null here would not show the judge is worthless — the gate operates downstream of retrieval that has already conditioned on the repository — but it would show the judge is not the stage doing that conditioning.

## 8. Procedure and cost

1. Merge this file. **Then** fetch the pulse and write `SEED_XREPO`.
2. Draw controls under the seed; write the committed URL-free half beside `controls-arxiv-window.json` and the payload to `.work/`.
3. Buy verdicts for the control arm only — **1,076 calls, ≈ $7** at the measured token volumes with prompt caching. Positive verdicts are reused, and a positive whose stored context digest no longer matches is re-bought rather than reused.
4. Compute the endpoints once, after coverage is 1.0 for both judges. No endpoint is inspected before then, enforced by the same gate as the companion study.

## 9. Deviations

Any departure from this file is recorded in it, with the date, what was known at the time, and which direction it moves the result — the standard the companion registration applied to its own target change and its two ledger repairs. A deviation that makes a null harder to fire is the only kind that can be taken without re-registering.
