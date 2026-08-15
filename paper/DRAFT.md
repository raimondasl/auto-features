# From Topic Match to Actionability: Building, Debugging, and Calibrating a Paper-Recommendation System for Code Repositories

**Raimondas L.**¹
¹ *RepoRadar project* — `github.com/raimondasl/auto-features`

*Draft of 2026-08-15. Comments welcome.*

---

## Abstract

We describe RepoRadar, a system that recommends arXiv papers a repository's maintainers should *act on* — a utility judgment, not a topical-relevance one — and the six-week campaign that measured it. Configured as this paper describes, it reaches mean **net@2 +5.42** against a strong agentic baseline's **+1.62** across 25 repositories (paired **+3.79**, 95% CI [+2.17, +5.58], sign *p* = 0.0001) at one-eightieth the cost per repository.

Three findings we believe generalize. **Retrieval for actionability fails on a register mismatch**: a codebase's vocabulary describes what it *has*, while the paper that would improve it describes what it should *adopt*. Every query channel derived from repository text reached 0–8% recall, while the two that route around the mismatch — a citation hop through the repository's own bibliography, and HyDE-style hypothetical abstracts against a dense index — reach 75% in union. **LLM judges of utility are near-binary in practice**: a 0–3 gate put almost every admitted paper on one score, hiding a per-repository precision spread from 0.00 to 1.00. The literature's remedies — listwise selection, pairwise tournaments, ensembles — all failed here with a consistent excess-strictness pathology, while re-asking the same question on a 0–9 scale and reading the *expectation over the answer token's distribution* ordered the hidden band at ROC-AUC 0.84. **Most of what we learned came from instruments, not ideas**: of roughly fifty measured claims, thirty-four are negative results and seventeen are corrections of our own published numbers.

That last category is the paper's spine, not its apology. Silent harness failures — caches outliving their assumptions, truncation correlated with the verdict, partial runs overwriting whole-set artifacts — repeatedly manufactured findings that survived until dedicated tests killed them. The most expensive has a name we now use as a checklist item, **void, not null**: an arm whose channel returned *nothing* scores identically to one that returned nothing *useful*. Two published findings were manufactured that way. Two more were figures measured under one configuration and quoted for another, which is why every number here carries its coordinates (§4.2) — including the one the headline omits: a user who installs the system and changes nothing gets **−8.12**, worse than emitting nothing, because the default enables none of the stages above.

---

## 1 Introduction

A maintainer of a mature open-source project has a discovery problem that ordinary literature search does not solve. The question is not *"which papers are about my topic?"* — for an active field, thousands are — but *"which paper, if I read it on Monday, would change what I build?"* The distinction is between **relevance** and **utility** [1], and it is unforgiving: a recommendation feed that surfaces ten topically-adjacent-but-unactionable papers is worse than one that stays silent, because the reader pays attention for nothing and stops trusting the feed.

RepoRadar is a CLI tool that profiles a repository (README, docs, dependency manifests), queries paper sources (arXiv primarily; Semantic Scholar, OpenAlex, bioRxiv, DBLP optionally), ranks candidates, gates them with an LLM actionability judgment, and emits a three-tier digest whose "Top Picks" section is allowed — encouraged — to be empty. This paper is not primarily a systems description. It is an account of a measurement campaign (2026-07-04 to 2026-08-13, 123 merged pull requests) organized around a single benchmark, in which nearly every architectural belief we started with was tested, and most were refuted. We think the record is more useful than the system: the negative results delimit a design space that others will otherwise re-explore, and the positive result — a distributional rescore that fixes a near-binary LLM gate — is simple, cheap, and, we suspect, applicable to LLM-judged selection pipelines generally.

**Contributions.**

1. **A task formulation and benchmark** for repository-conditioned paper recommendation with an abstention-aware, asymmetric metric (`net@2`), 25 real repositories spanning ML, systems, deliberate negative controls, and a thin-documentation cohort, a pooled LLM judge, and a strong agentic baseline (§3–4). We document the benchmark's own failure modes: at 12 repositories it was measurably overfit (a jackknife moved a headline result from a 70% pool reduction to 11% by dropping one repository), and we describe the expansion that fixed it (§4.5).
2. **A validity program for the LLM judge** rather than an assumption: a second-judge agreement study (κ = 0.51 binary, 0.73 quadratic-weighted; the disagreement is a strictness offset, not a ranking difference) and a model-free ground truth mined from git history — papers a repository *actually adopted* — against which the judge scores 61% actionable versus a 2% random floor (§4.4).
3. **A diagnosis of retrieval for utility** (§5): the register-mismatch account, the failure of every document-derived query channel, and two channels that work by routing around the mismatch — a bibliography-seeded citation hop (44% recall overall; 89% where seeds are plentiful) and blind HyDE against a 3.1M-vector public index (27/48, of which 15 are unreachable by the hop; union 75%).
4. **A diagnosis and repair of LLM utility gating** (§6–7): the gate's score distribution is near-binary, hiding a 0.00–1.00 per-repository precision spread in its modal band; five pre-registered ranking experiments, of which comparative (listwise selection, pairwise Bradley–Terry) and ensemble methods fail while a fine-grained distributional rescore succeeds (band AUC 0.84), converts to a calibrated probability with two fitted parameters, and survives out-of-family judge checks, out-of-run replication, and a live end-to-end run (§8).
5. **An audit of multi-source retrieval** (§9): five channels built to reach the literature arXiv does not carry, four of which had never worked and one of which — measured properly for the first time, at $0 before any judge call — delivers ~175 uncategorised papers per repository without measurably helping. Along the way, a ranker bias we predicted would matter (uncategorised papers escaping the category component) turns out to move 18 of 32 non-arXiv papers into or out of the top-10 while changing the output by **+0.00 net@2**.
6. **A catalogue of 38 negative results and 19 corrections** (§10), and a taxonomy of the silent harness failures that repeatedly manufactured wrong findings, with the defensive patterns that stopped them (§11) — including a $0 static audit that searches for the *one invariant, two implementations* defect on purpose, having twice found it by accident. The audit is also the source of our least comfortable result: run exhaustively over all 79 configuration fields and both files that can set them, it shows that the configuration a new user receives enables none of the stages the rest of this paper measures (§8.9).

Throughout, numbers marked **[NR-n]** refer to the negative-results catalogue in §10, and every measured claim carries the script that re-derives it in the project's `evals/` directory.

---

## 2 Related Work

**Paper recommendation and code–paper linkage.** Scholarly recommenders overwhelmingly target researcher-facing relevance (citation recommendation, related-work search); the specific direction *repository → papers ranked by actionability for that repository* appears essentially unexplored — prior code/paper linkage work such as paper2repo [2] runs the opposite direction (paper → its implementation). Practitioner-relevance studies are directly cautionary: Lo, Nagappan and Zimmermann found engineers' relevance ratings of research essentially uncorrelated with citation counts [3], which our feature experiments reproduce in miniature (§7, E5).

**Retrieval without shared vocabulary.** Our register-mismatch diagnosis is a species of the vocabulary-mismatch problem; the remedy that worked, generating hypothetical documents and searching in embedding space, is HyDE [4], run against a public binary-embedding snapshot of arXiv built with mxbai-embed-large-v1 [5, 6]. Hybrid lexical fusion uses BM25 [7] with reciprocal-rank fusion [8]. SPECTER2 provides citation-informed scientific embeddings [9]; cross-encoder rerankers, and their documented inability to follow *non-relevance* criteria zero-shot, are analyzed in FollowIR [10].

**LLMs as rankers and judges.** Listwise LLM reranking (RankGPT [11], LRL [12]), pairwise ranking prompting [13], setwise selection [14], and tournament schemes [15] all report large gains over pointwise scoring for *relevance*. The study closest to our task, utility judgment over already-relevant candidates [16], finds pointwise utility scoring collapses while listwise-set selection works — a prediction our E1 experiment *fails to reproduce* in the repository setting (§7). Judge reliability and its biases are documented around MT-Bench [17]; comparative judgments' advantages over absolute scores [18] and pairwise-consistency methods [19] motivated our E4. Calibration of LLM confidence spans verbalized probabilities [20, 21], panels and ensembles [22], and escalation cascades [23]. The mechanism our winning method uses — reading a score's token distribution rather than its sample — is G-Eval's probability-weighted scoring [24], refined by TrustJudge's distribution-sensitive scale [25]; the finding that a sufficiently fine pointwise scale matches listwise ranking [26] anticipated our result, and Rank1 [27] uses the same first-token-probability reading for relevance. Permutation self-consistency [28] and position-bias measurements [17, 29] informed the mandatory shuffling/both-order protocols in E1/E4. Anchored pairwise scales [30] and conformal Elo calibration [31] were evaluated as calibration routes and rejected empirically.

**Methodology.** Our pre-registration discipline (success criteria and kill conditions written before any result), leave-one-repository-out cross-fitting, and the ban on per-repository knobs are standard prescriptions against benchmark overfitting; the specific trap we document — selecting a *calibration* by a rank-only criterion (AUC), which is provably blind to the decision threshold — deserves wider attention (§8.3).

---

## 3 Task and System

### 3.1 Task

Given repository *R* (its working tree; no stars, ratings, or telemetry — the cold-start case), return a set of papers *S(R)*, possibly empty, such that each paper "could genuinely improve *this* repository" — the wording of both the system's gate and the benchmark's judge. The empty set is a first-class answer: three of our 22 benchmark repositories are negative controls (Flask, Click, Requests) for which the correct output is nothing, and two more (pyca/cryptography, redis) have literatures centred off-arXiv (IACR; VLDB/OSDI), where near-abstention is correct for an arXiv-fed system. That label deserves a caveat we could only test later: `gold_n: 0` encodes *no gold arXiv papers*, which is a claim about coverage rather than about whether research exists that would improve the repository. Adding a second source and letting the judge decide on merit (§9.5) vindicated the label for Flask and refuted it for Requests, whose digest gained a dependency-bloat tool the judge scored actionable.

### 3.2 Pipeline

The shipped pipeline is a funnel:

```
profile(R) ──► queries ──► collect (~227 papers; range 56–296)
   ──► heuristic rank (keyword/category/recency/embedding + BM25-RRF fusion)
   ──► top-k (50) ──► LLM gate: 0–3 actionability, Haiku      [stage 1]
   ──► papers scoring exactly 2: fine-scale 0–9 rescore,
        expectation over answer-token distribution,
        frozen logistic → P(actionable), show iff P ≥ 2/3      [stage 2]
   ──► top-n (15) ──► three-tier digest (Top Picks / Maybe / Muted)
```

Stage 1 (0–3 gate; "triage") asks whether a paper proposes a method with a concrete implementation path for *this* repository, described by a keyword profile plus the first 300 characters of its README — a budget chosen by experiment (§6.3). Its depth of 50 is also a measured value, and only became one late: it shipped at 15 for most of the campaign, on no measurement at all, and moving it is worth +1.00 net@2 per repository (§8.9). Stage 2 exists because of the central selection finding of this paper (§7) and is detailed in §8. Politeness is engineered, not aspirational: one process-wide arXiv rate gate at the stated 1 request / 3 s ceiling, identifying User-Agent, 30 s minimum backoff on HTTP 429 with a 15-minute patience budget — added after sustained polling earned the development machine a ~70-minute IP block, and after a 429 storm was silently cached as seven empty candidate pools (§11). A rate gate is not a volume gate, and we later paid for the difference: a correct 3-second limiter cannot express *and no more than N per day*, and four sweeps of the same 174-query benchmark in one day were throttled at ~760 requests (§11, lesson 9).

---

## 4 Evaluation Methodology

### 4.1 Metric: asymmetric, abstention-aware

The headline metric is **net@2** over a system's *returned* set: (# genuinely actionable) − 2·(# not), where "genuinely actionable" is a pooled judge's score ≥ 2 on a 0–3 rubric. The asymmetry encodes the product stance that a junk recommendation costs more than a good one earns, and it gives abstention a defined value (0). Two consequences shape everything downstream. First, the expected contribution of a shown paper at true precision *p* is **3p − 2**: showing pays exactly above p = 2/3, which later becomes the system's *derived* — not tuned — display threshold. Second, on an unranked set there is no interior optimum in digest size: truncation scales the total without changing its sign, so "show fewer" is only coherent once a reliable within-set ordering exists (§6.5). We flag the metric's bias plainly: net@2 rewards shyness, so it structurally flatters precision-improving changes (§8.4–8.5) and understates the cost of the recall problem it cannot see.

### 4.2 Every headline in this paper, and the configuration it describes

This table exists because we got the same thing wrong twice. A mean net@2 is meaningless without four coordinates — how many repositories, how wide the digest, how deep the gate, and whether the candidate pool was collected live or reused — and we have twice quoted a figure measured under one setting as though it described another: a noise floor measured at width 10 applied to a width-15 arm **[C-8]**, and a four-repository July mean quoted against the 25-repository benchmark **[C-17]**. Both numbers were real and both were in our own results log. Neither belonged where we put it. The defence is not vigilance, it is a single place where every headline carries its coordinates.

| mean net@2 | repos | digest | gate | pool | date | what it is |
|---|---|---|---|---|---|---|
| **−11.0** | 4 | 10 | none | live | 07-05 | pre-gate heuristic digest — **superseded, see −8.12** [C-17] |
| +0.25 | 4 | 10 | 15 | live | 07-05 | the gate, first measured (§6.1) |
| +1.86 | 22 | 10 | 50 | live | 08-08 | gate without the rescore |
| +3.18 | 22 | 10 | 50 | live | 08-08 | + fine-scale rescore (§8.4) |
| **+4.55** | 22 | 10 | 50 | live | 08-09 | + HyDE — the figure first published as the headline |
| +3.91 | 22 | 10 | 50 | live | 08-10 | *same configuration re-run* — the +4.55 was a favourable draw [C-7] |
| +3.80 | 25 | 10 | 50 | live | 08-10 | the 25-repository headline (§8.7) |
| +3.88 / +4.12 / +4.12 / +4.16 | 25 | 10 | 50 | live | 08-11…13 | four further draws of one configuration |
| +2.72 / +3.16 / +3.72 | 25 | 10 | 15/25/50 | frozen | 08-14 | gate-depth arms (§8.9) |
| +4.96 | 25 | **15** | 50 | frozen | 08-15 | depth 50 at the shipped digest width |
| **+5.42** | **24** | **15** | 50 | frozen | 08-15 | vs the agentic baseline — 24, not 25, because one baseline run failed |
| **+5.12** | 25 | **15** | 50 | frozen | 08-15 | the same arm on all 25 — the figure to compare against a non-baseline number |
| **−8.12** | 25 | **15** | none | live | 08-16 | **what `rr init` ships** (§8.9) |

Three noise floors govern which differences above are readable at all, and they are *not* interchangeable — the second row is the one C-8 misapplied:

| minimum resolvable effect | repos | digest | pool |
|---|---|---|---|
| 1.04 net@2/case | 22 | 10 | live, 3 draws |
| **0.48** | 25 | 10 | frozen, 2 passes |
| **0.74** | 25 | **15** | frozen, 2 passes |

Two reading rules follow. **Levels are comparable only within a row of coordinates**; +5.42 and +5.12 differ by case set alone, and −11.0 and −8.12 are the same configuration measured a month and a benchmark apart. **Paired deltas are more portable than levels** — both arms share every coordinate by construction — which is why the campaign's conclusions rest on deltas and why every comparison in this paper is same-session where it could be.

### 4.3 Benchmark, judge, baseline

Each of 25 cases names a live repository (Table 1 in the artifact lists all; examples: ColBERT, detectron2, stable-baselines3, PEFT, diffusers, PyTorch-Geometric, Whisper, DuckDB, Qdrant, RocksDB, Arrow, LLVM-adjacent, ruff, SciPy, redis, plus three negative controls and a three-repository thin-documentation cohort added last, §12.2). **Every headline result in this paper was measured on the 22 cases that predate that cohort**, which is why the counts below read 22. Per case, RepoRadar's top-10 and the baseline's recommendations are pooled and judged **blind to source** by GPT-5.5 under a fixed rubric; verdicts are cached keyed by (case, paper, rubric-hash) so no system can influence its own grading and re-runs are paired. The baseline is Claude Opus 4.8 running agentically with web search over the same repository (~$0.80/repo) — a strong instantiation of "just ask a frontier model."

### 4.4 Is the judge measuring anything? A validity program

Every number in the project is agreement with one model, so we tested the instrument itself.

**Reproducibility (P7).** 200 verdicts re-judged by Sonnet under a byte-identical rubric: exact agreement 50%; binary (≥2) κ = **0.507**; quadratic-weighted κ = **0.727**. The confusion matrix is one-notch-diagonal — GPT's 0s are Sonnet's 0s *58 of 58 times*, GPT's 3s are mostly Sonnet's 2s — so the judges **rank identically and differ by a strictness offset** (base rates 40% vs 22% actionable). Paired arm-vs-arm comparisons under one judge therefore largely cancel the offset; absolute rates are judge-relative and reported as such. The pre-registered κ ≥ 0.60 bar was *missed*: the labelled set is noisier than predicted, and ±10-net@2 conclusions inherit that.

**Validity (P6).** A label no model produced: an arXiv ID present in a repository's docs at HEAD and absent 24 months earlier is a technique the project demonstrably adopted. Mining six repositories yielded 31 usable adoptions (self-citations filtered). The judge scores **61%** of them actionable against the repository *as it was* before adoption — versus **2%** for random arXiv papers — with the misses individually traceable (a "projects using us" page with reversed citation direction; tutorial background citations; one broken link). The judge rewards approximately the right thing; the pre-registered 70% bar was missed, the 40% invalidation bar comfortably cleared.

### 4.5 The benchmark had to be debugged too

At 12 cases the benchmark was **measurably overfit**: a jackknife on the P1 result (§5.4) showed that dropping the single repository `rag` moved the headline pool-reduction figure from 70% to **11%** — one fold held 28% of all gold targets. Effective repository count (inverse Simpson over target share) was 5.4 of 7. Ten cases were added against four named blind spots (thin docs; no arXiv bibliography; non-ML; citation-rich), doubling targets to 48 and raising the effective count to 15.2 of 17. Two predictions made during expansion were refuted by measurement and recorded: repositories we expected to be thin-docs cases were not (ruff's README is 4× PEFT's), and the six no-bibliography cases we expected to yield no targets supplied **12 of 22 new ones** — repositories can fail to cite an existing literature that would improve them, which made them the sharpest cases in the set. Cohort-1 numbers are flagged as resting on the concentrated set wherever quoted.

### 4.6 Guardrails

All experiments from §7 onward were **pre-registered**: success criteria and kill conditions written before any API call, one configuration per experiment, no prompt iteration against the benchmark. No per-repository knobs anywhere — every threshold is global, and the display threshold (2/3) is derived from the metric, not fitted. Anything fitted is leave-one-repository-out (LORO): fit on 21 repositories, predict the held-out one, pool. Frozen testbeds decouple method evaluation from collection stochasticity, and a positional band-reconstruction (needed because run artifacts recorded judge scores but not gate scores) is verified by tests against every case's own recorded aggregates.

---

## 5 Finding the Papers: Retrieval Under a Register Mismatch

### 5.1 The failure, measured

After the first benchmark runs looked respectable at the pool level, a $0 diagnostic asked the pointed question: of the 24 papers the Opus baseline recommended *and* the judge confirmed (the initial gold set), how many were anywhere in the 2,030 papers RepoRadar's own queries had fetched across those repositories? **Zero.** Not a ranking miss — a disjoint set, despite 23 of 24 targets lying inside the searched categories, and despite arXiv returning them at rank 1 given the right phrase. The mechanism: TF-IDF profiles emit generic terms (`model`, `image`, `torch`), each matching tens of thousands of papers, sampled by a lexical relevance with no impact weighting. Two compounding, previously silent bugs were found and fixed along the way: dependency manifests never parsed and boilerplate surviving stop-wording, so the benchmark had been transmitting queries like `(all:license)` and `(all:https)` — the fix changed **two-thirds of all transmitted queries** — and unquoted multi-word terms parsed as OR on the arXiv API, turning the profiler's most specific outputs into its broadest queries (`all:speech recognition` matches 246,802 papers; quoted, 6,845).

### 5.2 The register mismatch, and four negative results

Five candidate fixes were tried against the 24-target probe; all failed **[NR-1..4]**:

| approach | recovered / 24 |
|---|---|
| TF-IDF keyword queries (control) | 0 |
| harvest the repo's own bibliography as queries | 0 |
| LLM phrases: "name what the repo *uses*" | 2 |
| LLM phrases: "name what the repo *lacks*" | 0 (83% of phrases matched nothing) |
| citation-count-sorted search | 1 |
| fetch deeper (raise per-query results) | ≤3; 9/24 outside the match set at any depth |

The two LLM arms fail for opposite, diagnostic reasons. Asked what the repository *uses*, the model answers accurately and uselessly — detectron2 "uses Mask R-CNN, panoptic segmentation"; its targets were Soft-NMS, Copy-Paste augmentation, ConvNeXt, which it does not. Asked what it *lacks*, the model aims at the right *classes* of work and phrases them as descriptive compounds no title contains ("experience replay prioritization methods": 0 hits; the term of art "prioritized experience replay": found). We name the underlying invariant the **register mismatch**: *a codebase — and any text derived from reading it — describes what the project has; the valuable paper describes what it should adopt next.* These are disjoint registers, and negative result NR-7 later showed the failure is not fixable by better string matching: with stemming and BM25 closing the phrasing gap entirely, "lacks" queries rank true targets *worse than random* (percentiles 57/80/94), because they name a plausible *different* research agenda. Describing what the repo has (`uses`, or just pasting the keyword profile) remained within noise of the best arm at every depth **[NR-7]**.

### 5.3 What works, part 1: the citation hop

Repositories never cite what would improve them (NR-1) — but those papers are **one hop away**. Seeding on the arXiv IDs the repository itself cites and expanding one hop in both citation directions reached **18/24 (75%)** of the original gold set, against 0–2 for everything else. Honesty about that number took two more measurements. First, a transport bug had earlier produced 14/24 with a confident wrong mechanism attached (§11). Second, re-measured over the expanded 22-case benchmark, 75% was revealed as recall over a favourable subset: across all cases the hop reaches **21/48 = 44%**, with a clean structural predictor — repositories with ≥7 arXiv-cited seeds give 89% recall, thin bibliographies 33%, and six repositories with no arXiv bibliography 0% by construction. Three attempts to extend it failed: coupling-degree pool filtering did not replicate off the concentrated cohort (70% cut → 10%) **[NR-8]**; synthetic seeds from LLM phrases recovered 4/27 with all successes in the single arXiv-native domain (vector search) **[NR-6]**; gap-phrase re-ranking lost to pasting the keyword profile **[NR-7]**.

### 5.4 What works, part 2: HyDE against a public dense index

The remaining channel inverts the register problem: have an LLM write the *abstract of the paper it wishes existed* for this repository (HyDE [4]), and search a dense index with it. A public 3.1M-vector binary-embedding snapshot of arXiv abstracts [6] makes this free beyond one-time 432 MB of column-pruned Parquet range reads. The experiment was gated on verifying four dependencies (license, columnar fetch, latency, target coverage — all passed) plus a fifth we almost failed to name: that vectors we compute are **bit-identical** to the index's (they are, Hamming 0/1024 on held-out rows) — without which every query would have measured nothing while looking healthy. Blind (the generator saw only the repo context), four hypothesis abstracts per repository: union top-1k reaches **27/48**, median rank 837, versus 10/48 for embedding the README (bimodal: occasionally rank-1, median 46,656 — the register mismatch visible as a distribution) and 3/48 for keyword queries. Crucially the two working channels barely overlap: **hop ∪ HyDE = 36/48 (75%)**, with 15 targets only HyDE reaches — including repositories whose bibliography is empty. Of the two, only the dense channel is shipped and measured end to end (§8.5); the hop's remaining 9 complementary targets fall almost entirely in repositories that already saturate the digest, so its marginal value is real but currently unmeasurable on this benchmark.

### 5.5 The pool is dense, and the bottleneck moves

The first wild labelling of the candidate pool itself (P5: 1,200 papers gated, 320 judged, none of them gold targets) recalibrated the whole retrieval picture. The HyDE top-100 stratum is **58% actionable** (9% score-3) against a **2%** random-arXiv floor — a 29× separation — and the hop's high-coupling stratum 67%/27%. The scary sparsity figures earlier sections had internalized ("1 good paper per 5,111 candidates") measured distance to *known gold targets*, a top stratum, not to useful papers. Meanwhile the gate's confusion became precise: precision 0.97, **recall 0.60** on this distribution — the gate was rejecting actionable papers, not admitting junk, and tightening it (every prior instinct) had almost no precision left to buy. Retrieval was in better shape than believed; **selection became the bottleneck**.

---

## 6 Selecting the Papers: an LLM Gate and Its Ceiling

### 6.1 The gate transforms the product, then stalls

Adding the 0–3 gate over the ranker's head (2026-07-05) took the user-facing digest from mean net@2 **−11.0** — confident ten-paper dumps with zero actionable on two repositories — to **+0.25**, mostly by converting false-positive floods into correct abstentions. Every subsequent candidate-side improvement then hit the same wall: all-time discovery fixed a real paper-age artifact (the baseline's picks were uniformly old; a 90-day window structurally excluded them) and improved the pool while slightly *hurting* the headline; listwise reranking by gate score recovered it; BM25-RRF fusion improved nDCG on all cases while feeding the gate more borderline papers **[NR-11]**. The pattern, visible by 2026-07-12: *discovery ↑, ordering ↑, retrieval ↑ — headline pinned by gate precision.*

### 6.2 Gate-side levers, exhausted honestly

Three levers were tried on the gate itself. Threshold sweeps showed `min≥3` buys perfect precision only by abstaining on 3 of 4 cases — net@2's λ=2 makes "2 good + 1 dud" *tie* "nothing", which a user does not. A stronger gate model (Sonnet for Haiku) was **metric-identical** **[NR-9]**. A carefully engineered rubric rewrite targeting the observed false-positive modes missed its one target (the negative-control leak), cost recall elsewhere, and flattened the score distribution — the high-precision tier died **[NR-10]**; it was reverted the same day. Conclusion at the time: a capability floor. Conclusion in hindsight (§7): the information was present but *quantized away* by the scale.

### 6.3 What to tell the gate about the repository

A 602-paper labelled set (all cached verdicts) made prompt questions cheap ($0.10/arm, paired). The robust findings: telling the gate what the repository is **for** — any purpose statement — is worth +12 to +22 net@2 over a keyword profile alone (the profile actively misleads: ColBERT profiles as "web APIs" via its Flask dependency); the best measured budget is the README's first 300 characters (+22, Bonferroni-adjusted p = 0.031), though the curve's shape is unresolved and an early "more prose is worse" claim was retracted on a reader's challenge — the differences between budgets never reached significance **[C-2]**. Two sharper negatives: an LLM's *paraphrase* of the README scores below sending nothing at all, while *verbatim* sentence extraction recovers the entire gap (+21 paired) — rewriting destroys the term-of-art vocabulary the gate matches on **[NR-12]** — and semantic sentence selection does not beat taking the first 300 characters (−4, p = 0.78), a mechanism-backed prediction that failed **[NR-13]**. Confirmed end-to-end, prose-300 moved the 12-case benchmark +1.75 → +2.75 (P(Δ≤0)=0.032), halving the digest's junk at constant size.

The most instructive gate-context negative is **P8 [NR-14]**: appending the repository's own top-15 open issues *verbatim* — ground truth about what users want — is **the worst arm ever measured** (−38 net@2 vs prose-300, CI [−55, −21]). Precision unchanged, recall −0.27, damage proportional to how much there was to lose (r = −0.61 with base rate); an internal control (one repo surfaced zero issues; its prompt was byte-identical and its metrics didn't move) pins causation. A list of fifteen named wants *replaces the question*: the gate stops asking "would this improve the project" and starts asking "is this on the list."

### 6.4 Depth is not the answer either

Two symmetric experiments closed the "just show the gate more" direction. Widening the triage window 20 → 50 bought 2 actionable papers across 12 cases **[NR-15]**. Gating the **entire** candidate pool (~227/repo, trivially affordable at $0.05) against the shipped depth-50, both arms same session on all 22 cases: **−0.18 paired, 6 better / 6 worse / 10 ties — a wash** **[NR-16]**. (The first version of that comparison read −0.95; an arXiv throttle had zeroed two of the strongest cases, supplying −17 of the −21 total delta — corrected after collection failures were made loud and excluded rather than scored **[C-4]**.) The mechanism unifies §5.5 and §6: the digest shows 10; deeper gating changes *which* arbitrary admits fill the window, because **nothing orders what the gate admits**.

**Both of these were later reversed by the stage that made their premise false**, and we leave them here in their original form because the reversal is the point rather than an embarrassment. §8.5 records the first: HyDE's pool expansion converted at +1.36 once the rescore ordered what the gate returned, where the identical expansion had been NR-16's wash a month earlier. §8.9 records the second and sharper one — re-measured on a frozen pool with depth as the only variable, gating 50 rather than 15 is worth **+1.00 net@2 per case**. Every conclusion in this subsection was true of a system without a within-band ordering, and none of it survived acquiring one.

### 6.5 The distribution that explains the stall

The gate's scores are near-binary: on real pools 0–14% of admits receive a 3, everything else a 2. Within that score-2 band, true precision per repository runs **0.00 to 1.00** — `diffusion`'s ten band papers were all genuinely actionable (+10) while `vectordb`'s eight contained five duds (−5) — and the two are **indistinguishable at gate time**: the share of 3s among admits correlates with band precision at r = +0.30, the heuristic ranker is flat inside the band (its real contribution is concentration: top-50 density 1.7× the pool, with no ordering within), and the gate's own 3, though genuinely predictive (gate-3 → 76% judge-3), fires far too rarely to order anything. A back-of-envelope "adaptive digest size" proposal — show every 3; at most *k* 2s — died on arithmetic and on the data: with 3p−2 economics an unranked set has no interior optimum in *k*, per-repository *k* is reward hacking, and the abstention signal the rule needs does not exist (`diffusion`, 0×3/10×2/p=1.00, is identical at gate time to `numerics`, 0×3/10×2/p=0.60) **[NR-17]**. What survived was the requirement: **rank — or better, calibrate — the score-2 band.**

---

## 7 Ranking the Hidden Band: Five Pre-Registered Experiments

A structured literature review (six parallel sweeps; primary sources; the load-bearing citations spot-checked against arXiv, one misattribution found and flagged) produced five candidate mechanisms, each pre-registered with success criteria and kill conditions, all evaluated **offline against frozen labels** — zero new judge calls — on: Testbed A (a frozen 22-case live run; 220 shown papers, 105 in-band), a pool-300 replication arm, a 12-repo labelled band (judge-3 ordering target), and wild-pool sanity sets. Baselines fixed in advance: show-all +1.91, score-3-only +0.82 with 14/22 abstentions. Total spend ≈ $14.

| experiment | mechanism | band AUC (judge≥2) | policy net@2 | verdict |
|---|---|---|---|---|
| **E2: 0–9 rescore, logprob expectation** (gpt-4o-mini) | distributional pointwise | **0.841** | see §8 | **winner** |
| E1: subset-selection share, 15 shuffles (Sonnet / Haiku) | listwise-set + self-consistency | 0.635 / 0.616 | +1.23 / +1.18 | fail |
| E3: 10-persona ensemble votes + verbalized P (Haiku) | simulated annotators | 0.676 | +0.82 | killed (ECE 0.425) |
| E4: round-robin pairwise, both orders, Bradley–Terry (+ anchors) | comparative | 0.643 | +2.00 (21/22 ties) | fail |
| E5: metadata features, LORO logistic | learned combiner | 0.585 | +1.50 | below bar |

**The winner is the least structural change.** E2 re-asks the gate's own question on a 0–9 anchored scale (its bands are the 0–3 rubric's bands widened) and reads **the expectation over the answer token's probability distribution** — "mostly 7, maybe 8" becomes 7.3 instead of collapsing to a rung [24, 25, 27]. The de-quantization is real: the modal digit carries >0.9 probability for only 23% of papers (kill bar: 80%), and the motivating pair dissolves — `vectordb`'s three actionable papers are its top three band scores (8.0, 8.0, 7.9 against duds at 7.1→3.8) while `diffusion` sits uniformly high. It replicates on every testbed it never touched during development (pool-300 arm 0.761; judge-3 ordering 0.760; wild pools 0.949), and survives the pre-registered same-family landmine — the judge is also an OpenAI model — scoring *higher* against the Sonnet second judge (0.896) than against GPT-5.5 (0.843) on the 74 dual-judged rows.

**The failures have one shape.** E1's "select the papers a maintainer should act on — possibly none" abstains correctly on bad bands (`numerics`: empty in 13/15 shuffles) and is catastrophically strict on good ones — Sonnet selected *nothing* in 15/15 shuffles on two all-actionable repositories; the literature's top-half-selection prior [16] never appeared, its mirror image did **[NR-18]**. E3's consider-the-alternative ensembles collapse toward P̂ ≈ 0 regardless of label — the predicted overconfident consensus failure inverted **[NR-19]**. E4 survives its position-bias kill (swap-inconsistency 0.322 with mandatory both-order queries; Claude-family position bias is severe [17]) but misses its ordering bar, and its calibration mechanism fails absolutely: **128/128 real papers beat the synthetic borderline anchor** — a real abstract always beats a template survey, so anchoring [30] provides no threshold **[NR-20]**. Claude models under maintainer framing default to "no"; adding comparative or ensemble structure amplified the strictness rather than extracting signal. Metadata features (age, citations, influential citations, HyDE rank, hop coupling) carry weak signal (0.585), consistent with [3] **[NR-21]**, and the kitchen-sink combined model (features + all four method scores, LORO) *loses to E2's single score on every axis* — at 22 queries, every added column is another way to overfit **[NR-22]**.

**The Anthropic-native variant is measured and rejected, not merely untried.** Approximating the token distribution by sampling Haiku ten times on the identical prompt scores band AUC **0.590** vs 0.841 — because in 44% of papers, nine or ten of ten draws return the *same digit*. At default temperature the model is near-deterministic on this task; sampling re-reads the mode and cannot recover a distribution the sampler will not explore. The exact logprob reading is load-bearing, which makes the second-vendor dependency structural **[NR-23]**.

---

## 8 Calibration, Shipping, and Live Validation

### 8.1 From score to policy

The 0–9 expectation is an ordering; the metric's economics (§4.1) demand a probability. A **two-parameter logistic** — the only fitted numbers anywhere in the shipped method — maps expectation to P(actionable), evaluated LORO and shipped as the all-22 fit, with a single global display rule: a band paper reaches Top Picks iff **P ≥ 2/3**, the derived breakeven. On the frozen benchmark this rescues **every** net-negative repository (`compiler` −5→+2, `vectordb` −5→+1, `numerics`/`linter` → abstention) at a one-paper toll on the all-good tail; shown papers drop 132→102 while actionable shown drops only 97→91 (precision 0.73→**0.89**). Freezing the A-fitted map and applying it unchanged to the pool-300 arm — different run, different shown sets — replicates: +3.09 vs +1.73 (10+/3−, p=0.092).

### 8.2 Engineering against silent decalibration

The map is calibrated *to a prompt*, so the repository-description block is one shared function used by the product, the gate, and the benchmark harness; a test asserts the shipped prompt is **byte-identical** to the one the coefficients were fitted against. Failure is designed to be loud and conservative: a paper whose call fails is omitted, never scored (an unscored band paper does not reach Top Picks); a run that scores under half its band skips the gate with a warning rather than demoting everything into a fake abstention; scores are persisted only when the gate applied, so post-hoc consumers (archive, notifications) infer gating from data instead of trusting a flag — the flag version, our first attempt, did not even cross the two CLI commands involved, which a linter caught. A parametrized test renders every output format and asserts each honours the threshold; it immediately caught two generators the plumbing had missed.

### 8.3 A correction, and the trap behind it

Our first recorded policy number, +2.91, evaluated **a map that does not ship**: it came from the E5 harness, whose inner cross-validation selects L2 regularization by *AUC* and chose heavy shrinkage. AUC is rank-only — invariant to any monotone transform — so it is structurally blind to where P crosses 2/3, the single property a thresholded policy depends on; shrinking the slope leaves AUC untouched and moves the decision boundary. The shipped map's honest LORO figure is **+3.14**. Both computations are valid cross-validation of *different* estimators; selecting a calibration by a ranking criterion is the error, and we suspect it is common. (The correction is favourable, which is precisely when the mechanism, not just the revised number, should be published.) **[C-5]**

### 8.4 The live run

Everything above is offline replay. The shipped path was then run end-to-end — fresh collection, live ranking, live gate, live rescore through the shipped module (the harness calls it rather than reimplementing it; this project has already published one measurement of a harness's own prompt under the system's name **[C-3]**) — on all 22 cases:

| | offline replay | **live run** | Opus 4.8 baseline |
|---|---|---|---|
| RepoRadar + fine-scale | +3.14 | **+3.18** | — |
| RepoRadar show-all | +1.91 | +1.86 | — |
| baseline | — | — | +1.82 |
| paired vs baseline | +1.32 | **+1.36** (10 w / 6 l / 6 t; sign p = 0.45) | |
| paired vs show-all | +1.23 | **+1.32** (8+/2−; p = 0.109) | |
| digest precision | 0.89 | **0.91** | 0.94 |
| net-negative repositories | 0 | **0** | 1 |

Every aggregate reproduces within noise while the **per-case numbers move hard** (`ann` show-all +4→−2, `crypto` +3→−2; six net-negative repositories on this draw rather than four, all six eliminated by the stage): collection is stochastic, per-case values in any single run are draws, and the replay was measuring something stable rather than a lucky draw. At this point neither comparison cleared p < 0.05 on 22 paired cases — a clear mean improvement, measured twice, not established per-repository. The losses that remained were all recall: cases where the admitted set never contained what the agentic baseline found (one repository admitted nothing at all), the 0.60-recall problem a precision stage cannot touch.

### 8.5 Closing the recall half: the dense channel, measured end to end

§5.4's dense channel had been measured only as retrieval; it shipped next, and was evaluated in a second run with **both arms back to back in one session and HyDE the only variable** — deliberately not against the stored run above, whose per-case values had just been shown to swing by ±6 between draws, which a fresh-versus-stored comparison would have credited to the flag. A degraded-arm counter in the harness reports any case where the dense channel silently fell back to the keyword pool, because a degraded arm measures the *old* channel under the new flag's name; it fired on none of the 22.

| | control (keyword pool) | **+ HyDE** | Opus 4.8 baseline |
|---|---|---|---|
| RepoRadar + fine-scale | +3.18 | **+4.55** | — |
| RepoRadar show-all | −4.18 | +1.41 | — |
| baseline | — | — | +1.82 |
| paired vs baseline | +1.57 (11 w / 4 l / 6 t; p = 0.12) | **+2.73** (15 w / 3 l / 4 t; **p = 0.0075**) | |
| digest precision | 0.91 | **0.94** | 0.94 |
| shown / actionable | 97 / 88 | **121 / 114** | 49 / 46 |
| net-negative repositories | 0 | **0** | 1 |

This is the first result in the campaign that clears p < 0.05 against the baseline: the system matches the agentic baseline's precision while returning 2.5× as many papers. **Attribution matters more than the headline.** HyDE's own paired increment is +1.36 (10+/3−) at **p = 0.092** — not significant at n = 22. What crossed the line is the cumulative system: two shipped changes of about the same size, neither established alone. (The control arm's baseline is measurable on 21 of 22 cases — a transient arXiv verification failure on `speech` — which is why its paired figure reads +1.57 against +1.36 above on a different draw; the HyDE-versus-control comparison is RepoRadar-to-RepoRadar and unaffected.)

The pre-registered target was recall, not the mean: a gain without movement on the six recall-driven losses would not have been the win the channel was shipped for. Three are repaired outright (`speech` 0 → +10, `graph` +1 → +10, `rag` 0 → +4 — the repository that had been admitting nothing at all), and `llminfer` and `numerics` remain genuine losses. The two stages **compose rather than merely stack**, and the evidence is that precision *rose* while the shown set grew: the earlier finding that gating a wider pool was a wash concluded that "the bottleneck is not how many papers the gate sees, it is that nothing ranks what it returns" **[NR-16]** — the rescore made that sentence false, and only then did a larger pool convert. Without it, doubling the pool would have fed the near-binary gate more borderline papers, which is exactly what BM25-RRF fusion did earlier in the campaign, when better nDCG made the headline worse **[NR-11]** Fusion itself, however, does **not** survive the same scrutiny we applied to the other stages: ablated inside this configuration it is worth +0.00 net@2/case and puts 8.72 against 8.80 actionable papers in front of the gate, so it is carried rather than justified **[NR-35]**. The composition claim above is established for the pool expansion and the rescore, and *not* for every component of the configuration they sit in.

Three repositories returned *fewer* papers under the wider pool (`storage` 10→8, `diffusion` 10→9, `compiler` 2→1), all at 100% precision on what remained — the rescore declining newly-admitted band papers, not lost recall. And net@2's asymmetry still flatters precisely this kind of precision-preserving expansion; it flattered the previous result the same way.

### 8.6 Auditing the frozen map: decalibrated, and provably not worth fixing

The two fitted constants are the system's only tuned numbers and its most obvious liability, and §8.2's defences are all *textual* — they catch an edited prompt, not a shifted meaning. We therefore measured the map against the live run it had just decided. Every paper in the system's own top-10 (220 across 22 repositories) already carried a judge verdict, and the harness judges the whole of that window, so it is ground truth on exactly the population the map governs with no selection bias inside it. Re-gating and re-scoring those papers reproduced the live decision on **117 of 121** shown papers (97%; the four misses are one paper each, all conservative — the re-run noise floor of two temperature-0 calls), which is the precondition for reading anything else.

The map **is** decalibrated, in the direction of under-confidence: on the 126 papers it governs the observed actionable rate is 0.817 against a mean predicted P of 0.689, a paired gap of **−0.129, 95% CI [−0.187, −0.067]** (ECE 0.128). All five reliability bins under-predict and so do 16 of 19 repositories (residual sd 0.105 about a mean of −0.117) — a **global level shift, not the per-repository dispersion we opened the audit expecting**. The *ordering* is intact: band AUC 0.824 live against 0.841 at fit time.

And it is worth almost nothing. A leave-one-repo-out refit — fitted on 21 repositories and scoring the 22nd, so it never sees the repository it grades — moves eleven repositories and nets **+0.00 net@2 per case** (6+/5−/11 ties; 95% CI [−0.45, +0.45]). An *oracle* threshold chosen on the test set itself, reported only as an unachievable ceiling, buys **+0.27**: twelve papers flipped into the digest, ten of them actionable, +6 net across the whole benchmark against a base of +4.36.

Two mechanisms explain why an error this real is this cheap. The first is structural: P = 2/3 is *derived* from the metric (3p − 2 = 0), so a paper at the boundary is worth approximately zero by construction, and moving a correctly-derived threshold across a well-ordered set has near-zero expected value wherever it lands. The second is distributional, and is the near-binary pathology of §6.5 **reappearing one level down**: 63 of the 126 band papers sit in P ∈ [0.8, 0.9) and only 24 fall within ±0.1 of the threshold. The rescore de-quantized the gate enough to *order* it, not enough to spread it across the decision boundary — and a calibration error can only cost where the papers are.

We record the failed prediction, because the way it failed is the transferable part. The audit was opened on an unpaired reading: 43 judge-actionable papers sit in the system's own top-10 unshown, worth +11 net@2 across the six repositories where showing more would help. The same threshold move costs −80 across the nine where the gate is correctly strict, and showing the full top-10 everywhere scores +1.41 against +4.55. **An accounting of only the upside of a threshold move is not a counterfactual**; the LORO refit is the paired version of the identical question, and it is zero.

This closes recalibration as a direction rather than deferring it, and it leaves the monitoring question in a usable shape: **AUC is the alarm and the calibration gap is a gauge.** Ordering holding at 0.824 says the scorer still works; a drifting level says only that a number no decision depends on has moved. What the audit cannot do is attribute the gap — the fit population's band was 74.3% actionable against 81.7% live (Wilson [0.741, 0.875]), consistent with the dense channel having enriched the band but statistically indistinguishable at n = 126.

### 8.7 The result on the full benchmark, and how much of a headline is weather

Every figure above was measured on the 22 repositories that predate the thin-documentation cohort (§12.2). Those three were added *because* the system handles them badly, so re-running the headline over all 25 was an obligation rather than bookkeeping. The agent baseline was deliberately held at its original turn budget, since raising it would make the number non-comparable with the one it replaces.

| | 22 repositories | **25 repositories** |
|---|---|---|
| RepoRadar mean net@2 | +4.55 | **+3.80** |
| baseline | +1.82 | **+1.57** |
| paired | +2.73 | **+2.26** |
| sign test | 15 w / 3 l / 4 t, *p* = 0.0075 | 15 w / 5 l / 3 t, ***p* = 0.041** |
| digest precision | 0.94 | 0.898 |

**The result holds on the harder benchmark**, weaker and still under 0.05. But the decomposition is the part worth reporting, because it was not what we predicted. We expected the three hard repositories to account for the drop. They account for **one tenth of a point**: the same 22 repositories, same configuration, score **+3.91** today against the published +4.55. The remaining **−0.64 is run-to-run drift on identical inputs.**

Two things follow. First, **the published +4.55 was a favourable draw**, and we say so rather than leaving the more flattering number standing. Second, and more generally: a single benchmark mean in this setting is a draw from a distribution roughly ±0.6 wide, which is the aggregate shadow of the ±6 per-case swings we document throughout. Anyone reporting one number from one run of a 20-odd-case LLM-judged benchmark — ourselves included, repeatedly — is reporting weather as if it were climate. The defence we can afford is paired same-session arms, which is why every comparison in this paper is one.

**How much of a headline is weather, measured.** A third draw of the identical configuration (2026-08-11) puts the benchmark's own noise at a residual per-case sd of **1.23** (42 df) plus a whole-run shift of sd 0.27, giving a **minimum resolvable effect of 1.04 net@2 per case** for a paired same-session comparison and 1.07 against a stored run. Two consequences. First, the 22-case mean across three draws is **+4.18** with a range of +3.91 to +4.55, so the +4.55 we published was the highest of three. Second, and worse for single-run reporting: the 25-case sign test gives *p* = 0.0414 on one draw and *p* = 0.0001 on the next, **two orders of magnitude apart on identical inputs**. A single run's significance is not a property of the system. We also note what this floor implies about our own null results: the stated-intent experiments (+0.44, +0.12) were below the resolvable effect before they were run, so their nulls are real but uninformative. **[C-8]**

**Every number in this section describes a ten-paper digest, and the system ships fifteen.** The benchmark's returned-set cut and the product's `output.top_n` had drifted apart the same way the gate depth had (§8.9), and in the same audit. Measured on one frozen pool with the width as the only variable, widening it is worth **+1.24 net@2 per case** — so the figures above understate what a user receives, and the paired advantage over the agentic baseline grows with them. Re-measured against the cached Opus responses at the shipped width, RepoRadar reaches **+5.42 against +1.62, paired +3.79** (95% CI [+2.17, +5.58], 18 w / 1 l / 5 t, sign *p* = 0.0001) on the 24 repositories where the baseline completed.

We report that with the precision line first, because it is the more informative one: at fifteen the system returns **four times as many papers at five points lower precision** than the agent (0.888 against 0.938). net@2 rewards each actionable paper linearly and charges 2 for a dud, so a wider digest at slightly worse precision wins by a wide margin — and the metric's volume sensitivity, already flagged in §4.1 and §8.5, is larger at fifteen than at ten. What licenses showing those papers is that their marginal precision (0.855) clears the 2/3 breakeven the metric derives; what it does not license is the claim that the digest is *better to read*. Those are different propositions and only the first is measured here.

A methodological consequence we did not anticipate: **the noise floor is a property of the digest width too.** Re-measuring on the same frozen pool gives a minimum resolvable effect of **0.74 at fifteen against 0.48 at ten** — the residual per-case sd rises 0.61 → 0.93, because more shown papers means more chances for temperature-0 jitter to move one across the display threshold, and each is worth +1 or −2. Widening the digest made the instrument *less* sensitive, which inverts the usual more-data intuition, and it invalidated the floor our own reporting tool was deriving from pool provenance alone.

**A claim of ours that this draw falsifies.** We reported that the calibrated rescore eliminated every net-negative repository. In this run `numerics` returns a single paper, the judge scores it 0, and the case lands at **−2.0**. The claim was true of every run it had been measured on, and it is a *per-draw* property rather than a guarantee: one dud in a one-paper digest is sufficient to break it. **[C-7]** Later draws confirm it and sharpen it: the count of net-negative repositories runs 0, 1, 2 across draws of one configuration, and the identity changes every time — `numerics` on the 25-case headline, `speech` and `thin-lang` on 2026-08-12, none at all on 2026-08-13. *Which* repository fails is weather; that roughly one does is climate.

Two further draws of the 25-case configuration — one byte-identical (2026-08-12), one under a changed query default (2026-08-13, §9.3) — landed at **+4.12** and **+4.12**, so four draws of the shipped system now read +3.80, +3.88, +4.12, +4.12. Neither of the new ones re-measured the agentic baseline, so the paired advantage above still rests on the first two.

### 8.8 Lowering the floor: freeze the pool

A 1.04 floor makes most component-level work unmeasurable — a plausible ranking change is worth a few tenths, and no honest experiment at n = 25 can see it. Since the dominant variance term is *which candidates were collected*, the eval harness gained a mode that collects once and reuses the stored pool. Measured rather than asserted (the flag shipped with an unmeasured claim of "0.2–0.3" in its own help text, which is exactly the sort of assertion this project does not accept elsewhere):

| | live (3 draws) | **frozen (2 reuse passes)** |
|---|---|---|
| residual sd, per case per draw | 1.23 | **0.61** |
| whole-run shift | sd 0.27 | sd 0.10 |
| **MRE, paired same session** | 1.04 | **0.48** |
| cases identical across draws | 8 / 22 | **20 / 25** |

Freezing removes just over half the residual noise, confirming the diagnosis; the pre-registered prediction (MRE ≤ 0.42) still **missed**. What survives is temperature-0 model jitter in the gate and the rescore, concentrated in three repositories that carry 89% of the remaining variance. Two design points matter for reuse. The pool is frozen **before** ranking, not after, so ranking flags can vary across arms that share one pool — otherwise the mode is useless for exactly the experiments it was built for. And provenance is a refusal, not a label: the harness will not compare a frozen arm against a live one, and a report script derives its floor from provenance rather than accepting it as an argument, because a frozen comparison read against the live floor calls a real effect unresolvable. The first version of that guard was itself broken in a way its tests could not see (§11, lesson 10).

### 8.9 What the lower floor bought first: two shipped defaults nobody had measured

A static audit of where the benchmark stops measuring the product (§11, lesson 10) turned up a configuration field neither side could justify: the gate's depth shipped at 15 while every headline was measured at 50. Checking the record made it worse — **no experiment had ever included 15.** The nearest, NR-15, compared windows 20 and 50, so the shipped default was shallower than the shallowest arm ever run, in a comparison confounded with a prompt change and predating the rescore.

Three arms over one frozen pool, gate depth the only variable, pre-registered with its alarm written down first: *if 50 loses, the shallow default is vindicated and it is the benchmark that should change, because every headline since 2026-08-07 was measured at 50.*

| gate depth | mean net@2 | shown | actionable | precision | actionable in the returned top-10 |
|---|---|---|---|---|---|
| **15** (shipped) | +2.72 | 101 | 90 | 0.891 | 5.00 |
| 25 | +3.16 | 124 | 109 | 0.879 | 5.76 |
| **50** (measured) | **+3.72** | 141 | 125 | 0.887 | **6.52** |

Paired against 15, depth 50 is **+1.00 net@2 per case, 95% CI [+0.12, +1.92]** against a frozen floor of 0.48 — and **+1.23, CI [+0.27, +2.27], sign p = 0.035** on the 22 non-control repositories. Depth 25, at +0.44, sits inside the floor and is unresolved rather than equal, so only the endpoints are characterised.

Three things make this more than one number. The **gate-free measure moves with it**: actionable papers reaching the returned top-10 rise 5.00 → 5.76 → 6.52, and that same measure moved 0.00 in the ranking experiment of §9.6, where it correctly predicted a null for $0. **Precision is flat while the shown set grows 40%** (0.891 → 0.887 at 101 → 141 papers), which is the compositional signature of §8.5 rather than dilution. And the **mechanism was already established**: the rescore orders what a wider gate admits, which is exactly why NR-16's "nothing orders what the gate admits" had to be discarded rather than trusted.

The blemish is stated rather than averaged away. `webdev`, a negative control, goes 0.0 → −2.0 at depth 50: a deeper gate on a repository with no applicable literature has more chances to be wrong, and the controls-only delta is −0.67 on three cases. The 22 real repositories pay for that three times over.

**The same audit found the mirror image, and it caught us out the other way.** The digest's width shipped at 15 while every number in this paper was measured at 10 — so unlike the gate depth, here the *shipped* value was the unmeasured one and the *benchmark* was the conservative one. We predicted the difference was worth about +0.1 net@2 per repository and said so as a **ceiling**, on the reasoning that papers ranked 11–15 must clear the gate less often than those ranked 1–10. Measured on the same frozen pool with the window as the only variable, it is **+1.24 net@2 per case, 95% CI [+0.48, +2.08], sign p = 0.035** — twelve times the ceiling, and larger than the depth result above.

The error is instructive and it is ours. Two precision-by-rank curves were available and we extrapolated from the wrong one: the *raw window* decays 0.68 → 0.48 across ranks 1–10, but the *shown* curve — papers that clear the gate — is 0.75–1.00 and flat, and only shown papers enter net@2. The reason it is flat is the near-binary gate of §6.5: within the admitted set nearly everything scores 2, so "ranks 11–15 have lower gate scores" is true and nearly empty, and the stage that actually discriminates is the fine-scale rescore, which runs on every band paper regardless of rank. We had that finding and did not apply it. The added papers were 85.5% actionable (47 of 55), against a breakeven of 2/3.

Two consequences. **Every net@2 in this paper understates the shipped system**, because all of them were measured at a window the product does not use; the configuration reported as +3.72 above is +4.96 at the shipped width. And net@2's structural reward for precision-preserving expansion (§4.1) is doing real work in that number — what defends it is that the marginal precision, 0.855, sits well above the display threshold the metric itself derives.

**Asked exhaustively, the same question has a much worse answer.** Both findings above came from an audit comparing *twelve hand-listed* configuration fields against the dataclass defaults. Neither number was wrong, but the scope was chosen by whoever last edited the list — the C-14b defect one level up — and the object was wrong too: `rr init` writes a **template**, and where the template sets a value, the dataclass default is dead text. Re-run over all **79** configuration leaves and both surfaces, with every leaf required to be either compared or excused in writing, the audit reports seven divergences instead of zero.

| field | what a user runs | what we measured |
|---|---|---|
| `triage.enabled` | `False` | `True` |
| `suggestions.provider` | `template` | `claude` |
| `triage.finescale.enabled` | `False` | `True` |
| `hyde.enabled` | `False` | `True` |
| `ranking.hybrid` | `False` | `True` |
| `ranking.w_embedding` | `1.5` (template) | `0.0` |

**The configuration a new user receives enables none of the three stages this paper is about.** `rr init && rr update` is the ungated heuristic digest. We then measured it rather than citing §6.1's July figure, and the measurement is reported in full below because our pre-registered prediction was wrong: on the same 25 repositories at the same digest width the default scores **−8.12**, against **+5.12** for the configuration this paper describes. Each opt-in is individually defensible: the gate needs an LLM credential, the rescore needs a *second* vendor's (only OpenAI exposes the logprobs it reads), HyDE needs a 1.1 GB index sync, and a default that fails without a credential is worse than one that under-delivers. The documentation says all of this. What is not defensible is that the audit built to catch precisely this shape reported clean for as long as it was scoped by hand — and that `triage.enabled: true` on its own is a no-op, because the gate also requires `suggestions.provider`, which the configuration alone does not reveal.

We changed no default in response, and the restraint is the point. `ranking.hybrid` was the temptation — dependency-free plain Python, present in every headline — but NR-11 measured it as better nDCG everywhere and a *lower* headline, with the cost recovered only once the rescore ordered what the gate admits, a stage the default does not run. Enabling it for a product that gates nothing would ship the NR-11 loss: the gate-depth error run backwards. `ranking.w_embedding` is unmeasured in *both* directions — the one arm touching that channel scored README embeddings as a query (7/48 at top-100, median rank 46,656), which says nothing about weighting them against keywords at digest time — and it is the only field whose behaviour depends on the install, contributing nothing unless an optional extra is present. **Replacing an unmeasured default with a different unmeasured default buys nothing and destroys the record of which one shipped.** Both are declared with their reasons and their *values* pinned, so a later edit re-opens the question instead of inheriting the exemption. What the finding actually licenses is an experiment we have not run: the out-of-the-box arm.

**The out-of-the-box arm, and a prediction we got wrong by ten points.** We pre-registered −2 to +2, with "below −5 means the reasoning is wrong" written as the surprise condition. It came in at **−8.12**, and the condition fired.

| | default (`rr init`) | measured (`rr init --measured`) |
|---|---|---|
| mean net@2, 25 repositories | **−8.12** | **+5.12** |
| papers shown | 235 | **197** |
| of those, actionable | 89 | **174** |
| precision | **0.379** | **0.883** |
| net-negative repositories | **19 / 25** | 2 / 25 |

The reasoning that failed is worth stating because it is a metric error, not a systems one. We argued that retrieval had improved enormously since the July figure — the recency window that structurally excluded every gold target was gone, the query bridge was repaired — so the ungated arm should be far better than −11. Retrieval *had* improved: this arm surfaces **89 actionable papers**. But `net@2` pays `3p − 2` per paper *shown*, so at p = 0.379 **every displayed paper costs 0.86 on average**, and the heuristic tiering has no way to decline — it filled all fifteen slots in 17 of the 25 repositories. Better retrieval fed a display rule that cannot abstain, and the benchmark charged for every additional paper. The compressed version: **the measured configuration shows fewer papers (197 against 235) and delivers nearly twice as many actionable ones.** The gate's value is not that it finds papers; it is that it declines to show them — which is precisely what §6.1 concluded in July, and we still predicted wrong because we were reasoning about the numerator.

One further embarrassment belongs in the record. The stale figure we set out to correct (−11, four repositories, one of them a negative control) sits **2.9** from the true 25-repository value; our prediction's midpoint sits **8.1** away. The provenance objection was correct and the row had no business in a user-facing table — but the number we were replacing was the more accurate of the two on offer, and the honest summary is that we replaced a well-attributed guess with a badly-attributed measurement, not the reverse.

What this licenses is the claim we had been making without evidence: an ungated keyword digest is **worse than emitting nothing**, since abstaining everywhere scores 0 and this scores −8.12 while losing on 19 of 25 repositories. It is now measured. **[C-17]**

What the finding also licenses, and what we did instead of changing a default, is **making the measured configuration something a user can ask for**: `rr init --measured` writes it, every value annotated with the measurement that justifies it, alongside its prerequisites (two API keys, a 1.1 GB index sync) and its price (~$0.01–0.02 per repository per run, against ~$0.80 for the agentic baseline it beats). The mechanism that keeps this from becoming its own stale copy is the one worth reporting: the audit asserts that all 39 fields of the benchmark's configuration are reproduced by that file, **with no exemption mechanism at all** — the default template may differ from the benchmark for declared reasons, but a file whose entire purpose is to be the configuration behind a published number may not, so any difference is a documentation defect by construction. A recommended configuration is a claim about a specific run, and claims decay exactly the way code does; this one fails a test instead of aging quietly. The default, meanwhile, now states its own weakness in three places — its header, `rr init`'s output, and `rr update` at the point the gate would have run — because a user who never opens the configuration file should still learn once which of the two systems they are running.

**Two things generalize.** First, freezing the pool is what made this affordable to ask at all — at the live floor of 1.04 a +1.00 effect is unresolvable, and the question would have stayed closed on evidence that had already expired. Second, and less comfortable: *a default set before a system acquires a capability does not update itself when the capability arrives.* Both of this system's digest defaults were set when nothing ordered the gate's output, and both were still in place two shipped stages later — one too shallow, one too narrow, and neither discovered by suspecting it. What found them was auditing the configuration we ship against the configuration we measure, and the two were wrong in opposite directions, which is exactly what an audit catches and a hunch does not.

---

## 9 Beyond arXiv: Five Source Channels, and the Bridge That Broke All of Them

RepoRadar ships adapters for four non-arXiv paper sources (Semantic Scholar, OpenAlex, bioRxiv, DBLP), and gained a fifth late in the campaign for a named coverage gap (IACR ePrint). Every headline in this paper was nonetheless measured on arXiv alone. This section is the account of what happened when we finally asked why, and it is the largest single block of correction in the campaign: one defect that had made every non-arXiv channel return garbage since the day it was written, two published findings that turn out to have measured nothing, and — once the channels worked — a null where we had predicted a gain.

### 9.1 One translator, five call sites, two of them wired

arXiv's API takes a boolean grammar (`(all:"vectorized execution") AND (cat:cs.DB)`); keyword APIs take words. The bridge between them was a one-liner repeated at each call site — `q.replace("all:", "").strip('"')` — which leaves the parentheses, the `AND`, and the category clause intact. Every non-arXiv source has therefore been receiving arXiv boolean syntax as a keyword query, for the whole of the product's history.

It surfaced only because IACR ePrint, added last, returns *zero* for such a query rather than degrading quietly. Our first published account of the defect attributed it to drift — a transform written for an earlier query shape that went stale when query construction changed. **Git refutes that account.** The parenthesised builder is dated 2026-02-22 and the one-liner 2026-02-23; checked out at the one-liner's own commit, the builder already emitted the form it cannot parse. There is no working era and no regression window. It was wrong on the day it was written **[C-9]**.

The repair — one shared `to_plain_keywords`, placed beside the builder whose output it consumes — was published as "routed through all three call sites." There were **five**, and it routed **two**. The product's Semantic Scholar, OpenAlex and bioRxiv paths, and one of the two eval runners, went on hand-rolling the broken one-liner **[C-9a]**.

Why the tests did not notice is the transferable part. `to_plain_keywords` was correct, unit-tested, and *unused* by three of its callers; **a unit test of a translator is structurally incapable of detecting that nothing calls it.** The guard that replaced it parses `cli.py`, `harness.py` and `run_eval.py` and fails any comprehension over the query list that does not call the shared translator — a *wiring* test rather than a behaviour test, mutation-verified against the exact shipped state. It is the same shape as the harness-rebuilds-the-prompt failure of §11: one invariant, two implementations, and a test suite exercising the copy nobody runs.

### 9.2 Four sources, four ways of returning nothing useful

With the bridge repaired we could ask, for the first time, what each channel actually does. Every measurement here is $0 — real `build_queries` output, real APIs, no LLM call.

| source | on the malformed query | repaired |
|---|---|---|
| IACR ePrint | 0 papers | 5 on the identical query |
| DBLP | 0 hits, every query it answered | 0–4 hits (one genuine empty) |
| Semantic Scholar | 0 papers, 12 of 12 cases | 20 per query, **zero** overlap with the old transform |
| bioRxiv | **the entire recent window** | topically filtered |

Three sources return nothing, which is loud. **bioRxiv fails in the opposite direction, and it is worse.** Its adapter keeps a paper when any query token longer than two characters occurs in the title or abstract. In `(all:"duckdb sql") AND (cat:cs.LG OR cat:cs.CL)` the tokens `("duckdb`, `sql")` and the category fragments match nothing — but `and` is three characters and matches **90 of 90** abstracts; without it, 0 of 90. Enabling bioRxiv did not add biology papers to the candidate pool. It turned the topical filter off and merged the whole recent window at full volume **[NR-28]**. A source that returns nothing announces itself; **a source that returns everything looks like it is working.**

The blast radius is checkable rather than arguable: no `dblp:` or `biorxiv:` identifier appears in any of the 78 recorded run files, so neither source ever contributed a paper to a scored pool and no published net@2, precision or recall figure was computed from their output. Both are re-labelled from *shipped* to **built, wired, never validated**.

One superseded diagnosis is worth recording on its own. Four previous failures to measure DBLP had been closed with "DBLP exposes only a publication year, so it is structurally mismatched to a recency window." That remains true of the adapter and it was **not** what made DBLP return nothing: the measurement above ran at all-time lookback, removing the year filter entirely, and the malformed query still returned zero. A correct diagnosis of a real defect had been accepted as *the* explanation for a symptom it did not cause — which is why the entry's "and now we know why" was premature.

### 9.3 A second defect, visible at a glance, that the benchmark cannot see

`build_queries` pairs each profile keyword with its nearest TF-IDF neighbour and sends the pair as a quoted phrase. Nothing ever required the two words to co-occur; they merely scored adjacently. It emits `"use page"` for DuckDB, `"data cd"` for redis, `"server code"` for ruff — and three of the five queries every source receives are these, ahead of the single-term ones **[C-10]**. Asked of DBLP, `"use page"` returns a paper on simulating the economic performance of sow phenotypes; the benchmark's own hand-written query for that repository returns *Incremental Fusion: Unifying Compiled and Vectorized Query Execution*.

Three arms, 25 cases, one session, all valid (25/25 cases changed their returned top-10; mean Jaccard 0.50 and 0.47):

| arm | net@2 | precision | paired vs control | 95% CI |
|---|---|---|---|---|
| `adjacent` (the defect, control) | +4.12 | 0.914 | — | — |
| `verified` (only phrases the repo contains) | **+4.16** | **0.920** | **+0.04** | [−0.64, +0.88] |
| `none` (no phrase queries at all) | +3.64 | 0.880 | **−0.48** | [−1.00, +0.04] |

Both deltas sit inside the 1.04 floor, and a gate-free retrieval measure — actionable papers reaching the ranked top-10, 6.80 / 6.56 / 6.28 — gives the same ordering and the same non-significance **[NR-29]**.

**The mechanism explains why a benchmark cannot see a defect this obvious.** Every arXiv query carries `AND (cat:cs.DB)`, which keeps results in the right field however meaningless the phrase, so `"use page"` still returns database papers. The bridge correctly strips that clause for keyword sources, which have no equivalent — they receive the bare phrase and answer it literally. **The benchmark measures the one channel where the bug does not bite.** And the obvious repair is measured *backwards*: `none` is the worst arm on every axis, because a meaningless phrase is still a query returning up to 50 candidates, and dropping three of five shrinks the pool.

We changed the default to `verified` and flag it as the one place in this paper where the evidence does not compel the conclusion. +0.04 at p = 0.55 justifies nothing; the argument is that `adjacent` demonstrably asks for phrases the repository does not contain (an observation, not a statistic), that the CI bounds worst-case harm at −0.64/case, and that every non-arXiv channel receives the bare phrase. Every number published before 2026-08-12 was measured under `adjacent`.

**A second sizing error, and it is not the first one repeated.** We proposed this experiment as the highest-value direction available *because it touched 25 cases* rather than IACR's two. That reasoning was about coverage and ignored effect size: a defect the category filter neutralises has no headroom on arXiv however many cases it runs on. The IACR error (§9.4) was sizing an experiment against the *ceiling*; this one was assuming breadth implies power.

### 9.4 IACR ePrint: the register mismatch inside a source chosen to fix a coverage gap

Cryptography's literature is largely not on arXiv, and `crypto` and `encryption` have been the benchmark's steadiest under-performers, so the adapter was pre-registered on those two cases before it existed. Its first run scored both arms identically — because **zero IACR papers reached a top-10 in either case**. That is a void arm, not a null (§9.7), and the degraded-arm check is the only reason it was caught; the cause was the bridge of §9.1.

The valid measurement is **−0.50 net@2/case against a minimum resolvable effect of 3.44** — no detectable effect, and unsurprisingly so at n = 2 **[NR-27]**. The mechanism is more informative than the number. Six IACR papers reached the ranked top-10 and **five were judged 1**: topically exact, not actionable. ePrint is dominated by papers describing **attacks on** primitives rather than improvements a library should **adopt** — the register mismatch of §5.2 reappearing in a source picked specifically to close a coverage gap.

The sizing error we made here is worth stating because it is easy to make. The subset was justified by comparing the MRE (3.44) against the *headroom* (8.5, the distance to a perfect score). Headroom is a ceiling, not a plausible effect; a realistic channel gain of +1 to +2 per case was never detectable at n = 2, and detecting +1.5/case needs roughly n = 11. The adapter ships opt-in and off, documented as **built and unvalidated** — nothing here justifies enabling it, and nothing rules out a real effect below the floor either.

### 9.5 Semantic Scholar, measured three times

S2 is the only source with a *published* number downstream of the bridge, and resolving it took three measurements.

**First: the number was void, not null.** "Adding Semantic Scholar did not help" reported mean net@2 +0.83 → +0.58 and precision 0.91 → 0.76, with a mechanism attached: *S2 papers carry no arXiv categories, the ranker's absent-signal rule stops penalising them, so they compete harder and the gate lets a non-actionable one through.* The mechanism requires S2 papers in the pool. There were essentially none: across 12 cases the malformed query returned **one** paper in total, against 20 per query once repaired, with zero overlap between the two. The reported moves were run-to-run drift, later measured at ±0.6 at the mean and ±6 per case, attributed to a channel that never delivered a paper **[NR-30]**. The recommendation survives on inverted grounds: not *S2 hurts* but *S2 has never been tested*. (Getting there also required building the rate limiting the four S2-touching modules did not have — three had none at all, and the fourth spaced only the requests *within* a single call, so an eval sweeping 25 repositories opened each one unspaced. Retry backoff is not rate limiting; it fires after the server has already been hit too fast **[C-11]**.)

**Second, at $0: the channel works, and it floods the negative controls.** Before spending ~$18 on a judged A/B we ran the stage-1 yield probe the P4 protocol calls for. S2 delivers **218.6 papers per case**, 211 new after dedup, **174.5 of them non-arXiv**, and **73 reach the ranked top-10** across 16 of 23 cases — nothing like DBLP's zero or IACR's six. But **22 of those 73 appearances come from the three negative controls, which are 3 of 23 cases**. On Flask, S2 papers took *every slot in the top-10*, outranking 287 arXiv papers **[NR-31]**. The probe changed what the A/B was looking for, and produced a pre-registered prediction (the gate holds; the mean moves less than the floor) and an alarm (any control goes net-negative, or pooled precision drops below 0.85).

**Third: the judged A/B.** 25 cases, both arms in one session, and emphatically valid — 122 papers returned by the treatment that the control never returned, across 25/25 cases.

| arm | net@2 | shown | actionable | precision | net-negative |
|---|---|---|---|---|---|
| `arxiv` | **+4.12** | 142 | 129 | **0.908** | **0** |
| `+s2` | +3.24 | 144 | 123 | 0.854 | 2 |

Paired excluding the controls: **−1.05, 95% CI [−2.14, +0.00]** — past the 1.04 floor with an interval that still touches zero. Those are two different claims, and the first version of the report script conflated them, printing "RESOLVED" on magnitude alone and calling that interval one that excludes zero, because its containment test was really a sign-agreement test. The honest reading was *big enough to see, not yet established, and pointing down* **[NR-32]**.

The mechanism read as **displacement rather than addition**. `thin-kv` shows it cleanly: eight papers shown in both arms, actionable 8 → 6 — S2 did not add noise beside the good papers, it pushed two of them out of the window. `llminfer` is the extreme, supplying five quantization papers of which four were judged 1: topically exact, not actionable, the register mismatch again with a fuller pool to express itself through.

**The alarm did not fire, and we had aimed it at the wrong place.** Precision landed at 0.854 and no control went net-negative, so by the letter, nothing. Yet two *ordinary* repositories went net-negative where the control arm had none. We had pointed the alarm at the controls because that is where the $0 probe found S2 concentrated; the damage landed on the repositories with genuine literature, where S2's papers compete against *better* papers rather than against nothing.

**A benchmark premise, tested by accident.** Because Tier B never reads the `negative_control` label — the judge sees every paper blind and is asked only whether it could improve this repository — the 17 papers S2 added to those three cases answer the coverage question directly: **4 actionable, 13 loose**. Flask is a real negative control (seven papers on-topic enough to retrieve, every one judged unactionable). Requests is not: three papers scored 2, and *PyTrim: A Practical Tool for Reducing Python Dependency Bloat* reached its digest — a plausible change for that library. The label encodes arXiv coverage, and for one of the three it understates what exists. The density is still the point, though: 4 of 17 added papers, against a pooled precision of 0.854 elsewhere.

One harness defect surfaced from reading the shown lists rather than the metrics. The treatment showed **6 duplicate papers across 4 cases** and the control none; the identifiers gave it away (`2605.23815v1` beside `2605.23815`). The product version-strips arXiv identifiers before merging a non-arXiv source; the eval harness merged on the raw identifier, so both copies survived **[C-12]**. Removing each duplicate's contribution moves four cases and the corrections cancel exactly at the mean. Per-case results were contaminated; the conclusion was not.

### 9.6 The absent-category bias: real, large, and worth nothing

Both the void S2 finding and the live one blamed the same ranker rule, so we tested it directly. `score_paper` omits the category component when a paper has no categories — which is every paper from every non-arXiv source. Writing the two totals out, `(kw + w·cat)/(1 + w)` against `kw`, the uncategorised paper wins **iff `kw > cat`**: not the flat advantage we first claimed (a paper in exactly the right category still wins, 0.893 to 0.840), but the common case in a real pool, where category matches are partial or absent.

Three policies — `omit` (shipped), `zero` (score the absence as zero), `impute` (the pool's mean category score) — over **one frozen candidate pool**, which is what makes a ranking experiment affordable: MRE 0.48 rather than 1.04 (§8.8), one live seeding pass and three arms that collect nothing. Arms valid: 25/25 cases changed, mean Jaccard 0.66 and 0.69.

| arm | non-arXiv papers in the ranked top-10 | net@2 | precision | paired vs `omit` | 95% CI |
|---|---|---|---|---|---|
| `omit` | **32** | +4.04 | 0.893 | — | — |
| `zero` | **14** | +4.04 | 0.897 | **+0.00** | [−0.72, +0.76] |
| `impute` | **39** | +4.08 | 0.898 | +0.04 | [−0.40, +0.48] |

`zero` more than halves the non-arXiv presence in the top-10 — 18 of 32 papers moved out — and changes the output by nothing. A gate-free measure agrees: actionable papers reaching the ranked top-10 are 6.60 / 6.60 / 6.36 per case, so this is not the triage gate absorbing a ranking change. The change never mattered. **At that rank, non-arXiv and arXiv candidates are interchangeable in actionability** **[NR-33]**. Our hypothesis is refuted, and so is the explanation we reached for first (that the gate was absorbing it); the refutation is more useful than the fix would have been.

It also undermines the measurement that motivated it. The same `+s2` configuration scored **+3.24** live in the morning and **+4.04** on a frozen pool that evening — **0.80/case apart**, the second draw landing 0.08 from the arXiv-only arm. **NR-32's −1.05 is one draw and it does not reproduce**, and its displacement mechanism is independently undercut: displacing S2 papers back out, which `zero` does at scale, buys exactly nothing. We record this as a *correction* to NR-32 rather than a replacement, because settling it needs paired live arms in one session — which is what NR-32 was, and which one further draw cannot overturn. `sources: [arxiv]` stays the default on the weaker basis that S2 has not been shown to **help**.

### 9.7 What the section adds up to

**The coverage gap is not where we looked.** Five channels were built to reach literature arXiv does not carry. DBLP returned nothing for six months; bioRxiv returned everything; IACR is unvalidated at n = 2; Semantic Scholar, the one that competes broadly, has not been shown to help; and OpenAlex — measured last, at $0 — supplies the most genuinely non-arXiv content of any of them (~229 papers per repository, of which only 32 across the entire benchmark turn out to be a paper the arXiv pool already held) while winning **three** top-10 slots on merit across 25 repositories. Six of its fourteen slots are won in the negative controls, where the correct output is nothing, and five in the one repository whose arXiv pool was a quarter of the median, where there was nothing to outrank **[NR-34]**. The two standing recall losses of §8.5 — `llminfer` and `numerics` — are repaired by none of them.

**A ranking result we did not want.** Two policies producing visibly different top-10s produce statistically identical output. Everything downstream of the heuristic ranker — the triage gate, the fine-scale rescore, the derived 2/3 threshold — is doing the work that decides quality. A heuristic-ranker change now has to clear a high bar to be worth measuring at all, and the cheap pre-check is the gate-free count of actionable papers reaching the top-10: it moved 0.00 and −0.24 here, and would have predicted the null for $0.

**A failure mode with a name.** Two findings in this campaign were manufactured by arms whose channel returned nothing — IACR's first run and the S2 finding — and a third, the phrase-query arms, is known to be real only because the report had by then been built to check. §11 develops it as a rule.

---

## 10 Negative Results and Corrections, Catalogued

Positive results say what to build; negatives say where the walls are. Ours, with one-line mechanisms:

**Retrieval.**
- **NR-1** Repositories do not cite what would improve them: 0/24 targets in any repo's own docs. A bibliography indexes what a project already does.
- **NR-2** LLM "what does it use" phrases: 2/24 — accurate descriptions of the wrong register.
- **NR-3** LLM "what does it lack" phrases: 0/24 direct; worse than random as ranking queries even after string-matching is fixed — they name a plausible *different* agenda.
- **NR-4** Citation-count-sorted search: 1/24 — a multiplier on a match set that never contained the paper. Deeper fetching: ≤3/24, with 9 unreachable at any depth.
- **NR-5** Recency windows (the shipped 14-day default; the eval's 90-day mode): structurally exclude every gold target (all ≥11 months old); the shipped default became all-time/relevance.
- **NR-6** Synthetic hop seeds: 4/27, all in the one arXiv-native domain; hub-ranking the seeds grew pools 1.8–3.6× and recall by one — the wrong neighbourhood, enlarged.
- **NR-7** Gap-phrase re-ranking with stemmed BM25: best arm 1.1× a paste-the-profile control; ordering `uses` > `profile` > `lacks` > `gaps`.
- **NR-8** Coupling-degree pool filtering: 70% cut → 10% on the expanded benchmark; the filter was a property of the case set. (Its persisted pools remain load-bearing infrastructure.)

**Gate & context.**
- **NR-9** A stronger gate model (Sonnet): metric-identical to Haiku. Twice.
- **NR-10** Rubric engineering: missed its target leak, cost recall, destroyed the score-3 tier.
- **NR-11** BM25-RRF fusion (headline level): better nDCG everywhere, headline down — more candidates for the same imprecise gate. (Kept, for the ranking win; the headline cost was later recovered by the rescore.)
- **NR-12** LLM paraphrase of the README as gate context: below *no description at all*. Paraphrase destroys term-of-art vocabulary; verbatim extraction recovers +21.
- **NR-13** Semantic sentence selection vs first-300-chars: −4 (p = 0.78), a mechanism-backed prediction that failed.
- **NR-14** Verbatim user wants (top issues) in the gate: −38 net@2, recall −0.27 — a want-list *replaces* the question.
- **NR-15** Triage window 20→50: 2 papers across 12 cases. **Reversed 2026-08-14** (§8.9): re-measured on a frozen pool with depth the only variable, 50 beats 15 by **+1.00 net@2/case**, CI [+0.12, +1.92]. The original arm was confounded with a prompt change, ran at n = 12, and — decisively — predated the stage that orders what the gate admits.
- **NR-16** Gating the whole pool: −0.18 paired on 22 cases — a wash; nothing orders what the gate admits.
- **NR-17** Adaptive digest size: no interior optimum exists on an unranked set (3p−2), and the abstention signal it needs does not exist (r = +0.30).

**Band ranking (all pre-registered).**
- **NR-18** Listwise subset-selection with shuffle voting: both models far too strict; the published top-half-selection prior inverted. Contradicts the transfer of [16] to this setting.
- **NR-19** Persona-ensemble vote fractions with consider-the-alternative: killed at ECE 0.425 by chronic *under*-confidence.
- **NR-20** Pairwise Bradley–Terry: below the ordering bar at 7× E2's cost; synthetic borderline anchors rejected by 128/128 real papers — no absolute scale.
- **NR-21** Metadata features (citations, age, HyDE rank, coupling): AUC 0.585 — practitioner utility is not in the metadata, echoing [3].
- **NR-22** The combined model: loses to the single best score on every axis at n=22 queries.
- **NR-23** Monte-Carlo distribution reading via Haiku sampling: 0.59 vs 0.84 — a near-deterministic sampler re-reads its mode; exact logprobs are load-bearing.
- **NR-24** Recalibrating the frozen map: the decalibration is real (−0.129, CI [−0.187, −0.067]) and worth +0.00 net@2 under LORO, against an oracle ceiling of +0.27. The threshold sits in the trough of a bimodal score distribution, where a calibration error has nothing to move (§8.6).

**Register, documentation, and the benchmark's own blind spots.**
- **NR-26** Stated intent as a *register* fix: asking the same repository description a better question ("what would you improve?" instead of "what is this?") and feeding the answer to the query is worth **+0.12 net@2/case, 95% CI [−0.72, +0.96]** over 25 paired cases, with the thinness correlation at the wrong sign. Whatever benefit exists in the richer arm (+0.44, CI spanning 0) tracks the extra *information* — source code the profiler never reads — not the question.
- **NR-25** Graceful degradation under thin documentation — predicted, and false. Precision 0.925 → 0.636 and mean net@2 → −0.50; the gate's top tier falls to 0.53 while growing, and no internal signal moves. A similarity floor, the remedy the experiment was built to test, is refuted: the papers are close to the query, the query is wrong (§12.1).

**Sources beyond arXiv, and the queries sent to them (§9).**
- **NR-27** A domain source for the coverage gap (IACR ePrint for cryptography): shipped and measured on its two pre-registered repositories at **−0.50 net@2/case** against a minimum resolvable effect of 3.44 — no detectable benefit. Six of its papers reached contention and **five were judged non-actionable**: ePrint is dominated by *attacks on* primitives rather than improvements a library should *adopt*, which is the register mismatch of §5.2 reappearing in a source chosen to fix a coverage gap.
- **NR-28** bioRxiv as a source: its adapter keeps any paper containing a query token longer than two characters, and the malformed bridge query (C-9) supplied `and` — which matches **90 of 90** abstracts. The source did not add biology papers; it disabled the topical filter and merged the whole recent window. Checkably harmless downstream: no `biorxiv:` or `dblp:` identifier appears in any of the 78 recorded runs, so neither source ever reached a scored pool.
- **NR-29** Repairing the phrase-query generator (C-10): **+0.04 net@2/case**, CI [−0.64, +0.88], inside the 1.04 floor, because arXiv's category clause neutralises a meaningless phrase and the benchmark runs on arXiv. Deleting the phrase queries instead is measured **worse** (−0.48/case): a junk phrase is still a query returning 50 candidates.
- **NR-30** "Adding Semantic Scholar did not help" was **void, not null** — the malformed bridge query (C-9) returns one S2 paper across twelve cases against 20 per query repaired, so the reported −0.25 net@2 and the mechanism offered for it described a channel that had never delivered anything.
- **NR-31** S2 stage-1 yield ($0, no judge): the channel works — 218.6 papers/case, 174.5 of them non-arXiv, 73 reaching the ranked top-10 — but **30% of those top-10 appearances come from the three negative controls**, which are 3 of 23 cases. On Flask, S2 papers took all ten slots, outranking 287 arXiv papers.
- **NR-32** S2 measured end to end: **−1.05 net@2/case** on the 22 real cases, CI [−2.14, +0.00] — past the floor, interval touching zero. Read as displacement rather than addition (`thin-kv`: 8 papers shown in both arms, actionable 8 → 6). **Corrected by NR-33**: the same configuration scored +4.04 on a second draw, 0.08 from the arXiv-only arm, so the effect does not reproduce and the default survives only on "S2 has not been shown to help."
- **NR-34** OpenAlex, measured at $0 before proposing a judged A/B: it supplies the most genuinely non-arXiv content of any channel (~229 papers/repository; only 32 across the whole benchmark are a pool paper re-badged) and wins **3** top-10 slots on merit across 25 repositories. Six of its fourteen slots land in the negative controls and five in the one repository whose arXiv pool was a quarter of the median. The A/B was declined on this evidence — the third time a $0 probe has paid for itself.
- **NR-38** `ranking.w_embedding`, the last value shipping to users that no number covered: **1.5** in the config `rr init` writes, **0.0** in the dataclass and in every published result. Measured on one frozen pool it is worth **+0.64 net@2/case**, CI [−0.28, +1.80], sign p = 0.79 — **inside the 0.74 floor**, so unresolved rather than absent, and no default moves. Composition improved on every axis (208 papers shown against 195, 187 actionable against 173, precision 0.899 against 0.887), which is unusual: NR-35's fusion showed more at *lower* precision for +0.00. Our pre-registered prediction was −1.5 to +0.5 and **most likely negative**, reasoning from the one prior measurement of this channel — README embeddings as a *query* are bimodal, median rank 46,656. The transfer was invalid: as a query a bad match costs the whole result set, while as one weighted component over an already-retrieved pool a mediocre signal is diluted and its rank-1 tail is what shows. **Median rank is the wrong statistic for a component that only breaks ties near the top.**
- **NR-37** **Documentation volume predicts nothing on real repositories**, which closes the thin-documentation axis rather than one remedy on it. Corpus size is the one signal outside §12.1's coherent-failure loop — known before a profile exists, so it cannot be fooled by a plausible-but-wrong one — and on the *induced* grid it separates cleanly (median 115 characters at the 8 materially degraded points against 1,076 at the 16 intact). On the 25 real repositories it is **r = +0.14** (Spearman +0.20), and the thinnest quintile scores **+5.60 against +5.25** for the rest. `db` has 1,857 profiler-visible characters and no parseable manifest, and posts the **highest net@2 in the benchmark**; `webdev` (384,456 characters), `cli`, `linter` and `http` all score 0.0. Every abstention threshold is worth +0.00 to −2.20. The ablation and real repositories are **different populations**: §12.2 validated the ablation as a proxy for *precision* and that stands, but nothing licensed extending it to *identifying* badly-served repositories, and volume is not quality of description. The repositories this system serves worst are defined by their domain — the negative controls and the arXiv coverage gap — not by how much prose they carry.
- **NR-36** Letting the profiler read **source code** (`profiler.scan_source`, shipped and never enabled by any arm) is worth **−0.52 net@2/case**, CI [−1.72, +0.72], sign p = 0.33 — inside the floor, with the point estimate negative on all three measures (gate-free actionable-in-top-15 9.00 → 8.80; precision 0.904 → 0.878). Proposed as the thin-documentation remedy on the reasoning that such a repository is thin in prose but has code, it is **worst exactly there**: the thin cohort falls +4.33 → +2.33, a direction predicted in advance from a $0 profile probe which found `thin-kv` acquiring `grafanalib` and `executor_panels` — its Grafana dashboards rather than its storage engine. n = 3, so that half is suggestive, not established. The shipped default is thereby **vindicated**, the first never-measured default in this campaign that was already right (§8.9's two were wrong by +1.00 and +1.24). Measures source scanning *as shipped*: `source_extensions` covers Python and JavaScript only, so six of 25 cases profile identically either way.
- **NR-35** BM25-RRF fusion, ablated inside the *measured* configuration for the first time — it had been on in every headline since the day it shipped, kept on NR-11's pre-rescore argument. Removing it is worth **+0.00 net@2/case**, CI [−1.00, +0.96], sign p = 1.000, inside the 0.74 floor — while changing **25/25 returned sets (Jaccard 0.41)** and 17 of 25 case scores. The pre-registered gate-free measure settles the mechanism rather than the outcome: actionable papers reaching the ranked top-15 before gating are **8.80 with fusion against 8.72 without**, so the "it ranks better" argument does not convert into more actionable candidates where it would have to. Retained in the shipped preset because every published number was measured with it and the preset's value is reproducing that configuration — but retained as a component we cannot show earns its place, which is a weaker claim than this paper previously made for it (§8.5).
- **NR-33** The absent-category ranking bias: real and large in composition (`zero` halves non-arXiv papers in the top-10, 32 → 14) and **+0.00 net@2/case**, CI [−0.72, +0.76], with a gate-free measure agreeing (6.60 vs 6.60 actionable in the top-10). At that rank arXiv and non-arXiv candidates are interchangeable in actionability — and anything downstream of the heuristic ranker is doing the work.

**Corrections of our own published numbers** — each recorded in place, with the wrong number retained as a warning.

- **[C-1]** The citation hop's first figure (14/24) was a transport artifact of a batch API's silent nested truncation, and two mechanism claims drawn from it were withdrawn; the same defect was silently costing shipped code 29% of its citation edges.
- **[C-2]** "More prose is worse" — never statistically supported; the argmax of four noisy arms was over-read as a curve.
- **[C-3]** A "+16 for README context" result had actually measured a packaging one-liner *plus* deletion of the keyword block, in a harness-rebuilt prompt: the eval now calls the shipped prompt builder, by rule.
- **[C-4]** The full-pool comparison's first delta (−0.95) was three-quarters an unhandled arXiv throttle scored as an honest zero.
- **[C-5]** §8.3's calibration-by-AUC trap (+2.91 → +3.14).
- **[C-6]** The calibration audit of §8.6 was opened on a +11 net@2 estimate read off the unshown-but-actionable papers in our own top-10; that accounting counts a threshold move's gains without its losses (−80 on the repositories where the gate is correctly strict), and the paired version of the same question is zero.
- **[C-7]** "Zero net-negative repositories" held in every run it was measured on and broke in the 25-repository draw (`numerics`, one paper, judged 0, −2.0) — a per-draw property reported as a property of the method; and the +4.55 headline it accompanied was itself a favourable draw, the same 22 repositories scoring +3.91 on re-run (§8.7).
- **[C-8]** Single-run p-values were reported as if stable: the same 25-repository configuration gives *p* = 0.041 and *p* = 0.0001 on consecutive draws, and the benchmark's minimum resolvable effect (1.04 net@2/case) was never measured until after several experiments had been sized below it — and the floor turned out to depend on the digest width as well as on pool provenance, so the tool deriving it was returning a value 35% too tight for every experiment at the shipped width.
- **[C-9]** Every non-arXiv source (DBLP, bioRxiv, OpenAlex, Semantic Scholar, and later IACR) was sent arXiv boolean syntax as a keyword query, and only surfaced when a newly added source returned exactly zero rather than degrading quietly. **[C-9a]** Our first account of it — a translation step written for an earlier query shape, gone stale when query construction changed — is itself wrong, and the correction matters more than the bug: git dates the parenthesised builder to 2026-02-22 and the one-liner to 2026-02-23, so there was no working era and no drift. It never worked. **[C-9b]** The repair was published as "routed through all three call sites"; there were five and it routed two, with the shared translator correct, unit-tested, and unused by three of its callers (§9.1).
- **[C-10]** The phrase-query generator paired each keyword with its TF-IDF neighbour and quoted the pair without ever checking the two words co-occur, emitting `"use page"` and `"data cd"` as three of the five queries every source receives (§9.3).
- **[C-11]** Three of the four modules calling Semantic Scholar had **no** rate limiting, and the fourth spaced only the requests within a single call; retry backoff had been standing in for a rate limiter, which fires only after the server has been hit too fast. A shared process-wide gate replaced them, and two further defects showed up only against the live API — an interval that undershot its own target by 7 ms on Windows, and a disable switch that did not clear a pending throttle hold.
- **[C-12]** The eval harness merged non-arXiv sources on raw arXiv identifiers while the product version-strips them, so `2605.23815v1` and `2605.23815` both survived: 6 duplicate papers across 4 cases in the S2 treatment arm, none in the control. Per-case values were contaminated; the corrections cancel exactly at the mean (§9.5). **[C-12b]** A *third* runner had the same defect and the guard written to prevent it read one file by name, so it stayed green (§11, lesson 10).
- **[C-13]** The digest promoted an **ungated** paper to Top Picks on the heuristic `score_total ≥ 0.5` — the threshold the LLM gate was built to replace, at mean net@2 −11. It fires when the digest window is wider than the gate's window, or when a gate call fails; the gate's own docstring described the strict rule the benchmark scores, and the shipped tiering did the opposite. Blast radius on published numbers is zero and checkably so: across 87 recorded runs and 6,420 returned papers the gate failed on exactly none.
- **[C-14]** A **third** implementation of "is this the same paper": a bare `arxiv_id.split("v")[0]` at eight call sites beside `dedup_id`, which was itself too narrow — it left versions on pre-2007 identifiers, five of which sit in our judged pools, so a source merge and the judge pool five steps later disagreed about whether two records were one paper. **[C-14b]** The guard written for that fix inherited the same defect it was written to prevent: it read the five pipeline modules where the bug had been found, and a later sweep across all 102 modules turned up *three* competing rules — including one in two signal collectors that survives the truncation bug and then edits opaque identifiers instead. All three agree on a modern arXiv id, which is why only a survey could find it.
- **[C-19]** Adding a flag to the frozen-pool fingerprint's `POOL_FLAGS` **invalidates every stored pool**, and doing so for `rr_scan_source` (NR-36) silently removed the project's cheap, sensitive experimental mode — the one that took the floor from 1.04 to 0.48/0.74 — until the next experiment tried to use it and was refused. The refusal is correct: a pool carries no value for a dimension that did not exist when it was collected. Omitting default-valued flags from the hash would make new flags free but let a pool collected under an OLD default match a run under a NEW one, and this project changes defaults; that trades a loud failure for a silent one. What was wrong was the *diagnosis* — two opaque hashes — so pools now record the flag set they were fingerprinted over and a mismatch names what changed. A second guard in the same experiment was over-strict rather than wrong: it refused to compare an arm that *seeded* a pool against one that *reused* it, despite identical fingerprints, which would force a throwaway collection before every frozen pair; fixed at the comparability check and not in the shared provenance function, since the seeding run really did collect live.
- **[C-18]** The noise-floor guard repeated C-8 in the half its own repair did not reach. `mre_for` was made width-aware after a window-10 floor was applied to a window-15 arm — but only its *frozen* branch was; the **live** branch returned early, ignored width, and hard-coded the label "window 10". It printed a window-10 floor for the live window-15 arms of NR-36. The frozen floor rose 0.48 → 0.74 (×1.54) when the cut widened, so 1.04 **understates** the live floor at 15, and under-stating a floor turns noise into a finding — the direction the function's own docstring warns about. No published result moves (−0.52 is inside either value), but any live window-15 effect between 1.04 and roughly 1.6 would have been reported as resolved. The live branch now keys on width, an unmeasured width is labelled a **lower bound**, and a "past the floor" verdict against a lower bound prints *NOT resolved*. No scaled estimate was substituted: 1.04 × 1.54 would be an unmeasured number wearing a measurement's authority.
- **[C-17]** We quoted the ungated default at **−11** in a user-facing table headed "on the 25-repository benchmark". It was measured on **four** repositories in July 2026, one of them a negative control, before the retrieval repairs and at a narrower digest — it had never been on this benchmark at all. Measured properly it is **−8.12** (§8.9). Two things make this more than a citation slip: the false attribution was introduced by us in the very change that shipped a *warning about unmeasured configurations*, and our pre-registered prediction for the true value (−2 to +2) was **wrong by more than the stale number was** — we predicted retrieval gains would dominate and forgot that `net@2` charges for every additional paper a rule with no abstention will show.
- **[C-16]** The product/benchmark configuration audit reported "no undeclared divergence" while comparing **12 of 79** configuration fields, hand-listed, against the wrong object: `rr init` writes a template, and where that template sets a value the dataclass default the audit read is dead text. Made exhaustive over every leaf and both surfaces, it reports **seven** divergences — including that the shipped configuration enables neither the gate, nor the rescore, nor HyDE, nor the hybrid fusion, so a new user's digest is the ungated one measured at −11 (§8.9). The audit that found C-9's and C-12's shape had the shape itself. Its blast-radius pass then had a second one: the frozen-pool scanner knew only the v2 format, read a 1,250-paper v1 pool as **0 papers**, and printed "0 duplicates" about a file it had never parsed — *void, not null*, inside the very check written to quantify that class of error.
- **[C-15]** The two eval runners had opposite failure policies: the Tier B harness raises on a collection failure (the rule C-4 was paid for) while the Tier A runner warned and continued — which, with a second source enabled, scores a case on the non-arXiv half of its pool and prints a number indistinguishable from a real one.

---

## 11 Methodological Lessons: How Instruments Lie

Across six weeks, the experiments that nearly published wrong conclusions outnumber the model-capability surprises. The recurring classes, each now guarded by tests:

1. **Caches that outlive their assumptions.** A judge cache keyed without the rubric let a rubric-swap experiment silently overwrite the gold set's verdicts (nine targets vanished; caught by a test pinning the frozen target list). A baseline cache is keyed by a discriminator over mode/context/flags for the same reason. An empty result cached as success — seven empty candidate pools after a 429 storm — is worse than no cache: the next run skips it as done, and the repositories vanish from every downstream number while reports stay confident.
2. **Failure indistinguishable from absence.** A hop function dropped rate-limited chunks with a bare return: "throttled" and "these seeds cite nothing" were identical to every caller, and a 90%-undercounted pool *reported success*. The pattern's fix is mechanical: return failure structurally (`failed_chunks`), and refuse to persist partial results.
3. **Truncation correlated with the verdict.** Twice: judge responses truncated at max_tokens dropped scores of 2/2/3 (skewing base rates down), and a selection model's longest outputs — exactly the select-everything responses — were the ones that failed to parse. A dropout that correlates with the outcome is not noise.
4. **Partial runs overwriting whole-set artifacts.** Three separate scripts destroyed full-set results via a `--case` smoke run before merge-by-key became the standard write pattern.
5. **The harness measuring itself.** Rebuilding the system's prompt inside the eval produced a published result about a prompt the system never sends [C-3]. The rule since: the harness calls the shipped builder, and — for the calibrated stage — a test asserts byte-identity.
6. **An experiment that "confirms" on empty data.** A missing API key produced empty score lists, which averaged to 0.0, which satisfied the comparison the script then printed as a confirmed negative — agreeing with the conclusion already drafted. Verdict printing now requires every cell non-empty.
7. **Bug signatures in too-perfect agreement.** A loop variable shadowing a parameter made a variant silently re-run its control; byte-identical pools on 10 of 11 cases was the only tell. Perfect agreement between arms is a red flag, not a replication.
8. **Small-n statistics discipline.** Per-case sd on the headline metric is ≈1.7, so single-case swings flip 4-case means [NR-11]; the 12-case set was jackknife-fragile (§4.5); every headline since is paired, sign-tested, and reported with its non-significance where true. Corrections of over-claims from noisy argmaxes [C-2] and small-sample enrichment artifacts (a 0.00× that became 2.32× at full n) are recorded inline.
9. **Void, not null — an arm is not a measurement until you have counted whether the channel delivered anything.** This class cost two findings outright and nearly a third: IACR's first run scored both arms identically because zero IACR papers ever reached a top-10 [NR-27]; "adding Semantic Scholar did not help" described a source that returned one paper across twelve cases [NR-30]; and the phrase-query arms are known to be real only because the report had by then been built to check. A void arm's delta of 0.00 is indistinguishable from a null and reads as one. Every source arm now prints, before its delta, how many cases changed their returned top-10 and the mean Jaccard against the control; a report refuses to interpret an arm at 0/25.
10. **Correct code that nothing calls, and guards whose tests avoid the failing case.** A shared query translator was correct, unit-tested, and unused by three of its five callers [C-9b] — a unit test of a function cannot detect that nothing calls it, so the replacement is an AST-level *wiring* assertion over the call sites, mutation-verified. The same shape appeared in the frozen-pool provenance guard: it folded per-case pool fingerprints into a set, so any run over more than one case reported `mixed`, and two arms drawn from **different** pools would have matched and compared cleanly. It shipped because every test used a single-case run — the one shape where the bug is invisible. And it recurses: the guard written after the C-12 merge bug read *one file by name*, so it stayed green while a third runner two files away had the identical defect [C-12b]. **A guard scoped to the site you found the bug at is a guard against finding it again** — a rule we then broke ourselves within a day, writing a guard over the five modules the last two instances had lived in [C-14b]. Since two published corrections share this exact shape, we now search for it on purpose rather than by accident — a $0 static pass over every module in the pipeline that reports each place an invariant is implemented, which of the competing rules it uses, and how much of the difference reached a published number. Its first run found five divergences, one of them a live product bug [C-13]. The rule then caught the audit a third time, on the axis it had not thought to check: its configuration pass compared *twelve hand-listed fields* against the dataclass defaults, while the file `rr init` actually writes is a template that overrides some of them — so it was scoped by hand **and** reading the wrong object, and it reported clean while the shipped product ran none of the pipeline stages the paper measures [C-16]. The repair is the one generalizable move here: stop listing what to check, enumerate the space and require every element to be either compared or excused **in writing**, so a new field fails the audit until somebody decides. A list of what you checked cannot tell you what you did not.
11. **A rate limit is not a volume limit.** A correct process-wide 1-request/3-seconds gate was applied to every arXiv attempt, retries included, and we were throttled anyway. The signature distinguishes the two: a rate violation fails early and uniformly, while ours failed at cases 24 and 25 after ~162 successful requests in that run and ~760 across the day. A 25-case sweep is 174 byte-identical queries, and four sweeps in a day re-fetched the same pool four times. The fix was a response cache keyed on query, result count and sort order — 174 requests to 0 on a warm repeat — kept out of the product, because serving a six-hour-old answer to a daily digest is a behaviour change nobody measured. Its one subtlety is worth stating: the cache initially refused to store *any* empty result, on the sound rule that "found nothing" and "refused" are the same bytes on disk, but the one caller that raises on exhausted retries can *prove* its empties are answers, so the guarantee is now stated at that call site rather than assumed globally.
12. **Size against a plausible effect, not against the ceiling — and breadth is not power.** Two sizing errors, opposite in shape. The IACR subset was justified by comparing its resolvable effect (3.44) against the *headroom* (8.5, the distance to a perfect score); headroom is not a plausible outcome, and detecting a realistic +1.5/case there needs n ≈ 11, not 2. The phrase-query experiment was justified by touching 25 cases rather than 2 — which ignored that arXiv's category clause neutralises the defect, so the arm had no headroom however many cases it ran on.
13. **Magnitude and significance are two claims, and a report will happily conflate them.** Our source-comparison script printed "RESOLVED" when |mean| exceeded the noise floor, and its interval test was written as a sign-agreement check, so it called [−2.14, +0.00] an interval excluding zero. Both now print separately: whether the effect is large enough for this benchmark to see, and whether *this* draw established it.

None of these are exotic. They are what "evaluation infrastructure" actually consists of, and we would not trust the headline numbers of any comparable system whose reports do not mention fighting them.

---

## 12 Limitations

**The judge defines the task.** All labels are one frontier model's rubric-guided opinion; the validity program (§4.4) bounds this — a second judge ranks identically at different strictness; adopted papers score 61% vs a 2% floor — but does not remove it. Absolute rates are judge-relative (a ~0.55 multiplier under the stricter judge); paired comparisons largely cancel the offset.

**The metric rewards shyness.** net@2 charges 2 per false positive, so a precision stage — our main result — is exactly what it flatters, and it flatters the precision-preserving pool expansion of §8.5 the same way. The recall deficit it structurally undervalues (gate recall 0.60) is narrowed but not closed: three of the six pure-retrieval losses are repaired, two remain, and both are repositories whose bibliographies are too thin for the citation hop and whose targets the dense channel does not reach either. A user who wants discovery may prefer a different λ, and nothing here optimizes reading-time value directly.

**n = 22, and the significance is cumulative.** The final configuration clears p = 0.0075 against the baseline, but neither shipped stage clears p < 0.05 on its own increment (the rescore's live run p = 0.45; HyDE's p = 0.092) — what is established is the system, not either component. The benchmark is English and arXiv-centric; 22 of its 25 cases are popular OSS. It is also running out of headroom in the direction we improved: four repositories now return ten actionable papers out of ten, so further recall gains there are invisible to the metric.

**The target user was not in the benchmark, and the system degrades sharply and silently on them (§12.1–12.2).** *Degrades*, not *fails*: the thin trio scores **+2.00**, positive but roughly three points under its thick partners, and this preamble previously said "fails on them silently" — contradicting the section it cites. A later measurement narrows it further and is the more useful statement: **documentation volume does not predict outcome across the 25 real repositories (r = +0.14), and the thinnest quintile scores slightly better than the rest** [NR-37]. The deficit the thin trio shows is real, but "thin documentation" is not the axis it lies on — the ablation that named that axis induces a damage real repositories do not exhibit. Across the original 22 repositories the documentation corpus the profiler actually reads — README plus `docs/` — has a floor of 1,857 characters and a median of **194,999**, so every measurement of what to tell the system about a repository was made under a documentation surplus of three orders of magnitude. Three real thin-documentation repositories have since been added (§12.2), the thinnest at 108 characters.

**One source, and a benchmark shaped like it.** Everything here is measured on arXiv, and §9 is the account of why: of five channels built to reach other literature, four had never worked and the fifth has not been shown to help. That leaves two coupled limits. The system's coverage is arXiv's — for the cryptography and storage repositories in the benchmark, near-abstention is the best it can structurally do, and the `gold_n: 0` label on the negative controls turns out to encode arXiv coverage rather than the absence of useful research (§9.5). And the benchmark inherits the same shape, so it is **blind to defects that only bite elsewhere**: a query generator emitting phrases no repository contains is invisible on arXiv, where a category clause keeps results in the right field regardless, and clearly broken the moment the same query reaches a keyword API (§9.3). We would expect any arXiv-only evaluation of a multi-source system to share this blind spot.

**The measured system is not the shipped default.** Every number in this paper describes a configuration with the gate, the rescore, HyDE and the hybrid fusion all enabled; the configuration `rr init` writes enables none of them, because each needs a credential, a second vendor, or a 1.1 GB index (§8.9). The gap is documented and each opt-in is defensible in isolation, but it means our headline measures what a *fully configured* installation achieves, and the digest a new user sees on day one is the ungated one this paper measures at −11. We have since run the out-of-the-box arm rather than guessing, and it lands at −8.12 (§8.9) — net-negative on 19 of 25 repositories, and ten points below our own pre-registered prediction. What we have done is remove the *ambiguity*: the measured configuration ships as a named preset whose fields are asserted against the benchmark's, and the default announces its own weakness at install and at run time. That narrows the limitation to what it really is — an unmeasured operating point — rather than leaving it as a reader's incorrect assumption about which system the headline describes.

**Calibration freshness.** The two fitted parameters are frozen against a specific prompt and judge vintage. Tests catch prompt edits — the shipped block is asserted byte-identical to the one the coefficients were fitted against — but not slow semantic drift in what "actionable" means, which is why the map is re-measured against live judge verdicts rather than assumed (§8.6).

**Vendor coupling.** The winning mechanism requires token logprobs, which only one major API exposes (§7, NR-23); the method is one `logprobs=true` flag away from vendor-neutral, but today the dependency is structural.

### 12.1 The thin-documentation failure, measured

We ablated the profile's sources — capping the README and withholding `docs/` for the system while the judge continued to see the real repository — across four budgets on six repositories, one session, documentation the only variable. Mean net@2 falls +5.17 → +3.00 → +3.17 → **−0.50**, and pooled precision **0.925 → 0.636**. Both pre-registered alarm conditions fired at the floor; the pre-registered prediction (decay toward zero *by abstention*, precision holding above 0.85) failed. The actionable share of the judged pool falls monotonically, 0.840 → 0.568, so this is retrieval degradation rather than a selection failure.

**The mechanism is our own founding diagnosis under laboratory conditions.** Strip a repository's prose and its profile collapses to a bare self-description, so retrieval returns what the repository *already is*. At the floor, `speech`'s digest opens with **the Whisper paper itself** — judged unactionable for the Whisper repository — followed by seven near-neighbours of Whisper's own method. The register mismatch (§5.2) is not a quirk of keyword search; it is what happens whenever the only thing describing a repository is that repository.

**The danger zone is a little documentation, not none.** `db` (DuckDB, no parseable manifest) profiles to *literally nothing* at the floor — zero keywords, zero domains, zero prose — and the gate scored all ten candidates 0 and abstained: net@2 0.0, safe. `speech`, whose thin profile was entirely plausible, reached **−13.0**. The failure requires enough information to form a plausible but *wrong* question, which inverts the intuition that more context is monotonically safer.

**Nothing inside the system notices.** The gate's top confidence tier — 76% judge-3 in our wild-pool labelling, 1.00 precision in the control arm — falls to **0.53** at the floor while issuing *more* 3s, and the calibrated probability barely moves (0.799 → 0.709). Queries, hypotheses, gate and rescore all consume the same impoverished profile and therefore fail **coherently**; internal consistency is preserved exactly. The judge is the only component seeing the real repository, and thus the only available detector. This also refutes the remedy the experiment was designed to test: we expected a similarity floor on the dense channel's top-*k*, but the retrieved papers are not distant — they are the correct answer to the wrong question, and no distance threshold separates those.

We report this as the sharpest limitation in the paper rather than a solved problem. Three caveats bound it: n = 6 with `speech` supplying most of the magnitude (excluding it, the floor is +2.00 at precision 0.800); the per-case sign tests do not reach significance (p = 0.125 at the floor), so the alarm is a pre-registered decision rule and not a demonstrated effect; and **ablation yields a ceiling, not an estimate** — removing detectron2's README does not make our models forget detectron2, and a genuinely obscure private codebase receives none of that help. **[NR-25]**

### 12.2 Three real thin-docs repositories, and how far the ceiling was off

Because ablation cannot answer its own question, we added three real repositories to the benchmark — obscure (14–19 stars, so plausibly unmemorised), actively maintained, and each **paired with a thick-docs case in the same domain**, so a thin-versus-thick difference is not confounded by domain or by whether literature exists at all. All six ran in one session under a uniform agent-baseline budget.

This first required correcting how "thin" was being measured. We had quoted the benchmark's thinnest README as 1,639 characters; that governs only the 300-character prose block, while the profiler reads README **plus `docs/`**, and that repository carries 78 documentation files totalling 384,456 characters. By corpus the true floor was 1,857 and the median case is **194,999**. The new cases sit at **108**, 1,073 and 3,556 — the first is roughly 1,800× below the median.

| | RepoRadar net@2 | agent baseline | digest precision | pool actionable share |
|---|---|---|---|---|
| thick trio | **+7.00** | +4.00 | **21/21 = 1.000** | 0.833 |
| thin trio | **+2.00** | +1.67 | **14/18 = 0.778** | 0.622 |

Precision is the clean signal and it lands **inside the band the ablation predicted** (0.853 at its 300-character rung, 0.636 at zero). Fisher's exact on shown papers gives p = 0.037, which we report alongside the objection that papers are not independent units — they cluster by repository, so the figure overstates its power. This was the pre-registered check on NR-25: had the thin cases scored like their thick partners, the ceiling would not have been one. They did not, so the finding stands, and the ablation looks like a *reasonable* proxy rather than a wildly optimistic one.

Two further observations. **Both systems degrade, and RepoRadar degrades about twice as hard** (−5.00 against the baseline's −2.33): thin documentation is a property of the task, not a defect unique to this system, but the +3.00 margin RepoRadar holds over the baseline on thick repositories collapses to **+0.33** on thin ones. And the agent baseline **failed outright on two of three thin cases** at the turn limit that has never once bound on the 22 thick cases — reading unfamiliar code to discover what a project is turns out to be expensive for an agent, which is the same information deficit expressed as compute rather than as precision.

**The +2.00 replicates, and it decomposes in a way that matters for what to do next.** Three further draws of the current shipped configuration on the same three repositories give a cohort mean of **+1.67, +2.00 and +2.67** against **+5.32, +5.27 and +5.45** for the other 22 — so the roughly three-point deficit is climate rather than weather, which is not something a single draw of this benchmark can establish (§8.7). But the cohort mean conceals its own composition: `thin-kv` scores **+5.0 in all three** and `thin-gnn` +2.0 to +5.0, while `thin-lang` — the 108-character case — scores **−2.0, −4.0, −2.0**. One repository of three carries the entire deficit.

That has a direct methodological consequence we state rather than discover later: **this cohort cannot support a measured thin-documentation remedy.** At n = 3 with one dominant case, against a floor of 0.74, only an effect of several points per case is detectable, and a subgroup claim resting on one repository is precisely the fragility a jackknife exposed at n = 12 (§4.5). Any remedy we test on these three will be a hypothesis, not a result, until the cohort is large enough to carry a subgroup. We record the deficit as real and its remediation as currently unmeasurable — two different claims that a single "thin docs is broken" would have merged.

One claim from the ablation did **not** survive contact with real repositories. We had reported that a repository whose profile ablates to nothing abstains safely, and the 108-character case did abstain in its first draw — the gate admitted five papers and the calibrated rescore cleared none, which read as a clean demonstration that the second stage protects the first. In a second draw it returned one paper (actionable). Both outcomes are safe, but *"it abstains"* is not supported; *"it does not produce junk"* is.

---

## 13 Conclusion

The system-level story is a funnel whose bottleneck moved twice: from *retrieval* (solved not by better queries but by routing around a register mismatch — citation hops and hypothetical-document search), to *selection* (an LLM gate that transformed the product and then stalled), to *ordering within the gate's own blind spot* — a near-binary score distribution hiding a 0.00–1.00 precision spread. That the two shipped stages **compose** is the part we would not have predicted: the pool expansion had been measured as a wash a month earlier, and only became worth +1.36 once something ranked what the gate admitted. Neither stage is significant alone; together they are (p = 0.0075), which is an argument for measuring systems rather than components. The method-level story is that, for this utility-judgment setting, the literature's comparative machinery (listwise selection, pairwise tournaments, ensembles) failed with a consistent strictness pathology, while the smallest possible change — the same pointwise question at finer granularity, read as a distribution rather than a sample, calibrated with two parameters to the metric's own breakeven — beat a strong agentic baseline at one-eightieth its cost ($0.01 vs $0.80 per repository). The last stretch of the campaign added a third story we would have preferred not to tell: the four alternative paper sources built to cover what arXiv misses had, between them, never delivered a usable paper, because a one-line bridge between two query grammars was wrong from the day it was written — and the finding that "a second source did not help" had been published about a channel that returned nothing at all. When the sources finally worked, the ranking bias we had blamed for their behaviour turned out to move half the non-arXiv papers into or out of the top-10 while changing the output by exactly nothing, which says something uncomfortable and useful about where the value in this pipeline actually sits: not in the heuristic ranker, but in everything downstream of it. The meta-level story, which we believe most transferable: with a benchmark honest enough to be debugged itself, pre-registration cheap enough to be habitual, and negative results recorded at the same fidelity as wins, six weeks of measurement can replace an architecture of accumulating beliefs with a short list of things that are actually true — provided you check, every single time, that the arm you are measuring returned anything at all.

---

## Reproducibility

Every quantitative claim maps to a script in `evals/`, and the section-by-section index of which script re-derives which result is in `evals/README.md`. That index is itself checked: a test asserts every script it names exists, and — the direction that matters — that every script in `evals/` is either indexed or explicitly declared non-paper with a reason, so the index cannot quietly cover whichever subset someone remembered. It is worth saying why this exists at all: for several weeks this sentence promised scripts "named per section in the artifact" while no such naming existed anywhere in the repository, which is the same unverifiable-assertion shape §10 catalogues, committed about our own reproducibility claim. Scripts run against cached labels where possible ($0 re-derivation), and each result is recorded with its cost. The benchmark definition, 5,100-line results log with inline corrections, frozen testbeds, and all experiment code are in the repository. Total measured spend for the full campaign is on the order of $240, dominated by judge calls and agentic baselines; the three $0 stage-1 probes of §9 (source yield, query-transform audit, dependency verification) are the reason it is not considerably more.

## Acknowledgements

The experimental campaign was executed in collaboration with Anthropic's Claude (Opus 4.x / 5 family) operating as an autonomous research-engineering assistant under human direction and review; all merges, and therefore all errors of judgment that survived review, are the human author's.

---

## References

[1] Z. Zhang, et al. *Are Large Language Models Good at Utility Judgments?* SIGIR 2024. arXiv:2403.19216.

[2] H. Shao, et al. *paper2repo: GitHub Repository Recommendation for Academic Papers.* WWW 2020. arXiv:2004.06059.

[3] D. Lo, N. Nagappan, T. Zimmermann. *How Practitioners Perceive the Relevance of Software Engineering Research.* ESEC/FSE 2015.

[4] L. Gao, X. Ma, J. Lin, J. Callan. *Precise Zero-Shot Dense Retrieval without Relevance Labels* (HyDE). ACL 2023. arXiv:2212.10496.

[5] S. Lee, et al. *mxbai-embed-large-v1.* Mixedbread AI model card, 2024. `huggingface.co/mixedbread-ai/mxbai-embed-large-v1`.

[6] *arXiv abstract embeddings, binary MRL snapshot (3.1M vectors, Apache-2.0).* HF dataset `bluuebunny/arxiv_abstract_embedding_mxbai_large_v1_milvus_binary`, 2026-07 snapshot.

[7] S. Robertson, H. Zaragoza. *The Probabilistic Relevance Framework: BM25 and Beyond.* FnTIR 2009.

[8] G. Cormack, C. Clarke, S. Buettcher. *Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods.* SIGIR 2009.

[9] A. Singh, et al. *SciRepEval / SPECTER2: scientific document representations.* 2022. arXiv:2211.13308.

[10] O. Weller, et al. *FollowIR: Evaluating and Teaching Information Retrieval Models to Follow Instructions.* 2024. arXiv:2403.15246.

[11] W. Sun, et al. *Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents* (RankGPT). EMNLP 2023. arXiv:2304.09542.

[12] X. Ma, et al. *Zero-Shot Listwise Document Reranking with a Large Language Model.* 2023. arXiv:2305.02156.

[13] Z. Qin, et al. *Large Language Models are Effective Text Rankers with Pairwise Ranking Prompting.* NAACL 2024. arXiv:2306.17563.

[14] S. Zhuang, et al. *A Setwise Approach for Effective and Highly Efficient Zero-shot Ranking with Large Language Models.* SIGIR 2024. arXiv:2310.09497.

[15] Y. Chen, et al. *TourRank: Utilizing Large Language Models for Documents Ranking with a Tournament-Inspired Strategy.* 2024. arXiv:2406.11678.

[16] See [1].

[17] L. Zheng, et al. *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.* NeurIPS 2023. arXiv:2306.05685.

[18] A. Bansal, et al. *Peering Through Preferences: Unraveling Feedback Acquisition for Aligning Large Language Models.* 2023. arXiv:2308.15812.

[19] Y. Liu, et al. *Aligning with Human Judgement: The Role of Pairwise Preference in Large Language Model Evaluators* (PairS). COLM 2024. arXiv:2403.16950.

[20] K. Tian, et al. *Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from Language Models Fine-Tuned with Human Feedback.* EMNLP 2023. arXiv:2305.14975.

[21] M. Xiong, et al. *Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs.* ICLR 2024. arXiv:2306.13063.

[22] P. Verga, et al. *Replacing Judges with Juries: Evaluating LLM Generations with a Panel of Diverse Models.* 2024. arXiv:2404.18796.

[23] J. Zhang, et al. *Trust or Escalate: LLM Judges with Provable Guarantees for Human Agreement.* 2024. arXiv:2407.18370.

[24] Y. Liu, et al. *G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment.* EMNLP 2023. arXiv:2303.16634.

[25] *TrustJudge: Inconsistencies of LLM-as-a-Judge and How to Alleviate Them.* 2025. arXiv:2509.21117.

[26] *Likert or Not: LLM Absolute Relevance Judgments on Fine-Grained Ordinal Scales.* 2025. arXiv:2505.19334.

[27] O. Weller, et al. *Rank1: Test-Time Compute for Reranking in Information Retrieval.* 2025. arXiv:2502.18418.

[28] R. Tang, et al. *Found in the Middle: Permutation Self-Consistency Improves Listwise Ranking in Large Language Models.* NAACL 2024. arXiv:2310.07712.

[29] *Large Language Models are not Fair Evaluators.* 2023. arXiv:2305.17926.

[30] *JP-TL-Bench: Anchored Pairwise LLM Evaluation for Bidirectional Japanese-English Translation.* 2026. arXiv:2601.00223.

[31] *From Uncertain Judgments to Calibrated Rankings: Conformal Elo Estimation for LLM Evaluation.* 2026. arXiv:2606.13221.

[32] J. Carbonell, J. Goldstein. *The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries.* SIGIR 1998.
