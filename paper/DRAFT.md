# From Topic Match to Actionability: Building, Debugging, and Calibrating a Paper-Recommendation System for Code Repositories

**Raimondas L.**¹
¹ *RepoRadar project* — `github.com/raimondasl/auto-features`

*Draft of 2026-08-09. Comments welcome.*

---

## Abstract

We describe RepoRadar, a system that recommends arXiv papers a software repository's maintainers should *act on* — a utility judgment, not a topical-relevance one — and the five-week measurement campaign that turned it from a confidently wrong keyword ranker (mean net@2 of −11 on its user-facing output) into a system that beats a strong agentic baseline on a 22-repository benchmark (**+4.55 vs +1.82** mean net@2, paired +2.73, 15 wins to 3, sign test *p* = 0.0075). The campaign produced three findings we believe generalize. First, **retrieval for actionability fails on a register mismatch**: a codebase's vocabulary describes what it *has*, while useful papers describe what it should *adopt*; every query channel derived from repository text failed (0–8% recall), while channels that route around the mismatch — a citation hop through the repository's own bibliography, and HyDE-style hypothetical abstracts against a dense index — reach 75% in union; shipping the latter is worth +1.36 mean net@2 end to end and repairs three of the six repositories that had been losing to the baseline on pure retrieval misses. Second, **LLM judges of utility are near-binary in practice**: a 0–3 actionability gate concentrated almost all admitted papers on one score, beneath which true precision varied from 0.00 to 1.00 per repository with no visible signal. Comparative and ensemble judging — the literature's standard remedies — failed in our setting (three of five pre-registered experiments; a consistent failure mode of excess strictness), while simply re-asking the question on a 0–9 scale and reading the *expectation over the answer-token distribution* ordered the hidden band at ROC-AUC 0.84 and, through a frozen two-parameter calibration, eliminated every net-negative repository. Third, **most of what we learned came from instruments, not ideas**: of roughly forty measured claims, over twenty are negative results, five are corrections of our own published numbers, and a recurring class of silent harness failures (caches that outlive their assumptions, truncation correlated with the verdict, partial runs overwriting whole-set results) manufactured findings that survived until dedicated tests killed them. We report all of them.

---

## 1 Introduction

A maintainer of a mature open-source project has a discovery problem that ordinary literature search does not solve. The question is not *"which papers are about my topic?"* — for an active field, thousands are — but *"which paper, if I read it on Monday, would change what I build?"* The distinction is between **relevance** and **utility** [1], and it is unforgiving: a recommendation feed that surfaces ten topically-adjacent-but-unactionable papers is worse than one that stays silent, because the reader pays attention for nothing and stops trusting the feed.

RepoRadar is a CLI tool that profiles a repository (README, docs, dependency manifests), queries paper sources (arXiv primarily; Semantic Scholar, OpenAlex, bioRxiv, DBLP optionally), ranks candidates, gates them with an LLM actionability judgment, and emits a three-tier digest whose "Top Picks" section is allowed — encouraged — to be empty. This paper is not primarily a systems description. It is an account of a measurement campaign (2026-07-04 to 2026-08-08, 102 pull requests) organized around a single benchmark, in which nearly every architectural belief we started with was tested, and most were refuted. We think the record is more useful than the system: the negative results delimit a design space that others will otherwise re-explore, and the positive result — a distributional rescore that fixes a near-binary LLM gate — is simple, cheap, and, we suspect, applicable to LLM-judged selection pipelines generally.

**Contributions.**

1. **A task formulation and benchmark** for repository-conditioned paper recommendation with an abstention-aware, asymmetric metric (`net@2`), 22 real repositories spanning ML, systems, and deliberate negative controls, a pooled LLM judge, and a strong agentic baseline (§3–4). We document the benchmark's own failure modes: at 12 repositories it was measurably overfit (a jackknife moved a headline result from a 70% pool reduction to 11% by dropping one repository), and we describe the expansion that fixed it (§4.4).
2. **A validity program for the LLM judge** rather than an assumption: a second-judge agreement study (κ = 0.51 binary, 0.73 quadratic-weighted; the disagreement is a strictness offset, not a ranking difference) and a model-free ground truth mined from git history — papers a repository *actually adopted* — against which the judge scores 61% actionable versus a 2% random floor (§4.3).
3. **A diagnosis of retrieval for utility** (§5): the register-mismatch account, the failure of every document-derived query channel, and two channels that work by routing around the mismatch — a bibliography-seeded citation hop (44% recall overall; 89% where seeds are plentiful) and blind HyDE against a 3.1M-vector public index (27/48, of which 15 are unreachable by the hop; union 75%).
4. **A diagnosis and repair of LLM utility gating** (§6–7): the gate's score distribution is near-binary, hiding a 0.00–1.00 per-repository precision spread in its modal band; five pre-registered ranking experiments, of which comparative (listwise selection, pairwise Bradley–Terry) and ensemble methods fail while a fine-grained distributional rescore succeeds (band AUC 0.84), converts to a calibrated probability with two fitted parameters, and survives out-of-family judge checks, out-of-run replication, and a live end-to-end run (§8).
5. **A catalogue of 20+ negative results and 5 corrections** (§9), and a taxonomy of the silent harness failures that repeatedly manufactured wrong findings, with the defensive patterns that stopped them (§10).

Throughout, numbers marked **[NR-n]** refer to the negative-results catalogue in §9, and every measured claim carries the script that re-derives it in the project's `evals/` directory.

---

## 2 Related Work

**Paper recommendation and code–paper linkage.** Scholarly recommenders overwhelmingly target researcher-facing relevance (citation recommendation, related-work search); the specific direction *repository → papers ranked by actionability for that repository* appears essentially unexplored — prior code/paper linkage work such as paper2repo [2] runs the opposite direction (paper → its implementation). Practitioner-relevance studies are directly cautionary: Lo, Nagappan and Zimmermann found engineers' relevance ratings of research essentially uncorrelated with citation counts [3], which our feature experiments reproduce in miniature (§7, E5).

**Retrieval without shared vocabulary.** Our register-mismatch diagnosis is a species of the vocabulary-mismatch problem; the remedy that worked, generating hypothetical documents and searching in embedding space, is HyDE [4], run against a public binary-embedding snapshot of arXiv built with mxbai-embed-large-v1 [5, 6]. Hybrid lexical fusion uses BM25 [7] with reciprocal-rank fusion [8]. SPECTER2 provides citation-informed scientific embeddings [9]; cross-encoder rerankers, and their documented inability to follow *non-relevance* criteria zero-shot, are analyzed in FollowIR [10].

**LLMs as rankers and judges.** Listwise LLM reranking (RankGPT [11], LRL [12]), pairwise ranking prompting [13], setwise selection [14], and tournament schemes [15] all report large gains over pointwise scoring for *relevance*. The study closest to our task, utility judgment over already-relevant candidates [16], finds pointwise utility scoring collapses while listwise-set selection works — a prediction our E1 experiment *fails to reproduce* in the repository setting (§7). Judge reliability and its biases are documented around MT-Bench [17]; comparative judgments' advantages over absolute scores [18] and pairwise-consistency methods [19] motivated our E4. Calibration of LLM confidence spans verbalized probabilities [20, 21], panels and ensembles [22], and escalation cascades [23]. The mechanism our winning method uses — reading a score's token distribution rather than its sample — is G-Eval's probability-weighted scoring [24], refined by TrustJudge's distribution-sensitive scale [25]; the finding that a sufficiently fine pointwise scale matches listwise ranking [26] anticipated our result, and Rank1 [27] uses the same first-token-probability reading for relevance. Permutation self-consistency [28] and position-bias measurements [17, 29] informed the mandatory shuffling/both-order protocols in E1/E4. Anchored pairwise scales [30] and conformal Elo calibration [31] were evaluated as calibration routes and rejected empirically.

**Methodology.** Our pre-registration discipline (success criteria and kill conditions written before any result), leave-one-repository-out cross-fitting, and the ban on per-repository knobs are standard prescriptions against benchmark overfitting; the specific trap we document — selecting a *calibration* by a rank-only criterion (AUC), which is provably blind to the decision threshold — deserves wider attention (§8.3).

---

## 3 Task and System

### 3.1 Task

Given repository *R* (its working tree; no stars, ratings, or telemetry — the cold-start case), return a set of papers *S(R)*, possibly empty, such that each paper "could genuinely improve *this* repository" — the wording of both the system's gate and the benchmark's judge. The empty set is a first-class answer: three of our 22 benchmark repositories are negative controls (Flask, Click, Requests) for which the correct output is nothing, and two more (pyca/cryptography, redis) have literatures centred off-arXiv (IACR; VLDB/OSDI), where near-abstention is correct for an arXiv-fed system.

### 3.2 Pipeline

The shipped pipeline is a funnel:

```
profile(R) ──► queries ──► collect (~227 papers; range 56–296)
   ──► heuristic rank (keyword/category/recency/embedding + BM25-RRF fusion)
   ──► top-k (15) ──► LLM gate: 0–3 actionability, Haiku      [stage 1]
   ──► papers scoring exactly 2: fine-scale 0–9 rescore,
        expectation over answer-token distribution,
        frozen logistic → P(actionable), show iff P ≥ 2/3      [stage 2]
   ──► three-tier digest (Top Picks / Maybe / Muted)
```

Stage 1 (0–3 gate; "triage") asks whether a paper proposes a method with a concrete implementation path for *this* repository, described by a keyword profile plus the first 300 characters of its README — a budget chosen by experiment (§6.3). Stage 2 exists because of the central selection finding of this paper (§7) and is detailed in §8. Politeness is engineered, not aspirational: one process-wide arXiv rate gate at the stated 1 request / 3 s ceiling, identifying User-Agent, 30 s minimum backoff on HTTP 429 with a 15-minute patience budget — added after sustained polling earned the development machine a ~70-minute IP block, and after a 429 storm was silently cached as seven empty candidate pools (§10).

---

## 4 Evaluation Methodology

### 4.1 Metric: asymmetric, abstention-aware

The headline metric is **net@2** over a system's *returned* set: (# genuinely actionable) − 2·(# not), where "genuinely actionable" is a pooled judge's score ≥ 2 on a 0–3 rubric. The asymmetry encodes the product stance that a junk recommendation costs more than a good one earns, and it gives abstention a defined value (0). Two consequences shape everything downstream. First, the expected contribution of a shown paper at true precision *p* is **3p − 2**: showing pays exactly above p = 2/3, which later becomes the system's *derived* — not tuned — display threshold. Second, on an unranked set there is no interior optimum in digest size: truncation scales the total without changing its sign, so "show fewer" is only coherent once a reliable within-set ordering exists (§6.6). We flag the metric's bias plainly: net@2 rewards shyness, so it structurally flatters precision-improving changes (§8.4) and understates the cost of the recall problem it cannot see.

### 4.2 Benchmark, judge, baseline

Each of 22 cases names a live repository (Table 1 in the artifact lists all; examples: ColBERT, detectron2, stable-baselines3, PEFT, diffusers, PyTorch-Geometric, Whisper, DuckDB, Qdrant, RocksDB, Arrow, LLVM-adjacent, ruff, SciPy, redis, plus the three negative controls). Per case, RepoRadar's top-10 and the baseline's recommendations are pooled and judged **blind to source** by GPT-5.5 under a fixed rubric; verdicts are cached keyed by (case, paper, rubric-hash) so no system can influence its own grading and re-runs are paired. The baseline is Claude Opus 4.8 running agentically with web search over the same repository (~$0.80/repo) — a strong instantiation of "just ask a frontier model."

### 4.3 Is the judge measuring anything? A validity program

Every number in the project is agreement with one model, so we tested the instrument itself.

**Reproducibility (P7).** 200 verdicts re-judged by Sonnet under a byte-identical rubric: exact agreement 50%; binary (≥2) κ = **0.507**; quadratic-weighted κ = **0.727**. The confusion matrix is one-notch-diagonal — GPT's 0s are Sonnet's 0s *58 of 58 times*, GPT's 3s are mostly Sonnet's 2s — so the judges **rank identically and differ by a strictness offset** (base rates 40% vs 22% actionable). Paired arm-vs-arm comparisons under one judge therefore largely cancel the offset; absolute rates are judge-relative and reported as such. The pre-registered κ ≥ 0.60 bar was *missed*: the labelled set is noisier than predicted, and ±10-net@2 conclusions inherit that.

**Validity (P6).** A label no model produced: an arXiv ID present in a repository's docs at HEAD and absent 24 months earlier is a technique the project demonstrably adopted. Mining six repositories yielded 31 usable adoptions (self-citations filtered). The judge scores **61%** of them actionable against the repository *as it was* before adoption — versus **2%** for random arXiv papers — with the misses individually traceable (a "projects using us" page with reversed citation direction; tutorial background citations; one broken link). The judge rewards approximately the right thing; the pre-registered 70% bar was missed, the 40% invalidation bar comfortably cleared.

### 4.4 The benchmark had to be debugged too

At 12 cases the benchmark was **measurably overfit**: a jackknife on the P1 result (§5.4) showed that dropping the single repository `rag` moved the headline pool-reduction figure from 70% to **11%** — one fold held 28% of all gold targets. Effective repository count (inverse Simpson over target share) was 5.4 of 7. Ten cases were added against four named blind spots (thin docs; no arXiv bibliography; non-ML; citation-rich), doubling targets to 48 and raising the effective count to 15.2 of 17. Two predictions made during expansion were refuted by measurement and recorded: repositories we expected to be thin-docs cases were not (ruff's README is 4× PEFT's), and the six no-bibliography cases we expected to yield no targets supplied **12 of 22 new ones** — repositories can fail to cite an existing literature that would improve them, which made them the sharpest cases in the set. Cohort-1 numbers are flagged as resting on the concentrated set wherever quoted.

### 4.5 Guardrails

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

Repositories never cite what would improve them (NR-1) — but those papers are **one hop away**. Seeding on the arXiv IDs the repository itself cites and expanding one hop in both citation directions reached **18/24 (75%)** of the original gold set, against 0–2 for everything else. Honesty about that number took two more measurements. First, a transport bug had earlier produced 14/24 with a confident wrong mechanism attached (§10). Second, re-measured over the expanded 22-case benchmark, 75% was revealed as recall over a favourable subset: across all cases the hop reaches **21/48 = 44%**, with a clean structural predictor — repositories with ≥7 arXiv-cited seeds give 89% recall, thin bibliographies 33%, and six repositories with no arXiv bibliography 0% by construction. Three attempts to extend it failed: coupling-degree pool filtering did not replicate off the concentrated cohort (70% cut → 10%) **[NR-8]**; synthetic seeds from LLM phrases recovered 4/27 with all successes in the single arXiv-native domain (vector search) **[NR-6]**; gap-phrase re-ranking lost to pasting the keyword profile **[NR-7]**.

### 5.4 What works, part 2: HyDE against a public dense index

The remaining channel inverts the register problem: have an LLM write the *abstract of the paper it wishes existed* for this repository (HyDE [4]), and search a dense index with it. A public 3.1M-vector binary-embedding snapshot of arXiv abstracts [6] makes this free beyond one-time 432 MB of column-pruned Parquet range reads. The experiment was gated on verifying four dependencies (license, columnar fetch, latency, target coverage — all passed) plus a fifth we almost failed to name: that vectors we compute are **bit-identical** to the index's (they are, Hamming 0/1024 on held-out rows) — without which every query would have measured nothing while looking healthy. Blind (the generator saw only the repo context), four hypothesis abstracts per repository: union top-1k reaches **27/48**, median rank 837, versus 10/48 for embedding the README (bimodal: occasionally rank-1, median 46,656 — the register mismatch visible as a distribution) and 3/48 for keyword queries. Crucially the two working channels barely overlap: **hop ∪ HyDE = 36/48 (75%)**, with 15 targets only HyDE reaches — including repositories whose bibliography is empty.

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

Every aggregate reproduces within noise while the **per-case numbers move hard** (`ann` show-all +4→−2, `crypto` +3→−2; six net-negative repositories on this draw rather than four, all six eliminated by the stage): collection is stochastic, per-case values in any single run are draws, and the replay was measuring something stable rather than a lucky draw. We state the significance honestly: on 22 paired cases neither comparison clears p < 0.05 — the result is a clear mean improvement, measured twice, that is *not* established as reliably per-repository — and net@2's asymmetry flatters precisely this kind of precision-stage change. The remaining losses to the baseline are all recall: cases where the admitted set never contained what the agentic baseline found (one repository admits nothing at all), the 0.60-recall problem a precision stage cannot touch.

---

## 9 Negative Results and Corrections, Catalogued

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
- **NR-15** Triage window 20→50: 2 papers across 12 cases.
- **NR-16** Gating the whole pool: −0.18 paired on 22 cases — a wash; nothing orders what the gate admits.
- **NR-17** Adaptive digest size: no interior optimum exists on an unranked set (3p−2), and the abstention signal it needs does not exist (r = +0.30).

**Band ranking (all pre-registered).**
- **NR-18** Listwise subset-selection with shuffle voting: both models far too strict; the published top-half-selection prior inverted. Contradicts the transfer of [16] to this setting.
- **NR-19** Persona-ensemble vote fractions with consider-the-alternative: killed at ECE 0.425 by chronic *under*-confidence.
- **NR-20** Pairwise Bradley–Terry: below the ordering bar at 7× E2's cost; synthetic borderline anchors rejected by 128/128 real papers — no absolute scale.
- **NR-21** Metadata features (citations, age, HyDE rank, coupling): AUC 0.585 — practitioner utility is not in the metadata, echoing [3].
- **NR-22** The combined model: loses to the single best score on every axis at n=22 queries.
- **NR-23** Monte-Carlo distribution reading via Haiku sampling: 0.59 vs 0.84 — a near-deterministic sampler re-reads its mode; exact logprobs are load-bearing.

**Corrections of our own published numbers** — each recorded in place, with the wrong number retained as a warning: **[C-1]** the citation hop's first figure (14/24) was a transport artifact of a batch API's silent nested truncation, and two mechanism claims drawn from it were withdrawn; the same defect was silently costing shipped code 29% of its citation edges. **[C-2]** "More prose is worse" — never statistically supported; the argmax of four noisy arms was over-read as a curve. **[C-3]** A "+16 for README context" result had actually measured a packaging one-liner *plus* deletion of the keyword block, in a harness-rebuilt prompt: the eval now calls the shipped prompt builder, by rule. **[C-4]** The full-pool comparison's first delta (−0.95) was three-quarters an unhandled arXiv throttle scored as an honest zero. **[C-5]** §8.3's calibration-by-AUC trap (+2.91 → +3.14).

---

## 10 Methodological Lessons: How Instruments Lie

Across five weeks, the experiments that nearly published wrong conclusions outnumber the model-capability surprises. The recurring classes, each now guarded by tests:

1. **Caches that outlive their assumptions.** A judge cache keyed without the rubric let a rubric-swap experiment silently overwrite the gold set's verdicts (nine targets vanished; caught by a test pinning the frozen target list). A baseline cache is keyed by a discriminator over mode/context/flags for the same reason. An empty result cached as success — seven empty candidate pools after a 429 storm — is worse than no cache: the next run skips it as done, and the repositories vanish from every downstream number while reports stay confident.
2. **Failure indistinguishable from absence.** A hop function dropped rate-limited chunks with a bare return: "throttled" and "these seeds cite nothing" were identical to every caller, and a 90%-undercounted pool *reported success*. The pattern's fix is mechanical: return failure structurally (`failed_chunks`), and refuse to persist partial results.
3. **Truncation correlated with the verdict.** Twice: judge responses truncated at max_tokens dropped scores of 2/2/3 (skewing base rates down), and a selection model's longest outputs — exactly the select-everything responses — were the ones that failed to parse. A dropout that correlates with the outcome is not noise.
4. **Partial runs overwriting whole-set artifacts.** Three separate scripts destroyed full-set results via a `--case` smoke run before merge-by-key became the standard write pattern.
5. **The harness measuring itself.** Rebuilding the system's prompt inside the eval produced a published result about a prompt the system never sends [C-3]. The rule since: the harness calls the shipped builder, and — for the calibrated stage — a test asserts byte-identity.
6. **An experiment that "confirms" on empty data.** A missing API key produced empty score lists, which averaged to 0.0, which satisfied the comparison the script then printed as a confirmed negative — agreeing with the conclusion already drafted. Verdict printing now requires every cell non-empty.
7. **Bug signatures in too-perfect agreement.** A loop variable shadowing a parameter made a variant silently re-run its control; byte-identical pools on 10 of 11 cases was the only tell. Perfect agreement between arms is a red flag, not a replication.
8. **Small-n statistics discipline.** Per-case sd on the headline metric is ≈1.7, so single-case swings flip 4-case means [NR-11]; the 12-case set was jackknife-fragile (§4.4); every headline since is paired, sign-tested, and reported with its non-significance where true. Corrections of over-claims from noisy argmaxes [C-2] and small-sample enrichment artifacts (a 0.00× that became 2.32× at full n) are recorded inline.

None of these are exotic. They are what "evaluation infrastructure" actually consists of, and we would not trust the headline numbers of any comparable system whose reports do not mention fighting them.

---

## 11 Limitations

**The judge defines the task.** All labels are one frontier model's rubric-guided opinion; the validity program (§4.3) bounds this — a second judge ranks identically at different strictness; adopted papers score 61% vs a 2% floor — but does not remove it. Absolute rates are judge-relative (a ~0.55 multiplier under the stricter judge); paired comparisons largely cancel the offset.

**The metric rewards shyness.** net@2 charges 2 per false positive, so a precision stage — our main result — is exactly what it flatters, and the recall deficit it structurally undervalues (gate recall 0.60; six baseline losses that are pure retrieval misses) remains the open problem. A user who wants discovery may prefer a different λ, and nothing here optimizes reading-time value directly.

**n = 22.** The headline improvement is a mean effect measured twice (offline and live) with consistent direction and non-significant per-repository sign tests (p = 0.109 / 0.45). The benchmark is popular-OSS-biased: well-maintained READMEs (the prose-300 bet pays 11/12; a genuinely thin-docs private codebase — the actual target user — is underrepresented), English, arXiv-centric.

**Calibration freshness.** The two fitted parameters are frozen against a specific prompt and judge vintage; drift is unmeasured. Tests catch prompt edits, not slow semantic drift in what "actionable" means.

**Vendor coupling.** The winning mechanism requires token logprobs, which only one major API exposes (§7, NR-23); the method is one `logprobs=true` flag away from vendor-neutral, but today the dependency is structural.

---

## 12 Conclusion

The system-level story is a funnel whose bottleneck moved twice: from *retrieval* (solved not by better queries but by routing around a register mismatch — citation hops and hypothetical-document search), to *selection* (an LLM gate that transformed the product and then stalled), to *ordering within the gate's own blind spot* — a near-binary score distribution hiding a 0.00–1.00 precision spread. The method-level story is that, for this utility-judgment setting, the literature's comparative machinery (listwise selection, pairwise tournaments, ensembles) failed with a consistent strictness pathology, while the smallest possible change — the same pointwise question at finer granularity, read as a distribution rather than a sample, calibrated with two parameters to the metric's own breakeven — beat a strong agentic baseline at one-eightieth its cost ($0.01 vs $0.80 per repository). The meta-level story, which we believe most transferable: with a benchmark honest enough to be debugged itself, pre-registration cheap enough to be habitual, and negative results recorded at the same fidelity as wins, five weeks of measurement can replace an architecture of accumulating beliefs with a short list of things that are actually true.

---

## Reproducibility

Every quantitative claim maps to a script in `evals/` (named per section in the artifact), runs against cached labels where possible ($0 re-derivation), and is recorded with its cost. The benchmark definition, 3,400-line results log with inline corrections, frozen testbeds, and all experiment code are in the repository. Total measured spend for the full campaign is on the order of $150, dominated by judge calls and agentic baselines.

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
