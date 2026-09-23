# Registration of the five band-ordering mechanisms (E1 to E5)

Verbatim excerpt of `evals/RESEARCH-score2-ranking.md`, lines 106-186, from the development repository. Sections 4 to 6: the ranked shortlist with each experiment's hypothesis, procedure, success criterion and kill condition, the calibration-versus-ordering distinction, and the guardrails. TIMELINE.md gives when it was committed.

---

## 4. Ranked experiment shortlist

All experiments run on the frozen Testbeds from the internal grounding — zero new judge calls. Primary: **Testbed A** = `results\judge-gpt-5.5-20260807T180938Z.json` (22 cases, 220 shown papers, 105 within-band, band membership reconstructed positionally from the sweep counts per the stable-sort argument; abstracts joined from `.work\full_pool\{case}.jsonl`, repo profiles from `.work\{case}`). Replication: the pool-300 arm (164310Z + the four db/storage single-case re-run files). Secondary: **Testbed B** = `diag_triage_prose300.json` (125-paper admitted band, 12 repos; target judge==3 ordering only, due to top-stratum saturation at judge ≥ 2). Sanity: **Testbed C** pooled (33 + 66 judged admits). Baselines fixed in advance: show-all +1.91 mean net@2, score-3-only +0.50, oracle = per-case act10; per-case sd 1.67, so all headline claims use paired per-case deltas plus a sign test.

### Experiment 1 — Shuffled subset selection with vote-share threshold (run this first)

**Hypothesis.** A single comparative selection call over the whole admitted set, repeated over shuffles, separates diffusion-like from vectordb-like bands where pointwise scores cannot, and the "possibly none" option abstains on all-bad bands.

**Procedure.** For each of the 17 pool-50 cases with a non-empty admit set: one Sonnet prompt containing the prose-300 repo profile plus all admitted papers (title+abstract, randomized ID labels), instruction: "select only the papers this repo's maintainers should actually act on — possibly none," with the evidence-first rubric (name the repo component and the concrete Monday-morning change before selecting). R = 15 shuffles of paper order at temperature 1. Score = selection share ∈ {0/15 … 15/15}. Pre-registered policy: digest = gate-3 papers + band papers with share ≥ 2/3. Haiku arm as cost ablation. Calls: 22 × 15 = 330, ~5-8k tokens each.

**Evaluated against.** Testbed A stored judge scores: (i) pooled within-band ROC-AUC of share vs judge ≥ 2, precision@k, cumulative net@2@k; (ii) headline policy net@2 vs +1.91; (iii) replication on the pool-300 arm; (iv) Testbed B: share-ordering scored on judge==3 density@10 vs reference points 2.4 / 3.1 / 5.3; (v) Testbed C pooled AUC.

**Success criterion (in advance).** Pooled within-band AUC ≥ 0.65 AND policy mean net@2 > +1.91 with ≥ 14/22 cases at delta ≥ 0, AND diffusion's full +10 retained (all 10 band papers clear 2/3), AND net improvement on at least two of {numerics −2, compiler −5, vectordb −5, linter −2}.

**Cost.** ~$3-6 (Sonnet), ~$0.50 (Haiku). **Kill condition.** Shares saturate (pooled mean > 0.8 with sd < 0.1); empty selections never occur on the linter/compiler bands across 15 shuffles; or AUC ≤ 0.55.

### Experiment 2 — Fine-scale distributional pointwise via OpenAI logprobs

**Hypothesis.** The score-2 pile-up is a quantization artifact; an 11-point anchored scale read as a logprob expectation, plus a reason-then-true/false p(true), recovers a usable continuous ordering and a thresholdable pseudo-P at near-zero cost.

**Procedure.** gpt-4o-mini with logprobs. Per paper (repo profile + title/abstract), two variants: (a) anchored 0-10 actionability score, anchors written from the existing rubric text (not from benchmark failures); score = Σ p(sᵢ)·sᵢ over the token distribution; (b) brief forced reasoning, then "true/false: maintainers should act on this"; P = normalized p(true). ~220 calls (Testbed A) + 125 (B) + 99 (C) ≈ 450 calls.

**Evaluated against.** Same Testbed A metrics as E1; calibration of p(true): Brier + reliability curve on all 220 shown papers; policy net@2 at pre-registered global P ≥ 2/3.

**Success criterion.** Pooled within-band AUC ≥ 0.65; promotion to calibration candidate iff Brier ≤ 0.22 and policy net@2 > +1.91.

**Cost.** <$1. **Kill condition.** Modal score token carries p > 0.9 for > 80% of papers (distribution degenerate, same collapse in finer clothes); AUC ≤ 0.55. **Mandatory validity check.** Judge is GPT-5.5 (same family): any win must survive intersection with the 200-row `second_judge.json` Sonnet verdicts before being believed.

### Experiment 3 — Ensemble vote fraction + verbalized P with the 2/3 rule (Anthropic-native calibration)

**Hypothesis.** Simulated-Annotators-style agreement over varied prompts yields an approximately calibrated P(actionable) that the utility-derived 2/3 threshold converts directly into the digest policy.

**Procedure.** Per band paper: N = 10 Haiku calls at temperature 1, each with a different rubric persona / few-shot set; each call must first state the strongest reason the paper is NOT actionable for this repo, then output verdict (act / not) + verbalized P (0-100). P̂ = mean of vote indicator and verbalized-P/100 (Avg-Conf). Optional panel arm: average with E2's p(true) (two families; disagreement logged as an uncertainty signal). Pre-registered policy: show iff P̂ > 2/3. ~2,200 Haiku calls (A) + 1,250 (B).

**Evaluated against.** Brier / log-loss / reliability curve on the 220 shown papers (the direct test of goal b); within-band AUC; policy net@2; LORO isotonic map as a diagnostic only (does the raw 2/3 rule's realized risk match nominal?).

**Success criterion.** Pooled ECE ≤ 0.15 and policy net@2 > +1.91 with the raw (un-recalibrated) 2/3 threshold; linter's band paper falls below threshold.

**Cost.** ~$2-4. **Kill condition.** P̂ clusters in 0.7-0.9 for nearly all papers regardless of judge label (the cross-family confident-wrong consensus pattern); ECE > 0.3; ordering AUC ≤ 0.55.

### Experiment 4 — Round-robin pairwise + Bradley-Terry, with template anchors (ordering backbone / fallback)

**Hypothesis.** Comparative utility judgments carry the within-band signal that absolute scores discard (the literature's largest measured deltas); anchors give the ordering an absolute zero for abstention.

**Procedure.** Per case: all pairs of admitted papers, both orders, Haiku, evidence-first comparative prompt; inconsistent pairs = ties; BT by MLE. Anchored arm: add 3 fixed-template anchor abstracts instantiated from the repo profile (clearly-actionable pattern / borderline / topically-adjacent-not-actionable); pre-registered policy: show iff P_BT(beats borderline anchor) ≥ 2/3. ~1,200 candidate pairs + ~600 anchor comparisons ≈ 1,800-2,500 calls.

**Evaluated against.** Testbed A within-band Kendall tau and AUC (primary — this experiment's job is ordering); Testbed B judge==3 ordering; anchored policy net@2 (secondary).

**Success criterion.** Pooled within-band AUC ≥ 0.70 (should beat pointwise per PRP/PairS if the mechanism transfers to utility) — that alone earns it the ordering-backbone role even if anchoring fails. Anchored policy graduates only if net@2 > +1.91.

**Cost.** ~$1-3. **Kill condition.** Swap-inconsistency > 45% of pairs (position bias drowning signal — a real risk given Claude-family measurements); anchor win-rates swing wildly under paraphrase of the anchor template (scale is fragile, calibration claim dead).

### Experiment 5 — Free-feature logistic chassis with LORO (cheapest calibration; also the ablation harness)

**Hypothesis.** Repo-independent metadata (has-code, stars, S2 influentialCitationCount, months-since-release, hop-coupling degree, HyDE rank, SPECTER2 cosine to repo wants) carries some within-band signal, and an L2 logistic regression over ≤ 8 features — including E1-E4 scores as columns — is the honest path to a pooled P(actionable).

**Procedure.** Assemble the feature matrix for the 220 Testbed A papers ($0: HF/S2 lookups, CPU embeddings, existing `.work\hyde_topk` and `.work\hop_pool` structural features); fit L2 logistic with LORO across the 22 repos (fit on 21, predict held-out, pool the 22 held-out sets); regularization strength by nested CV, never by benchmark score.

**Evaluated against.** LORO-pooled AUC and Brier vs judge ≥ 2; incremental AUC of each feature over the best single method from E1-E4.

**Success criterion.** Features-only LORO AUC ≥ 0.60 (else metadata is confirmed dead, a useful negative); combined model must beat the best single method's Brier to justify existing in the pipeline.

**Cost.** ~$0. **Kill condition.** Features-only AUC ≈ 0.5 — the expected outcome per FSE 2015 for citation-flavored features and the coupling/HyDE features' derivation-on-these-labels caveat (pitfall 7); in that case E1-E4's raw scores ship un-combined.

## 5. The calibration-vs-ordering distinction

These are different products and the metric treats them differently. **Pure ordering cannot rescue all-bad sets**: a perfect ranking of linter's band (precision 0.00) still shows papers at −2 each; permutation-style listwise and un-anchored pairwise BT are structurally forced to nominate a "best" paper even when the correct digest is empty. Ordering also cannot choose k: under net@2 each shown paper is worth 3p−2, so on an unranked or uniformly-scored set the optimal policy is show-everything (if p ≥ 2/3) or show-nothing — there is no interior optimum, and reordering a fully-shown top-10 cannot change net@2@10 at all (pitfall 8; every experiment must therefore report ordering metrics — AUC, precision@k — separately from policy value). **Only a calibrated P(actionable) with the global 2/3 threshold handles both tails simultaneously**: diffusion's band (true p = 1.00) all clears the threshold and keeps its +10; linter's band clears nothing and the case abstains at 0 instead of −2; and the +10/+4/+3 cases the score-3-only policy throws away are recovered because calibrated P, unlike the score-3 proxy, does not over-abstain. Mapping the shortlist: **E3** targets calibration directly (vote fraction + verbalized P is the construction with the best published ECE numbers); **E2's** p(true) is a cheap pseudo-P that needs a global recalibration check before its 2/3 threshold is trusted; **E1** delivers ordering plus *approximate* abstention — its selection share is peer-relative, so the linter test hinges entirely on whether "possibly none" fires, which is exactly what its kill condition probes; **E4** is ordering-first, gaining an absolute zero only through the unvalidated anchor mechanism; **E5** is calibrated by construction (logistic output) but only as good as its features. The pragmatic composite, if no single method wins both: E4-or-E1 ordering with E3-or-E2 supplying the absolute gate.

## 6. Guardrails

- **No per-repo knobs, all thresholds global.** The show/abstain threshold is 2/3, derived from net@2's payoff structure (threshold = FP-cost/(FP-cost+TP-gain) = 2/(2+1)), not from the benchmark. R (shuffles), N (votes), anchor templates, checklist items, and MMR lambda are single global constants fixed before any benchmark score is seen.
- **Pre-register one configuration per experiment.** The success criteria above are the pre-registration. No prompt iteration against the 22-case benchmark; if any fitting occurs (isotonic/Platt/logistic weights), it is LORO cross-fitted — fit on 21 repos, predict the held-out repo, pool — and the fitted map is global and frozen.
- **Offline-first.** Testbeds A/B/C cost zero judge calls; only ranker inference is spent. New judge labels are needed only if a policy admits papers outside the labeled shown/sampled sets — e.g., re-ranking full admitted pools at pool scale (12% label coverage, top-stratum-biased; the honest version costs ~$1.15 gate + ~$18 judge and is explicitly out of scope for this round). Keep all candidate selection inside the frozen labeled sets.
- **Label leakage.** Judge justifications and `proposed_change` sit beside the labels in `cache\judge\v1` and in the results files; no ranker prompt or feature may touch them. The coupling≥3 / HyDE-rank features and the r = +0.30 finding were derived on these same labels — anything built on them is LORO-only and reported as diagnostic.
- **Judge blindness.** The judge cache is keyed by (case, paper, rubric-hash), so verdicts are frozen per paper and method-agnostic — no ranker can influence its own grading. Preserve this: never re-judge a paper with method-specific context.
- **Judge-relativity and family confounds.** All labels are GPT-5.5-relative (second judge: same ranking, ~one notch stricter). Relative comparisons (method vs show-all) are safe; absolute net@2 claims and any OpenAI-model winner (E2) must be spot-checked against the 200 dual-judged rows in `.work\second_judge.json` to rule out same-family bias inflation.
- **Noise floors before victory laps.** Bound gate run-to-run churn with `diag_triage_prose.json` vs `diag_triage_prose300.json` (two draws of the same arm) before attributing small deltas to a new method; with per-case sd 1.67 over 22 cases, report paired per-case deltas and a sign test, never bare means. Per-repo judge==3 metrics are noise (1-14 positives per case) — pool always.
- **Bias mitigations are mandatory plumbing, not options.** Both-order querying for every pairwise call (Claude-family swap consistency measured as low as 23.8%), shuffle-aggregation for every listwise call (zero-shot listwise collapses under random input order), evidence-first rubric naming the repo component and concrete change before any score (+7.3 to +11.5pp for Claude judges), and no venue/author metadata ever shown to any judge (authority and style bias are the dominant measured biases).

---

