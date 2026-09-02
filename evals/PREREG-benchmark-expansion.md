<!--
DRAFT — NOT YET REGISTERED.

This document was produced on 2026-09-02 by a six-designer / twelve-refuter / one-synthesizer
pass over the project's measured record, then read and edited by hand. It becomes a
pre-registration only when (a) every `____` blank is filled at the step that produces it,
(b) the RepoRadar tag `rr-frame60-freeze` exists, and (c) it is committed with the word DRAFT
removed from this banner. Until then nothing in it binds, and no candidate repository has
been enumerated, cloned, profiled, or looked at.

Two corrections applied by hand to the synthesizer's inputs, both verified against the tree:
  * NR-37 (documentation volume predicts nothing) was measured 2026-08-16 under a corpus rule
    the profiler widened on 2026-08-19 (commit e0cb9b5, "The profiler reads the wrong files").
    thin-gnn's corpus read 1,073 chars then and reads 107,895 now. P0.4's re-run is therefore
    mandatory, not hygiene.
  * The "comparator exhausts its turn budget and returns nothing" caveat in SUBMISSION.md
    describes the Opus 4.8 / 12-turn arm. The Opus 5 draw this frame compares against is
    `ok` on all 37 legacy cases (one genuine abstention, bio-mdtraj, no cutoffs), so the
    VOID-OPUS rule in §5.6 is a precaution for the held-out science cases, not an
    observed failure mode of this comparator.
-->

# Pre-registration — benchmark expansion 37 → 60 under a frozen sampling frame [rung 10]

**File:** `evals/PREREG-benchmark-expansion.md`. **Status:** written before any candidate repository was enumerated, cloned, profiled, or looked at. Every blank (`____`) is filled at the step that produces it and never earlier; the git history of this file is the audit trail. Rules are numbered so the ledger can cite them.

---

## 0. Preconditions, purpose, and the arm

**P0.1 Rung 1 passed.** `evals/PREREG-rung1.md` (commit `7ce7a35`) registered a KILL that would bar this expansion. Recorded verdict (RESULTS.md, NR-52): consensus margin +0.57 vs GPT +0.32, |Δ| = 0.25 ≤ 0.5, sign preserved, 6/6 science losses persist — **PASS**. The expansion is not barred. It is also recorded there that the bar was near-tautological (Sonnet ≥ 1 demotes 0.7 % of papers); this frame states the power of every bar it sets.

**P0.2 What this buys, said in advance.** The ladder (`RESEARCH-net2-directions.md` §7) placed the n = 60 expansion as step (6), after product gates raise the mean, to "make the final claim at ≥ +1.2/case". This frame executes it **now and for a different purpose**: (a) the benchmark's first held-out set under a frozen system, (b) a stated population and sampling procedure for the 60, and (c) a datasheet. Its own prediction (§8, P1–P2) is that **the 95 % CI still spans zero**. It is a validity purchase, not a margin purchase, and the paper must say so.

**P0.3 The arm.** Two RepoRadar configurations exist at n = 37: the shipped arXiv arm (+0.32/+0.54 vs Opus 5 draw 1 depending on which of two identical-config runs is used, 0.22/case apart — C-7) and the unshipped arXiv+EPMC arm (+1.08, CI [−0.97, +3.16], sd 6.41). Naming the better arm here would be arm selection on the development set. **Primary arm = the configuration shipped at tag `rr-frame60-freeze` (§7.1).** The arXiv+EPMC arm at the same tag is **co-registered** and run on every held-out case (incremental cost ~$25 + judging), reported beside it, labelled unshipped unless its own gate has passed before the freeze. All power arithmetic below is given for both m37 = +0.54 (shipped) and m37 = +1.08 (arXiv+EPMC) with sd 6.41–6.56.

**P0.4 NR-37 re-run.** NR-37 (2026-08-16) was n = 25 (thin-gnn corpus 1,073 chars, not 107,895), before any materials case existed. Before this file is committed: `uv run python evals/thin_docs_detector.py` over all 37, result recorded here: Spearman ρ(log corpus, net@2) = `____`. Documentation volume stays a **covariate**, never a stratum or exclusion (§2.3).

**P0.5 The adoption channel is decoupled** (§6). No benchmark case is selected for adoption yield. Why: adoption mining is git + regex and judging adopted/control papers needs only the judge on a T0 context — neither arm runs — so the ~40 extra adoptions never had to enter the headline (all four refutations of the coupled designs agree on this point).

---

## 1. Population

The 23 held-out cases are a stratified random sample of:

> Public, non-fork, non-archived, non-mirror GitHub repositories that, on the snapshot date **D** (§4.1), (i) carry at least one of the topic tags committed in `evals/frame/topics.json` (§2.1), (ii) have ≥ 100 stargazers, (iii) were created ≥ 180 days before D, (iv) were pushed to within the 365 days before D, (v) report a primary language in {Python, C, C++, Rust, Go, Julia, JavaScript, TypeScript, R, Fortran}, (vi) have a README whose prose is English (§3, X5), and (vii) are software projects under the mechanical rule X4.

**What is claimed.** Estimates over the 23 are estimates over *this* population under the stratum composition fixed in §2.2 — a composition-weighted mean, not a GitHub-population mean (no Horvitz–Thompson weights are used; the per-cell universe counts are published so a reader can reweight).

**What is not claimed.** Nothing about private repositories, non-English documentation, non-GitHub hosting, untagged repositories (topic tagging correlates with maintainer polish — the coverage check in §4.2 reports how many of the 37 legacy repos the frame would have captured), or the paper's stated target user. The **small-star band (100–999)** is the nearest public proxy for an unmemorised, thinly-covered codebase and is named a proxy, not the target user. The three thin legacy repos (fireball, sekas, distributed_graph_flow) are below the star floor and outside this population; they stay in the benchmark under the legacy flag.

The 37 legacy cases are a **convenience sample** (three hand-curated waves). The pooled 60 is reported as "37 convenience + 23 frame-sampled" and is never called representative.

---

## 2. Strata, quotas, and the adoption/target-user resolution

### 2.1 Stratification axes (both computable from the API before anything is cloned)

**Axis 1 — literature-venue locus** (the measured axis of difficulty: controls score 0 by abstention, off-arXiv CS is a coverage test, dense on-arXiv physical science is where Opus 5 wins). Assigned by a script from the repository's GitHub topics against the committed lists:

| stratum | operational definition | topic list (committed verbatim in `topics.json`) |
|---|---|---|
| **L1** arXiv-native ML | ≥ 1 L1 topic, majority rule | machine-learning, deep-learning, neural-network, nlp, natural-language-processing, computer-vision, object-detection, image-segmentation, reinforcement-learning, llm, large-language-models, language-model, transformers, diffusion-models, generative-model, generative-ai, information-retrieval, neural-search, vector-database, vector-search, similarity-search, approximate-nearest-neighbor, embeddings, graph-neural-networks, speech-recognition, text-to-speech, audio-generation, time-series, tabular-data, gradient-boosting, quantization, model-compression, llm-inference, inference-engine, robot-learning, recommender-system, anomaly-detection, automl, federated-learning, multimodal, ocr |
| **L2a** physical/computational science (on-arXiv, mixed) | same | materials-science, materials-informatics, computational-chemistry, quantum-chemistry, density-functional-theory, dft, molecular-dynamics, condensed-matter, condensed-matter-physics, physics, computational-physics, physics-simulation, astronomy, astrophysics, cosmology, high-energy-physics, particle-physics, hep, lattice-qcd, climate, climate-model, climate-science, weather, atmospheric-science, ocean-modeling, geoscience, geophysics, seismology, hydrology, fluid-dynamics, cfd, computational-fluid-dynamics, finite-element, finite-element-analysis, pde, differential-equations, numerical-methods, numerical-analysis, linear-algebra, sparse-matrix, scientific-computing, plasma-physics, optics, photonics, quantum-computing, quantum-simulation, monte-carlo, statistical-mechanics, crystallography, spectroscopy |
| **L2b** life science (bioRxiv/PMC/journal-first) | same | bioinformatics, computational-biology, genomics, genome-assembly, sequencing, ngs, single-cell, scrna-seq, transcriptomics, proteomics, metabolomics, structural-biology, protein-structure, protein-design, molecular-dynamics, cheminformatics, drug-discovery, neuroscience, computational-neuroscience, neuroimaging, electrophysiology, systems-biology, epidemiology, phylogenetics, metagenomics, microbiome, medical-imaging, biomedical, ecology, population-genetics, crispr, mass-spectrometry |
| **L3** off-arXiv CS (VLDB/OSDI/PLDI/IACR/ACM SE) | same | database, databases, dbms, sql, key-value-store, storage-engine, embedded-database, columnar, column-store, query-engine, olap, oltp, transactions, compiler, compilers, jit, llvm, interpreter, programming-language, type-system, static-analysis, linter, formatter, type-checker, cryptography, encryption, tls, security, fuzzing, symbolic-execution, program-analysis, distributed-systems, consensus, raft, paxos, replication, networking, network-protocol, quic, operating-system, kernel, filesystem, file-system, message-queue, stream-processing, dataframe, build-system, build-tool, package-manager, testing |
| **L4** no research-literature adjacency ("controls") | ≥ 1 L4 topic **and 0** topics from L1∪L2a∪L2b∪L3 | web-framework, web-development, webapp, http, http-client, http-server, rest-api, api-client, cli, command-line, command-line-tool, terminal, tui, orm, template-engine, templating, logging, logger, configuration, config, dotenv, serialization, json, yaml, toml, markdown, static-site-generator, task-runner, scheduler, argument-parser, date-time, validation |

Collisions among L1/L2a/L2b/L3 (a topic such as `molecular-dynamics` is deliberately on two lists): assign to the stratum with the most matching topics; ties broken by `sha256(SEED || full_name) mod k`; every collision logged. Why not "sparser-tier wins": that precedence would push arXiv-dense repos into the abstention-shaped strata and flatter the system (frame refutation). L4 held-out draws are **judge-scored like every other case**; `negative_control: true` is *not* set on them; "control" is a stratum name, and a post-hoc `verified_empty` flag is descriptive only.

**Axis 2 — star band**: small = 100–999, popular = ≥ 1,000. Why: the small band is the honest public proxy for the target user, is API-visible, and confounds neither domain nor literature presence (the fatal flaw of the post-cutoff/recall-probe stratum). Recognisability is recorded as a **covariate** (§7.4), not an inclusion rule.

**Covariates recorded for all 60, never used for selection:** stars, contributors, created_at, primary language, profiler-visible doc chars (`_collect_text_corpus`), arXiv/hf.co id count in docs at HEAD (extractor v2, §6.1), recall-probe score, `previously_inspected` (§3, X2 list), owner.

### 2.2 Quotas — the rule, then the numbers

**Allocation rule (fixed here, blind to any held-out result):** the two abstention-shaped strata (L3, L4) are held at their legacy share of the benchmark (rounded), and the remaining held-out slots are split equally across the three research strata. Why: the whole legacy margin lives in five over-answer cases in L3/L4 (31 % of the paired sum of squares), so the one way an expansion could manufacture margin is by enlarging those strata; the rule caps them, and the extra weight it puts on science is the direction that lowers the expected pooled mean, not raises it.

| stratum | legacy 37 (assigned by the §2.1 script; expected assignment shown) | share | held-out quota (small / popular) | total at 60 | share at 60 |
|---|---|---|---|---|---|
| L1 | 11: rag, cv, rl, peft, diffusion, graph, speech, llminfer, vectordb, ann, thin-gnn | 29.7 % | **5** (3 / 2) | 16 | 26.7 % |
| L2a | 7: numerics, mat-mlip (mace), mat-chgpot (chgnet), dscribe, mat-toolkit (pymatgen), mat-featurize (matminer), mat-phonon (phonopy) | 18.9 % | **5** (3 / 2), ≤ 2 tagged `materials-science` | 12 | 20.0 % |
| L2b | 6: bio-align, bio-singlecell, bio-scvi, bio-mdsim, bio-mdtraj, bio-kmer | 16.2 % | **5** (3 / 2) | 11 | 18.3 % |
| L3 | 10: crypto, systems, db, storage, compiler, linter, encryption, columnar, thin-kv, thin-lang | 27.0 % | **6** (3 / 3) | 16 | 26.7 % |
| L4 | 3: webdev, cli, http | 8.1 % | **2** (1 / 1) | 5 | 8.3 % |
| | 37 | | **23** (13 / 10) | **60** | |

Abstention-shaped share (L3+L4): 35.1 % → 35.0 %. Legacy assignment is produced by the same script (topics at HEAD; fallback for untagged legacy repos: `expected_categories` in `benchmark.yaml` through a committed category→stratum map); if the script disagrees with the table above, **the script wins and the disagreement is logged**. Legacy per-stratum rows are **descriptive only** (their scores were known when the strata were written — the ladder's "post-hoc stratification" caveat applies to them); confirmatory per-stratum statements are made on held-out cases only. The materials cap exists because 6 of 7 legacy L2a cases are one sub-field and the sign flip is to be observed in new fields; it is a coverage rule written before the draw.

### 2.3 The adoption-yield / target-user tension, resolved

- **No benchmark stratum carries the adoption requirement.** Zero of the 23 are selected on citation richness, history length, or any label-derived property. Star band and topic are the only selection axes.
- **The adoption requirement is carried by a separate judge-validity pool (§6)** of research-stratum repositories with ≥ 30 months of history and ≥ 10 identifiers in their docs, drawn from the same universe in the same seeded order, judged at T0 with no arm run, never added to `benchmark.yaml`, never entering any net@2 figure. Its selection bias (mature, citation-tracking, ML-heavy) is a limit of the *validity study*, stated as such, with a pre-registered heterogeneity check between the 9 legacy adoption repos and the new pool.
- **Headline reporting with and without the abstention-shaped strata.** Every pooled figure is reported four ways, all pre-declared: all cases; without L4; without L3+L4; and legacy-37 / held-out-23 separately. `evals/frame/analyze.py` refuses to emit the all-cases row without the others.

---

## 3. Inclusion/exclusion rules, pre-screens, and the ledger (datasheet form)

Applied in this order to every walked candidate; the **first failing rule is logged and the walk moves on**. No rule references either arm's output, any judge verdict, documentation volume, bibliography presence, or citation count.

| id | rule | how checked | cost |
|---|---|---|---|
| **X1** | In the universe snapshot (`universe-D.csv`): public, non-fork, non-archived, `mirror_url` null, ≥ 100 stars, created ≤ D−180 d, pushed ≥ D−365 d, language in the §1 set, ≥ 1 committed topic | API fields at enumeration | $0 |
| **X2** | Not previously exposed: not one of the 37 legacy cases; not in `evals/frame/prior_exposure.txt` = every `github.com/<owner>/<repo>` string in the tree at the freeze commit (`grep -rhoE 'github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+' evals paper src PLANS.md ROADMAP.md README.md --exclude-dir=__pycache__ \| sort -u`, 64 entries today, incl. lammps/lammps, openkim/kim-api, pachterlab/kallisto, samtools/htslib, tblite/tblite, satijalab/seurat, nf-core/rnaseq, zaeleus/noodles, SciML/DifferentialEquations.jl) plus, by name, deepmodeling/deepmd-kit and mir-group/nequip | string match | $0 |
| **X3** | Owner not already present among the 37 legacy cases or earlier selections (one repository per owner among new draws; legacy owner duplicates — facebookresearch, pallets, huggingface, scverse — stay and are recorded) | string match | $0 |
| **X4** | Software project: `topics ∪ name ∪ description ∪ README[:300]` does not match the committed regex `/(awesome|curated[- ]list|paper[- ]list|reading[- ]list|tutorial|course|homework|lecture|book|cheat[- ]?sheet|interview|roadmap|template|boilerplate|starter|dotfiles|dataset[- ]only|official (implementation\|code)|code for (the\|our) paper|implementation of (the\|our) paper)/i`. Script only; **no manual override in either direction**; a misclassified repo that passes stays in and is flagged | regex | $0 |
| **X5** | README prose in English: fastText `lid.176` p(en) ≥ 0.8 on README text with code blocks, badges and URLs stripped, **applied only when ≥ 300 characters remain**; shorter READMEs pass with flag `lid_na` (so the rule cannot cull the thin proxy) | script | $0 |
| **X6** | Blobless clone at the pinned SHA ≤ 2 GB on disk and succeeds within 2 attempts (arm-neutral infrastructure limit) | `git clone --filter=blob:none` | minutes |
| **X7** | ≥ 20 files with a source extension of the reported primary language at HEAD (software floor; a docs-only or data-only repo fails here) | `git ls-tree -r` | $0 |
| **X8** | L2a only: at most 2 of the 5 selected carry `materials-science` | count | $0 |
| **X9** | Repository deleted or made private **before any arm has run on it** → `VOID-PRE`, replaced by the next reserve in seeded order, both rows kept | GitHub API | $0 |

**Not exclusions, by design:** profiler failure (`rr profile` raising is a RepoRadar outcome, §5.6); Opus turn-budget exhaustion (§5.6); prose volume; absence of a bibliography; "primary literature already represented"; any human judgement of interest.

**Pre-screens at $0 (recorded, not selective):** for every *selected* repo, after selection and before either arm runs: (a) `ids_v2(HEAD)` count (extractor v2, §6.1) and, where ≥ 24 months of history exists, `ids_v2(HEAD) − ids_v2(T0)` — recorded as covariates and as **incidental** adoptions (they never count toward §6's target); (b) arXiv-coverage covariate = fraction of DOI/arXiv/PMID identifiers extractable from docs at HEAD that resolve (OpenAlex) to a work with an arXiv version — reported per stratum as a manipulation check of §2.1 (expected order L1 > L2a > L2b > L3; L4 mostly undefined 0/0), never used to reassign a stratum.

**Ledger** `evals/frame/ledger.csv`, one row per walked candidate: `cell, seeded_rank, full_name, stars, language, topics, created_at, pushed_at, decision{SELECTED,RESERVE,EXCLUDED,VOID-PRE}, rule_id, evidence (URL / SHA / path), checked_on, minutes`. Counts per rule per cell are a datasheet table. **Nothing is ever excluded after either arm has run on it** (§5.6 governs arm failures).

---

## 4. Selection procedure

### 4.1 Freeze, commit, snapshot (order is mandatory)
1. Tag the system: `git tag -a rr-frame60-freeze` on the shipped configuration; record here the tag SHA `____`, the config hash asserted by `evals/audit_product_divergence.py` `____`, the HyDE index directory hash `____`, the enabled source set `____`, the judge model strings (GPT-5.5 `____`, claude-sonnet-5 `____`), the comparator settings (`claude-opus-5`, v2 prompt hash `____`, `max_turns` 30, effort `____` — pinned this time), and the legacy RepoRadar run whose 37 scores enter the pooled analysis `____` (the rung-1 control `20260830T034455Z` unless §7.3's re-run replaces it).
2. Commit this file, `topics.json`, `prior_exposure.txt`, the category→stratum map, and the scripts `evals/frame/{enumerate,classify,draw,walk,analyze}.py`. Frame commit **H = `____`**.
3. **D** = the calendar day after H. Enumerate: for each stratum, each topic, each band, query `topic:T stars:100..999|>=1000 archived:false fork:false pushed:>=D−365 created:<=D−180` (sort `stars`, any order — the seed re-randomises), bisecting the `created` range until every query's `total_count` < 1,000 (the API cap); union, dedupe by `full_name`, filter language and mirror; write **`universe-D.csv`** with every API field and the query id, plus per-query `total_count`. Commit it **before any exclusion runs**.

### 4.2 Seed and draw
4. **SEED** = `outputValue` of the NIST Randomness Beacon v2 pulse at the first hour boundary ≥ 24 h after the commit time of step 3 (`https://beacon.nist.gov/beacon/2.0/pulse/time/<ms>`); committed as `evals/frame/SEED`. Why not the commit hash: an author can re-commit until the draw looks right; a future beacon pulse cannot be chosen.
5. `classify.py` assigns stratum (§2.1) and band; `draw.py` orders each of the 10 cells by `sha256(SEED || full_name)` ascending and writes **`draw_order.json`** (the full order per cell). Coverage check: which of the 37 legacy repos appear in `universe-D.csv` → `coverage.json` (prediction P11).
6. `walk.py` walks each cell in order applying X2–X8; first *q* passing rows = SELECTED, next 2 passing rows = RESERVE; ledger committed. Pin `pinned_sha` = HEAD of each selected repo as of D (via the API commit list) and commit **`selected.json`**.

### 4.3 Closing the case set before any score exists
7. Clone each selected repo at `pinned_sha` into `repos/<case>`; add to `benchmark.yaml` with `wave: held-out`, `stratum`, `band`, `pinned_sha`, `frame_rank`, `expected_categories` from the committed stratum map (never per repo by hand — F13 in `RESEARCH-scientific-software.md` made this a manual step; it is a shopping surface and is closed), `gold_queries`/`distractor_queries` from the stratum defaults (offline fixtures only). Compute and commit every case's context hash through `second_judge.verify_contexts` (`sha256(RUBRIC \0 context)[:12]`) **for all 23 before either arm runs on any of them**.
8. Run order = round-robin over strata in the fixed cycle L1, L2a, L2b, L3, L4 (skipping exhausted strata; within stratum alternate small, popular), committed as `run_order.json`. The first 11 (≈ L1 3, L2a 2, L2b 2, L3 2, L4 2) are stage 1; the remaining 12 are stage 2.

---

## 5. Statistical design

### 5.1 What n buys (sd 6.41; t-based 95 % half-widths)
| set | n | half-width | mean needed to exclude 0 |
|---|---|---|---|
| legacy | 37 | 2.14 (bootstrap 2.11 as published) | +2.14 |
| interim pooled | 48 | 1.86 | +1.86 |
| **held-out** | **23** | **2.77** | +2.77 |
| pooled | 60 | 1.66 | +1.66 |

- Power of the pooled-60 two-sided test at the observed effect: **~25 %** at +1.08, **~10 %** at +0.54. MDE at 80 % power: +2.35. n for 80 % power: ~280 at +1.08, ~1,100 at +0.54. **n = 60 is a budget number, not a power number.**
- Held-out 23 at its expected effect (+0.78 arXiv+EPMC / +0.43 shipped, §8): power ~8 % / ~6 %.
- Stratification buys ≤ 1.5 % on the standard error (between-stratum share of the paired sum of squares 8.5–13 %; within-stratum sd 6.40–6.43). Its value is fixing the composition so the estimand cannot drift, not precision.
- The matsci sign flip (−3.17, sd 7.4 on n = 6) needs ~43 physical-science cases to resolve at 80 % power; 5 held-out L2a cases (SE 2.9) **cannot** confirm or refute it. Stratum predictions in §8 are directional and labelled so.
- Expected pooled sd rises slightly (science share 35 % → 38 %); the half-width figures above are therefore optimistic by ≤ 0.03.

### 5.2 Pre-registered estimands and tests
- **Primary (generalisation):** mean paired net@2, RepoRadar(tag) − Opus 5 draw 1, **GPT-5.5 label**, over the 23 held-out cases; 95 % CI from the project's bootstrap helper (same helper as every published figure); paired sign-flip permutation p (10,000 draws, two-sided); win/loss/tie.
- **Co-primary label:** the same under **Sonnet ≥ 2**; consensus (GPT ≥ 2 ∧ Sonnet ≥ 1) as the third PREREG-rung1 label. **All three always reported; the primary label is GPT-5.5 for continuity and is not switchable** (§6.4 says why no validity result can license a switch).
- **Secondary (magnitude):** the same statistics over the pooled 60, always reported, labelled "37 convenience + 23 frame-sampled", **never gated on the held-out result** (a gate keyed to an observed mean is a data-dependent selection rule — reviewer refutation). Instead a pre-registered **heterogeneity test**: Welch t between legacy-37 and held-out-23 paired deltas, p reported.
- **Pre-registered reading rule for the held-out estimate** (so the sentence cannot be chosen afterwards): m23 ≥ 0 and CI23 ∋ m37 → "consistent with the development-set estimate; not independently significant"; m23 ≥ 0 and CI23 ∌ m37 → "held-out advantage smaller than the development estimate"; m23 < 0 → "the development-set advantage is not supported on held-out data" and the paper's headline sentence is the held-out number. Applied under each label.
- **Cuts (all pre-declared):** without L4; without L3+L4; per stratum × band (held-out cells have n = 1–3; reported as points, no confirmatory claim); leave-k-out removing the k largest positive **and, separately, the k largest negative** deltas, k ∈ {3, 5}; Wilcoxon signed-rank as a secondary test (ranks bought nothing at n = 37: t p = 0.32 vs Wilcoxon 0.37).
- Legacy cases carry a three-way flag: `tuned` (core 25), `pre-registered-observed` (bio 6, matsci 6: predictions were recorded before their measurement, but the shipped configuration was confirmed with their scores visible), `held-out` (the 23). Why: a blanket "37 = tuning set" over-concedes the science 12 (reviewer refutation).

### 5.3 Futility interim
One look, after stage 1 (11 held-out cases; pooled n = 48), **non-binding futility only, no efficacy stop, no alpha spent**; the final tests keep two-sided 0.05 unadjusted.
- **Statistic:** m11 = mean paired delta over the 11 held-out cases, primary label. Why not the ladder's pooled ≥ +0.8 bar: 37 × 1.08 = 40.0 already exceeds 48 × 0.8 = 38.4, so that bar fires only if the 11 new cases average below −0.15 — it cannot fail on new evidence; applied to the 11 alone it would false-stop ~45 % of the time under this frame's own prediction.
- **Boundary:** stop enrolment if **m11 ≤ −2.0**. Operating characteristics (SE 6.41/√11 = 1.93): P(stop) = 7.5 % at true +0.78, 10 % at +0.43, 15 % at 0, 50 % at −2.0, 78 % at −3.5. Meaning: stop only when the held-out data already refute the development-set advantage by ≥ 1.4 SE and the remaining 12 cases would only size the failure.
- **Conditional bias if enrolment continues** (true +0.78): +0.30 on m11, ≤ 0.15 on the 23-case mean, ≤ 0.06 on the pooled mean; the unconditional 11-case estimate is reported beside the final one.
- **On stop:** the 11 cases are fully reported; the 12 unrun cases are listed with their pinned SHAs as *not run* (never zero); the quota table shows the unmet cells; the benchmark is frozen at 48 as a descriptive set.

### 5.4 Second comparator draw (rung 9, kept as a gated module)
Module R: inside window W, re-run **both** arms at the tag on a seeded 12-case subset of the legacy 37 (`sha256(SEED||case)` order; both judges pinned). This measures in one purchase (i) Opus 5 draw-to-draw + time-drift sd (inseparable; stated), (ii) RepoRadar's at the tag (2.23 measured), (iii) comparator drift since the legacy runs. **Rule (the ladder's registered 25 % bar):** if the Opus 5 noise variance share ≥ 25 % of the paired variance 41.1 (noise sd ≥ 3.2), buy a second pinned draw on the 23 held-out cases (draw-averaged and draw-1-only both reported); the remaining 25 legacy cases' second draw is optional and priced (§9). Below the bar, the module's sd is reported as the comparator reproducibility figure and no second draw is bought. The 12-case sd estimate carries ~±20 % relative SE; a result within ±0.3 of the bar is reported as "at the bar" and the draw is bought (pre-decided tie rule). The second draw is non-blind (draw 1 known), as the project recorded on 2026-08-16.

### 5.5 Comparator and judge pinning
All held-out arm runs, module R, module J (§7.3), and all judging occur inside **one window W ≤ 21 days**; every run records the exact model version string the API returns, timestamp, prompt hash, settings. Stage-2 comparator runs may be pre-bought before the interim (the interim then governs only judging and analysis) so W is not stretched by the look.

### 5.6 Void, not null — the mechanical rules
- **RepoRadar arm:** a completed run with an empty digest = abstention, **0**. A profiler or pipeline exception after 2 retries = **VOID-RR**, traceback path logged, case *not* replaced, Opus 5 still run; count reported per stratum and language. Why not exclude at screening: the profiler is the first stage of the arm, and removing repos it cannot ingest selects the sample toward RepoRadar (three refutations).
- **Opus 5 arm:** up to **3 attempts** at the frozen settings. Any attempt returning `ok` is the draw. All three ending in `error_max_turns` or an API error = **VOID-OPUS** — the comparator is unmeasured on that case (the paper's stated convention, SUBMISSION §3), the case is excluded from the paired analysis, **not replaced**, counted per stratum, and RepoRadar's score on it is reported descriptively. **Pre-registered sensitivity:** the same analysis scoring a VOID-OPUS case as comparator abstention 0 (what the maintainer received). Prediction P9 sizes the expected loss; the frame states plainly that voids will fall disproportionately on science cases and that both conventions are shown so the reader can see the selection.
- **Papers:** a returned paper first posted after the RepoRadar index cutoff recorded at the tag is removed from **both** arms' returned sets before judging (`VOID-PAPER`, counted); a paper without a verdict after 2 judge retries is excluded from that label's arithmetic, never scored non-actionable.

---

## 6. Judge-validity pool (decoupled from the benchmark)

### 6.1 Label
Adoption = `ids_v2(HEAD) − ids_v2(T0)`, T0 = D − 24 months, mined by `evals/mine_adoptions.py` with **extractor v2** = the existing arXiv regex ∪ `huggingface.co/papers/<id>` ∪ `hf.co/papers/<id>` (diffusers' docs fell from 99 to 11 arXiv-regex ids through link migration; v2 is applied to **both** ends so a migrated id cannot become a false adoption), plus the existing self-citation (`CITE_HEADING`) and 182-day too-new filters, a **reverse-citation path filter** (ids whose only occurrences are under paths matching `/(projects|showcase|used[-_ ]by|gallery|community|awesome)/i` are dropped), and a **doc-genesis guard** (`ids_v2(T0) ≥ 1`; positives from repos with an empty T0 bibliography are flagged `genesis` and excluded from the primary). Label v1 numbers (NR-56/57: 35 usable over 9 repos — graph 13, diffusion 7, peft 5, rag/rl/llminfer/bio-scvi 2, bio-singlecell/mat-phonon 1) are reported unchanged as v1; v2 is recomputed over the legacy 37 (P14).

### 6.2 Universe, order, stop rule
Rows of `universe-D.csv` in L1 ∪ L2a ∪ L2b with `created_at ≤ D − 30 months` and `ids_v2(HEAD) ≥ 10` (a blobless clone and a doc-glob grep, $0). **Mine every qualifying row** (the yield distribution over the qualifying population is itself a reported result and dissolves the yield-prediction problem); X2 and X4 apply; the 9 legacy adoption repos form a separate `legacy` cluster. Judge positives in `sha256(SEED||full_name)` order until **≥ 60 new usable positives** (after per-repo cap and cross-repo dedup) or the list is exhausted — exhaustion below 60 is a recorded negative result, not a reason to widen the rule. Per-repo cap **8** in the primary (seeded subset; surplus kept for sensitivity; graph 13 → 8, so the legacy base is 30). Positives sharing an arXiv id across sibling repos are assigned to one repo by seed and counted once.

### 6.3 Controls and judging
Four controls per positive, **arm-neutral**: arXiv listings (API) in the positive's primary category, submitted in the same half-year, not cited anywhere in the repo at HEAD, seeded per case. Why not `pool-cut100`: it is RepoRadar's own HEAD-seeded pool, so a judge harsher on RepoRadar-shaped papers (Sonnet, 2.3×) would be credited with "validity" (both adoption refutations). The legacy 35 are re-run under this control scheme; the pool-control result stays reported as NR-57. Both judges, byte-identical rubric, `t0_context` (README excerpt, manifests, listing at T0 — no arm), `use_cache=False`, T0 verdicts never entering `evals/cache/judge/`. The pool runs and is analysed **before any held-out benchmark case is judged** (runbook §10).

### 6.4 Endpoints, and what they cannot decide
- **Primary per judge:** AUC of the judge's ordinal rubric score against adopted/control, with a **repo-cluster bootstrap** (and paper-level dedup) — level-free discrimination.
- **Legacy statistic:** P(actionable | adopted) − P(actionable | control) at each judge's own threshold, Wilson intervals, cluster bootstrap; and P(actionable | control) per judge reported as the level-sensitive descriptive.
- **Pre-committed consequences.** If the primary judge's AUC CI includes 0.5 on the enlarged pool, the paper states that its primary label shows no demonstrated discrimination against the only model-free label, and the abstract leads with the two-judge interval rather than a single number. If both judges' AUC CIs exclude 0.5, the paper says both order papers meaningfully and the base-rate disagreement remains unresolved. **No outcome switches the primary label**, because the gap at a judge's own threshold is Youden's J at that operating point: a judge at an 87 % positive rate sits in the top-right of its ROC and is bounded low by geometry, so "Sonnet's larger gap" restates the base rates and identifies neither of them as correct; adoption is a lower bound on actionability and cannot calibrate a level. Any decision rule that reads the gap difference against PREREG-rung1's 0.15 bar is therefore **retired here, before the data**, and stated as such.
- **Transportability:** heterogeneity of AUC between the legacy cluster and the seeded pool, and between the small and popular bands, reported; validity is established on citation-tracking research repos and is an **assumption** for L3/L4 and the small band — stated in the limitations.
- **Contamination sensitivity:** positives split by adoption commit date relative to each judge's published training cutoff (recorded here at commit: GPT-5.5 `____`, Sonnet 5 `____`); AUC reported on the post-cutoff subset.

---

## 7. Contamination and leakage statement

7.1 **Development vs held-out.** Every configuration decision in the system (w_embedding 1.5, digest window 15, gate calibration, finescale map, source set) was made with the 37 legacy cases' outputs and judge caches visible; the 12 science cases were pre-registered additions on which those decisions were confirmed, not fitted. The 23 are the first cases whose scores no configuration decision has seen, under tag `rr-frame60-freeze`. **This licenses** the held-out estimate as a generalisation check of the tagged system at half-width 2.8. **It does not license** a population claim beyond §1, a significance claim (§5.1), or any per-stratum inference at n = 5–6. **The held-out property is consumed once:** the first configuration decision made with any held-out score visible converts the 23 into development data, and the paper must then say so; a later "final claim" at n = 60 with a changed configuration is a new held-out problem.

7.2 **Training-data exposure.** Both arms and both judges are frontier models trained on public GitHub; popular-band repositories and their citing papers are almost certainly in their pretraining, and adopted papers appear in HEAD docs that the judges may have memorised (§6.4 sensitivity). The small band is a partial control; the recall probe (7.4) quantifies exposure per repo. Unquantifiable residual exposure is stated, not estimated.

7.3 **Time and drift.** Repositories are pinned at `pinned_sha` (D). Opus 5 with live web search sees the web at run time, including each repo's own HEAD docs and post-D papers; RepoRadar's index is frozen at the tag. The paper-date cutoff of §5.6 removes post-index papers from both arms; the asymmetry that remains (web-search reads the repo's current citations) favours the comparator and is stated. Comparator drift since the legacy runs (`claude-opus-5` is an alias, not a snapshot) is measured by module R, reported beside the pooled row, and not gated. **Module J:** a seeded 40-paper subset of legacy GPT-5.5 and Sonnet verdicts is re-judged inside W; a label-flip rate > 15 % (the measured Sonnet self-disagreement floor is 8.4 %) marks legacy and held-out labels as different instruments in every table. The legacy RepoRadar scores entering the pooled analysis must come from the tagged configuration; if the run named in §4.1 predates the tag, RepoRadar is re-run at the tag on all 37 inside W and re-judged (priced in §9).

7.4 **Recall-probe covariate.** Each of Opus 5, GPT-5.5, Sonnet 5 (tools off, temperature 0) is asked "In one sentence, what is `<owner>/<repo>`?"; the answer is scored against the repo's GitHub `description` by embedding cosine, with the threshold **calibrated on the 37 legacy repos first** (the value at which ≥ 34/37 are "recognised"), then frozen and applied to the 23. Cost < $5. Reported per band; used in no rule.

---

## 8. Predictions (recorded now; each checked and listed as confirmed/refuted)

| id | prediction | falsifiable at this n? |
|---|---|---|
| P1 | Held-out mean, GPT label: **+0.8** (arXiv+EPMC arm) / **+0.4** (shipped), band [−0.6, +1.9] / [−1.0, +1.5]; **CI spans zero** | yes (point and band) |
| P2 | Pooled-60 mean, GPT label: +1.0 / +0.5, band [+0.6, +1.4] / [+0.1, +0.9]; CI spans zero | yes |
| P3 | Under Sonnet ≥ 2: held-out mean ≤ 0 (point −2.5, band [−5, +0.5]); consensus within ±0.6 of GPT | yes |
| P4 | Directional, held-out only (n = 5–6, SE ≈ 2.7–2.9; **no confirmatory claim**): L3 ≥ 0; L4 ≥ 0; L2a ≤ 0 (the flip observed in new fields, not only materials); L1 and L2b within ±2 | direction only |
| P5 | Over-answer mechanism: 2–4 of 23 held-out cases have comparator net@2 ≤ −2 with RepoRadar ≥ 0, and they carry ≥ 60 % of any positive held-out sum; on the other cases the arms are level within ±1 | yes |
| P6 | Small band paired delta ≥ popular band paired delta (comparator's web search blunted); RepoRadar's absolute net@2 does not differ between bands by more than 2 | direction |
| P7 | Non-Python-primary cases (profiler prose-only for C/C++/Go/Fortran): RepoRadar absolute net@2 ≤ Python cases' | direction |
| P8 | Heterogeneity legacy vs held-out: Welch p > 0.05; NR-37 re-test on the 23: Spearman ρ(log corpus, net@2) in [−0.3, +0.4] (underpowered; recorded, not claimed) | yes / weak |
| P9 | Voids: VOID-OPUS on 2–4 of the 10 science cases and ≤ 1 elsewhere; VOID-RR ≤ 2 (most likely on Fortran/Go); VOID-PRE ≤ 1 | yes |
| P10 | Screening: pass rate through X2–X8 40–60 % (popular) and 25–45 % (small); 60–110 rows walked for 43 passing (23 + 20 reserves) | yes |
| P11 | Coverage check: ≥ 28 of the 37 legacy repos appear in `universe-D.csv`; the three thin repos do not | yes |
| P12 | Validity pool: 150–400 qualifying rows; usable v2 adoptions per qualifying repo median 0–1, mean 0.8–1.8; ≥ 60 new usable positives reached within the first 80 repos in seeded order | yes |
| P13 | Validity endpoints at ≥ 90 capped positives: AUC(GPT) 0.60–0.70, AUC(Sonnet) 0.62–0.72, both CIs exclude 0.5, their difference does not exclude 0; P(actionable\|control): GPT ≥ 0.70, Sonnet ≤ 0.55 | yes |
| P14 | Extractor v2 raises the legacy usable count by ≥ 5 (diffusers alone) and creates no adoption absent under v1 for a repo whose links did not migrate | yes |
| P15 | Module R: Opus 5 draw+drift sd 3–5 (point 4.0 — the 25 % bar fires); RepoRadar sd at the tag ≤ 2.5; module J flip rate ≤ 15 % | yes |
| P16 | Recall probe: ≥ 34/37 legacy recognised by all three models; held-out popular ≥ 80 %, small ≤ 50 % | yes |

---

## 9. Cost and labour

Per-case rates: Opus 5 ~$9.50 notional (~$19 science), RepoRadar ~$1, judging $5–10 per judge per arm per case (cache hits reduce re-runs).

| item | compute | conditional? |
|---|---|---|
| A. Held-out 23: Opus 5 (13 × $9.50 + 10 × $19 = $314) + RepoRadar shipped + arXiv+EPMC ($46) + judging 2 judges × 3 arms (EPMC arm shares most of the pool; count 2.3 arm-equivalents) $530–1,060 | **$890–1,420** | no |
| B. Module R (12 legacy cases, both arms, both judges): $140 + $12 + $240–480 | **$390–630** | no |
| C. Second Opus 5 draw on the 23 + judging 2 × 1 arm | $540–770 | if R's bar fires (predicted yes) |
| C′. Second draw on the other 25 legacy cases | $300 + $250–500 | optional |
| D. Validity pool: mining $0; ~60 positives + 240 controls + legacy 35 + 140 re-controls ≈ 475 papers × 2 judges at the measured ~$0.03/verdict | **$30–100** | no |
| E. Module J (40 papers × 2 judges), recall probe (60 × 3), coverage/enumeration API | **< $20** | no |
| F. RepoRadar at the tag on all 37 + re-judge (≈ 60 % cache hits) | $37 + $150–300 (worst $740) | if §4.1's named run predates the tag |
| G. Optional human slice: 60 held-out papers stratified arm × stratum, two non-author annotators blind to arm, same rubric; reported as ordering agreement (Spearman vs each judge's score) and κ — **not** as a base-rate arbiter (a third rater picks a level exactly as a third model would) | $0 compute; 30 h or $600–1,200 hired | optional |
| **Modal total (A+B+D+E)** | **$1,330–2,170** | |
| **Maximum (all conditionals, no G)** | **$2,750–3,900** | |

**Labour (hours):** frame finalisation, topic lists, prior-exposure list, scripts 16; enumeration, snapshot, coverage check 8; classify/draw/seed 4; screening walk 60–110 rows at 15–25 min (clone only after the cheap rules) 20–35; case builds (clone at SHA, yaml, contexts, hashes) 23 × 40 min ≈ 15; validity pool supervision and analysis 10; arm runs, interim, final analysis 12; datasheet, ledger tables, paper edits 12. **Total ≈ 95–115 h** (+30 h for G). **Calendar:** validity pool week 1–2; window W (≤ 21 days) weeks 2–5; analysis and write-up week 6.

---

## 10. Runbook (executable order; nothing out of sequence)

1. Verify P0.1 (grep NR-52 verdict), run P0.4, fill §0 blanks. Tag the system; fill §4.1 step 1.
2. Commit this file + `evals/frame/` scripts and lists → H. Do not clone anything.
3. Day D: `uv run python evals/frame/enumerate.py --date D --topics evals/frame/topics.json --out evals/frame/universe-D.csv`; commit CSV + `coverage.json`.
4. After the beacon pulse: write `evals/frame/SEED`; commit.
5. `uv run python evals/frame/draw.py --universe ... --seed $(cat evals/frame/SEED) --quotas evals/frame/quotas.json --out draw_order.json`; commit.
6. `uv run python evals/frame/walk.py` (X2–X8, ledger, reserves, pinned SHAs); commit `ledger.csv`, `selected.json`, `run_order.json`.
7. Build the 23 cases (§4.3); `verify_contexts` hashes committed for all 23. **From here no case may be excluded or replaced except X9 before its first arm run.**
8. Validity pool: `mine_adoptions.py --mine --extractor v2` over qualifying rows → `validity_screen.csv` committed; `judge_validity_adoption.py --controls arxiv-window --judge` in seeded order to ≥ 60; analysis committed as NR-`__` **before step 9**.
9. Window W opens: stage 1 (first 11 in `run_order.json`), both arms, both judges; modules R and J; RepoRadar-at-tag on 37 if F applies. Interim (§5.3) computed by `analyze.py --interim`; decision committed. Stage 2. Second draw if R's bar fired.
10. `analyze.py --final`: every table in §5.2, predictions P1–P16 scored, datasheet (`universe-D.csv`, `draw_order.json`, `ledger.csv`, `selected.json`, per-run model strings and timestamps, both judges' raw scores, adoption sets and controls, void lists) published with the paper.

---

## Conflicts between the six proposals, decided

| conflict | decision | why (one sentence) |
|---|---|---|
| Adoption repos inside vs outside the benchmark | outside (§6) | Judge validity needs no arm run, so putting citation-rich repos in the headline buys bias for nothing. |
| Primary = precision (pooled) vs held-out generalisation | held-out primary, pooled always-reported secondary | Half-width ≤ 1.7 is reachable only by pooling the development set, which is the named reviewer penalty. |
| Gate the pooled number on the held-out result vs always report | always report + heterogeneity test + reading rule | A publication gate keyed to an observed mean is winner's-curse selection. |
| Interim on pooled ≥ +0.8 vs CP < 0.20 vs held-out boundary | held-out m11 ≤ −2.0, OCs stated | The pooled bar cannot fail; a CP rule is incoherent when the pre-registered aim is not significance. |
| Post-cutoff "unrecognised" stratum vs star band | star band + recall covariate | The creation window is empty at 2026 cutoffs and the stratum confounds recognisability with literature absence. |
| Profiler failure as exclusion vs outcome | outcome (§5.6) | Excluding what the product cannot ingest selects the sample toward the product. |
| Controls from RepoRadar's pool vs arm-neutral | arm-neutral arXiv-window controls | A judge harsher on RepoRadar-shaped papers must not be credited with validity for it. |
| Choose the primary judge by adoption gap vs fixed | fixed GPT-5.5, consequences pre-committed | The gap is an operating-point statistic that restates the base rates and cannot pick one. |
| Seed from commit hash vs beacon | beacon | A commit hash can be re-rolled by re-committing. |
| Quotas: coverage targets vs legacy-share rule | legacy-share rule for L3/L4, equal split elsewhere | Designer-chosen targets after seeing stratum deltas are case-shopping one level up; the rule caps the favourable strata. |

---

## 11. Paper paragraphs

**Methods.** Cases were assembled in two waves. Wave A (n = 37) is a convenience sample of public GitHub repositories curated during development; the system's configuration was fixed with its scores visible (25 tuned, 12 pre-registered additions), and it is reported separately. Wave B (n = 23) is a stratified random sample drawn under a frame committed before any candidate was enumerated (commit H): the population is public, non-fork, non-archived GitHub repositories carrying at least one of a committed topic list, with ≥ 100 stars, ≥ 6 months of history, a push in the preceding year, an English README, and a primary language in a stated set; the frame was enumerated on date D from the GitHub search API with every query sliced below the 1,000-result cap (raw snapshot published), stratified by literature-venue locus (arXiv-native ML, physical/computational science, life science, off-arXiv CS, no-literature) crossed with star band (100–999, ≥ 1,000) under quotas fixed by a rule that holds the abstention-shaped strata at their Wave-A share (5/5/5/6/2), ordered within cells by SHA-256 of a NIST beacon pulse issued after the snapshot commit, and walked under nine mechanical exclusion rules with every candidate logged (Appendix: N excluded per rule per cell). RepoRadar was frozen at a tagged configuration before the frame commit; repositories were pinned at their HEAD commit on D and all Wave-B contexts hashed before either system ran; both systems and both judges ran within one 21-day window under recorded model version strings, with a 12-case Wave-A re-run measuring comparator drift and draw noise; a run that returned nothing is scored as abstention, a run that failed after three attempts is reported as unmeasured and never as zero, and no case was excluded after either system had run on it. One non-binding futility look (held-out mean ≤ −2.0 after 11 cases; no alpha spent) was pre-registered, as were sixteen predictions checked in §X. The primary estimate is the paired RepoRadar − Opus 5 net@2 mean over Wave B under the GPT-5.5 label with the same under claude-sonnet-5 and a consensus label always reported; the pooled 60 is reported as a labelled mixture with a heterogeneity test, and every pooled figure is also given without the no-literature stratum and without the two abstention-shaped strata. Judge validity was assessed on a separate seeded pool of citation-tracking repositories using adoption (identifiers present in a repository's documentation at HEAD and absent 24 months earlier, mined from git with no model) against window-matched arXiv controls, reported as per-judge discrimination with repository-clustered intervals.

**Limitations.** The benchmark's population is public, English-documented, topic-tagged GitHub repositories with ≥ 100 stars in a stated language set; it says nothing about private repositories, non-English or non-GitHub projects, or untagged repositories, and the small-star band is a proxy for, not a sample of, the thinly-documented private codebase the system is built for. Wave A remains a convenience sample; adding a frame-sampled wave does not make the pooled 60 representative, and the held-out estimate (half-width ≈ 2.8 net@2 at n = 23) has roughly 8 % power at the effect observed in development, so the expansion establishes a stated population, a held-out check, and a datasheet, not significance; at the observed effect ~280 cases would be needed. Per-stratum results on 5–6 held-out cases are descriptive, and the materials-science sign reversal is neither confirmed nor refuted here. The two judges order papers alike but disagree on base rate, the sign of the paired margin depends on which is used, and the model-free adoption label can test discrimination but not level; judge validity is measured on citation-tracking research repositories and is an assumption for the off-arXiv, no-literature, and small-star cases. Both systems and both judges are frontier models trained on public GitHub; exposure is quantified by a recall probe but not removed, and the comparator's live web search can read each repository's current citations. Comparator runs that exhaust their turn budget fall disproportionately on scientific repositories and are reported as unmeasured, with a sensitivity that scores them as abstentions. The held-out property of Wave B is consumed by the first configuration decision made with its scores visible.