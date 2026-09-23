# Benchmark-expansion rules (a draft that was never registered)

Verbatim excerpt of `evals/PREREG-benchmark-expansion.md`, lines 1-2, 69-82, 126-149, 184-191, from the development repository. This draft was never registered. It is included because the adoption walk's registration (PREREG-judge-validity-pool.md) adopts its eligibility rules by reference: section 1 and section 3 (rules X4, X5 and X7), which `evals/frame/eligibility.py` implements. Section 5.2 is included for the estimands it names. TIMELINE.md gives when it was committed.

---

<!--
DRAFT — NOT YET REGISTERED.

[...]

## 1. Population

The 23 held-out cases are a stratified random sample of:

> Public, non-fork, non-archived, non-mirror GitHub repositories that, on the snapshot date **D** (§4.1), (i) carry at least one of the topic tags committed in `evals/frame/topics.json` (§2.1), (ii) have ≥ 100 stargazers, (iii) were created ≥ 180 days before D, (iv) were pushed to within the 365 days before D, (v) report a primary language in {Python, C, C++, Rust, Go, Julia, JavaScript, TypeScript, R, Fortran}, (vi) have a README whose prose is English (§3, X5), and (vii) are software projects under the mechanical rule X4.

**What is claimed.** Estimates over the 23 are estimates over *this* population under the stratum composition fixed in §2.2 — a composition-weighted mean, not a GitHub-population mean (no Horvitz–Thompson weights are used; the per-cell universe counts are published so a reader can reweight).

**What is not claimed.** Nothing about private repositories, non-English documentation, non-GitHub hosting, untagged repositories (topic tagging correlates with maintainer polish — the coverage check in §4.2 reports how many of the 37 legacy repos the frame would have captured), or the paper's stated target user. The **small-star band (100–999)** is the nearest public proxy for an unmemorised, thinly-covered codebase and is named a proxy, not the target user. The three thin legacy repos (fireball, sekas, distributed_graph_flow) are below the star floor and outside this population; they stay in the benchmark under the legacy flag.

The 37 legacy cases are a **convenience sample** (three hand-curated waves). The pooled 60 is reported as "37 convenience + 23 frame-sampled" and is never called representative.

---


[...]

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

**Not exclusions, by design:** profiler failure (`rr profile` raising is a Anonymous outcome, §5.6); Opus turn-budget exhaustion (§5.6); prose volume; absence of a bibliography; "primary literature already represented"; any human judgement of interest.

**Pre-screens at $0 (recorded, not selective):** for every *selected* repo, after selection and before either arm runs: (a) `ids_v2(HEAD)` count (extractor v2, §6.1) and, where ≥ 24 months of history exists, `ids_v2(HEAD) − ids_v2(T0)` — recorded as covariates and as **incidental** adoptions (they never count toward §6's target); (b) arXiv-coverage covariate = fraction of DOI/arXiv/PMID identifiers extractable from docs at HEAD that resolve (OpenAlex) to a work with an arXiv version — reported per stratum as a manipulation check of §2.1 (expected order L1 > L2a > L2b > L3; L4 mostly undefined 0/0), never used to reassign a stratum.

**Ledger** `evals/frame/ledger.csv`, one row per walked candidate: `cell, seeded_rank, full_name, stars, language, topics, created_at, pushed_at, decision{SELECTED,RESERVE,EXCLUDED,VOID-PRE}, rule_id, evidence (URL / SHA / path), checked_on, minutes`. Counts per rule per cell are a datasheet table. **Nothing is ever excluded after either arm has run on it** (§5.6 governs arm failures).

---


[...]

### 5.2 Pre-registered estimands and tests
- **Primary (generalisation):** mean paired net@2, Anonymous(tag) − Opus 5 draw 1, **GPT-5.5 label**, over the 23 held-out cases; 95 % CI from the project's bootstrap helper (same helper as every published figure); paired sign-flip permutation p (10,000 draws, two-sided); win/loss/tie.
- **Co-primary label:** the same under **Sonnet ≥ 2**; consensus (GPT ≥ 2 ∧ Sonnet ≥ 1) as the third PREREG-rung1 label. **All three always reported; the primary label is GPT-5.5 for continuity and is not switchable** (§6.4 says why no validity result can license a switch).
- **Secondary (magnitude):** the same statistics over the pooled 60, always reported, labelled "37 convenience + 23 frame-sampled", **never gated on the held-out result** (a gate keyed to an observed mean is a data-dependent selection rule — reviewer refutation). Instead a pre-registered **heterogeneity test**: Welch t between legacy-37 and held-out-23 paired deltas, p reported.
- **Pre-registered reading rule for the held-out estimate** (so the sentence cannot be chosen afterwards): m23 ≥ 0 and CI23 ∋ m37 → "consistent with the development-set estimate; not independently significant"; m23 ≥ 0 and CI23 ∌ m37 → "held-out advantage smaller than the development estimate"; m23 < 0 → "the development-set advantage is not supported on held-out data" and the paper's headline sentence is the held-out number. Applied under each label.
- **Cuts (all pre-declared):** without L4; without L3+L4; per stratum × band (held-out cells have n = 1–3; reported as points, no confirmatory claim); leave-k-out removing the k largest positive **and, separately, the k largest negative** deltas, k ∈ {3, 5}; Wilcoxon signed-rank as a secondary test (ranks bought nothing at n = 37: t p = 0.32 vs Wilcoxon 0.37).
- Legacy cases carry a three-way flag: `tuned` (core 25), `pre-registered-observed` (bio 6, matsci 6: predictions were recorded before their measurement, but the shipped configuration was confirmed with their scores visible), `held-out` (the 23). Why: a blanket "37 = tuning set" over-concedes the science 12 (reviewer refutation).

