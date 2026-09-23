# Registration of the scientific cohort (12 repositories) and its predictions

Verbatim excerpt of `evals/RESEARCH-scientific-software.md`, lines 1394-1598, from the development repository. Section 14: population, strata, configuration, endpoints, predictions and bars, written before the benchmark measurement and after the six-repository pilot the paper mentions. TIMELINE.md gives when it was committed.

---

## 14. PRE-REGISTERED — cohort 3, and the re-baseline it is bundled with (2026-08-19)

Written **before the first paid call**, like §9, so no bar can be chosen after seeing the
answer. Everything below — the twelve repositories, their profile strata, the endpoints, the
predictions and the kill bars — was fixed while the spend was still zero.

### 14.1 The question, and why it cannot be deferred again

Two questions, and the run answers both because they need the same session:

1. **Does Anonymous work on non-ML scientific software?** Six live runs (§5) said "on three of
   six", on a single draw, with no baseline, before nine profiler fixes. That is the demo's
   whole premise and it has never been measured properly.
2. **What are this project's numbers now?** §11 measured that the profiler work moved the
   keyword profile of **16 of the 25 benchmark cases**. Every live figure in `RESULTS.md`
   describes a pipeline that no longer ships. This is not optional maintenance; it is the debt
   §10.4 recorded and §11 quantified.

Bundling them is the cheap direction. The legacy 25 have to be re-run anyway; running the new
twelve in the same session costs one baseline pass instead of two and puts every number in one
artifact under one config fingerprint.

### 14.2 Population: twelve cases, fixed before judging

Six bioinformatics and six materials science, all drawn from the nineteen repositories scouted
in §4 — so the pool was defined before any of them was judged, and no case was added after
seeing a result.

| case | repository | domain | criteria | profile stratum |
|---|---|---|---|---|
| `bio-align` | minimap2 | long-read alignment | N, C | **defective** |
| `bio-singlecell` | scanpy | single-cell analysis | N, C | clean |
| `bio-scvi` | scvi-tools | probabilistic omics | N, C | polluted |
| `bio-mdsim` | OpenMM | molecular dynamics | N, B | polluted |
| `bio-mdtraj` | MDAnalysis | trajectory analysis | N, B | clean |
| `bio-kmer` | sourmash | k-mer sketching | N, B | clean |
| `mat-mlip` | MACE | equivariant potential | N, C | polluted |
| `mat-chgpot` | CHGNet | charge-informed potential | N, C | clean |
| `mat-descriptors` | DScribe | atomistic descriptors | N, C | clean |
| `mat-toolkit` | pymatgen | materials analysis | N, C | polluted |
| `mat-featurize` | matminer | materials informatics | N, C | **defective** |
| `mat-phonon` | phonopy | lattice dynamics | N, C | clean |

**Excluded, and this matters more than the inclusions.** htslib, kallisto, tblite, kim-api,
LAMMPS, deepmd-kit and nequip are not here. The first five are the compiled, manifest-less,
`doc`-less repositories where Anonymous is *known* to be weakest — install boilerplate as
queries (`pre-commit`, `mamba`, `gnu lesser`), a citation id as a query (`giab007`), a package
that loses its own name (`__kallisto__`). Dropping them makes this cohort **optimistic about
the population it claims to represent**, and any sentence written from these results has to
carry that. deepmd-kit and nequip were trimmed only to reach six per domain.

**Nobody is here for having a good profile.** `bio-align` and `mat-featurize` are the two worst
profiles of the twelve and both are in. Removing them would raise the mean by construction,
which is the shape of result this document exists to refuse.

### 14.3 Strata, assigned 2026-08-19 from profiles, before any judging

Measured with the shipping profiler on the cloned repositories, written down here so that
"which cases did well" cannot become a post-hoc story:

- **clean (6)** — subject words dominate and the prose describes the project:
  `bio-singlecell`, `bio-mdtraj`, `bio-kmer`, `mat-chgpot`, `mat-descriptors`, `mat-phonon`.
- **polluted (4)** — subject words present with a named defect beside them: `bio-scvi`
  (`customcard`, a sphinx-design directive, keyword #5), `bio-mdsim` (zero anchors, no domain —
  a C++ project, so the manifest channel is a structural zero), `mat-mlip` (prose is the table
  of contents — A6, deferred in §10.3), `mat-toolkit` (`directory`, `following`, `support`,
  `default` in the top twenty).
- **defective (2)** — the profile does not describe the project: `bio-align` (prose is a
  phishing warning; keywords are CLI flags `fa`, `fq`, `aln`, `ax`), `mat-featurize` (`module`,
  `tests`, `utils`, `test module` — the Sphinx module index and the test tree, because all 30
  of its doc files are generated `docs/_sources/` duplicates, §10.4).

Three defects deferred with measurements in §10.3 are confirmed still live on this cohort:
A6 on `mat-mlip`, B4's anchor-only bigram (`wandb crystal-toolkit`) on `mat-chgpot`, and
`docs/_sources/` on `mat-descriptors` and `mat-featurize`. None is fixed for this run; fixing
them mid-cohort would make the pre-registration meaningless.

### 14.4 Configuration and command

The shipped, measured configuration, unchanged from the last full runs in `RESULTS.md`:

```bash
uv run python evals/run_judge_eval.py --baseline api \
    --rr-pool 50 --rr-rerank --rr-all-time --rr-hybrid --rr-sweep --rr-finescale --rr-hyde \
    --rr-frozen-pool evals/.work/pool-cohort3
```

`--rr-frozen-pool` is not optional. Two runs of an identical configuration overlap only **0.49
by Jaccard** on the ranked top-10, and that is the largest variance term in every paired
comparison this project has run. Freezing the pool makes every later arm — the Europe PMC arm
below, any threshold sweep — a comparison against the same candidates rather than against a
different draw.

`--baseline api` keeps the Opus arm. Without it there is a description and not a comparison.

**Order, as a cost guard:** the twelve new cases first (`--case bio-align,bio-singlecell,…`),
then the legacy 25 into the same frozen-pool directory. Splitting costs the whole-run shift
(sd **0.27**), which is small against the 1.04 floor, and buys a checkpoint before the larger
half of the spend.

### 14.5 Endpoints, declared now

**Primary:** mean `net_value@2` over the **twelve scientific cases**, with pooled precision.

**Secondary, all pre-declared so none is a fishing expedition:**
- the legacy-25 mean, re-measured in the same session (the re-baseline);
- the bio-6 and mat-6 means separately;
- the three profile strata;
- the all-37 mean, which is what the benchmark's headline becomes.

The legacy-25 arm is re-measured rather than compared against a stored number, so **no
conclusion here depends on which historical figure was right**.

### 14.6 What the instrument can and cannot resolve

Per-case sd is **1.73**; the benchmark's minimum resolvable effect is **1.04 net@2** per case
for a paired same-session comparison (`noise_floor.py`, three draws, 42 df).

| quantity | n | SE | resolves |
|---|---|---|---|
| a domain mean (bio-6, mat-6) | 6 | 0.71 | effects above ~1.4 |
| the scientific-12 mean | 12 | 0.50 | effects above ~1.0 |
| scientific-12 vs legacy-25 | — | 0.61 | a gap above ~1.2 |
| a stratum mean (defective, n=2) | 2 | 1.22 | **nothing**; reported as description only |

So this run can establish *"scientific software is not materially worse than ML/CS"* only down
to about 1.2 net@2, and the two-case defective stratum is an anecdote by construction. Stated
here so that neither is over-read later.

### 14.7 Predictions

Written from the §5 single draw (bio three at +6.67 / precision 0.92; matsci three at +3.00 /
0.744) and from the published ML headline of +5.12 over 25 cases.

| | prediction |
|---|---|
| scientific-12 mean | **+4.0 to +6.0** |
| bio-6 | **≥ +5.0** |
| mat-6 | **+2.5 to +4.5** |
| legacy-25, re-measured | **+4.0 to +6.5** — wider than it sounds, because 16 of 25 profiles moved |
| clean stratum (n=6) | ≥ +5.0 |
| defective stratum (n=2) | ≤ +2.0 |
| pooled precision, scientific-12 | ≥ 0.80 |

The gate remains the predicted failure mode: §5.1 found 11 of 12 non-actionable Top Picks were
**score-3 papers**, and §9.4 killed both candidate fixes against their own bars. Expect the
same signature, now with n=12 instead of n=6 behind it.

### 14.8 Bars

- **WIN** — scientific-12 mean **≥ +4.0** and pooled precision **≥ 0.80**. "Anonymous works on
  scientific software" becomes a supportable claim for repositories of this kind, with §14.2's
  exclusion stated alongside it.
- **KILL** — scientific-12 mean **≤ +2.0** or pooled precision **≤ 0.65**. The demo is scoped to
  named repositories and the general claim is dropped, not softened.
- **BETWEEN** — reported as "works on the clean stratum; not established generally", and the
  demo leads with `mat-descriptors` and `bio-singlecell`.

A **legacy-25 mean below +3.5** is its own alarm, independent of the cohort: it would mean the
profiler work cost the ML benchmark something that Tier A, being four hand-written fixtures and
a frozen pool, cannot see.

### 14.9 A second arm, declared now so it can be run later

`--sources arxiv,europepmc` on the **bio-6**, against the same frozen pool. This is what B1
(§13) was built for and it has never been judged. Declared here rather than after the primary,
so the bar predates the number: **the paired delta must be ≥ +1.0 net@2 on the bio-6** to
justify recommending the source, since anything smaller is inside the floor. Its sign is
genuinely unknown — the fine-scale map has never been fitted on a bio abstract, and Europe PMC
papers reach the gate with **no HyDE** behind them.

Not funded by this pre-registration. Its cost is small because the arXiv pool is frozen.

**§20 supersedes this paragraph, and three of its four sentences of design are wrong.** The
frozen pool cannot be reused — `sources` is in `POOL_FLAGS`, so the harness refuses, and it is
right to: that pool holds no Europe PMC paper, so the arm would have measured zero by
construction. The cost is therefore not small. And the +1.0 bar has no power behind it — the
bio-6 per-case sd is 3.78, which was measurable from the artifact when this was written. The
*question* stands and is now designed in §20.

### 14.10 Cost

Estimated from call counts, not invoices, at $1.25–3.00 per case (judge + Opus baseline):

| | cases | estimate |
|---|---|---|
| cohort 3 alone | 12 | **$15–36** |
| legacy re-baseline | 25 | **$31–75** |
| both | 37 | **$46–111** |

**The 69 verdicts cached under `evals/cache/judge/v1/gpt-5.5/scisoft-*/` are NOT reusable**,
checked 2026-08-19 before quoting them as a saving. Two independent reasons: the cache path is
keyed on the case label, and those runs used `scisoft-minimap2` where the benchmark case is
`bio-align`; and each entry is validated against `sha256(RUBRIC + repo_context)`, which the
profiler work changed for every one of these repositories. §8 listed them as a saving. They are
not one, and the estimate above assumes no reuse.

### 14.11 What this run does not measure

The compiled/manifest-less population (§14.2). R, Julia, Rust and Nextflow repositories, none
of which has ever been profiled. Journal-only literature. The recency path. A second draw — every
number from this run is a single draw, and the §5 matsci values (+1, −1, +9) had a per-domain
sd of 5.3, which is the strongest argument for reading the twelve-case mean and not the cases.

---
