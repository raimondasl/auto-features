<!--
REGISTERED 2026-09-02. This file binds from the commit that removed the word DRAFT.

A correction made in the act of registering: the DRAFT banner said registration required the
candidate list `pool-universe-Dp.csv` to be committed. That was backwards and contradicted
this file's own runbook, where enumeration is step 3 and registration is step 1. Registration
comes FIRST -- it is what makes the enumeration a procedure carried out under a fixed design
rather than a search. The candidate list is an OUTPUT of the registered procedure.

WHAT IS FROZEN FROM NOW ON
  * the population rule (section 2), the label and every filter (section 1), the control
    scheme (section 4), the endpoints and their pre-committed consequences (section 5), the
    budgets and stop rule (section 3), and predictions P1-P9 (section 8);
  * the code that implements them, committed alongside: mine_adoptions.py (extractor v2),
    frame/enumerate_pool.py, frame/walk_pool.py, judge_validity_adoption.py
    (arxiv-window controls, cluster_bootstrap_auc, the cache gate) and their tests.

WHAT IS NOT YET WRITTEN, and the one honest gap at registration: the analysis wrapper that
assembles pool verdicts into section 5's tables does not exist. cluster_bootstrap_auc and
roc_auc do, and are tested; the glue does not. It must be written BEFORE any verdict is
bought, and it must implement section 5 as registered here rather than as convenient later.

BLANKS still to fill, each at the step that produces it and never earlier: Dp (section 2.1),
SEED_POOL and its pulse timestamp (section 2.4), the two judge training cutoffs (section 5).

MEASURED BEFORE REGISTRATION, and labelled as such wherever it is used:
  * the `diffusion` link-migration premise (section 1);
  * `ids_v2(T0)` and `ids_v2(HEAD)` counts across the 37 legacy benchmark repositories,
    which derive the PP2 threshold (section 2.3) and size the life-science blind spot
    (section 6, item 2).

NOT MEASURED, so the predictions that depend on them are genuinely open: no HEAD-minus-T0
adoption set has been taken over the legacy 37 (P1, P2 unanswered); no AUC has been computed
on any real verdicts, legacy or otherwise (P6, P8 unanswered -- the cluster bootstrap is
tested only against synthetic fixtures, deliberately); no candidate repository outside the
legacy 37 has been enumerated, cloned or looked at (P3, P4, P5, P9 unanswered).

MAINTAINER DECISIONS of 2026-09-02, all recorded before registration:
  * the life-science blind spot is SIZED, not closed (section 6, item 2);
  * the judging target is 100 new positives, with 60 as the reporting minimum (section 3.3);
  * the primary endpoint measures ADOPTION DISCRIMINATION, not matching, and a null is
    three-ways ambiguous (section 4, section 5) -- raised by the maintainer before
    registration and written into the pre-committed consequences rather than the limitations;
  * the 37 -> 60 benchmark expansion is on hold and runs only on an explicit directive, which
    is why this file exists separately (section 11).
-->

# Pre-registration — judge-validity pool from repository adoption

**File:** `evals/PREREG-judge-validity-pool.md`. **Status: REGISTERED 2026-09-02.** Written and registered before any candidate outside the legacy 37 was enumerated, cloned, profiled or looked at, and before any adoption set or AUC was computed on any data. Every remaining blank (`____`) is filled at the step that produces it and never earlier; the git history of this file is the audit trail.

---

## 0. Why this exists

Every quality number in this project is agreement with a judge. NR-52 measured GPT-5.5 and Sonnet 5 disagreeing about a comparator margin badly enough to flip its sign; NR-53 showed the disagreement is a property of the judges (self-kappa 0.798 against cross-judge 0.199) rather than sampling. NR-59 located it precisely: the two judges **order** papers almost identically — AUC against the fine-scale score differs by **0.027** — and disagree about **level** by nearly a factor of two, base rates **0.874** and **0.494**. Every threshold in the system is therefore a bet on a base rate.

No third judge fixes this. Consensus, majority-of-three and a tiebreaker all *pick* a level by construction; none *measures* one. That needs a label from outside the models, and there is exactly one: **adoption**. An identifier present in a repository's documentation at HEAD and absent 24 months earlier is a technique the project verifiably took up, mined from git history with no model in the loop.

NR-56 and NR-57 built that label and could not settle the question: 35 usable positives across 9 repositories, `graph` supplying 13, gaps 0.143 and 0.243, neither excluding zero. NR-57 established that **this benchmark cannot supply more** — mining all 37 cases moved usable adoptions from 31 to 35. So the positives must come from a population enumerated for the purpose. That is this pool.

**What it is not.** It is not a benchmark, it produces no net@2 figure, and no repository in it enters `benchmark.yaml`. It runs no RepoRadar arm and no comparator: mining is git plus a regular expression, and judging shows each judge a repository as it stood at T0. That is why it can run entirely on its own, and why this is its own file (section 11).

---

## 1. The label

**Adoption = `ids_v2(HEAD) − ids_v2(T0)`**, where **T0 = the pinned HEAD commit's date − 24 months** (`mine()` computes `head_date − 720 days`; T0 is anchored to the repository, not to a calendar date, because the label asks what a repository took up in *its own* last two years).

**Extractor v2** = the arXiv regex ∪ `huggingface.co/papers/<id>` ∪ `hf.co/papers/<id>`, applied at **both** ends.

**Why both ends, measured rather than asserted.** `diffusion` holds **99** arXiv-regex identifiers at T0 = 2024-08-16 and **11** at HEAD, while the Hugging Face form goes **66 → 163**; the union goes 127 → 172. Projects migrate their paper links wholesale, and v1 reads that as a project that stopped citing papers. The damaging direction is the other one: **28 of T0's identifiers are reachable only through a Hugging Face link**, so under v1 any of them re-linked to arXiv by HEAD scores as a fresh adoption — a fabricated positive in the one label here that no model produced. `tests/test_adoption_extractor_v2.py` pins that failure.

**Filters, all applied at HEAD unless stated.**

| filter | rule | why |
|---|---|---|
| self-citation | `CITATION*` files and identifiers within 800 characters of a `Citation`/`Cite`/`BibTeX` heading, under **both** extractors | a reference implementation always cites its own paper and did not *adopt* it; under v2 the project's own paper in HF form would otherwise be the strongest-looking adoption in the pool |
| too-new | paper posted < 182 days before `head_date` | a paper the project could not reasonably have evaluated yet |
| reverse-citation path | identifiers whose **every** occurrence is under a path matching `/(projects\|showcase\|used[-_ ]by\|gallery\|community\|awesome)/i` | those papers cite the repository, not the reverse |
| doc-genesis | `ids_v2(T0) ≥ 1`; rows from repositories below it are flagged `genesis` and excluded from the primary | with no bibliography at T0 there is no "before", so every identifier at HEAD reads as an adoption |

**The reverse-citation filter is applied to the HEAD side only.** Dropping an identifier from the T0 bibliography would *manufacture* an adoption, which is the one direction a ground-truth label cannot afford to be wrong in.

**HEAD is pinned to a SHA and recorded on every row.** Not a precaution: `diffusion` measured hours apart differs by one paper, while T0 — pinned — reproduced exactly.

**The v1 record is immutable.** `mine_adoptions.py` writes v2 to `adoptions-v2.json`; NR-56/57's `adoptions.json` is never overwritten, because it is the record v2 is compared against.

---

## 2. Population

### 2.1 Enumeration
`pool-universe-Dp.csv`, produced on day **Dp = 2026-09-02** (snapshot taken 2026-09-02T21:56:31Z; **17,888 distinct repositories from 266 queries**, 0 rows from a truncated slice, 0 rows created after the cutoff of 2024-03-02) by `evals/frame/enumerate_pool.py` against the committed topic list (`evals/frame/pool/topics.json`, 32 topics) and rule set, from the GitHub search API with every query sliced below the 1,000-result cap. The raw query URLs, raw JSON responses, response `Date` headers and per-query `total_count` are archived alongside, because the snapshot can never be re-queried.

**Snapshot log (append-only). One snapshot was taken, found defective, and discarded; the record is here rather than in a commit message, because a re-taken snapshot is exactly what shopping for a population would look like.**

* **Run 1 — 2026-09-02T21:25Z, DISCARDED.** 17,779 rows from 266 queries. The query builder emitted **two** `created:` qualifiers when subdividing an over-cap slice — `created:<=2024-03-02` plus `created:<start>..<end>` — and GitHub honours only one. Two measured consequences, both in the preserved `coverage-DISCARDED-run1.json`: every year subdivision of `machine-learning stars:100..149` returned the **identical** `total_count` of 1196 as the un-subdivided query, so subdivision narrowed nothing and the 1,000-result cap was never escaped; and the final slice returned **1377**, *more* than the un-subdivided query, because its year range replaced the cutoff — admitting **192 repositories created after 2024-03-02**, the latest 2026-08-10, into a universe defined as capped there. The run is discarded for violating its own eligibility rule, not for the population it produced, and its `coverage.json` is committed under that name so the claim can be checked. Fixed in `_query` (one `created:` clause, upper bound `min(year_end, cutoff)`), pinned by a sweep over the whole slice × year grid.
* **Run 2 — the snapshot of record.** Same Dp, same topics, same rules; only the query builder changed.

**The committed archive is trimmed, and that is required rather than convenient.** The raw responses are ~250 MB and every payload carries `html_url` for every repository — tens of thousands of `github.com/<owner>/<repo>` strings, which is precisely what the no-URL rule below keeps out of this tree. Committing them would break, at scale, the rule the CSV obeys. `trim_archive` keeps the query, its response `Date`, the API's own `total_count`, and the ordered `full_name`s each slice returned — enough to check that the CSV is exactly the union of the slices and that no slice was silently truncated. The untrimmed raw stays on disk, untracked.

**The enumeration artefact carries no `html_url` and no `url` column** — repositories are identified by `full_name` only. This is a rule, not an implementation detail: any artefact containing `github.com/<owner>/<repo>` strings inside this repository would be swept up by a later prior-exposure grep, and `mine_adoptions.SCREEN_COLUMNS` is already URL-free. `tests/` pins it.

### 2.2 Eligibility
Public, non-fork, non-archived repositories carrying at least one of the committed topics, with ≥ 100 stars, an English README, a primary language in the committed set, and:

* **PP1** `created_at ≤ Dp − 30 months` as an API pre-filter **and**, checked after the clone, `head_date − created_at ≥ 30 months`. The API filter alone does not guarantee history before T0, because T0 is anchored to `head_date` (section 1) and a repository may have been quiet for a year. `thin-gnn` and `thin-lang` are NR-57's record of what that costs.
* **PP2** `ids_v2(T0) ≥ 3`, with T0 resolved exactly as `mine()` resolves it (section 2.3).
* **PP3** at most **3 repositories per owner**, applied along the frozen seeded order.
* Not one of the 37 legacy benchmark cases.
* Software-project and README-quality rules as committed (a curated paper list is precisely the artefact carrying two hundred identifiers and zero adoption semantics; README quality matters because section 4 puts a README excerpt into both judges' prompts).

**Implementation note, 2026-09-03, before the walk.** A pre-flight audit found that none of the rules in this list was implemented — the walk asked only PP1 and PP2 — so a curated paper list could have cleared `ids_v2(T0) ≥ 3` on a bibliography of two hundred identifiers and dominated the positives. All four are now in `evals/frame/eligibility.py`, each recording its own ledger outcome so a rejection says *which* rule rejected it. Two carry recorded deviations:

* **X4 reads `topics ∪ name ∪ README[:300]`, not the repository description.** The benchmark frame's X4 includes the description; §2.1 forbids it from entering this tree because it routinely carries a repository URL, so the enumeration never recorded it. This narrows the rule's input, and the narrowing can only let *more* repositories through, never fewer.
* **X5 is scaffolded but NOT APPLIED.** The registered rule is fastText `lid.176` at p(en) ≥ 0.8 on stripped prose, applied only above 300 characters. The stripping, the threshold and the `lid_na` pass-through are implemented exactly as registered; the detector is injectable and is **absent**, because `lid.176` is a 126 MB model behind a package this project does not depend on. Every candidate therefore passes flagged `lid_na_no_detector`, and the flag is on every row. A substitute detector was deliberately not written: it would be a different rule wearing X5's name, culling a different set with no way for a reader to tell. Supply the model and the rule applies as written; until then the pool's population is **not** filtered for English prose, and §6 gains that as a stated limitation — both judges are shown a README excerpt (§4), so a non-English README makes the judging task harder rather than the label wrong.

### 2.3 Why the screen is at T0, and where the constant comes from
`ids(HEAD)` **contains the outcome.** Screening on it conditions eligibility on a quantity that includes the adoptions being counted, which makes the reported yield distribution partly circular. The T0 form costs the identical clone, is a pre-T0 property like every other rule here, and subsumes the doc-genesis guard.

Because it is a **different quantity**, a threshold cannot be inherited. It is derived from the only adoption dataset that exists — all 37 legacy clones under extractor v2, $0, **measured 2026-09-02 and published here before the pool runs**:

| threshold on `ids_v2(T0)` | qualifying repos | of those, contributing ≥ 1 positive | v1 usable positives held | capped at 8 |
|---|---|---|---|---|
| ≥ 10 | 5 | 4 | 27 / 35 | 22 |
| ≥ 5 | 6 | 5 | 29 / 35 | 24 |
| **≥ 3** | **10** | **7** | **33 / 35** | **28** |
| ≥ 2 | 15 | 8 | 34 / 35 | 29 |
| ≥ 1 (the genesis guard alone) | 19 | 9 | 35 / 35 | 30 |

**The move from HEAD to T0 is yield-neutral; only the constant costs anything.** At a fixed threshold of 10 the two screens select the *identical* five repositories — `cv`, `diffusion`, `graph`, `peft`, `rl` — so the eight positives lost at that row belong to the threshold, which the merged text already had, and not to the T0 form.

**The constant moves because the primary endpoint is clustered by repository**, and a clustered interval's precision is governed by the number of clusters, not the number of papers (section 5). Per clone screened, `≥ 10` buys **0.108** contributing repositories and `≥ 3` buys **0.189**; going further, from 3 to 2, buys five more qualifying rows for one more contributing repository, which is the point where screening stops paying. Hence PP2 = 3.

**PP1 is measured too, not hypothetical:** `thin-gnn` and `thin-lang` have **no commit at all before their own T0**, so they can carry no label under any threshold. That is what an eligibility rule on `created_at` alone fails to catch.

**`ids_v2(T0) ≥ 10` is retained as a pre-declared sensitivity.** It is a strict subset of the judged set, so it costs nothing to report.

### 2.4 Order
**SEED_POOL** = the `outputValue` of the NIST Randomness Beacon v2 pulse at an **absolute UTC timestamp named in the commit that fixes `pool-universe-Dp.csv`**, at least 24 hours after it.

> **The pulse is hereby named: `2026-09-04T00:00:00Z`.** Fixed in the same commit that fixes the candidate list (taken 2026-09-02T21:56:31Z), which is ~26 hours earlier — comfortably past the 24-hour bar with margin for the push and merge to settle. `SEED_POOL` is the `outputValue` of `https://beacon.nist.gov/beacon/2.0/pulse/time/1788480000000`. Nothing in the walk may begin before it, and this timestamp is not to be re-named: doing so would require re-committing an artefact this file declares immutable, and the re-commit would be visible. Naming the pulse in the commit is what stops a re-roll: taking a later pulse requires re-committing an artefact this file declares immutable, and the re-commit is visible. A commit hash would not do — an author can re-commit until the order looks right; a future beacon pulse cannot be chosen.

Candidates are walked and positives are judged in `sha256(SEED_POOL ‖ full_name)` order, **flat** — not interleaved by stratum. Round-robin would make the pooled endpoint a designer-weighted mixture whose weights depend on which stratum exhausts first; an empty heterogeneity cell is a better outcome than a re-weighted primary.

---

## 3. The walk

### 3.1 One pass
Evaluating PP2 requires resolving T0 and grepping at T0, which is most of the mining work — so there is no two-stage economy and screening *is* mining. For each candidate in seeded order the walk clones blobless, resolves and records `head`, `head_date`, `t0`, `t0_commit_date` and the **realised window** `head_date − t0_commit_date` (the nominal cutoff is not the realised one: `rev-list -1 --before` can land years earlier across a history gap, and controls are matched on the realised date), greps `ids_v1`/`ids_v2` at both ends, records **DOI and PMID counts at both ends as a covariate only** (section 6.2), mines the adoption rows if PP2 holds, persists the T0 context with its hash so judging never re-clones, appends every row, and deletes any clone it made.

A per-row timeout of **300 s**. A timeout is a recorded outcome, never a silent skip. Clone failures are recorded, never dropped.

### 3.2 Budgets
**B₀ = 300 rows are walked unconditionally**, and the qualifying rate and per-repository yield are estimated over exactly those 300. A fixed prefix of a seeded order is a uniform random sample of the population; a prefix whose length is set by accumulated yield is inverse sampling and biases the rate upward.

The walk then continues to **B = 1,200 rows** or until cumulative capped-usable new positives reach **100**, whichever comes first. A yield curve — rows walked, rejects by rule, clone failures, timeouts, qualifiers, gross adoptions, capped-usable positives — is committed every 50 rows, so a shortfall is visible in hours rather than at the ceiling.

### 3.3 Stop rule
Per-repository cap **8**. Identifiers shared across repositories are assigned to one repository by SEED_POOL and counted once, legacy winning ties.

**Implementation note, 2026-09-03, before the walk. The tie is won by the papers the legacy cluster actually contributes, not by all 94 it mined.** `walk_pool.legacy_ids` passed every usable legacy positive into the contest, but this section's own cap of 8 means only **32 of the 94** ever enter the analysis — `diffusion` holds 46 and contributes 8, `peft` 27 and contributes 8, `graph` 13 and contributes 8. So a pool repository adopting one of the other **62** lost the tie to a paper the legacy cluster is not using, and was then counted in **neither** stratum. Sixty-two identifiers counted zero times is not "counted once", and the loss is not random: those 62 are the most widely adopted ML papers in the legacy set, which is exactly what a newly enumerated ML repository is most likely to have taken up.

The cap is therefore applied **before** the contest, by the same seeded rule a pool repository's own cap uses — `sha256(SEED_POOL ‖ case:id)`, first 8 — so both strata are capped by one implementation and which 8 survive is a function of the pulse rather than of whoever wrote the analysis. This can only *raise* the pool's positive count, which is the anti-null direction; it is taken because the alternative loses papers to nobody, not because of its effect on n, and it is recorded here before any positive exists.

**Judging stops at 100 new usable positives. 60 is the reporting minimum, not a stopping point.** Fixed by the maintainer on 2026-09-02, before any verdict exists, for a power reason stated in section 9: at 90 capped positives with legacy-like concentration the primary interval can include 0.5 for a sampling reason, which would fire section 5's pre-committed null branch on an artefact. The extra 40 positives cost roughly 400 verdicts, about **$8–12**.

**No endpoint is inspected before the stop rule fires.**

### 3.4 Extension and exhaustion
If B is reached below 60 new positives the walk continues **down the already-frozen seeded order**, to the end of the list. Taking more of an order fixed by an unchoosable pulse before anything was cloned introduces no discretion.

Changing a **rule** after exhaustion remains barred: the star floor is never lowered, no topic is added, the T0 window is never lengthened, the 182-day and self-citation filters are never relaxed, and PP2 is never moved. If the whole list is exhausted below 60, that is the recorded negative result — the analysis runs at whatever *n* exists, the shortfall is reported against section 9's minimum detectable AUC, and section 5's null branch fires.

---

## 4. Controls

Four controls per positive, **arm-neutral**: arXiv listings in the positive's primary category, submitted in the same half-year, not cited anywhere in the repository at HEAD, drawn per positive under SEED_POOL. The per-(category, half-year) listing is archived, because it is the negative class of the primary endpoint and must be reproducible.

**Implementation note, 2026-09-03, before any listing was archived. The draw spans the window; it used to come off the end of it.** `arxiv_window_listing` asked for 200 results sorted by `submittedDate`, and arXiv's default sort order is **descending** — so it returned the *newest* 200 of the half-year. Measured: `cs.LG` H1-2021 holds **13,262** papers, so every control for a positive in that window came from its last few days. This section registers "submitted in the same half-year" and names no cap and no ordering, so the 200 was an unregistered narrowing that made the negative class arXiv's index order rather than the seed's.

It also ran in a knowable direction. NR-43 measured actionability rising steadily with recency, **0.31 (2013) to 0.64 (2025)**, so controls drawn systematically months *newer* than the positive they are matched against are scored higher — compressing the gap toward the null §5 is pre-committed to reporting. Conservative, but by accident.

The window is now split into **six contiguous, non-overlapping monthly slices** covering every day of it, with the draw taken proportionally from each, at the same request cost. Residual skew is to the end of each slice — days rather than months. The archive records every sub-query, what it requested, what it returned and whether it was cut off at its own cap, because "200 of 13,262" and "200 of 200" are the same number in an archive that stores only a count.

**Full enumeration was the maintainer's stated preference and was rejected on measured cost, not on principle.** At arXiv's enforced 3 s minimum request interval and a 100-record page, one busy window is ~6.6 minutes and a run needs 40–80 of them: **4.4 to 8.8 hours** of continuous third-party API access, on the critical path *after* the walk, since the windows are not known until the positives exist. arXiv returned HTTP 429 during the measurement that produced these figures and subsequent fetches did not complete. `depth="full"` is implemented and available, so the choice stays open rather than being closed by this paragraph; the default is the stratified draw.

**Why not the shipped candidate pool.** A pool built by RepoRadar is RepoRadar's own HEAD-seeded output, so a judge harsher on RepoRadar-shaped papers — Sonnet, by a factor of 2.3 — would be credited with "validity". Both adoption refutations agree on this. The legacy 35 are re-run under this control scheme; the pool-control result stays reported as NR-57.

**What "matched" does not mean, and it is the largest limitation in this file.** A control is a paper from the same field and the same half-year that the repository did not cite. **Nothing about that makes it a worse paper for the repository than the one it did adopt.** Quite possibly it is better: the maintainers may never have seen it, may have had nobody free to do the work, may have been locked into an existing dependency, or may have chosen on grounds that have nothing to do with technical merit. Adoption records what a project *did*, not what would have helped it most, and no procedure available here can tell those apart — asking a model which paper would have helped more is precisely the circularity this pool exists to escape.

So the primary endpoint is **adoption discrimination**, not matching, and this file uses that name. It answers "does this judge rank papers a project went on to take up above comparable papers it did not?" It does **not** answer "does this judge identify the most useful paper available?" The two coincide only insofar as projects adopt what helps them most, which is an assumption, is certainly false in individual cases, and is not tested anywhere here.

**The direction of the resulting bias is knowable, and it is the reason this is still worth running.** Genuinely useful papers sitting in the control set can only pull a judge's positives and controls closer together, never further apart — so the measured AUC is a **lower bound** on discrimination. That makes the two outcomes asymmetric, and §5's pre-committed consequences are written to respect it: an interval that **excludes 0.5 is conservative evidence** that the judge tracks something real, while an interval that **includes 0.5 cannot separate** "this judge does not discriminate" from "the controls were genuinely good papers this project never got to". A null here is therefore weaker than it looks, and the paper must say so rather than reporting it as a clean negative.

**Implementation note, 2026-09-03, before the first control was drawn. Two things this section registers were not implemented, and one of them put an arm marker in the prompt.**

* **The scheme was never selected.** `CONTROL_SCHEME` was assigned from `--controls` and read by nothing, so `arxiv-window` chose the output filename and the provenance label while the pool scheme drew the papers. `arxiv_window_controls`, `arxiv_window_listing` and `enrich_positives` had no production caller anywhere in the tree: the scheme this section registers had never run on real data. It now dispatches, and the arm-neutral draw lives in `judge_validity_pool.draw_controls`, which is where the v2 positives, the materialised HEAD citation sets and the verified `SEED_POOL` are.
* **The match was on the wrong category.** This section registers "the positive's **primary** category". The draw matched `categories[0]` — feed tag order, not a claim about which is primary — against a `cat:` query that arXiv satisfies from *any* category including cross-lists. So a `cs.LG` positive could draw controls whose primary is `stat.ML` or `cs.CV`, and the AUC would partly measure primary-versus-cross-list. `collector._result_to_paper` now keeps arXiv's own `primary_category` and the draw filters on equality.
* **The prompt carried a deterministic arm marker, which is the one thing this section forbids.** "No arm" was honoured for the T0 context and not for the paper. `collector._result_to_paper` stores `result.get_short_id()`, which is *versioned* — measured, **674 of 674** candidates in `.work/pool-cut100/ann.json` — while every mined positive id is unversioned by construction, since the extractor regexes capture `\d{4}\.\d{4,5}` with no version group (**0 of 120** legacy rows). `judge._build_user_prompt` prints the id verbatim, so in NR-56/57 **every control prompt read `arXiv: 2409.11629v1` and every positive `arXiv: 2409.11629`**. Separately, `diagnose_triage.fetch_papers` collapses whitespace and `_result_to_paper` does not, so 82 of those 674 abstracts carry embedded newlines a positive never can. Both arms are now assembled by one function that canonicalises the identifier and normalises title and abstract identically, and a check refuses to buy a verdict on a set whose arms remain distinguishable by shape. Whether either judge keyed on the marker is **unmeasurable after the fact**, which is why this is stated as a defect in NR-56/57's prompts rather than as a correction with a known effect. The 496 control verdicts already bought under the old shape are **not reusable for this pool**; they remain NR-56/57's record and are not rewritten.

Both judges, byte-identical rubric, a T0 context (README excerpt, manifests, file listing at T0 — no arm), `use_cache=False`, and T0 verdicts never entering the shared judge cache. That last one is mandatory rather than an optimisation: `judge_paper` keys its cache on `(model, repo, paper_id)` and **not** on the context, so a T0 verdict written into the gold cache overwrites the HEAD verdict for the same paper. That exact write once took `rag` from 5 targets to 0. Cache roots are hashed before and after; a change is a blocking failure.

---

## 5. Endpoints, and what they cannot decide

* **Primary, per judge:** **AUC of the judge's ordinal rubric score against adopted/control, with a repository-cluster bootstrap** and paper-level dedup. Level-free by construction — adding a constant to every score, or moving a judge's bar, leaves it unchanged, which is the entire point given NR-59.
  Reported with it: cluster count, largest cluster's share, the **realised** design effect (the ratio of clustered to paper-level bootstrap variance, measured rather than assumed from an ICC), and the minimum detectable AUC at 80 % power — so an interval spanning 0.5 can be read as *no discrimination* or as *too few repositories* instead of the two collapsing into one sentence.
  `evals/judge_validity_adoption.py::cluster_bootstrap_auc`, pinned by `tests/test_adoption_cluster_auc.py`.
* **Secondary:** `P(actionable | adopted) − P(actionable | control)` at each judge's own threshold, Wilson intervals, cluster bootstrap; and `P(actionable | control)` per judge as the level-sensitive descriptive.
* **Transportability:** heterogeneity of AUC between the legacy cluster and the seeded pool, and between star bands.
* **Contamination sensitivity:** positives split by adoption commit date relative to each judge's published training cutoff (recorded at registration: GPT-5.5 `____`, Sonnet 5 `____`); AUC on the post-cutoff subset.

  **Maintainer decision, 2026-09-03, before the walk and before any positive existed: the blanks are not filled, and this endpoint is VOID for this pool.** No published training cutoff for either judge is recorded anywhere in this repository, and none will be supplied. Filling them from a model's observed behaviour would be the circularity this pool exists to escape, and guessing them is something this project has already refused once on the record: `judge_date_stratify.py` (NR-43) states that its design "does not need to know the cutoff, and deliberately does not guess it", because testing for a discontinuity *anywhere* is strictly stronger than testing at a date that had to be assumed.

  The endpoint is therefore reported as **not computed, with this reason attached**, and never as a computed null. §6 item 6 already names the contamination split as the only instrument available against recognition bias, so the consequence is that **recognition remains an unmitigated confound in this pool**, stated as such beside the primary rather than left implicit. That is a real weakening of what the pool can claim, and recording it here — before any positive is visible — is what keeps it from being discovered afterwards and written up as a limitation that was always known.

  Nothing else changes in the design: the adoption commit date is still recorded on every capped positive by `walk_row`, so the split remains computable by anyone who later supplies two dated, sourced cutoffs.

  **Measured 2026-09-03, and it weakens even that fallback.** The legacy materialisation pass recovers an adoption date for **44 of the 94 usable legacy positives (47 %)**, and the shortfall is concentrated in the largest cluster: `diffusion` yields 9 of 46, `llminfer` 0 of 2, `mat-phonon` 0 of 1, against `peft` 22 of 27 and `graph` 8 of 13. The date comes from `git log -S<id>` over the whole T0→HEAD window, which diffs every commit's documentation, and each blob on a `--filter=blob:none` promisor clone is a lazy fetch: a single identifier on `huggingface/diffusers` exceeds **300 s**, so its 46 would run for hours. The pass therefore bounds the search at 20 s per identifier and records every miss as `null` with its reason rather than as a blank. So if two sourced cutoffs ever arrive, the contamination split on the legacy stratum would run on a **47 % subset that is not missing at random** — it is missing where repositories are largest and their histories longest. That is recorded here as a second, independent reason the endpoint is void, and it is not fixable by patience.

**Pre-committed consequences, and they are deliberately asymmetric.** §4 establishes that the measured AUC is a *lower bound*: a control may be a better paper than the positive it is matched against, which can only compress the two classes together. So:

* **Interval excludes 0.5** — conservative evidence that the judge tracks something real. Reported as demonstrated adoption discrimination, with the lower-bound argument stated so the number is not read as an upper limit on the judge's quality either.
* **Interval includes 0.5** — the project states that its primary label shows **no demonstrated discrimination** against the only model-free label available, and every headline figure downstream carries that beside it. It must **also** state that this outcome cannot separate "the judge does not discriminate" from "the controls were genuinely good papers this project never got to" from "too few repositories" (the minimum detectable AUC, above). A null here is three-ways ambiguous and is not to be reported as a clean negative.
* **Both judges exclude 0.5** — both order papers meaningfully and the base-rate disagreement remains unresolved.

**No outcome switches the primary label.** The gap at a judge's own threshold is Youden's J at that operating point: a judge at an 87 % positive rate sits in the top-right of its ROC and is bounded low by geometry, so "Sonnet's larger gap" restates the base rates and identifies neither as correct. Adoption is a **lower bound** on actionability and cannot calibrate a level. Any decision rule reading the gap difference against PREREG-rung1's 0.15 bar is **retired here, before the data**.

---

## 6. What this cannot establish

1. **Which judge's level is right.** Section 5, by geometry. Pre-committed.
2. **Validity on the life-science channel.** Measured, not conjectured: of the 37 legacy repositories, **0 of 6 `bio-*` and 0 of 6 `mat-*` clear `ids_v2(HEAD) ≥ 10`** (bio: 2, 0, 0, 0, 5, 3; materials: 1, 2, 0, 4, 2, 1), against 5 that do (`cv`, `diffusion`, `graph`, `peft`, `rl`). Life-science and materials documentation cites DOIs and journal references; this label reads arXiv and Hugging Face links only. **On the maintainer's decision of 2026-09-02 the blind spot is sized, not closed**: DOI and PMID counts are recorded at both ends as covariates, the pool licenses the judge on arXiv-citing research repositories, and validity on Europe PMC is an explicit **assumption**. Widening the extractor to DOIs is deliberately *not* done here — Zenodo software DOIs would be a large self-citation false-positive class, the filters and tests would have to be rebuilt, and the change would land after the premise was measured.
3. **Whether the adopted paper was the *best* paper.** §4 in full: a matched control may well have served the repository better than the paper it actually took up, and adoption cannot distinguish "not useful" from "never seen", "nobody free to do the work", or "chose otherwise for non-technical reasons". The endpoint is adoption discrimination; a judge could be right about value and wrong about adoption, and would score badly here for being right. This is the largest limitation in the design and it has no fix available — the alternative label would have to come from a model, which is the circularity being escaped.
4. **Ordering among topically close papers.** Category-window controls are a broad negative. An AUC excluding 0.5 against them is not evidence about the discrimination the product actually performs, which is among retrieved, on-topic candidates.
5. **Anything about repositories with no literature to track.** Excluded by construction. Bibliography-maintaining repositories also skew popular, so the small-star band will be thin.
6. **Freedom from recognition.** Both judges have almost certainly seen these repositories and the papers they adopted. The T0 screen removes the *outcome* from eligibility; it does not remove *fame*. Recognition inflates both judges' rates and compresses their difference toward zero, so a null on the difference is partly a property of the frame. The contamination split is the only instrument against it and will be underpowered.
7. **That the runner's knowledge did not reach the walk.** The frozen list, the unchoosable seed and the mechanical rules make it inert, not absent.

---

## 7. Contamination and leakage

The pool runs no arm, so there is no held-out set to protect and no configuration decision it can reach. What remains:

* **Judge cache.** Section 4: `use_cache=False`, a separate namespace for the second judge, and before/after hashes of both cache roots as a blocking gate.

  **Implementation note, 2026-09-03, before any pool verdict was bought.** As written, this bullet's two clauses contradict each other, and the committed code could not satisfy both. `second_verdict(cache_as=…)` writes the second judge's separate namespace *inside* `.work/second_judge`, which is one of the two hashed roots — so hashing that root whole made the gate fire on precisely the runs that bought something, and because it raised first, the gold-set guard after it never executed on any such run. The gate had a true-negative rate of zero and had never passed.

  The unit is therefore a **partition, not a root**: `evals/cache/judge` is hashed in full with no exclusion ever, and `.work/second_judge` is hashed except for the declared T0 namespaces, which are the "separate namespace" this section registers as legitimate. Only Sonnet's is declared — GPT runs `use_cache=False` and writes nothing, so `gpt-5.5#t0` is protected rather than permitted. An excluded namespace must not exist or must carry an ownership marker this study wrote, because an exclusion justified by spelling rather than by ownership would make overwriting somebody else's 1,504 verdicts the one thing the gate cannot see. Both guards are now evaluated and both outcomes reported, since neither implies the other: `resolve_targets` sees only ids the baseline picked whose gold verdict scores ≥ 2, so a gold-cache write for an unpicked paper moves the hash while leaving the gold set identical, and a score crossing 2 for a picked paper does the reverse.

  **This narrows what the hash covers**, and is recorded here rather than in a commit message for that reason. It does not narrow what is *protected*: nothing that was watched and is not a declared, owned T0 namespace has been excluded. `tests/test_pool_controls.py::TestTheCacheIsolationGate` pins each clause.
* **The legacy re-mine** pins each legacy case's HEAD to the last commit on or before the `head_date` already recorded in `adoptions.json`, so `t0` reproduces to the SHA and the stored T0 verdicts remain verdicts about the same prompt.
* **Training cutoffs** are recorded at registration, not after the positives are visible.

---

## 8. Predictions, registered before the data

| id | prediction | falsifiable at this n? |
|---|---|---|
| **P1** | Extractor v2 raises the legacy usable count by ≥ 5, driven by `diffusion`. *Check, not prediction* — it follows from the premise measured in section 1 | no (already implied) |
| **P2** | **≤ 3 of v1's 35 legacy positives are false adoptions** revealed by the widened T0 bibliography (an identifier cited at T0 only in HF form and at HEAD in arXiv form). Above 3, NR-56/57's gaps are re-reported under v2 as primary and the v1 numbers are labelled as containing migration artefacts | yes |
| **P3** | Qualifying rate on the enumerated population `q ∈ [0.08, 0.30]`, point **0.15** — measured over the unconditional 300-row prefix | yes |
| **P4** | Capped-usable positives per qualifying repository `y ∈ [0.8, 2.5]`, point **1.5** | yes |
| **P5** | 100 new positives are reached within B = 1,200 rows | yes |
| **P6** | AUC(GPT-5.5) 0.60–0.70 and AUC(Sonnet 5) 0.62–0.72 at ≥ 130 capped positives; **both intervals exclude 0.5; their difference does not exclude 0** | yes |
| **P7** | `P(actionable \| control)`: GPT ≥ 0.70, Sonnet ≤ 0.55 — the NR-59 level disagreement reproduces on a population neither judge's threshold was tuned on | yes |
| **P8** | The realised design effect is ≥ 1.5, i.e. the paper-level interval the previous estimator would have reported is materially too narrow | yes |
| **P9** | The reverse-citation and doc-genesis filters, which fired **zero** times across the legacy 37, together remove ≥ 5 % of gross adoptions on the enumerated population | yes |

**Scored so far** (predictions above are left exactly as registered; outcomes are recorded here and in `RESULTS.md`):

* **P1 — met, 2026-09-02, runbook step 2 [NR-60].** Legacy usable count 35 → 94 (+59); `diffusion` +39 and `peft` +22 carry all of it, and 61 of the 94 are reachable only through a Hugging Face link. A check rather than a prediction, as registered.
* **P2 — met, 2026-09-02 [NR-60]. 0 of v1's 35 positives are migration artefacts**, scored at each case's own recorded T0 commit. NR-56/57's gaps stand as measured and the >3 consequence does not fire.
* **Recorded alongside, which P2 was not testing:** v2 rejects 2 of v1's 35, both in `rl`, both papers that appear only on that project's "Projects using…" page — papers citing the repository rather than adopted by it. NR-56/57's 35 were 33 adoptions and 2 reverse citations.
* **P9's premise corrected before the fact.** "Fired zero times across the legacy 37" was true only because v1 had no such fields. Under v2 the two filters fire 10 times in 120 gross adoptions (8.3 %) on the legacy set. P9 concerns the *enumerated* population and stays open, but it is no longer predicting an unobserved event.
* **P3–P8 remain open.** No candidate has been enumerated and no verdict bought; no AUC has been computed on real data, deliberately (P6 registers an AUC over a set that includes these repositories).

---

## 9. Cost and yield

**The unknown that decides everything is the qualifying rate on an enumerated population, and it has never been measured.** The legacy 37 are a hand-curated, arXiv-oriented sample, so their rate is an upper bound, not a forecast. Hence P3's bracket and the unconditional 300-row prefix.

| scenario | q | y | positives/row | rows for 60 | rows for 100 |
|---|---|---|---|---|---|
| optimistic | 0.30 | 2.5 | 0.75 | 80 | 133 |
| **central** | **0.15** | **1.5** | **0.225** | **267** | **444** |
| conservative | 0.10 | 1.0 | 0.100 | 600 | 1,000 |
| pessimistic | 0.08 | 0.8 | 0.064 | 940 | extension |

**Power, for the registered endpoint rather than inherited.** The ≥ 60 / ~75 targets in NR-57 and NR-59 are standard errors of the *unclustered* discrimination gap, which section 5 demotes to secondary. The primary is a repository-clustered AUC, whose precision is governed by clusters, not papers.

| capped positives (new + legacy) | SE(AUC = 0.60) | 95 % interval at 0.60 | minimum detectable AUC |
|---|---|---|---|
| 90 (60 new + 30 legacy) | 0.047–0.057 | [0.489, 0.711] — **may include 0.5** | ≈ 0.62 |
| 130 (100 new + 30 legacy) | 0.036–0.044 | [0.514, 0.686] — excludes | ≈ 0.578 |

**Money is not the constraint.** ~1,300 verdicts at the measured $0.0195 (NR-57: 256 ≈ $5) is about **$26**; at $0.03, **$40**.

**Wall-clock is.** Per row: a blobless clone plus two `git grep` passes that lazily fetch every doc blob at that revision, plus the self-citation read. At a 60–120 s median that is 8–25 h serial for 300–1,200 rows, or 2–7 h at four-way parallelism, with the 300 s timeout bounding the tail. Disk ~2 GB with delete-after-mine. `self_cited()` previously ran one `git show` per documentation file — thousands of lazy fetches on a large repository — and the walk would not have finished without that fix.

---

## 10. Runbook

1. ~~**Registration first.**~~ **Done 2026-09-02**: this file was committed with DRAFT removed, together with the topic list, the enumerator, the walk, the control scheme, the endpoint and the cache gate. Changing the instrument mid-pool is an instrument change. The one piece still to write is the analysis wrapper (banner), and it must exist before any verdict is bought.
2. **`--extractor v2` over the legacy 37** (`$0`): scores P1 and P2, and measures the filter survival rate the yield arithmetic assumes — the reverse-citation and doc-genesis filters have fired **zero** times so far, and both will fire on an enumerated population (P9). A prerequisite, not hygiene.
   *This step is deliberately numbered after registration.* An earlier draft put it first, which would have scored P1 and P2 out of the very file that registers them — the defect this project has corrected repeatedly, and the one the §6.1 premise measurement already forced P1 to be demoted for.
3. Day Dp: enumerate; commit `pool-universe-Dp.csv`, the raw response archive, and **the absolute UTC timestamp of the pulse that will seed it**.
4. After that pulse: write `SEED_POOL`; commit.
5. Walk B₀ = 300 unconditionally; commit the yield curve and the measured `q` and `y`. Continue to B = 1,200 or 100 positives.
6. Draw controls; judge both judges in seeded order to 100 new positives.
7. Analysis: section 5's endpoints, P1–P9 scored, and the datasheet — candidate list, seed and pulse timestamp, walk ledger, positives and controls, both judges' raw ordinal scores, DOI/PMID covariates, void and timeout lists — published with it.

---

## 11. Relationship to the benchmark expansion

`evals/PREREG-benchmark-expansion.md` (rung 10) proposes expanding the benchmark from 37 to 60 repositories. **That expansion is on hold by the maintainer's directive of 2026-09-02 and runs only on an explicit instruction.** This pool does not wait for it and does not depend on it: it draws its own candidate list, fixes its own seed, runs no arm, and touches no benchmark artefact.

Two consequences, stated because they are the price of running first.

* **The benchmark's selection inputs are not frozen when this pool runs.** If the expansion later proceeds, that ordering is a fact about it and must be disclosed there: parameters fixed after this pool's results were visible cannot claim to have been chosen blind. The mitigation available to the expansion is to freeze its inputs in a pushed commit before reading anything here — which is its decision to make, not this file's.
* **This pool must not contaminate a later prior-exposure rule.** The expansion's X2 excludes repositories "previously exposed", defined as every `github.com/<owner>/<repo>` string in the tree. Section 2.1's no-URL rule is what keeps this pool's thousands of candidate names out of that grep. Without it, running the pool first would silently strip citation-rich research repositories out of the expansion's own population — the coupling this separation exists to remove, arriving through a shell command.
