# Retrieval design — where RepoRadar's candidate pool comes from, and what to do about it

**Status: three designs proposed 2026-08-02; all three now measured (2026-08-06/09).** This
records the candidate architectures and the evidence for and against each, so the design work
is not lost and is not redone. Read [`evals/RESULTS.md` → Candidate-pool
diagnosis](evals/RESULTS.md) first — it holds the measurements this is reasoning about.

> ### Verdicts, so nobody reads the sketches as open questions
>
> | design | verdict |
> |---|---|
> | **1 — coupling funnel** | **Filter refuted; channel confirmed.** The 70%-cut filter was a property of the 7-case set and fell to 10% on 22 cases. The *hop itself* is real (21/48 = 44%, and 89% where a repo has ≥7 arXiv-cited seeds) and remains unshipped. |
> | **2 — HyDE against a dense index** | **Verified 4/4, replicated blind, best measured channel.** 27/48 in top-1k, median rank 837; **15 targets no other channel reaches**; union with the hop **36/48 = 75%**. Unshipped as of this writing. |
> | **3 — persistent neighbourhood** | **Not built, and its own negative result held up.** Similarity — in *any* space, including citation-trained SPECTER2 — is the wrong relation for "would improve this repo". |
>
> **The question all three deferred to has been answered.** Every design ends in triage, and
> this document said triage caps them all. It does not: the gate's ceiling was scale
> quantization, not capability, and `reporadar/finescale.py` now ships the fix
> (band ordering AUC 0.84; live 22-repo net@2 +3.18 vs an Opus baseline's +1.82).

Provenance is marked throughout. **VERIFIED** = re-run by hand against the live API in the
main session. **REPORTED** = measured by an investigating agent and not independently
re-checked. The distinction matters: one agent estimate in this project was off by 10x, and
another turned out to be measuring a transport bug.

---

## The problem, in one paragraph

RepoRadar builds arXiv queries from TF-IDF keywords of a repo. **VERIFIED:** across 9
benchmark repos it fetched 2,030 papers and reached **0** of the 24 an Opus baseline
recommended and a GPT-5.5 judge confirmed genuinely actionable. Not a category-filter problem
(23 of 24 are inside the searched categories) and not an arXiv limitation (a precise phrase
returns them at rank 1). The root cause is a **register mismatch**: a codebase's vocabulary
describes what it *has*; the useful paper describes what it should *adopt*. Two LLM
query-generation prompts, tested blind, recovered 2/24 and 0/24 — the second aimed at exactly
the right techniques and phrased them as compounds no paper title contains.

**The one channel that reaches these papers is the citation graph**, because it maps
identifier to identifier and never constructs a query, so the asymmetry has nothing to
corrupt.

---

## Design 1 — Bibliographic Coupling Funnel

Keep the citation hop as the recall channel; spend the entire precision budget on a free
structural signal: how many of the repo's own seeds co-cite each candidate.

**REPORTED:** a direction-aware filter (`forward_degree >= 2` **or** a middle band of
`backward_degree`) cut `cv` from 14,867 candidates to 2,115 while retaining 3/3, moving
density from 1:4,956 to 1:705.

**The insight worth keeping**, and it is measured: splitting the two hop directions separates
two very different sets. The **backward** set (papers the seeds cite) is the repo's own
intellectual ancestry — ResNet, MS-COCO, Mask R-CNN — and is only ~2.5% of the pool. The
**forward** set is later work. Ranking by coupling degree alone surfaces the ancestors, i.e.
exactly what the repo already knows.

**Weakest link (REPORTED, and the author says so):** the funnel reaches a top-150 band and
stops, and the head of that band is actively wrong — Dense Passage Retrieval, BEIR, BERT for
`rag`; ResNet, COCO, Mask R-CNN for `cv`. Coupling is a filter, never a sort key.

**Fatal for two repos:** `crypto` and `systems` have no arXiv-indexed bibliography, so there
is nothing to seed from. Structurally 0, not merely bad.

---

## Design 2 — "Wanted Poster": HyDE hypotheses against a local dense index

Have the LLM write the *abstract of the paper it wishes existed* for this repo, embed that,
and search it against a precomputed binary index of all arXiv.

**Why this attacks the actual constraint.** The register mismatch is only fatal against a
*lexical* index. "experience replay prioritization methods" and "prioritized experience
replay" are far apart as strings and close as vectors. A hypothesis abstract is written in
the literature's register by construction, because that is what the model was asked to
produce.

**REPORTED, blind protocol** (hypotheses generated from the repo profile alone; the
generating process never saw the targets):

| query | top-100 | top-1k | median rank |
|---|---|---|---|
| TF-IDF keywords (today) | 1/24 | 3/24 | 10,041 |
| repo README (today's `w_embedding`) | 5/24 | 9/24 | 19,461 |
| **HyDE-4** | **8/24** | **10/24** | **1,584** |

Pool sweep, HyDE-16: 3,164 candidates → 10/24; 26,572 → 13/24; 480,391 → 24/24. Against the
citation hop's 28,598 → 14/24, that is **71% of the recall from 11% of the candidates**.

**Two properties that matter more than the recall number:**

1. **It covers the repos the citation hop cannot.** At the 3,164 operating point it recovers
   `crypto` 2/2 and `systems` 1/1 — precisely where Design 1 is structurally zero. The two
   channels are **additive**: REPORTED union ≥ 17/24 from ~31,800 candidates.
2. **Discovery becomes fully offline and sends nothing.** This is a *net privacy improvement*
   over the status quo, where repo-derived keywords are transmitted to arXiv on every run.
   It is Feature 13's deferred `--local-only` arriving by a different route.

**Cost (REPORTED):** ~370 MB one-time index sync + ~670 MB model weights; ~100 s per repo per
run on CPU, no network; ~$0.01 per run for hypotheses ($0 on Ollama).

**Weakest link (REPORTED):** the bottleneck moves from the index to the hypothesis. Recall is
capped by whether the sampled gap-guesses happen to include the gap the judge rewarded — at
the 3,164 operating point, `cv` 1/3, `rl` 1/3.

**Unverified and load-bearing.** Before building any of this: does the HuggingFace dataset
exist under the stated licence, is the columnar range-fetch real, and is 1.87 s/query over
3.1M vectors reproducible? Every one of those is a single check, and this project has lost
work to exactly this class of assumption.

> **VERIFIED and REPLICATED, 2026-08-06** — ROADMAP P4, `evals/verify_hyde_deps.py` and
> `evals/hyde_replication.py`, write-up in `evals/RESULTS.md`. All four dependencies hold
> (apache-2.0, 3,106,925 rows, 15.9%-of-shard column fetch, **1.21 s/query**, 48/48 targets
> present). Two corrections: the "~370 MB sync" is **432 MB** and only under column pruning
> (the full dataset is 2,542 MB), and a **fifth** dependency this document does not name is
> more load-bearing than any of the four — that a locally computed vector is comparable to
> the stored ones. It is: mxbai-embed-large-v1 over the abstract alone, binarised at >0,
> Hamming **0/1024**.
>
> Blind replication on 48 targets: HyDE-4 **27/48 in top-1k, median rank 837**, against
> 10/48 for a README query and 3/48 for keywords in the same index. The **additivity claim
> is the one that holds**: hop 21/48, HyDE 27/48, **union 36/48 (75%)** with **15 targets
> reachable only by HyDE** — against a REPORTED ≥71% on half as many targets. The claim that
> does **not** hold is the per-repo one quoted above: `crypto` is **1/2** and `systems`
> **0/1**.

---

## Design 3 — Persistent per-repo neighbourhood, no query at all

Build a citation neighbourhood once per repo, persist it, rank by RRF of SPECTER2
seed-proximity and coupling, and let it accumulate across runs.

**Its most valuable contribution is a negative result.** The author fetched SPECTER2 vectors —
citation-trained, so a *vocabulary* mismatch provably cannot apply — and reports that
seed-proximity still does not rank the targets highly. If that holds, the gap is **not merely
lexical**: "similar to what this repo does" is not the same relation as "would improve this
repo", and no embedding of the repo, in any space, expresses the second one. That would
constrain Design 2 as well, whose ranking is ultimately a similarity.

**Weakest link (REPORTED, and decisive):** it ends by handing ~100 papers to `triage.py`, and
triage was measured at chance on this exact benchmark — 50% precision against a 50% base rate,
40% recall, on the `speech` case. Every design in this document terminates in triage, and
none of them fixes it.

> **WITHDRAWN, 2026-08-02 and 2026-08-08.** "Triage is at chance" was an n=10 reading from one
> case. On all 428 labelled papers the gate runs **precision 0.81 / recall 0.78** against a 32%
> base rate, and on wild pools **0.97 / 0.60**. It is well above chance and its failure is
> recall, not precision. The "none of them fixes it" clause is also now false — see the verdict
> box at the top of this file. What survives is the *ordering* of concerns: a retrieval
> improvement terminating in an unordered gate delivers a better pool and the same digest,
> which is exactly what gating the whole pool measured (a wash, −0.18 paired).

---

## What all three agree on

1. **The citation graph is the only channel with demonstrated recall.** Every design keeps it.
2. **Coupling degree, SPECTER2 similarity and citation count all surface the repo's own
   ancestry** at the head of the ranking. Similarity is the wrong relation, however it is
   computed.
3. ~~**Triage is the terminal stage of every design and is currently at chance.**~~ Withdrawn
   (see above). What holds is the ordering: improving retrieval raises the ceiling and does
   not deliver a better digest on its own — measured, as a wash, when the whole pool was
   gated.
4. **All of it must live under `--foundational`.** Every target is ≥11 months old and
   `lookback_days` defaulted to 14, so the default path could not see any of them.
   **Resolved 2026-08-07**: the shipped default is now all-time relevance
   (`lookback_days: 36500`, `sort_by: relevance`, `w_recency: 0.0`) — the configuration every
   benchmark number had actually been measured under for a month.

## What to settle before building — all three settled

- ~~**Re-measure the citation-hop baseline.**~~ **Done.** The 14/24 was a batching artifact;
  corrected to 18/24, then re-measured on the expanded benchmark as **21/48 = 44%**, with seed
  count as the whole story (≥7 seeds → 89%; no arXiv bibliography → 0% by construction).
- ~~**Verify Design 2's index dependency.**~~ **Done, 4/4**, plus a fifth dependency this
  document failed to name and which mattered more than the four: that a locally computed
  vector is comparable to the stored ones. It is — Hamming 0/1024 — and without that check
  every query would have measured nothing while looking healthy.
- ~~**Decide whether triage is fixable.**~~ **Yes.** See the verdict box at the top.

**What to settle before building is now empty. What remains is building it** — Designs 1 and 2
are both verified channels living only in `evals/`, and every remaining loss to the Opus
baseline is a recall failure they address.
