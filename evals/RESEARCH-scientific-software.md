# RepoRadar on scientific software: bioinformatics and materials science (demo plan + preliminary investigation)

**Date:** 2026-08-18. **Question:** can RepoRadar be shown *working well* on non-ML scientific
software — bioinformatics repositories against bioRxiv literature, and materials-science
repositories against materials literature — for an audience of scientific-software people?
**Method:** a 9-agent pass — a code audit of every ML/CS-domain assumption in `src/reporadar`
plus an adversarial verification of it, three $0 empirical probes (bioRxiv/Europe PMC/OpenAlex/S2
API behaviour, arXiv-category volume, HyDE-index presence and reach with hand-written
hypotheses), two scouting agents that cloned and profiled 19 candidate repositories with the
shipping profiler, one agent that ran the **measured configuration** (`rr init --measured`)
end to end on 6 of them, one that judged the shown papers with the project's own GPT-5.5 judge
under the shipped rubric, and a completeness critic. **Cost:** ~$0.16 for the six product runs
(Haiku gate + HyDE + gpt-4o-mini rescore + digest suggestions), ~$2.80 for the judge (69
GPT-5.5 calls, cached under `evals/cache/judge/v1/gpt-5.5/scisoft-*/`), $0 for everything else. No Opus baseline was run. Nothing in the repository was edited by the
agents; every artefact is under the session scratchpad (paths in the appendix).

Read the numbers here as **single-draw, six-repository, no-baseline** evidence. The benchmark's
own noise floor is 1.03 net@2/case on a 22–25 case mean (`evals/noise_floor.py`); nothing below
is a 25-case mean, and none of it has been replicated.

---

## 0. TL;DR

**Materials science: retrieval works as shipped; the gate over-admits, and the fix is small.**
The literature is on arXiv (`cond-mat.mtrl-sci` alone is 111,471 papers; 682 hits for
*"machine learning interatomic potential"* in that category), the HyDE index is a full arXiv
snapshot (35/35 known bio+matsci papers present, cutoff mid-July 2026), and hand-written HyDE
hypotheses for a universal-potential library returned M3GNet at rank 1 with 26/30 top-15
neighbours in `cond-mat.*`. The three live runs (chgnet, mace, dscribe) produced digests whose
Top Picks are recognisably the right reading list — MACE-POLAR-1, *Speeding Up MACE*,
*Overcoming systematic softening in universal MLIPs by fine-tuning*, LATTE descriptors,
*Compressing local atomic neighbourhood descriptors* — with HyDE supplying 8 of dscribe's 12 Top
Picks. **Judged (GPT-5.5, shipped rubric, one draw): dscribe +9 net@2 at precision 0.92, but
chgnet +1 (0.69) and mace −1 (0.64)** — mean +3.00, pooled precision 0.744 against the ML
benchmark's +5.72 / 0.89. Every one of the losses is the same failure: 11 of the 12
non-actionable Top Picks across the six runs are **gate-score-3 papers** — five are the
repository's *own* paper (5 of 6 repos, judged 1 every time) and six *use* CHGNet/MACE and
name it in the abstract (rare-earth oxide diffusion, Mn-rich cathodes, defective COFs,
twisted 2D materials, "Evaluation of the MACE architecture", the sibling library MatGL) — the
Haiku gate scores the name-match 3, the judge scores it 1. Papers
that came through the score-2 band **and the fine-scale rescore** were actionable 28 of 29
times (0.966). That is the reverse of the benchmark's finding that the score-3 band can be
trusted, and it points at two cheap fixes (§6 D4, G1) that this draw says are worth ~+2 net@2
per materials repo. What can go wrong on stage otherwise is cosmetic and known: three
off-topic *withdrawn* papers head the MACE digest because generic queries (`all:model`,
`all:use`) fetched them, and MACE profiles with **zero anchors** because its dependencies live
in `setup.cfg`.

**Bioinformatics against arXiv: good on this draw; against bioRxiv as the user asked: not
today.** The arXiv-native corners of bioinformatics scored *better* than materials —
minimap2's digest is aligners, seed filters and GPU X-drop (**+6, 0.89**); OpenMM's is OpenMM 8,
Grappa, alchemical transfer, MiMiC (**+9, 9/9 actionable**); scvi-tools' is scVI itself plus
diffusion/VAE single-cell models, 6 of 8 Top Picks from HyDE (**+5, 0.88**) — mean +6.67,
pooled precision 0.923, and not one Top Pick in any of the six runs was judged 0. But *bioRxiv
papers* are unreachable by every channel that makes the measured configuration good: **HyDE is
arXiv-only by construction**, and the shipped bioRxiv adapter is broken in a way the C-9 audit
did not reach —
bioRxiv's details API is a date listing served **oldest-first, 30 per page**, the adapter caps
at 40 pages, and under the product default `lookback_days: 36500` it fetches **1,200 preprints
from 2013–2016 and nothing newer**; at a 90-day window it fetches the oldest six days of the
window; and its topical filter (any query token >2 chars, substring match) passes a third of
*all* bioRxiv postings. The two keyword-searchable routes to bioRxiv that do exist in the
codebase are also closed: OpenAlex is called with `filter=type:article`, which excludes every
preprint (0 bioRxiv hits for *single-cell clustering*; 62,946 with `type:preprint`), and
Semantic Scholar is unmeasured. **The fix is small and measured to be feasible**: Europe PMC's
REST search is keyless, indexes 345,390 bioRxiv + 87,352 medRxiv preprints with abstracts and
DOIs, supports `PUBLISHER:"bioRxiv"` and date filters server-side, and covers ~98% of new
bioRxiv preprints at a 2–3 day lag. A one-day adapter plus three small ranking fixes (§6, B1–B4)
would give the bio demo a real bioRxiv channel — but that channel would then be keyword +
embedding + gate only, with no HyDE and a fine-scale calibration that has never seen a bio
paper, so "works well on bioRxiv" is a claim to *measure* after the fix, not to make at the demo.

**Do before the demo, in this order (each is small):** hide off-topic withdrawn papers (§6 D1);
exclude or label the repository's own paper (D4); suggest arXiv categories from the profile and
add bio/matsci packages to the domain map (D2); read `doc/`, `setup.cfg`, skip release notes,
strip MyST roles and reference-style badges (D3); build the Europe PMC bioRxiv source and fix
category scoring for non-arXiv papers (B1–B3); pre-register and run **cohort 3** (4 bio + 4
matsci) on the benchmark, with "rescore the score-3 band too" (G1) as its one arm (§8).
Success chances, stated as probabilities in §7 (subjective, reasoning given): materials
~0.8 today for a descriptor/analysis library like dscribe but ~0.45 for a universal-potential
library whose name every application paper cites (chgnet/mace), ~0.8–0.85 after D4+G1;
bio-on-arXiv ~0.65 for the three tested repos re-run from their stores, ~0.45 for a bio repo
the audience picks cold; bio-on-bioRxiv ~0.1 today / ~0.45 after B1–B4.

---

## 1. What the demo has to show, and the script

The audience builds and maintains scientific software. What they will want to see is not a
benchmark table but *their kind of repository* producing a digest they would act on:

1. `rr init --measured` in a repo they recognise; **set `arxiv.categories`** (the one field the
   measured config says to change — for materials `[cond-mat.mtrl-sci, physics.comp-ph,
   physics.chem-ph, cs.LG]`; for genomics tools `[q-bio.GN, q-bio.QM, cs.DS]`; for MD/structure
   `[q-bio.BM, physics.chem-ph, physics.comp-ph]`).
2. `rr update` — ~4 minutes wall on this machine (§5; ~1.5–2 min of it is HyDE on CPU: 41 s
   model load, 30 s encoder verification that touches huggingface.co, ~8 s per hypothesis).
3. `rr digest --format html` — the page, Top Picks first, each with the gate's one-sentence
   reason and the action ideas.
4. `rr audit` — what left the machine (the queries, 300 README characters, paper abstracts to
   the two LLM vendors); the privacy story matters more to people with unpublished code than to
   ML users.
5. Optionally `rr search "<phrase>" --hybrid` over the accumulated store, and the MCP tools.

"Working well" for this audience means: (a) the Top Picks are papers a domain expert nods at,
(b) it says *nothing* rather than padding when there is nothing, (c) it runs in a few minutes
without surprises (keys, downloads, throttles), and (d) the profile it prints (`rr profile`)
does not embarrass the repository — `smaller`, `pr`, `func` as scanpy's top keywords would.

**Questions this audience will ask, and what the evidence lets us answer** (the critic's
list, checked against §2–§5):

| question | answer today |
|---|---|
| "Why is my paper not there?" | Mechanism is explainable (`rr queries` prints the exact queries; categories → keyword pool + HyDE top-100 → RRF → gate on the top 50 → rescore → 15-paper window), and §5.2 shows the honest answer is usually *not in the pool* or *RRF 54–80, just below the gate*; there is no per-paper command yet (D7). If the paper is bioRxiv- or journal-only: unreachable, say so |
| "My literature is on bioRxiv / journals" | A limitation today (§3.2); Europe PMC is measured feasible and unbuilt (B1); journals not investigated |
| "My repo is C++ / Fortran / R" | minimap2 (C) scored +6 and OpenMM (C++) +9 from README + name query alone; but the profiler reads Python/JS manifests at the root and `docs/` only, so a manifest-less repo with a thin README profiles as boilerplate; R was never tested |
| "Does it understand formulas / gene names / formats?" | Yes for `Li7La3Zr2O12`, `LiFePO4`, `CO2`, `VASP`, `POSCAR`, `CIF`, `BAM`, `VCF`, `h5ad`, `RNA-seq`, `TP53`, `CRISPR-Cas9`, `k-mer` (verified through TF-IDF); no for digit-leading and Unicode tokens (`2D`, `16S`, Greek) and `__kallisto__`-style markdown; embedding-model handling of formulas unverified |
| "Why is this paper here? It just *uses* MACE" | Explainable (gate scores the name match 3, no rescore on score-3), not defensible on this draw; fix identified (D4, G1), not done |
| "Why is this an ML paper?" | `arxiv.categories` left at `cs.LG, cs.CL`; nothing suggests otherwise yet (F13/D2). With domain categories set, 65/65 judged Top Picks were on-topic |
| "What does it cost, how long?" | ~$0.03 per run measured here *including* digest suggestions (the README's $0.01–0.02 excludes them); ~4 min wall on a CPU laptop; two vendor keys; the keyless default was not run on these domains and scores worse than nothing on the benchmark |
| "Privacy of unpublished code?" | Strong: `rr audit` lists every destination and string; 300 README chars (settable to 0), keywords, queries and paper abstracts leave; `scan_source` off. Caveat to state: HyDE contacts huggingface.co on every run today (§3.1) |
| "Is this measured or cherry-picked?" | Six repos, one draw, one judge, no baseline, chosen by the scouts as clean profiles; scanpy and pymatgen held back by the profiler; the 25-repo benchmark contains no bio/matsci repo. Cohort 3 (§8) is the answer |
| "What is new this month?" | Not answerable from evidence: every run here is all-time/relevance; the recency path is unbenchmarked |

---

## 2. What is domain-neutral already, and what is not

The audit read every module in the pipeline and verified the load-bearing claims by running the
code on the cloned repositories. Twenty-one findings; the adversarial pass **confirmed 18,
marked 3 partial (F11, F17, F21 — impact overstated), refuted none**, and added ten the audit had
missed. Full text with file:line evidence is in the scratchpad (`wf/audit_rendered.md`); the
table below is what matters for the demo.

**Domain-neutral and safe** — `KNOWN_ARXIV_PREFIXES` includes `q-bio`, `cond-mat`, `physics`
(config.py:678–701) and validation only warns on unknown prefixes; `build_queries` and the
category clause assume nothing about `cs.*`; the gate rubric (triage.py:29–42), the fine-scale
rubric, the LLM-suggestions prompt, the judge rubric and `repo_context_block` are all worded for
"a software repository"; the HyDE encoder path stores old-style ids with subject class
(`cond-mat/0501001`); the store accepts any id; the `':'` guards keep synthetic ids out of the
arXiv-only integrity/citations/HF calls; the BM25 tokenizer keeps `2d`; formulas and format
tokens survive TF-IDF (`li7la3zr2o12`, `lifepo4`, `co2`, `vasp`, `poscar`, `cif`, `bam`, `vcf`,
`h5ad`, `rna-seq`, `k-mer`, `tp53`, `crispr-cas9` all verified).

**Not neutral** (severity after verification; B = bio, M = matsci):

| id | where | assumption | hurts | fix / effort |
|---|---|---|---|---|
| F1 **blocker** | `sources/biorxiv.py:25,73–102`; `pipeline.py:352,374` | bioRxiv treated as a searchable source sharing `arxiv.lookback_days`. API is a date listing, **oldest-first, 30/page**; cap 40 pages = 1,200 items; all-time default ⇒ 2013–2016 only; no `category`/server config | B | own `biorxiv:` config (server, category, window ≤14 d, `?category=bioinformatics` verified: 299 items/14 d fits under the cap) — or replace with Europe PMC (§3). 1–2 h |
| F2 major | `sources/openalex.py:183` | `filter=type:article` — excludes every preprint | B (M minor) | `type:article\|preprint`, optional `primary_location.source.id` for bioRxiv S4306402567 / ChemRxiv S4393918830. 1 line |
| F3 major | `profiler.py:39–78` | `PACKAGE_DOMAIN_MAP` has no bio/matsci packages; `sklearn`→"machine learning", `dask`→"distributed computing". Domains flow into the gate/HyDE prompt (triage.py:56–59) and the BM25 query (retrieval.py:38–40) | B, M | add bio + matsci entries; measured: scanpy → *machine learning, distributed computing*; pymatgen → *scientific computing*; mace/minimap2/htslib/kallisto/lammps → *general*. 30–60 min |
| F4 major | `ranker.py:60–79,187–194` | non-arXiv `categories` (`'Bioinformatics '`, OpenAlex topic names) never match ⇒ `category_score=0` at full weight; an uncategorised S2 paper has the term *omitted*. At kw=0.6: arXiv 0.733, S2 0.600, bioRxiv/OpenAlex **0.400** | B (M if OA/S2 on) | store source category separately and leave `categories` empty, or map bioRxiv/OA categories → arXiv equivalents; strip the trailing space. 1 h |
| F5 major | `profiler.py:259–273,100,118` | only `requirements.txt`, `pyproject.toml`, `setup.py`, `package.json`, at repo **root**. Not `setup.cfg` (MACE ⇒ 0 anchors), `environment.yml`, conda `meta.yaml`, R `DESCRIPTION`, `Cargo.toml`, `Project.toml`, CMake; sub-directory manifests (openmm `wrappers/python`, mdanalysis `package/`) invisible; `~=` not stripped (matminer `scikit-learn~`) | B, M | parsers for setup.cfg + environment.yml first (1 h), the rest 2–3 h |
| F6 major | `profiler.py:471–478,440–448,555` | reads `docs/` only (not `doc/` — phonopy, LAMMPS, deepmd-kit, tblite, sourmash), no exclusions (scanpy: 69 release-note files ⇒ `smaller`/`pr`/`func`), MyST `{role}` syntax not stripped, anchor pseudo-document forms phantom bigrams (`seekpath pypolymlp`) | B, M | add `doc/`; exclude release-notes/changelog/api; strip `{role}`; separate anchors. 1–2 h |
| F7 minor | `profiler.py:305–437,565` | stopwords kill `index`, `reference`, `parameters`, `system`, `thin`, `build`, `bin` ⇒ *thin film*, *crystal system*, *lattice parameters*, *reference genome*, *k-mer index* can never be keywords; sklearn list also lacks `using`, `tools`, `code`, `data` (kallisto queries `all:using`, `all:tools`) | B, M | trim/extend the lists. 30 min |
| F8 minor | `profiler.py:570`; `ranker.py:15` | tokens must start with a letter: `2D`, `3D`, `16S`, `10x`, single-letter elements, Greek dropped; `__kallisto__` yields no token (kallisto's own name missing from its profile) | B, M | unicode-aware pattern, kept identical in profiler and ranker. 20 min |
| F9 major (structural) | `hyde.py:71,407,452` | index is arXiv-only; only arXiv ids leave `discover` | **B** | HyDE-lite: send hypotheses as `search=` to OpenAlex/S2, or a small local bioRxiv-bioinformatics index (~587 records/30 d). 2–4 h / 1–2 d |
| F10 minor | `hyde.py:87–88` | "read exactly like an arXiv abstract … 'on benchmark X we obtain'" | B (mild) | reword to the field's own preprint register. 10 min, then re-run one replication |
| F11 partial | `suggestions.py:13–58` | template suggestions are ML-flavoured (*swap your optimizer/loss*, *SOTA*); only in the template config, but the LLM path falls back to them after one failure | B, M | neutral templates; gate ML ones on ML domains. 1 h |
| F12 minor | `source_analysis.py:138–207` | ML-only patterns; parses `.py/.js/.ts` only | B, M (only with `scan_source`) | domain patterns + C/R/Fortran/Rust imports. 1–3 h |
| F13 major | `sources/suggest.py:35–96,171` | no **category** suggestion anywhere although categories are the field the measured config says to change; no matsci list; C/C++ bio tools score 1 point < threshold 2 | B, M | `suggest_categories(profile)` printed by `rr profile`/`rr update` when categories are still `cs.LG, cs.CL`; matsci anchors; bio terms. 1–2 h |
| F14 minor | `pipeline.py:150–151` | arXiv fetched unconditionally; a `CollectionError` aborts before other sources run | B | honour `sources`; continue past an arXiv failure. 30 min |
| F15 minor | `paper_id.py:31–64` | no DOI canonical id: one preprint can enter as `biorxiv:`, `ss:`, `oa:` | B | `doi:` id when a DOI is known. 1–2 h |
| F16 minor | `sources/biorxiv.py:113–122` | any-token substring filter (`func` ⊂ *function*, `cell` ⊂ *excellent*); measured pass rate 33–39% of all postings | B | word-boundary, all tokens of one query. 20 min |
| F17 partial | `config.py:187`; `pipeline.py:256–267` | HF Papers enrichment on by default (~45 requests/run); on chgnet it attached *MACE* model cards to the CHGNet paper | B, M (cosmetic) | skip when no `cs.*` category configured |
| F18 cosmetic | templates, `cli.py:116` | "arXiv" wording; link label is the raw `biorxiv:10.1101/…` id | B | source label + DOI |
| F19 risk | `finescale.py:58–70` | logistic map fitted on 219 papers / 22 ML-systems repos; transfer to bio/matsci **unmeasured** | B, M | label a small band per domain and check calibration before claiming the threshold |
| F20/F21 cosmetic | `typed_anchors.py:62`; `suggest.py:46–85` | DB/ML examples; `rdkit`/`openmm` trigger the bioRxiv hint | — | examples; require a second bio term |

**Missed by the audit, found by the verifier** — three of these matter for the demo repos:
`embeddings.py:52–56` embeds the *raw* README with a 256-token model, so for badge-heavy READMEs
(MACE, scanpy, matminer) the repo vector that carries `w_embedding: 1.5` is an embedding of
shields.io URLs; `_repo_prose` takes the first 300 characters without skipping heading/TOC-only
lines, so MACE's "what this project is" block is its table of contents and scvi-tools' is
reference-style badge markup (`[![Stars][gh-stars-badge]]…`, which `_MD_IMAGE_RE` does not
strip); and non-arXiv sources receive only `queries[:5]` (`pipeline.py:351`), i.e. three
phrase queries and two single terms — for scanpy that is `scanpy pr`, `pr func`, `func pp`,
`smaller`, `scanpy`.

None of the ML/CS assumptions is in the *ranking or gating logic itself*; they are in what the
repository is made to look like (profile, prose, domains, queries) and in which sources can be
reached. That is why retrieval on the materials runs works as shipped and why the bio runs
work only where arXiv is the venue. The one loss the judge did find (§5.1) — the gate scoring
tool-named application papers 3 — is not a domain assumption in the code; it is a property of
repositories whose *name is a method* that every application paper cites, and it would bite an
ML repo of the same shape.

---

## 3. Where the literature lives, and what each channel can reach

### 3.1 arXiv and the HyDE index

- Category volume (arXiv API `totalResults`): `q-bio.GN OR q-bio.QM` = **16,722** papers
  (0.54% of the index); `cond-mat.mtrl-sci` = **111,471** (3.6%); `cond-mat.mtrl-sci` alone
  received **2,377** submissions in the last 90 days; *"machine learning interatomic potential"*
  in `cond-mat.mtrl-sci` = 682, in the three-category union 844.
- The local HyDE index (`evals/.work/hyde_index`, 36 shards, **3,106,925 rows**, matching the
  count in hyde.py:69) is a full arXiv snapshot with newest id `2607.15279` (mid-July 2026;
  July holds 15,279 rows against 32,040 for June — half a month). **35 of 35** verified
  domain papers are present: bio 14/14 (minimap2 1708.01492, BWA-MEM 1303.3997, kallisto
  1505.02710, Sailfish, freebayes, minimap/miniasm, hifiasm, scVI 1709.02082, DiffDock,
  EquiBind, UMAP, Leiden, Louvain), matsci 21/21 (MACE, CHGNet, M3GNet, NequIP, CGCNN,
  SchNet ×2, MEGNet, ALIGNN, Matbench, Matbench Discovery, CDVAE, MatterGen, MatterSim, OMat24,
  Allegro, DeePMD, GAP, SOAP, ACE, phonopy). Presence is not the constraint; *what is on arXiv
  at all* is.
- **HyDE reach with hand-written hypotheses (no LLM, shipped code path, `top_k=50`):**
  materials — a universal-potential abstract returned **M3GNet at rank 1**, 12/15 `cond-mat.*`
  primary, PET-MAD rank 12, CHGNet rank 42; a phonon-acceleration abstract returned Pheasy
  (rank 1), *Accelerating high-throughput phonon calculations via universal MLIPs* (rank 5),
  CSLD (rank 9), phonopy (rank 28), 14/15 `cond-mat.*` primary. Bio — a doublet-detection
  abstract returned the arXiv literature *about* doublet detection (a benchmark protocol and a
  hyperparameter study, ranks 1–2, 13/15 `q-bio` primary) then drifted to single-cell reviews;
  a batch-integration abstract returned 15 single-cell papers of which only **3/15** were
  `q-bio` primary and 6/15 `cs.LG` primary — the canonical methods it paraphrases (Harmony,
  Scanorama, BBKNN, MNN, Scrublet, DoubletFinder) are bioRxiv/journal-only and cannot appear.
- Timing on this machine (CPU-only torch): model load 41.5 s, `verify_encoder` 30.5 s **and it
  needs the network** (HEAD + Range GETs to huggingface.co; `load_encoder` also touches the Hub),
  encoding ~8 s per hypothesis, search 1.7–2.9 s per hypothesis over 36 shards. HyDE is not
  offline as run today; `HF_HUB_OFFLINE=1` plus a cached verify sample would make it so.

### 3.2 bioRxiv, empirically

| call | result |
|---|---|
| `details/biorxiv/1926-09-12/2026-08-18/0/json` (what `lookback_days: 36500` sends) | `total=475,692` (versions; 345,684 new preprints), **30 per page**, page 0 dated 2014-01-09..2014-10-03 |
| cursor 1170 (the adapter's 40th and last page) | 2014-09-22..2016-09-30 |
| 14-day window | total 2,808; page 0 all 2026-08-04 (oldest end) |
| 90-day window | total 18,514; the adapter's 1,200 cover 2026-05-20..05-25 (6.5%) |
| `?category=bioinformatics`, 90 d / 14 d | 1,867 / **299** (14 d fits under the 40-page cap) |
| shipped `collect_papers` at 36500 / 90 | 40 requests, 123 s / 204 s; kept **463/1,200** and **399/1,200** — `sequence` alone matched 25.7% / 13.9% of everything |
| documented `N most recent` / `Nd` forms | `Both dates must be in yyyy-mm-dd format` |

New bioRxiv DOIs carry prefix **10.64898** (older 10.1101); nothing in `src/reporadar` hard-codes
either. Latency 1.1–16.7 s per request. **The channel has never delivered a topical recent
paper under any shipped configuration** — consistent with the C-9 finding that no `biorxiv:`
id appears in any of the 78 recorded run files.

### 3.3 The keyword routes to bioRxiv that actually work (all keyless or already keyed)

- **Europe PMC** `rest/search?query=…&format=json&resultType=core&pageSize=100`:
  `PUBLISHER:"bioRxiv" AND SRC:PPR` = **345,390** records (medRxiv 87,352); `single-cell
  clustering leiden AND SRC:PPR` = 528 hits, 25/25 with `abstractText` and DOI, publisher
  identifiable per record (bioRxiv 18, Research Square 5, medRxiv 2); `FIRST_PDATE:[a TO b]`
  and `sort=P_PDATE_D desc` work server-side; `cursorMark` paging; ~0.5–1.3 s per call, 26 calls
  without refusal (rate-limit policy could not be read — **unmeasured**); 90-day bioRxiv count
  13,197 vs bioRxiv's own 13,431 vs OpenAlex 13,135 ⇒ ~98% coverage; **2–3 day indexing lag**
  (nothing for 08-16/17 while bioRxiv lists 08-17); 35/100 abstracts carry HTML tags. This is a
  one-GET-per-query adapter.
- **OpenAlex** (key present): `type:preprint` for *single-cell clustering* = 245,774, top-25 all
  preprints with `abstract_inverted_index`, 15/25 bioRxiv; source S4306402567 has 340,678
  works, newest 2026-08-15; costs 10 of 10,000 free daily credits per call.
- **Semantic Scholar** (key present, 4/4 answered without 429): `venue=bioRxiv&year=2026`
  filter works (4,929 hits, all bioRxiv DOIs, abstracts 20/25). Keyless S2 was refused 20/20
  in earlier sessions; the key is the difference.
- **ChemRxiv** (materials/chemistry): the public API answers scripts with an HTTP 403
  Cloudflare challenge; reach it via OpenAlex source S4393918830 (63,067 works, newest
  2026-08-14, 1,028 for *machine learning interatomic potential*). Europe PMC's ChemRxiv
  coverage stops in 2021-06.

---

## 4. Candidate repositories

Nineteen were cloned and profiled with the shipping profiler (`rr profile`, `rr queries`,
$0). Full rows in `wf/bio_scout_rendered.md` and `wf/mat_scout_rendered.md`; the short version:

**Bioinformatics** (`repos/`: scanpy, scvi-tools, minimap2, htslib, kallisto, sourmash, openmm,
mdanalysis) — bioRxiv hint fired only for scanpy and scvi-tools (a listed package must be an
anchor; the six manifest-less repos each scored one point against a threshold of two).
Profiles: **minimap2** mixed (`minimap2`, `reads`, `sam`, `alignment`, `splice`, `nanopore`
survive; `python` is #2 because the setup.py description is its own document; prose is a
phishing alert); **openmm** thin README, no anchors, clean prose, 4/5 single-word queries
generic; **scvi-tools** 82 anchors, domains include *deep learning*, but prose is 100% badge
markup and `customcard` (a Sphinx-design directive) is keyword #5 and two queries; **scanpy**
poor keywords (`smaller`, `pr`, `func`, contributor surnames from 69 release-note files) with
excellent anchors and prose — *not usable live until release notes are excluded*; **kallisto**
loses its own name (`__kallisto__`); **htslib** gets `giab007` (a citation id) as a query;
**sourmash** 16/20 keywords are Sphinx/pytest extras because `doc/` is unread; **mdanalysis**
`image` ties for #1 (rst `|sub| image::` lines). Where the literature lives: arXiv `q-bio.GN`
for Heng Li-style algorithm papers, kallisto, scVI, UMAP/Leiden; bioRxiv for most single-cell
and third-party genomics methods (a hand-made minimap2 target list is 3/7 on arXiv — the
author's own papers — and 4/7 bioRxiv-only); OpenMM's ML-potential-era literature is 7/7 on
arXiv.

**Materials** (`repos/`: pymatgen, mace, chgnet, phonopy, matminer, nequip, dscribe, tblite,
+ deepmd-kit, kim-api, lammps) — **chgnet** is the cleanest profile of the 19 (keywords are the
subject: *universal neural network potential, charge-informed atomistic modeling*); **mace**
zero anchors (`setup.cfg`), prose = table of contents, 4/8 generic queries, but excellent
keyword recall on its own name; **dscribe** clean prose, subject keywords survive (*descriptors,
kernels, soap, mbtr*), `module`/`code` pollution from 246 Sphinx files; **pymatgen** flagship
name, excellent anchors that map to no domain, generic keywords (`following directory`);
**nequip** clean anchors/prose, generic keywords, false DBLP hint via `lmdb`; **phonopy**,
**tblite**, **kim-api**, **lammps** are the honest blind-spot exhibits (`doc/` unread, no
manifest, install boilerplate as queries: `pre-commit`, `mamba`, `gnu lesser`, `files`).
Hand-made targets: 18/18 for CHGNet+MACE on arXiv, 16/18 inside the three-category set,
18/18 with `cs.LG` added.

**Chosen for the live runs:** minimap2, openmm, scvi-tools (bio); chgnet, mace, dscribe
(matsci). scanpy and pymatgen are the names the audience knows best and both are held back by
the profiler (§6 D3), not by the literature.

---

## 5. End-to-end runs of the measured configuration

Six sequential `rr init --measured` → `rr update` → `rr digest` runs, four fields patched from
the measured file (`repo_path`, `arxiv.categories`, `hyde.index_dir` → the eval index,
`output.digest_path`); everything else at measured values. 82 arXiv requests total, **zero
429s**, no stage skipped, HyDE encoder verified on every run, gate on 50 papers, fine-scale
persisted. Wall 223–254 s per repo (23.5 min for six).

| repo | domain | keyword pool | HyDE new | pool | gate 0/1/2/3 | admitted | HyDE among gated / ≥2 | band rescored / kept | Top / Maybe / Muted | HyDE in Top |
|---|---|---|---|---|---|---|---|---|---|---|
| minimap2 | bio | 193 | 225 | 418 | 21/19/7/3 | 10 | 10 / 0 | 7 / 6 | 9 / 6 / 1 | 0 |
| openmm | bio (MD) | 295 | 284 | 579 | 11/20/13/6 | 19 | 16 / 5 | 9 / 3 | 9 / 6 / 0 | 1 |
| scvi-tools | bio | 121 | 385 | 506 | 9/15/24/2 | 26 | 32 / 17 | 13 / 6 | 8 / 7 / 1 | **6** |
| chgnet | matsci | 199 | 352 | 551 | 0/11/28/11 | 39 | 30 / 20 | 4 / 2 | 13 / 2 / 2 | 3 |
| mace | matsci | 356 | 332 | 688 | 5/10/23/12 | 35 | 23 / 17 | 3 / 2 | 14 / 1 / 3 | 4 |
| dscribe | matsci | 151 | 274 | 425 | 4/21/23/2 | 25 | 37 / 17 | 13 / 10 | 12 / 3 / 1 | **8** |

("Muted" entries are all withdrawn papers; Top+Maybe = 15 = `top_n` in every run.)

What the digests contain (full top-15 lists in `wf/runs_rendered.md`, the digests themselves in
`runs/<repo>/digest.md`):

- **minimap2** — *Nucleotide String Indexing using Range Matching*, minimap2's own paper,
  *Improving spliced alignment by modeling splice sites with deep learning*, Genome-on-Diet,
  SneakySnake, HQAlign, LOGAN (GPU X-drop), FASTR, BLEND. Every on-topic pick came from the
  single `all:minimap2` query (16 hits); 3 of 8 queries returned nothing; HyDE added 225 papers
  and **0** reached Top Picks. The gate was the strictest of the six (10/50). The action ideas
  assume "your minimap2 Python wrapper" because the only anchor is `cython`.
- **openmm** — OpenMM-Python-Force, OpenMM-MiMiC, OpenMM 8, force-field parameter
  optimisation, alchemical transfer, autoencoder enhanced sampling, Grappa, GCMe, and one HyDE
  pick (a performance-portable MD DSL). Several score-3 picks *use* OpenMM rather than propose
  changes to it. `all:high`, `all:performance`, `all:code`, `all:library` (14,703 / 12,538 /
  4,020 / 1,035 hits) are noise.
- **scvi-tools** — the original scVI paper (via HyDE), its DE-genes companion, amortised GPLVMs,
  latent-diffusion and masked-diffusion single-cell generators, scMEDAL, sparse-mechanism-shift
  VAE. Keyword pool only 121; **HyDE supplied 385 of 506 and 6 of 8 Top Picks** — the best
  HyDE showcase with dscribe. 24/50 gated at score 2, so tiering rests on the rescore. The
  withdrawn entry at the top is *Computational Model of Music Sight Reading* (via `all:model`).
- **chgnet** — CHGNet, FastCHGNet, MatGL, fourth-generation HDNNP with non-local charge
  transfer (HyDE), *Overcoming systematic softening in universal MLIPs by fine-tuning*,
  cross-functional transferability, Allegro (HyDE) — and four application papers (Li₃YCl₆
  electrolytes, hcp-zinc EXAFS, rare-earth oxide diffusion, Mn-rich rocksalt cathodes) at
  gate score 3. Gate admitted 39/50, 0 at score 0. HF enrichment attached MACE model cards to
  the CHGNet entry.
- **mace** — MACE-POLAR-1, MACE, *Systematic Fine-Tuning of MACE for Catalysis*, MACE4IRmol,
  *Better without U* (HyDE), *Speeding Up MACE: Low-Precision Tricks*, PE-MACE, *Evaluation of
  the MACE Force Field Architecture*, tensor ACE (HyDE). Three off-topic withdrawn papers
  (hydraulic fracturing, opioid use disorders, music sight reading) **head the digest** because
  `all:model` / `all:use` / `all:models` each hit >100k arXiv papers.
- **dscribe** — *Updates to the DScribe Library*, Ceriotti/Willatt/Csányi on physical
  principles (HyDE), LATTE, spherical Bessel descriptors, tensor-reduced density
  representations, *Compressing local atomic neighbourhood descriptors*, *Inversion of the
  chemical environment representations*. 37 of 50 gated papers and 8 of 12 Top Picks from
  HyDE. The original DScribe paper was gated **0** ("describes the package itself") while
  CHGNet's, MACE's, minimap2's and OpenMM's own papers were gated 3.

### 5.1 Judged actionability (GPT-5.5, shipped rubric, single draw)

Every Top Pick, plus the Maybe-tier papers needed to fill a diagnostic top-10, was scored by
`evals/judge.py::judge_paper` (rubric v1, gpt-5.5) with `harness.assemble_repo_context` on the
full clone — the same judge and rubric as the benchmark, but a richer repo context than the
benchmark mini-repos and **no Opus baseline in the pool**, so these are absolute levels, not a
paired comparison. 69 papers, 0 judge failures, ~$2.80.

| repo | domain | Top Picks shown | actionable | **net@2** | precision | judge 0/1/2/3 | top-10 net@2 |
|---|---|---|---|---|---|---|---|
| minimap2 | bio | 9 | 8 | **+6** | 0.89 | 0/1/0/8 | +7 |
| openmm | bio | 9 | 9 | **+9** | 1.00 | 0/0/3/6 | +7 |
| scvi-tools | bio | 8 | 7 | **+5** | 0.88 | 0/1/5/2 | +7 |
| chgnet | matsci | 13 | 9 | **+1** | 0.69 | 0/4/5/4 | +1 |
| mace | matsci | 14 | 9 | **−1** | 0.64 | 0/5/3/6 | +1 |
| dscribe | matsci | 12 | 11 | **+9** | 0.92 | 0/1/7/4 | +7 |
| **mean / pooled** | | 65 | 53 | **+4.83** | **0.815** | 0/12/23/30 | +5.00 |
| bio (n=3) | | 26 | 24 | +6.67 | 0.923 | | |
| matsci (n=3) | | 39 | 29 | +3.00 | 0.744 | | |

Reference: the 25-repo ML/systems benchmark at the same width scores +5.72 / 0.89. With three
repositories per domain neither "bio above the benchmark" nor "matsci below it" is established;
the per-case sd on the benchmark is 1.73, so a three-case mean carries an SE of ~1.0 before any
domain effect. What *is* informative is the shape of the errors:

- **No Top Pick was judged 0** (65/65 ≥ 1): every shown paper was on-topic. All twelve losses
  are judge-1 — "same topic, nothing to port".
- **The losses sit in the gate's score-3 band, not the rescored band.** Gate-3 papers (trusted
  on the gate's word, fine-scale not applied): 36 shown, 25 actionable, **precision 0.694**
  (bio 9/11, matsci 16/25). Gate-2 papers that cleared the fine-scale threshold: 29 shown, 28
  actionable, **precision 0.966**. On the ML benchmark the score-3 band was the reliable one
  and the score-2 band was the coin flip (RESULTS.md, "Ranking the score-2 band"); here it is
  the other way round.
- **Two concrete patterns account for 11 of the 12 misses.** (a) The repository's *own* paper:
  minimap2 1708.01492, scVI 1709.02082, CHGNet 2302.14231, MACE 2206.07697, *Updates to the
  DScribe Library* — all gate 3, all judged 1 ("describes what the repository already
  implements"); OpenMM 8 was the exception (judged 3, its abstract enumerates new features).
  Excluding these five is a post-hoc adjustment, not a measurement, but it would put pooled
  precision at 0.883 and the mean at +6.5. (b) Materials **application and benchmark papers
  that use MACE/CHGNet and name it in the abstract** — chgnet: rare-earth oxide diffusion,
  Mn-rich rocksalt cathodes, and MatGL (a sibling library); mace: defective COFs, *Evaluation
  of the MACE Force Field Architecture*, twisted 2D materials, the double-halide lattice-dynamics
  benchmark. The gate scores the name match 3; the judge says "application, no method to
  port". dscribe, whose method (SOAP) is not the repo's name, had 10 of 12 Top Picks come
  through the fine-scale path and lost only its self-paper.
- **The fine-scale rejections were mostly wrong on this draw** (small n): of the four Maybe-tier
  papers the rescore dropped (P 0.37–0.45), three were judged actionable (minibwa for minimap2;
  HEIST and the sparse-mechanism-shift VAE for scvi-tools) and one not (MiMiC/CP2K for OpenMM).
- **The judge's `proposed_change` lines are demo material.** minimap2: "add an optional
  Ranger-based index backend … selectable by a command-line flag"; OpenMM: "add a PythonForce
  plugin/API that lets users register a Python callback returning energies and forces";
  chgnet: "add charge-state-aware atom encoding so Fe²⁺ and Fe³⁺ are distinct input
  environments"; mace: "add a preprocessing flag that detects selectively Hubbard-U-corrected
  configurations and applies the per-U-atom energy shift before training foundation models";
  dscribe: "an optimized SOAP backend in `dscribe.ext` computing expansion coefficients
  analytically … exposed as `method='optimized'`". Full per-paper verdicts with justifications
  are in `judge_scisoft.json` / `wf/judge_rendered.md`.

What this draw supports: on arXiv-covered scientific repositories the measured pipeline
produces digests at roughly benchmark quality **where the gate's score-3 band is not inflated by
name matches**, and the one place it fails (universal-potential libraries whose name appears in
every application paper) has a plausible cheap fix that the project's own machinery can test.

Two labels the critic insisted on, both right: "HyDE supplied N of M Top Picks" is
**provenance** (which channel fetched the paper first), not measured contribution — no
keyword-only, gate-off or fine-scale-off arm was run on these repos; and every number in §5 is
one draw on one machine (gate and suggestions are sampled, so a live re-run of the same repo
will differ in detail).

### 5.2 Recall diagnostic — the scouts' hand-made targets against the stores ($0)

Precision of what was shown is not recall. The critic joined the scouts' target lists (§4;
the scouts' opinion, not blind gold) with the run stores (`<clone>/.reporadar/papers.db`,
`paper_scores` by RRF rank; the gate is exactly the top 50 by RRF in all three stores):

| repo | targets | in pool | gated (top-50) | shown as a non-self pick | notes |
|---|---|---|---|---|---|
| chgnet | 9 | 5 | 2 | 1 (*systematic softening* 2405.07105, RRF rank 45) | Matbench Discovery 2308.14920 at RRF **54** — four places outside the gate; OMat24 rank 269; SevenNet 374; MatterSim, Orb, MACE-MP-0, *uMLIPs ready for phonons* not in the pool |
| mace | 9 | 5 | 1 (*Evaluation of MACE* 2305.14247, rank 44, shown, judged 1) | 0 | design-space paper 2205.06643 rank 80; CACE 176; OpenEquivariance 227; MACE-OFF 371; LES, e3nn, EMFNN, sublimation fine-tuning not in the pool |
| minimap2 | 4 | 3 | 2 | 0 | 2108.03515 (*New strategies to improve minimap2 alignment accuracy*) was **gated 0** with the reason "not novel methods that could be applied to this repository's Python wrapper/bindings" — the gate believes minimap2 is a Python wrapper because the profile's only anchor is `cython` (F5); minigraph 2003.06079 rank 292 |
| total | 22 | 13 | 5 | 2 | |

So the answer to "why is my paper not there" on these runs is, most often, *not in the pool*
(9/22) or *in the pool but below the gate window* (RRF 54–80), and once *the profile
misdescribed the repository to the gate*. Diagnostic, not a recall number — but it is the
shape of the question the audience will ask, and it is why D7 (a per-paper `rr why`) is on the
list.

---

### 5.3 What was then fixed, and what it did to the demo repo (2026-08-18)

Three of §6's presentation-only items were implemented so a single-repo demo could be
recorded: **D1** (withdrawn scope), **D4** (papers the repository already cites), **D5**
(suggestions addressed to the maintainer). All three change *rendering and prompting*, not
retrieval, ranking or gating — so the cached judge verdicts still describe exactly what is
on screen, and the tally below is recomputed from them rather than re-judged.

- **D1** — `digest_window` now stops walking once the window is full, so only retractions
  that were competing for a slot are listed, and the rendered list is read off that window
  instead of re-scanning every scored paper. The second derivation was a genuine
  duplicate-invariant bug of the C-9/C-12 shape: `_build_digest_context` rebuilt the list
  from `scored`, so fixing `digest_window` alone would have changed nothing on screen.
- **D4** — `profiler.cited_arxiv_ids_of` collects arXiv ids from the repository's own
  README, `CITATION*` and `docs/` (including `*.bib`, read raw because `_clean_document`
  strips the URLs most citations arrive in). `categorize_papers` takes `cited_ids`, flags
  those papers `already_cited` and mutes them *inside* the window — so excluding one cannot
  promote an unjudged paper from outside the digest — and they render in an "Already cited
  by this repository" section. The month is validated (01–12) in the id pattern, which is
  what separates a citation from a DOI: it rejects `2019.10694` (dscribe's README) and
  `1029.28096`/`1484.11876`/`2042.03300` (MACE's) while keeping every real id in the same
  files. `rr notify`, `rr watch` and the archive index pass the same set, so no count can
  disagree with the digest.
- **D5** — the suggestions prompt now says the reader *maintains* the repository, and
  carries the repo's own prose. It used to open "a developer whose project uses:
  {anchors}", which is why minimap2's action ideas addressed "your minimap2 Python wrapper".

**Effect on the demo repository (dscribe), recomputed from the cached verdicts:**

| | before | after |
|---|---|---|
| Top Picks shown | 12 | **10** |
| judged actionable | 11 | **10** |
| net@2 | +9 | **+10** |
| precision | 0.917 | **1.00** |
| off-topic retraction heading the digest | yes (1) | none |
| set aside as already cited | — | 2 (`2303.14046`, the project's own follow-up, judged 1; `1601.04077`, judged 2) |

The second exclusion is the honest cost of the rule: the judge scored `1601.04077`
actionable, not knowing dscribe's own tutorial teaches its method and its bibliography
cites it. The rule uses information the judge does not have; it is not free.

Not done, and still carrying measurement debt: **D2/D3** (profiler hygiene). `rr profile`
on dscribe still shows `module` and `code` at ranks 2–3 and the digest header still prints
`(all:module)` and `(all:code)`. Fixing that changes the profile → the queries → the pool,
so it needs a re-run and a re-judge before any number here could still be quoted. It is the
one visible blemish left in the recording.

Tests: `tests/test_digest.py::TestWithdrawnSectionScope` and `::TestAlreadyCitedPapers`,
`tests/test_profiler.py::TestCitedArxivIds`. Full suite 1910 passed.

**Demo assets** (outside the ephemeral scratchpad, so they survive the session):
`evals/.work/demo-dscribe/` — the clone, its `.reporadar.yml` (measured config, categories
`cond-mat.mtrl-sci, physics.comp-ph, physics.chem-ph, cs.LG`, index pointed at
`evals/.work/hyde_index`) and the seeded store whose **run 1 is the judged run**;
`demo_recorder/scenarios/reporadar-scisoft-demo-frozen.json` (~90 s, no network, renders run
1 — record this one) and `reporadar-scisoft-demo.json` (adds a live `rr update`, ~4 min,
cut between its two markers in post; a fresh run is a new draw, so the "ten of ten" line
would then describe a run nobody judged);
`demo_recorder/assets/reporadar_scisoft_env.ps1` (loads the keys, prints only which were
found — without it `rr digest` falls back to the ML-flavoured template action ideas).

---

## 6. Obstacles, ranked, with what to do about each

**D — do before the demo (all small, all cosmetic-to-major on stage)**

- **D1. Off-topic withdrawn papers head the digest.** `digest.py:145` lists every withdrawn
  paper in the run regardless of rank; generic queries put *opioid use disorders* at the top
  of the MACE digest. Fix: show the withdrawn section only for papers that would otherwise have
  been in the top-N or admitted by the gate; move it below Top Picks. Trivial.
- **D2. Categories and domains.** Add `suggest_categories(profile)` and print it from
  `rr profile`/`rr update` when `arxiv.categories` is still `cs.LG, cs.CL` (F13); add bio and
  matsci packages to `PACKAGE_DOMAIN_MAP` (F3); add matsci anchors and lower the term
  threshold for bio tools in `sources/suggest.py`. 1–2 h. Without D2 an audience member who runs
  `rr init --measured` on their own repo gets an ML pool.
- **D3. Profiler hygiene that decides the demo repos' queries.** Read `doc/`; skip
  release-notes/changelog; strip MyST `{role}` and reference-style badges; skip TOC-only lines
  in `_repo_prose`; parse `setup.cfg` and `environment.yml`; strip `~=`; keep anchors out of
  the bigram corpus; add `using/tools/code/module` to the boilerplate list (F5–F8, verifier
  items). Each 20 min–1 h; together they turn scanpy, pymatgen, MACE and phonopy from
  embarrassing profiles into usable ones. **Every one is domain-neutral**, which is why they
  are worth doing regardless of the demo — but each changes the profile the benchmark measures,
  so re-run the 25-case Tier A gate afterwards.
- **D4. Self-paper handling.** The repo's own paper is a score-3 Top Pick in five runs
  (judged 1 in all five — a −2 apiece under net@2) and a score-0 reject in one (dscribe's
  original paper). Exclude it when the title or arXiv id appears in the README/CITATION block,
  or label it "the project's own paper" outside the tiers, and make it consistent. Small;
  worth ~+2 net@2 per repo on this draw and it removes the single most predictable "why is
  my own paper a recommendation?" question.
- **D5. The repository described as a *user* of the tool.** `llm_suggestions.py:18` addresses
  "a developer whose project uses: {anchors}" — for minimap2 that is `cython`, hence "your
  minimap2 Python wrapper" in the action ideas, and the same misdescription reached the gate:
  2108.03515 (*New strategies to improve minimap2 alignment accuracy*) was gated 0 as "not
  applicable to this repository's Python wrapper/bindings" (§5.2). The suggestions prompt
  should state that the reader *maintains* the profiled repository; the gate side is fixed by
  D2/D3 (anchors and prose that describe the repo). Small; the suggestions prompt is not in the
  benchmark metric, so it needs only a spot check.
- **D6. Demo hardening — timing, network, variance.** Budget ~4 min per `rr update`
  (measured once each, 223–254 s, CPU torch); pre-warm the model cache; **test**
  `HF_HUB_OFFLINE=1` after one online run — it is asserted, not tried, and `verify_encoder`
  needs a cached verify sample to work offline; keep pre-populated `.reporadar/` stores for
  every demo repo so `rr digest` runs even if arXiv throttles (13–14 arXiv requests per run at
  1 req/3 s from one IP — several attendees behind one conference NAT will hit 429s, and
  `pipeline.py:150–151` aborts the whole run on a `CollectionError` with no fallback, F14);
  never run two updates at once; time `rr sync-index` (~1.1 GB) on a fresh machine once —
  the six runs used an index and weights that were already on disk; and expect a live re-run
  to differ from the judged digest, since the gate and suggestions are sampled — narrate a
  frozen store, re-run one repo live.
- **D7. `rr why <arxiv_id>`.** There is no per-paper explanation command, yet `papers.db`
  already holds the RRF rank and components, `matched_query`, gate score and reason,
  fine-scale P and tier for every run. A small read-only command over that table is the only
  credible answer to "why is my paper not there / why is it here" (§5.2) — and to "why is
  this an ML paper" (`arxiv.categories` left at the default). Small.

**B — needed for "bioRxiv papers" specifically**

- **B1. A keyword-search bioRxiv source.** **Landed 2026-08-19 (§13):** shipped as a NEW
  source `europepmc`, not as a replacement for `biorxiv`; the probe corrected four points of
  the spec below, two of which would have silently emptied or mutilated the channel. Replace the body of `sources/biorxiv.py` (or add
  `sources/europepmc.py`) with Europe PMC: `(<plain keywords>) AND SRC:PPR AND
  (PUBLISHER:"bioRxiv" OR PUBLISHER:"medRxiv") [AND FIRST_PDATE:[start TO end]]`,
  `resultType=core`, `pageSize=100`, one GET per query, ~1 s apart, `email=`; id
  `biorxiv:<doi>` (10.1101 and 10.64898), strip HTML from `abstractText`. Keep the bioRxiv
  details endpoint only if a category-listing mode is wanted (with `?category=`, a 14-day
  window, and newest-first paging). ~1 day including tests and the privacy-registry entry.
- **B2. Fix category scoring for non-arXiv papers (F4)** so a bioRxiv paper is not ranked
  0.33 below an arXiv paper with identical keyword overlap. 1 h. **Landed 2026-08-19 (§12.3):**
  the ranker now asks whether a paper's categories are in the target list's *vocabulary*, so
  bioRxiv and OpenAlex move 0.400 -> 0.600 and join Semantic Scholar on the
  `absent_category` path.
- **B3. DOI canonical id (F15)** so the same preprint from Europe PMC/OpenAlex/S2 appears
  once. 1–2 h. **Landed 2026-08-19 (§12.4):** `paper_id.doi_key` across five adapters. It
  also broke two things that identified an arXiv paper by exclusion, both fixed with the new
  `paper_id.is_arxiv_id`. Existing stores are not migrated.
- **B4. OpenAlex `type:article|preprint` (F2)** — one line — and pass more than
  `queries[:5]` to keyword sources. **Landed 2026-08-19 (§12.1–12.2):** preprints are 26.7%
  of the last-30-day OpenAlex pool and this is currently the only wired route to bioRxiv; the
  query cap went 5 -> 8, which was withholding 28.6% of the benchmark's queries.
- **B5. Accept the structural limit and measure it.** After B1–B4 the bioRxiv channel is
  keyword + embedding + gate + rescore with **no HyDE** and a fine-scale map that has never
  been fitted on a bio abstract (F19 — although the 69 verdicts give a first read: the 29
  score-2 papers the rescore passed were actionable 28 times, bio 15/15, and 3 of the 4 it
  rejected into Maybe were actionable, so on this draw the threshold is if anything
  conservative on these domains, not permissive). Whether the bioRxiv channel is "good" is
  exactly the question the benchmark exists to answer; the honest demo statement until it is
  answered is "bioRxiv is a source we are adding; the arXiv channel is the one we have
  measured". Also unaddressed anywhere: **journal-only literature** (Bioinformatics, Genome
  Biology, NAR; PRB, JCTC) — the only wired route is OpenAlex `type:article`, "built, never
  validated", probed here only for preprints.

**M — needed for "materials-science papers"** — nothing structural. ChemRxiv via OpenAlex
(S4393918830) if wanted; `cs.LG` in the category set (2 of 18 MACE targets need it).

**Not exercised at all, stated as such:** any multi-source run (all six were
`sources: [arxiv]` + HyDE, so F4/F15/F16/F18 are code-reading plus API probes — the behaviour
of a non-arXiv paper *inside a digest* has never been observed); R/Bioconductor, Julia, Rust,
Nextflow/Snakemake and conda-`environment.yml` repositories (never profiled, let alone run —
half of a bio audience writes R); the recency path (`lookback_days: 14`, `sort_by: submitted`,
"supported but never benchmarked" per the README) although "what came out this month" is a
question this audience will ask; and the keyless default configuration on these domains.

**G — the gate on tool-named application papers (the one measured loss)**

- **G1. Rescore the score-3 band too, or gate applications.** The matsci gates admitted 39/50
  and 35/50 with 11–12 score-3s; the judge (§5.1) called gate-3 papers actionable 0.694 of the
  time and fine-scale-passed papers 0.966. On chgnet/mace only 3–4 band papers were rescored
  because score-3s filled the window, so the stage that works never touched the papers that
  were wrong. Two candidate fixes, both cheap, both to be *measured* rather than shipped on
  this draw: (i) apply the fine-scale rescore to score-3 papers as well (its calibrated map is
  fitted on the score-2 band, so this needs a calibration check on cohort 3, §8); (ii) a
  gate-prompt clause distinguishing "applies or benchmarks this repository's method" from
  "changes it" — but the gate prompt shares `repo_context_block` with the fine-scale map, so
  only the rubric half may change and the change must be re-measured on the 25-case benchmark
  (a prompt tweak below ~1 net@2 will not resolve there; `diagnose_triage.py` at ~$0.10/arm is
  the cheap first look). On this draw the effect is bounded: fixing all seven matsci
  application misses is +14 net@2 over three repos.
- **G2. Fine-scale rejections in the Maybe tier were 3-of-4 actionable** — too few to act on;
  cohort 3 will add labels.

---

## 7. Success chances

Stated as subjective probabilities that a demo run on a repo of that kind produces a digest the
audience accepts as *working well* (criteria of §1), given the evidence above. None is measured;
the reasoning is the point.

| scenario | now | after D1–D7 (+G1 if it measures) | after B1–B4 too | why |
|---|---|---|---|---|
| Materials-science repo, descriptor/analysis library class (dscribe, pymatgen-after-D3) | **~0.8** | ~0.85 | — | literature on arXiv at volume; HyDE reaches it (M3GNet rank 1); dscribe +9 at precision 0.92 with 10 of 12 Top Picks through the rescored band |
| Materials-science repo whose **name is a method** every application paper cites (chgnet, mace) | **~0.45** | ~0.8 | — | on-topic throughout (0 judge-0) but judged +1 / −1 at precision 0.69 / 0.64: self-paper and six tool-named application papers admitted at gate 3. A materials audience spots "that paper just uses MACE" immediately. D4 + G1 are the fix; withdrawn off-topic papers at the top and MACE's empty profile are the cosmetics |
| Bio repo whose literature is on arXiv (aligners, MD, probabilistic single-cell), **one of the three tested**, re-run from its store | **~0.65** | ~0.75 | ~0.75 | minimap2 +6, openmm +9 (9/9), scvi-tools +5 — mean +6.67 at precision 0.923 on one draw; recall rests on the project-name query (minimap2: 16 hits, 0 HyDE picks) or on HyDE alone (scvi-tools: 385 of 506 pool papers); the arXiv slice of bio is thin (16,722 papers in q-bio.GN+QM) |
| Bio repo of that kind **chosen by the audience**, cold | ~0.45 | ~0.6 | ~0.6 | untested profile (scanpy/kallisto/sourmash-style defects, §4), `arxiv.categories` must be set by hand (F13), 9 of 22 hand-made targets never entered a pool (§5.2) |
| Bio repo against **bioRxiv** (single-cell toolkit, genomics pipelines) | **~0.1** | ~0.1 | **~0.45** | today no channel can deliver a recent bioRxiv paper; after B1–B4 it becomes a real keyword source, but with no HyDE, no measurement, and journal-only literature still out of reach |
| Compiled/R/Fortran repo with no Python manifest (htslib, LAMMPS, tblite; any R package) | ~0.3 | ~0.5 | — | profile is README-only and boilerplate-heavy; D3 helps; HyDE + prose can carry a good README (openmm: no anchors, +9) but not a directory listing (LAMMPS); R `DESCRIPTION` unread and no R repo was even profiled |

Two things raise every row: a clean README first paragraph (the 300-character prose is what
the gate and HyDE see) and a repository whose name is a good query. Two things lower every
row: `arxiv.categories` left at the ML default, and a repo whose research is journal-only.

---

## 8. Plan

**Status as of 2026-08-19.** D1/D4/D5 landed (§5.3); D2/D3 landed and are recorded in §10,
with the two defects they introduced fixed in §11; B2/B3/B4 landed in §12; B1 landed in §13.

**Every change that alters what the pipeline produces has now landed.** Still open: **D7**
(`rr why`) which is read-only and touches no number, and **cohort 3**, which is the blocking
item — §11 measured that the profiler work moved the profile of 16 of the 25 benchmark cases,
so every live number in this document describes a pipeline that no longer ships.

**Cohort 3 is pre-registered in §14** and its twelve cases are in `evals/benchmark.yaml`
(37 total). Item 2 below is superseded by §14: six cases per domain rather than four, because
four resolves nothing below ~1.7 net@2; the legacy 25 are re-measured in the same session, so
no conclusion depends on a historical figure; and the Europe PMC arm is declared with a bar
rather than left to be decided afterwards. Nothing has been spent.

1. **Pre-demo engineering (2–3 days):** D1–D7 in §6, then B1–B4. Every change except D5 is
   domain-neutral or additive; run `uv run pytest` and the Tier A gate after D3. D4 alone
   moves this draw from +4.83 to +6.5 mean net@2; do it first. Whether to *exclude* or merely
   *label* the self-paper is a presentation choice — the metric says exclude, a live audience
   may read "it found our own paper" as a sanity check — decide it, don't leave it to the gate.
2. **Cohort 3 — scientific software — added to `evals/benchmark.yaml`, pre-registered.**
   Four bio (minimap2, scvi-tools, openmm, scanpy-after-D3), four matsci (chgnet, mace,
   dscribe, pymatgen-after-D3), each with `expected_categories`, `gold_queries`,
   `distractor_queries`, and a `criteria:` line naming the venue (arXiv-native / bioRxiv /
   journal). Predictions to write down before running, from this draw: matsci four ≥ +4 mean
   net@2 *with* D4, ≤ +3 without; bio four ≥ +5 under arXiv-only; a *paired* delta for
   `sources: [arxiv, europepmc]` on the bio four whose sign we do not know. **The G1 arm is
   no longer proposed**: §9.4 measured both candidates offline and neither cleared its bars —
   the rescore over the score-3 band catches 2 of 8 misses at zero cost, and the rubric clause
   catches 8 of 9 while discarding a third of the digest for +0.33 net@2/repo. Cohort 3 should
   measure the pipeline as it ships. With four cases per domain the per-domain SE is ~0.9 net@2 (per-case sd
   1.73), so only effects above ~2 resolve; report anything smaller as "below the floor" as the
   project already does. Cost: judge ~$1–2/case + Opus baseline ~$0.8/case ≈ $25 for one full
   pass; the frozen-pool mode halves the floor on re-runs; the 69 verdicts already cached
   under `scisoft-*` are reusable if the cache key is kept.
3. **Calibration check for the fine-scale map on the new domains (F19):** the cohort-3 run
   already labels every shown paper; `calibrate_finescale.py --analyse` on those verdicts
   answers "is 2/3 still where it should be" at $0 extra — required before G1(i) can be
   trusted on score-3 papers, whose gpt-4o-mini expectations were never in the fit.
   **DONE 2026-08-20, §18.2 — and the named command is the wrong one.** `--analyse` reads a
   per-case cache that covers only the 22 legacy cases; the artifact already carries
   `finescale_p`, so `evals/finescale_domains.py` does it for $0. §18.1.
4. **Demo day:** lead with **dscribe** (cleanest matsci digest on this draw, +9, 8 of 12 Top
   Picks from HyDE) or OpenMM (+9, bridges the bio and comp-chem halves of the room); show
   minimap2 and scvi-tools from pre-populated stores; show chgnet/mace only after D4+G1 (as
   shipped they are the digests a materials expert will pick apart); show scvi-tools with the
   bioRxiv source on *only if* B1–B4 landed and the cohort-3 bio delta is not negative,
   otherwise show it as the arXiv-only run and say plainly where bioRxiv stands. Do not run
   `sources: [arxiv, biorxiv]` live as shipped — it spends 40 requests and 2–3 minutes
   fetching 2013–2016 postings. Keep the blind-spot exhibits (LAMMPS, phonopy, htslib,
   tblite) as one slide: what the profiler cannot see and why. Have a hand-made 8–10 target
   list per demo repo checked with `rr why` beforehand so "why is my paper not there" has a
   concrete answer.
5. **Second draw.** Re-run the six (~$0.20) and re-judge (~$3) once before the demo so the
   variance of the §5 numbers is stated rather than guessed — matsci's three values (+1, −1,
   +9) have a per-domain sd of 5.3 on this draw.

---

## 9. PRE-REGISTERED — the score-3 band, and a correction to §5.3 (2026-08-18)

Written **before** the first paid call, so the bars cannot be chosen after seeing the
answer. Both probes score against the 69 GPT-5.5 verdicts already cached under
`evals/cache/judge/v1/gpt-5.5/scisoft-*/`; no new judging is paid for.

### 9.0 Correction: the shipped already-cited rule catches three of the five self-papers

§5.3 and PR #151 report that the repository's own paper was a gate-3 Top Pick on five of
six repositories and was judged 1 every time. That finding stands. What I then implied —
that the merged rule removes them — is **wrong, and measured wrong here**: executed
offline against the six checkouts, `cited_arxiv_ids_of` removes **4 of the 69 papers**,
and only 3 of those are self-papers.

| repo | own paper | removed? | why |
|---|---|---|---|
| minimap2 | 1708.01492 | yes | cited in `README.md` |
| chgnet | 2302.14231 | yes | cited in `README.md` and `citation.cff` |
| dscribe | 2303.14046 | yes | cited in `docs/` |
| **scvi-tools** | 1709.02082 | **no** | it cites nine arXiv ids — Adam, the VAE, seven others — and none is its own |
| **mace** | 2206.07697 | **no** | its README bibtex cites 2205.06643, 2312.15211, 2401.00096, not itself |

`openmm` contributes zero cited ids at all: it has no `CITATION*` file and its docs live in
`docs-source/`, which `cited_arxiv_ids_of` does not read. The fourth removal is
`dscribe`'s 1601.04077, judged **actionable** — the acknowledged cost.

So the shipped effect over the six repositories is **shown 65 → 61, actionable 53 → 52,
precision 0.815 → 0.852, mean net@2 +4.83 → +5.67**. My earlier "+6.5 / 0.883" described
removing all five self-papers, which is a *different rule* from the one that shipped.
Reading `docs-source/` and a `README`-bibtex title match would close most of the gap, and
that is now a candidate, not a claim.

**What remains, and it is what these probes are about:** among the 61 shown papers, 9 are
non-actionable — 6 materials **application papers that name the tool**, 2 **self-papers the
citation rule misses**, 1 **sibling library**. Eight of the nine are gate-score-3.

### 9.1 Probe A — does the fine-scale stage already separate the gate-3 band?

The rescore was fitted on the score-2 band and has never been run on a 3. Nothing in
`finescale.py` is score-2-specific; the restriction lives in the caller.

* **Population** 33 gate-3 papers surviving the already-cited rule (25 actionable / 8 not).
* **Measure** shipped `finescale` probability per paper; ROC-AUC against the judge's ≥2,
  and the confusion at the shipped `SHOW_THRESHOLD` of 2/3.
* **Kill** AUC < 0.60 — no separation, G1(i) refuted.
* **Win** AUC ≥ 0.70 **and** the 2/3 threshold drops ≥ 4 of the 8 misses while costing
  ≤ 2 of the 25 actionable.
* **Reference** E2 measured AUC 0.84 on the score-2 band; materially below that on score-3
  is itself worth recording.
* **Cost** ~33 gpt-4o-mini calls, ~$0.01.

### 9.2 Probe B — does a rubric clause separate "applies it" from "changes it"?

One variant, added to `triage._RUBRIC`; `repo_context_block` is **not touched**, because the
fine-scale probability map is fitted to those exact bytes.

* **Population** the 61 shown papers (52 actionable / 9 not).
* **Fix bar** ≥ 5 of the 9 non-actionable lose their admit under the variant.
* **Cost bar** ≤ 3 of the 52 actionable lose theirs. **Kill** > 5 actionable lost.
* **Control** the same two arms over the 602 cached ML-benchmark labels — the shipped arm
  is already on disk (`evals/.work/diag_triage_prose.json`), so only the variant is paid
  for. Pooled precision must not fall and recall must not fall by more than 0.05: a clause
  that fixes materials by making the gate stingy everywhere is not a fix.
* **Cost** ~$0.13 (61 scisoft + 602 control, one arm each).

### 9.3 What these probes cannot resolve

Nine misses. A change that converts five or more is visible here; a one-or-two-paper
improvement is not distinguishable from the gate's own sampling noise, and will be
reported as unresolved rather than as a small win. Neither probe measures net@2 on a live
run — that is cohort 3's job (§8). A win here buys the right to spend $25 there, nothing
more.

---

### 9.4 RESULT — both candidates fail their own bars, and the cheap one fails informatively

Run 2026-08-18, ~$0.03 total, `evals/probe_score3_band.py`. Populations exactly as
pre-registered: 33 gate-3 papers (25 actionable), 61 shown papers (52 actionable).

### Probe A — fine-scale over the score-3 band: **below the bar, at zero cost**

| | measured | bar |
|---|---|---|
| ROC-AUC | **0.710** | kill < 0.60 · win ≥ 0.70 |
| misses dropped at P ≥ 2/3 | **2 of 8** | ≥ 4 |
| actionable lost | **0 of 25** | ≤ 2 |

It clears the AUC bar and fails the conjunction, so it does not qualify. Two things are
worth keeping.

**It ranks this band far worse than the one it was fitted on** — 0.710 against E2's 0.84 on
the score-2 band — and its *threshold* is in the wrong place here regardless: gate-3 papers
produce digit expectations of 7.7–9.0 almost uniformly, so the calibrated map says "show"
for 31 of 33. Re-fitting the threshold on these 33 papers is exactly the move
`calibrate_finescale.py` refuses by name — the threshold that scores best on a set is that
set's metric fitted to itself — and 8 misses cannot support a second parameter.

**The rescore is fooled where the gate is fooled.** MACE's own paper scores P = 0.926, the
joint-highest in the band. The two stages share the failure rather than correcting it,
which is the more useful finding: they are not independent votes.

**But it costs nothing.** 0 of 25 actionable papers dropped. Extending the rescore to the
score-3 band is a free +2 misses if it ships beside something that does the real work.

### Probe B — a rubric clause: **KILLED by its own kill clause**

| | measured | bar |
|---|---|---|
| misses no longer admitted | **8 of 9** | ≥ 5 |
| actionable lost | **14 of 52** | ≤ 3 · **kill > 5** |

The clause finds the shape it was written for — 8 of 9, including both self-papers the
citation rule misses and every application paper — and then takes a third of the good
papers with it. It cannot tell "uses the repository" from "extends the repository":

    3 -> 0  OpenMM 8: Molecular Dynamics Simulation with Machine Learning Potentials
    3 -> 0  A deep generative model for scRNA-seq with application to detecting DE genes
    2 -> 0  BLEND: fuzzy seed matches in genome analysis
    3 -> 1  MACE4IRmol · Systematic Fine-Tuning of MACE · Grappa · the SOAP paper

Every one of those *is* integrated with, or evaluated on, the repository it would improve —
which is what a paper proposing a change to a tool looks like.

**What shipping it would actually do**, on the same 61 papers:

| | shipped rubric | with the clause |
|---|---|---|
| shown | 61 | **39** (−36%) |
| actionable | 52 | 38 |
| precision | 0.852 | **0.974** |
| net@2 | **+34** | **+36** |

+2 net@2 over six repositories, **+0.33/repo against a live floor of 1.03** — a third of the
smallest effect this benchmark can resolve, bought by discarding a third of the digest. The
precision number is the seductive one and it is not the metric: net@2 already prices a shown
paper at 3p − 2, and the clause is paying two actionable papers for every three misses.

Per domain it is worse than the total suggests: matsci +12 → +20 (the intended fix), bio
**+22 → +16** (pure damage — bio had one miss in 25 and lost six actionable papers to catch
it). A change that helps one domain by hurting the other at a benchmark-invisible net is not
a fix, and shipping it on this evidence would be the "over-firing gate" mistake with the
sign flipped.

### The ML control arm was not run, and that saved $0.10

Pre-registered as the check that a clause fixing materials does not degrade the 25-repo
benchmark. The kill clause fired on the scientific-software set alone, so the control could
not have rescued it — spending on it would have been buying detail about a refuted arm.

### What this leaves

**Neither G1(i) nor G1(ii) ships on this evidence.** The remaining loss is 9 papers across
six repositories, and both candidate fixes cost more than they return. Recorded as such
rather than tuned until one passes: with 9 misses, any variant that "works" after three
attempts is fitting this set.

Three things the run does establish, which is more than the arms themselves are worth:

1. **The gate and the rescore fail together.** Any fix that treats the rescore as an
   independent second opinion on the gate's admits is building on that assumption, and it
   is false here (MACE's own paper: gate 3, P 0.926).
2. **The failure shape is detectable at 8/9** — the clause is a good *detector* and a bad
   *policy*. A narrower form that fires only when a paper proposes no change to the codebase
   (rather than whenever it uses it) is the obvious next arm, and would cost ~$0.02. It is
   **not** pre-registered here, and a third variant on nine labels starts to fit them; if it
   runs, it needs new bars and a hold-out, not a rerun of these.
3. **The honest per-repo statement for the demo is unchanged**: `dscribe` +10 at precision
   1.00, `chgnet`/`mace` still carry application papers, and no cheap prompt fix removes them
   without removing the papers a maintainer actually wants.

Nine misses. A change that converts five or more is visible here; a one-or-two-paper
improvement is not distinguishable from the gate's own sampling noise, and will be
reported as unresolved rather than as a small win. Neither probe measures net@2 on a live
run — that is cohort 3's job (§8). A win here buys the right to spend $25 there, nothing
more.

---

## 10. Profiler hygiene: nine fixes, a byte-identical benchmark, and four of my claims refuted (2026-08-19)

§6's D2/D3, designed by measuring against the 19 cloned repositories rather than by reading
the code. The design pass refuted more of the plan than it confirmed, which is recorded here
in full because the rejections are the more useful half.

### 10.1 What shipped

Nine fixes, each justified by a named repository, and all nine verified to leave the four
Tier A fixtures **byte-identical**:

| | repository | before | after |
|---|---|---|---|
| read `doc/`, `docs-source/`, one level down | phonopy | 13 of its top 20 keywords were its own dependency names | `supercell, displacements, cell, calculation, force` |
| exclude release notes and changelogs | scanpy | `smaller, pr, func` + two maintainer surnames | `scanpy, anndata, pp, tl, pl, experimental` |
| strip MyST and reST role names | scanpy, mdanalysis, tblite | `{func}`/`{pr}`/`{smaller}`, `mod`, `footcite` as keywords | role gone, target kept (`scanpy.pp.pca` survives) |
| strip reference-style badges and link definitions | scvi-tools | prose 100% `[![Stars][gh-stars-badge]]` markup | *"…is a package for probabilistic modeling"* |
| match rst substitution definitions | mdanalysis | `image` tied for keyword #1 | `mdanalysis, analysis, topology, core` |
| parse `setup.cfg` | MACE | 0 anchors, domains "general" | 19 anchors, `deep learning, scientific computing` |
| nested `package/` fallback | mdanalysis | 0 anchors | 42 anchors, 4 domains |
| strip `~=` specifiers | matminer | anchors `requests~`, `scikit-learn~` | correct names, and `machine learning` finally inferred |
| citations follow the same doc roots | dscribe | — | bibliography reachable wherever the docs live |

**Tier A, before and after: `P@10=0.867  R@10=0.439  nDCG@10=0.909  MRR=1.000  MAP=0.859
sep=+0.217`, webdev PASS.** Every digit, every case. That is now a guard rather than a
claim: `tests/test_profiler_golden.py` pins the keywords, anchors, domains, prose and
corpus phrases of all four fixtures, so the next profiler change has to state its intent.

### 10.2 Four things I asserted that the measurement refuted

- **"The bioconda world needs `environment.yml`, conda `meta.yaml`, R `DESCRIPTION`,
  `Cargo.toml`, `Project.toml`."** Zero of the 19 clones has any of them at the repository
  root. All five parsers dropped.
- **"Stopping `index`, `reference`, `parameters`, `system`, `thin`, `build` costs the
  domain terms of art — *thin film*, *lattice parameters*, *k-mer index*."** Measured over
  the corpus: `thin` occurs **0 times**; `bin` appears 10 times and every collocation is
  shell furniture (`alias bin`, `neighbor bin`); `build` (126) and `system` (270) are
  dominated by *to build*, *the system*. Removing any of them measured harmful or empty.
  Not done.
- **"`setup.cfg` extras should come too."** On MACE `[options.extras_require]` adds pytest,
  black, isort, mypy and pre-commit — and `pytest` becomes the repository's **top keyword
  and an arXiv query**. Only `install_requires` is read.
- **"Reading `docs-source/` gives openmm its citations back" (§9.0).** It does not. With
  `docs-source/` now read, openmm still yields zero cited ids, because it cites no arXiv
  papers in its docs at all — the two number-like strings there are a DOI fragment with
  month 21 and a number inside an SVG logo, both correctly rejected. §9.0's implication was
  wrong; the directory omission was real, the consequence I attached to it was not.

### 10.3 Two fixes deferred, with the measurement that deferred them

- **A6, the prose lead** (MACE's prose is its table of contents). The proposal was measured
  to have two defects: a README opening with a feature bullet list has the bullets silently
  deleted — one of the commonest ML-repo shapes — and YAML front matter costs the title. It
  also cannot be validated for free: the offline gate is **prose-blind** (with A6 applied it
  reproduces the baseline to the digit), and prose's real consumer is
  `triage.repo_context_block`, which feeds the frozen fine-scale logistic. Needs paid
  measurement, so it waits for cohort 3.
- **B4, dropping anchor-only bigrams** (phonopy's `seekpath pypolymlp`). It **regresses the
  committed gate**: nDCG@10 0.909 → 0.902, MAP 0.859 → 0.849, and the webdev control's
  `mean_top10` rises 56%. The defect is real and universal — `rag`, `cv`, `rl` and `webdev`
  all carry dependency-name bigrams — but the proposed fix costs more than it returns on
  the only population that can currently score it.

### 10.4 The debt this creates, stated plainly

These changes alter the profile, therefore the queries, therefore the candidate pool. **Every
published benchmark number in this project describes the pre-2026-08-19 profiler.** Tier A
proves the four ML fixtures do not move; it says nothing about the 25 benchmark repositories,
whose live pools nobody has re-fetched, and nothing about whether scientific-software digests
improve — the whole point of the work. Cohort 3 (§8) is where both get measured, and it must
now run *after* these fixes rather than before them.

**§11 settles the first half of that for free: 16 of the 25 moved, and the diff caught two
defects these very fixes introduced.** The debt is confirmed, quantified, and one round of
repair the wiser.

Known and not fixed: `docs/_sources/` is a committed Sphinx HTML build (216 of dscribe's 246
doc files, 30 of matminer's 30), so those two repositories read generated duplicates of prose
they already read. Excluding it would leave matminer with a 2-document corpus, which is its
own unmeasured change.

---

## 11. The blast radius of §10, measured — and the two defects it introduced (2026-08-19)

§10.4 recorded a debt in words: the nine fixes "say nothing about the 25 benchmark
repositories." That is free to settle, because §10 touched exactly one source file. Profiling
every benchmark clone with `profiler.py` at `f4b6da4` and at `29f5edc` and diffing the two
answers the question that decides when cohort 3 can be paid for.

### 11.1 How much of the benchmark moved

| field | cases moved |
|---|---|
| keywords | **16 / 25** |
| corpus_phrases | 14 / 25 |
| prose | 2 / 25 |
| anchors | 0 / 25 |
| domains | 0 / 25 |

So the debt is real and now has a number. The movement is concentrated on the **query side**:
keywords become arXiv queries, so a moved keyword list is a moved candidate pool, and no live
figure in this document describes the shipping profiler. Anchors and domains did not move at
all — the manifest fixes were genuinely confined to repositories that had the defect.

The improvements are real too. thin-gnn shed seven dependency-name keywords (`orbax
frozendict`, `reportlab mmh3`, `pyvis tensorflow`, `networkx pyarrow`) for its actual subject
(`dgf, beam, jax, sampling, transform`); scipy's top keyword went `pooch` → `scipy`.

### 11.2 What the diff caught that the 19 clones could not

Two defects, both of them §10's own fixes firing on repositories §10 never scanned.

**LaTeX became vocabulary.** Reading `doc/` trees let mathematics into the corpus for the
first time. A backslash is not a word character and `token_pattern` opens a token at a
letter, so every control sequence donates its *name*. scipy's manual carries 2379 `\left`,
2379 `\right` and 1228 `\frac`, and `left`, `right`, `frac` and `eqnarray` became four of its
top-twenty arXiv query keywords.

**The history exclusion list was measured on the wrong corpus.** `_NON_TOPIC_DIRS` was fitted
to the 19 bio and materials clones and shipped. scipy spells it `doc/source/release/` (81 of
its 431 doc files) and numba `docs/source/release/` (17), with numba's towncrier fragments in
`upcoming_changes/`. `release-notes` matched none of it, so `bug`, `fix` and `maint` sat in
scipy's top-twenty and `pr` in numba's top three — the exact defect §10 fixed for scanpy,
surviving under a different directory name on the cases every published number comes from.

### 11.3 The fix I implemented first, and the measurement that rejected it

The obvious repair is to sweep `\[a-zA-Z]+` out of the text. It was implemented, and it is
wrong. Auditing what that pattern actually deletes across all 25 clones:

| clone | hits | what they are |
|---|---|---|
| numerics (scipy) | 13712 | `\left`, `\right`, `\frac`, `\mu` — mathematics |
| llminfer (llama.cpp) | 243 | `\Intel`, `\bin`, `\Program`, **`\llama`**, `\nWhat` — Windows paths and C escapes |
| crypto | 411 | `\x`, `\xe`, `\xd` — hex escapes in byte literals |
| linter (ruff) | 6 | `\ruff`, `\AppData`, `\Roaming`, `\Scripts` — paths |

It deleted `llama` from llama.cpp's own install instructions twelve times, and `ruff` from
ruff's. Measured the other way round — what fraction of macros sit inside a context that
*says* "this is mathematics" — the answer is unambiguous:

| clone | macros inside marked math |
|---|---|
| pytorch-geometric | 262 / 262 (100%) |
| scipy | 12909 / 12972 (99.5%) |
| dscribe | 1537 / 1545 (99.5%) |
| peft | 51 / 56 (91.1%) |
| **llama.cpp** | **0 / 243 (0%)** |
| **ruff** | **0 / 6 (0%)** |

So the rule strips the math **context**, never the macro, and the repositories whose
backslashes are paths are left completely alone. Removing the context is also better on its
own terms: `\begin{eqnarray}` leaves no `eqnarray` behind, and `\frac{a}{b}` leaves no stray
single letters.

Each of the three context forms had to earn its place, by marginal contribution over the
corpus: `.. math::` blocks 12298 macros, `:math:` roles 2388, `$…$` spans 53. A MyST
```` ```{math} ```` fence matches **zero** occurrences and is deliberately not implemented,
on the same rule §10 used for `releases`/`history`/`whatsnew`.

The `$…$` form needed a guard of its own. Of 507 dollar spans in the corpus, **454 are shell
and Make** — `$VERSION $`, `${ARROW_ROOT}$`, `$(CXX) -o $` — and an unguarded pattern matches
from one variable to the next and eats the prose between them. All 53 genuine ones carry a
control sequence, so carrying one is the discriminator.

### 11.4 What shipped, and what it did

Three changes: math contexts stripped, `_NON_TOPIC_DIRS` given `release` and
`upcoming_changes`, `_NON_TOPIC_STEMS` given `release`. Additions to the exclusion lists are
again only names that FIRE on the corpus.

Tier A is **byte-identical** — `tests/test_profiler_golden.py` passes unchanged, so all four
ML fixtures profile exactly as they did before §10 and after it.

Against the post-§10 profiler, 6 of 25 cases move, and the collateral damage of the rejected
sweep is gone (llama.cpp, ruff, crypto, requests, cli and webdev are now untouched):

| case | effect |
|---|---|
| numerics (scipy) | `left, right, frac, eqnarray, bug, fix, maint` all gone → `distribution, stats, implementation, shape, parameter, minimize, random` |
| compiler (numba) | `pr` gone → `compiler, compilation, value` |
| columnar (arrow) | one tail swap, `ipc` → `files` (its `developers/release.rst` is process, not topic) |
| diffusion, peft, rl | rank shuffles inside the top twenty, no membership change |

### 11.5 One question this deliberately does not answer

crypto's `:mod:` targets now tokenize into `cryptography hazmat`, `hazmat primitives` and
`primitives serialization`, which displaced `signature`, `public`, `certificate`, `security`
and `message` from its top twenty. That is §10's keep-the-API-name rule working exactly as
designed, and on this one repository it costs five topic words. It is 1 of 25 cases, the
alternative (keep only the leaf of a dotted path) has no free way to be validated, and the
offline gate cannot score it. Left alone, recorded here, and put to cohort 3.

---
## 12. The non-arXiv paper is a second-class citizen, in three places (2026-08-19)

§8's item 3 — B4, B2, B3 from §6, taken together because they are one defect wearing three
hats. Everything downstream of collection was written when arXiv was the only source, so each
stage treats a non-arXiv record as a malformed arXiv one: the collector filters it out, the
ranker docks it a third of its score, and the id layer lets it in three times.

### 12.1 B4 (F2) — the OpenAlex source was filtering out preprints

`filter=type:article` excludes every preprint, and OpenAlex has counted `preprint` as its own
type since 2024. So the one source whose purpose is literature arXiv does not carry was
excluding the preprint servers.

Probed live 2026-08-19 over six bio and materials queries, **with a date filter, which is how
the pipeline uses it** — the all-time relevance ranking hides this almost completely (7 of 200
results), and the recent windows are where preprints live:

| window | pool | preprints | share |
|---|---|---|---|
| last 30 days | 300 | 80 | **26.7%** |
| last 180 days | 300 | 66 | 22.0% |

The venues the wider filter brings are exactly the ones this work needs: bioRxiv (26 in the
30-day probe, 36 in the 180-day), arXiv (32), ChemRxiv (9), Research Square (8), medRxiv.
**With `sources/biorxiv.py` broken by construction (B1), this one line is currently the only
wired route to bioRxiv in the product.**

It substitutes rather than adds — `per_page` is unchanged, so a quarter of the recent pool
becomes preprints instead of the pool growing by a quarter.

### 12.2 B4, second half — the keyword sources saw five queries

`pipeline.py` sent `queries[:5]` to every non-arXiv source. Measured over the 25 benchmark
repositories: they build a mean of 7.0 queries each, and the cap **withheld 50 of 175
(28.6%)**. Raised to 8, the most any of them produces, so every benchmark query now goes out;
raised rather than removed, because the cap is a real bound — one HTTP request per query per
enabled source, and a keyless Semantic Scholar caller is the first to see 429s.

Both stage-1 yield probes carried their own `queries[:5]`. A probe that sends fewer queries
than the product understates the channel it exists to judge, which is the C-9/C-12 shape
inside the measuring instrument, so they now import the product's constant. **Their published
yields in `RESULTS.md` were measured at 5 and a re-run will not reproduce them.**

Not done, and recorded here instead: the truncation is a *prefix*, and `build_queries` emits
seeds, then bigram phrases, then single keywords — so the queries dropped first are the plain
ones, and `queries.bigrams` already documents that phrase queries are safe on arXiv *because*
a category clause catches a meaningless phrase, and that keyword sources have no such
fallback. Changing the order changes retrieval; that needs a measurement, not an opinion.

### 12.3 B2 (F4) — a paper was ranked by which adapter found it

`score_paper` asked `if paper.get("categories")` — truthiness. Three of the six non-arXiv
adapters populate that field from a *different* taxonomy: OpenAlex its `primary_topic` display
name (`Machine Learning`), bioRxiv its subject area (`Bioinformatics `, trailing space
included), DBLP the venue key. None can intersect a list of arXiv categories under any
config, so the match is a guaranteed 0 averaged in at full `w_category`.

`ranking.absent_category` exists precisely to decide how a paper with no category signal is
treated — and those three adapters routed around it. The same paper at keyword 0.6:

| found by | category field | score |
|---|---|---|
| arXiv | `q-bio.GN` (matches) | 0.733 |
| Semantic Scholar | empty → `absent_category` | 0.600 |
| bioRxiv / OpenAlex | foreign taxonomy → hard 0 | **0.400** |

The 0.33 gap is not a judgement about the paper; it is the taxonomy its adapter happened to
read. So the question asked is now "are these categories in the vocabulary the target list is
written in", against `config.KNOWN_ARXIV_PREFIXES`, and a paper that fails it joins the rest of
the non-arXiv pool on the `absent_category` path. bioRxiv and OpenAlex move 0.400 → 0.600.

Two things deliberately not done. **Blanking `categories` in the three adapters** is the
audit's own suggested fix and is worse: the field is exported to the digest CSV, so blanking
destroys information a user can see to diagnose a ranking bug they cannot; and it puts one
invariant in three adapters plus every adapter written later — Europe PMC (B1) is next, and it
should be impossible for it to get this wrong. **Forgiving a non-match**: an arXiv paper in
`cs.CL` against a `q-bio.GN` target still scores 0, because that is evidence. A rule that
forgave it would pass every test above and destroy the component.

### 12.4 B3 (F15) — one preprint, three ids

Every adapter minted an id from its own API's handle: `oa:W4392847362`,
`ss:649def34...`, `biorxiv:10.1101/...`, `dblp:conf/vldb/X`. Three of them return the same
preprint, so it entered the pool three times, was gated three times (three API calls) and
could occupy three slots of a ten-paper digest.

A DOI is the identifier all of them already agree on, so when one is known it is the id —
lowercased, because DOI names are case-insensitive by specification and the sources disagree
in practice. That is the same trap `_extract_arxiv_id` fell into once for arXiv DOIs.

Arriving at three adapters, one preprint now yields one id. Sources with no DOI keep their
synthetic ids: DOI-first is not DOI-only.

**Two things this broke, both caught before merge and both the same shape** — code that
identified an arXiv paper *by exclusion*, which is correct exactly until the next id scheme:

- `sources/semantic_scholar.py` built an abstract URL for any id not starting with `ss:`. It
  would have emitted `arxiv.org/abs/doi:10.1101/...`. Now `paper_id.is_arxiv_id`, which also
  replaces the copy of the two arXiv id eras that `s2_recommendations.py` kept.
- `evals/openalex_yield.py` and `evals/s2_yield.py` counted non-arXiv papers as ids beginning
  `oa:`/`ss:`. Both would have reported their channel delivering ~nothing while it delivered
  the same as before. A measurement instrument that reads zero because the thing it measures
  was renamed is worse than no instrument.

**Not migrated.** A store holding `oa:W...` rows will see those papers once more under their
DOI ids. `biorxiv:<doi>` could be rewritten mechanically and the other two cannot, and a
migration that fixes one source of three is worse than none. The blast radius is small by
construction: every non-arXiv source is off by default, and §6 records that the behaviour of a
non-arXiv paper *inside* a digest has never been observed.

### 12.5 What is still not measured

All of it. Tier A is byte-identical (its fixtures are arXiv papers with arXiv categories), which
proves these changes do not disturb the ML benchmark and says nothing about whether they
improve a scientific-software digest. Every number here is an API probe or arithmetic, not a
judged outcome. **The non-arXiv channel still has no judged result of any kind** — that is
cohort 3, and this is the third of the changes that must land before it runs.

---
## 13. B1 — a bioRxiv source that can be searched, and the four things the spec got wrong (2026-08-19)

§6's B1. The spec was written from Europe PMC's documentation; every clause of it was checked
against live responses before the adapter was written, because §11 and §12 each turned up a
defect that only a probe found. **The probe changed four things.** Total cost $0, 0 hard
failures once requests were spaced.

### 13.1 What the spec got right

The query works exactly as written:

| clause | hits, `single cell rna sequencing` |
|---|---|
| keywords alone | 92,091 |
| `AND SRC:PPR` | 8,186 |
| `AND PUBLISHER:"bioRxiv"` | 5,932 |
| `AND (PUBLISHER:"bioRxiv" OR PUBLISHER:"medRxiv")` | 6,378 |
| `AND FIRST_PDATE:[2026-01-01 TO 2026-12-31]` | 1,005 |

`resultType=core` carries `abstractText`, `pageSize=100` is served, and the publisher clause
is doing real and correct work: the 1,808 preprints it drops are Research Square (10.21203),
Preprints.org (10.20944), F1000 (10.12688) and Authorea (10.22541) — none is bioRxiv or
medRxiv. That is the intended scope, but it *is* a choice, so it lives in one named constant.

### 13.2 Correction 1 — the keywords must not be quoted, and quoting is silent

The spec says `(<plain keywords>)`, unquoted. It is right, and getting it wrong would have
emptied the channel without any error. A quoted string is an exact-phrase match, and what
this adapter receives from `collector.to_plain_keywords` is a bag of words:

| product-shaped query | quoted | unquoted |
|---|---|---|
| sequence alignment long reads | **0** | 2,239 |
| molecular dynamics gpu simulation | **0** | 148 |
| rna velocity trajectory inference | **0** | 85 |
| protein language model structure | 1 | 1,099 |
| genome assembly nanopore | 1 | 1,204 |
| cryo em particle picking | 4 | 489 |
| spatial transcriptomics deconvolution | 7 | 263 |
| single cell variational inference | 17 | 206 |

This is C-9 inverted — that defect made bioRxiv return *everything* because the surviving
query word was `AND`; this one would have made Europe PMC return *nothing*, and a source that
returns nothing looks exactly like a source that is switched off.

### 13.3 Correction 2 — titles carry markup, not just abstracts

The spec says "strip HTML from `abstractText`". Over 785 sampled records, **18% of titles**
carried `<i>` (202) or `<sup>` (4), and 36% of abstracts carried `<h4>` (764). A title like
`Mapping  <i>trans</i>  -eQTLs at single-cell resolution` is what the gate reads.

### 13.4 Correction 3 — the obvious way to strip it destroys the abstract

`<[^>]+>` is the pattern anyone writes, and biology abstracts are full of `p < 0.001`. That
`<` opens a span the regex closes at the **next real tag**, taking every character between.
Measured over the same 785 records: the naive pattern removes 9,277 characters where a
tag-shaped pattern removes 5,367 — **3,910 characters of real abstract**, and the abstract is
what the gate, the ranker and the embedder read. One record lost 240 characters, its entire
results sentence:

> `…discrimination of SNPs from sequencing errors (t = 14.80, p ` **[240 chars gone]** ` Availability  Source code…`

Requiring a tag *name* — a letter first, then word characters, no spaces before the close —
strips all 1,246 real tags in the sample and spares all 10 of the non-tags, which are
`p < 0.001`, `<<1% have tri-kinetochores`, `<Choloepus didactylus>`, `<1% Wolbachia reads`,
`<3 years survival`. Whitespace is collapsed afterwards: the markup is padded, and 91 of 400
titles were left with a double space by tag removal alone.

### 13.5 Correction 4 — the URL and the id

The spec says id `biorxiv:<doi>`. **B3 superseded that**: `paper_id.doi_key` gives
`doi:<doi>`, so a preprint arriving from Europe PMC and from OpenAlex is one paper rather
than two. Both prefixes are live and both normalise identically — a 785-record sample held
10.1101 (284, the original) and 10.64898 (216, bioRxiv's current one).

The URL is `https://doi.org/<doi>`, not `biorxiv.org/content/<doi>` as `sources/biorxiv.py`
builds: the response cannot say which server a record came from (`publisher` and `pubType`
are **null** under `resultType=core` even though `PUBLISHER:` filters correctly), and the
10.64898 dois do not use that path shape.

### 13.6 What else the probe settled

- Every one of 785 records had a DOI, an abstract and a `firstPublicationDate`. No record
  needed dropping for missing fields.
- Europe PMC returns **no subject classification** for preprints, so `categories` is left
  empty — which, after §12.3, is the right thing: an empty list takes `ranking.absent_category`
  rather than introducing a fourth foreign taxonomy.
- An honest miss is `hitCount: 0` with a `resultList` present, so emptiness is
  distinguishable from refusal. `_request_json` therefore **raises** instead of returning
  `[]` like every other adapter in the package, and `collect_papers` raises if *every* query
  was refused — a caller must not be able to record "bioRxiv contributed nothing" about a
  conversation that never happened.
- A burst of six unspaced requests drew 504s and then a 503; 22 consecutive spaced ones
  completed clean, as did 8 more later. The flakiness is real and transient, so retry with
  backoff is mandatory and 4xx is not retried.
- The date clause is only added for windows ≤ 365 days. The measured configuration runs
  `lookback_days: 36500`, and asking for everything since 1926 is a slower way to ask for
  everything.

### 13.7 Shipped as a new source, not as a replacement

`sources/biorxiv.py` is untouched and `sources: [biorxiv]` still means what it meant. Silently
repointing an existing config value at a different API with different coverage is a change a
user cannot see. Instead `europepmc` is a new source, and `validate_config` now warns when
`biorxiv` is enabled without it, naming the defect: bioRxiv's endpoint is a date-interval
listing, so under an all-time lookback it returns the oldest postings in the window rather
than papers about the repository.

**No email is sent.** Europe PMC accepts one as politeness, and the obvious source for it is
`openalex.email` — but that address was given to this project for OpenAlex's polite pool, and
forwarding it to a second service is a data flow the user did not agree to. The privacy
registry entry records that this source sends repo-derived query strings and no email, which
is a higher sensitivity than the `sources.biorxiv` entry above it precisely because this one
can search.

### 13.8 Still not measured

Everything that matters. Tier A is byte-identical and cannot be otherwise — it is arXiv
fixtures and this adds a non-arXiv source. **No bioRxiv paper has ever been judged by this
project**, so whether the channel improves a digest is unknown, and §6's B5 stands: the
honest demo statement remains "bioRxiv is a source we are adding; the arXiv channel is the
one we have measured". Cohort 3 is where that changes, and B1 was the last change that had to
land before it.

---
## 14. PRE-REGISTERED — cohort 3, and the re-baseline it is bundled with (2026-08-19)

Written **before the first paid call**, like §9, so no bar can be chosen after seeing the
answer. Everything below — the twelve repositories, their profile strata, the endpoints, the
predictions and the kill bars — was fixed while the spend was still zero.

### 14.1 The question, and why it cannot be deferred again

Two questions, and the run answers both because they need the same session:

1. **Does RepoRadar work on non-ML scientific software?** Six live runs (§5) said "on three of
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
`doc`-less repositories where RepoRadar is *known* to be weakest — install boilerplate as
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

- **WIN** — scientific-12 mean **≥ +4.0** and pooled precision **≥ 0.80**. "RepoRadar works on
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
## 15. RESULT — cohort 3 clears its bars, and the misses are where the profile is bad (2026-08-20)

The twelve scientific cases of §14, run at the pre-registered configuration on 2026-08-20.
Artifact `evals/results/judge-gpt-5.5-frozenpool-bigrams_verified-20260820T060917Z.json`.
203 GPT-5.5 verdicts, 12 Opus baseline passes, **0 judge failures, 0 hallucinations, 0
abstentions, no HyDE degradation, no collection failure**. Estimated ~$18 from call counts,
inside §14.10's $15–36. The legacy 25 are §16; read §16.3 before quoting the primary, because the power analysis §14.6 declared was optimistic by roughly 2.5x.

### 15.1 The primary endpoint, against the bar it was given

| | measured | pre-registered |
|---|---|---|
| **scientific-12 mean net@2** | **+5.33** | +4.0 to +6.0 |
| **pooled precision** | **0.857** | ≥ 0.80 |
| **verdict** | **WIN** | ≥ +4.0 and ≥ 0.80 |

The Opus baseline over the same twelve cases means **+1.33**. RepoRadar is **+4.00 net@2 above
the strong baseline** on scientific software, which is the comparison that makes this a claim
rather than a description.

| case | net@2 | returned | actionable | precision | baseline | stratum |
|---|---|---|---|---|---|---|
| bio-scvi | +11.0 | 11 | 11 | 1.00 | +0.0 | polluted |
| mat-descriptors | +9.0 | 12 | 11 | 0.92 | +1.0 | clean |
| mat-phonon | +9.0 | 12 | 11 | 0.92 | +2.0 | clean |
| mat-chgpot | +8.0 | 11 | 10 | 0.91 | +5.0 | clean |
| bio-mdsim | +7.0 | 7 | 7 | 1.00 | +3.0 | polluted |
| bio-mdtraj | +7.0 | 7 | 7 | 1.00 | +1.0 | clean |
| bio-singlecell | +5.0 | 8 | 7 | 0.88 | +0.0 | clean |
| mat-toolkit | +4.0 | 4 | 4 | 1.00 | +0.0 | polluted |
| bio-kmer | +3.0 | 3 | 3 | 1.00 | +3.0 | clean |
| mat-mlip | +1.0 | 13 | 9 | 0.69 | +2.0 | polluted |
| bio-align | **+0.0** | 12 | 8 | 0.67 | −2.0 | **defective** |
| mat-featurize | **+0.0** | 12 | 8 | 0.67 | +1.0 | **defective** |

### 15.2 Every prediction, scored

Six of seven held. The one that missed, missed upward.

| prediction | predicted | measured | |
|---|---|---|---|
| scientific-12 mean | +4.0 to +6.0 | +5.33 | ✓ |
| pooled precision | ≥ 0.80 | 0.857 | ✓ |
| bio-6 | ≥ +5.0 | +5.50 | ✓ |
| **mat-6** | **+2.5 to +4.5** | **+5.17** | **✗ high** |
| clean stratum | ≥ +5.0 | +6.83 | ✓ |
| defective stratum | ≤ +2.0 | +0.00 | ✓ |
| failure mode is the score-3 band | — | confirmed, §15.4 | ✓ |

**Why mat-6 beat its band, stated carefully.** The prediction was anchored on §5's +3.00, which
was three cases on one draw with a per-domain sd of **5.3** — an anchor that could not have
supported a ±1 band, let alone been missed by one. The two matsci cases that failed there both
improved: CHGNet +1 → +8, MACE −1 → +1. It is tempting to credit the profiler and digest work
of §10–§13, and that may be right, **but this is a different draw of a different pipeline and
nothing here separates the two**. A causal claim would need the old pipeline re-run on the same
frozen pool, which was not done. The honest reading is that the miss says as much about the
anchor as about the improvement.

### 15.3 The strata ordered exactly as declared

| stratum | n | mean net@2 | pooled precision |
|---|---|---|---|
| clean | 6 | **+6.83** | 0.925 |
| polluted | 4 | +5.75 | 0.886 |
| defective | 2 | **+0.00** | 0.667 |

The ordering is monotone and the gap between clean and defective is 6.83 net@2. The defective
stratum is n=2 and §14.6 said in advance that it resolves nothing; it is reported as
description. What it describes is stark: both cases returned 12 papers, both had 8 judged
actionable, both scored exactly 0.0, and both are the two repositories whose profiles do not
describe the project — minimap2, whose 300 characters of prose are a phishing warning, and
matminer, whose top keywords are the Sphinx module index and the test tree.

### 15.4 Two separable causes of the sixteen misses

112 papers were returned across the twelve cases and 16 were judged non-actionable. They are
not spread evenly, and the two concentrations are independent of each other.

**By gate score — §14.7's predicted failure mode, confirmed at n=12:**

| gate score | actionable | non-actionable | share non-actionable |
|---|---|---|---|
| 2 | 71 | 5 | **7%** |
| 3 | 25 | 11 | **31%** |

Score-3 papers are non-actionable **four and a half times as often** as score-2 papers, and 11
of the 16 misses are score-3. §5.1 found 11 of 12 on six cases; this is the same signature with
twice the cases behind it. §9.4 killed both candidate fixes against their own bars, so this
remains a known, measured, unrepaired weakness rather than a new discovery.

**By profile quality:** the two defective-profile cases supplied **8 of the 16 misses from 24 of
the 112 returned papers** — a 33% miss rate against roughly 5% everywhere else.

So the misses have a *where* and a *which*: they concentrate in repositories the profiler
cannot describe, and within any repository they sit in the gate's score-3 band.

### 15.5 An observation that is NOT a result

`--rr-sweep` re-gates a wider candidate pool at each `min_actionable` threshold. At `min>=2`
that selection means **+6.58** against the shipped Top Picks' +5.33 — but it returns more
papers from a 20-candidate rerank pool and is **not the shipped configuration**. The difference
is +1.25, barely over the 1.04 floor, and it was not pre-registered.

**§18.3 corrects the sentence above.** The pool is *not* wider: `sweep_top_picks` filters the
same ranked window on the same triage scores, so `min>=2` is the shipped Top Picks before the
fine-scale stage runs — verified identical to the recorded arm on 37 of 37 cases. The arm
isolates exactly one stage. The disposition here is unchanged and now rests on better reasons
(§18.2): post-hoc selection, one draw, and a judge-swap check the "+1.25" reading fails.

It is recorded here as a candidate for a future pre-registration and **must not be quoted as a
result of this run**. Reading a sweep arm as a headline after the fact is exactly the shape §14
exists to prevent.

### 15.6 What this does and does not license

**Licensed.** "RepoRadar returns papers that a neutral judge calls genuinely actionable for
non-ML scientific software, at +5.33 net@2 and 0.857 precision over twelve repositories,
against +1.33 for an Opus baseline on the same repositories" — provided §14.2's exclusion is
said in the same breath.

**Not licensed, and each was declared in advance:**
- **The comparison with ML/CS.** The legacy 25 have not been re-measured. There is still no
  number in this project describing the current pipeline on the ML benchmark.
- **The population.** htslib, kallisto, tblite, kim-api and LAMMPS were excluded, and they are
  where this is known to be weakest. This cohort is optimistic about scientific software in
  general.
- **A second draw.** Every number here is one draw. §5's matsci values (+1, −1, +9) are the
  standing warning about reading individual cases.
- **Any causal claim for §10–§13.** See §15.2.

---
## 16. RESULT — the re-baseline, a power analysis I got wrong, and four silent misses (2026-08-20)

The legacy 25 at the §14 configuration, into the same frozen pool as the twelve. Artifact
`judge-gpt-5.5-frozenpool-bigrams_verified-20260820T172033Z.json`. 25 cases, **0 judge
failures**, every baseline ok. With §15 this gives the project its first current number for its
own benchmark since the profiler work.

### 16.1 The alarm is clear

§14.8 declared a separate alarm: a legacy-25 mean below **+3.5** would mean the profiler work
cost the ML benchmark something Tier A — four hand-written fixtures against a frozen pool —
structurally cannot see.

**legacy-25 = +5.88.** Clear, with room. The nine profiler fixes of §10 and the two repairs of
§11 did not cost the ML benchmark anything detectable. The old published figure was +5.12 over
25 cases, so this is +0.76 higher — **inside the 1.04 floor, and it describes a different
pipeline anyway, so it is context and not a control.**

### 16.2 The pre-registered endpoints

| endpoint | n | mean net@2 | sd | precision | pre-registered |
|---|---|---|---|---|---|
| scientific-12 | 12 | +5.33 | 3.75 | 0.857 | +4.0..+6.0 ✓ |
| legacy-25 | 25 | **+5.88** | 5.08 | 0.914 | +4.0..+6.5 ✓ |
| bio-6 | 6 | +5.50 | 3.78 | 0.896 | ≥ +5.0 ✓ |
| mat-6 | 6 | +5.17 | 4.07 | 0.828 | +2.5..+4.5 ✗ high |
| **ALL-37** | 37 | **+5.70** | 4.64 | **0.894** | the new headline |

**scientific-12 − legacy-25 = −0.55** (SE 1.48). **No difference is established.** For the demo
question that is the answer worth having: on this benchmark, scientific software is not
measurably worse than the ML/CS repositories the project was built on. It is equally true that
a gap of two or three net@2 in either direction would not have been detected — see §16.3.

Against the Opus baseline, which is what makes any of this a claim:

| | baseline | RepoRadar | delta |
|---|---|---|---|
| scientific-12 | +1.33 | +5.33 | **+4.00** |
| legacy-25 | +0.64 | +5.88 | **+5.24** |
| all-37 | +0.86 | +5.70 | **+4.84** |

### 16.3 CORRECTION — §14.6's power analysis was wrong, and optimistic

§14.6 built its resolution table on a per-case sd of **1.73**, taken from §8. The measured sd is
**3.75** on the scientific cases, **5.08** on the legacy ones, **4.64** over all 37 — between two
and three times the assumption. Every "resolves" figure I declared was therefore too small:

| quantity | §14.6 said | actually |
|---|---|---|
| a domain mean (n=6) | ~1.4 | **~3.0** |
| the scientific-12 mean | ~1.0 | **~2.1** |
| scientific-12 vs legacy-25 | ~1.2 | **~2.9** |

Two consequences, and I would rather state them than let a reader find them.

**The primary endpoint's interval includes sub-bar values.** scientific-12 is +5.33 with a 95%
interval of **[+3.21, +7.45]**, whose lower bound is below the +4.0 WIN bar. The rule declared in
§14.8 was a bar on the mean, the mean cleared it, and §15's WIN stands as written — but a
replication could land below the bar, and the result should be read as "cleared the declared
bar on one draw", not as "established with margin".

**"No difference established" in §16.2 is a weak statement, not a strong one.** With SE 1.48 the
interval on the −0.55 gap spans roughly ±3 net@2. It rules out a *large* penalty for scientific
software. It does not rule out a moderate one.

Where 1.73 came from matters for not repeating this: it is the figure §8 quoted for the
*within-domain* spread of a three-case draw, and I used it as the between-case sd of a
37-case benchmark. Those are different quantities. `noise_floor.py` measures the right one —
residual per-case sd **1.23** for a *paired same-config re-run*, which is the variance of a case
against itself, not of cases against each other. A future pre-registration should take the
between-case sd from a completed run, which now exists.

### 16.4 Four cases returned nothing, and all four had good papers available

The metric is abstention-aware, so returning nothing scores 0 and looks harmless in a mean.
These four are not harmless:

| case | returned | actionable papers in its pool | pool |
|---|---|---|---|
| linter | **0** | **9** | 16 |
| webdev | 0 | 3 | 15 |
| http | 0 | 2 | 15 |
| cli | 0 | 1 | 15 |

`pool_has_relevant` is **true for all four**. **§17 corrects this paragraph**: the four are
three different things, http and cli are close to correct abstentions, and the strong claim
made about linter below does not survive a second judge. Read §17 with this section. ruff's case had **nine actionable papers sitting in the pool and returned none of
them** — the single largest recall failure in the run, and it is invisible in a +5.88 mean
because a zero is exactly what a correct abstention also scores.

Two further cases returned a single paper (`compiler`, `encryption`, both +1.0 at precision
1.00), which is the same shape a step less severe.

This is a **selection** failure, not a retrieval one: the pool had the papers. Nothing in §14
predicted it and nothing here diagnoses it. **`rr why` (§6 D7) was built next for this reason**
and now names the stage that claimed any given paper — read-only, offline, free. Applied to
these four it would say whether the gate scored them 0, whether `triage.rerank` demoted them
out of the window, or whether they never reached the gate at all; those are different defects
with different fixes and the metric cannot tell them apart.

### 16.5 The score-3 problem is specific to scientific software

Not pre-registered, and the most interesting thing in the run.

| | score-2 non-actionable | score-3 non-actionable |
|---|---|---|
| scientific-12 | 5/76 (7%) | **11/36 (31%)** |
| legacy-25 | 15/168 (9%) | **2/30 (7%)** |

Fisher exact on the score-3 rows: **p = 0.027**. On ML/CS repositories the gate's score-3 band
is as trustworthy as its score-2 band. On scientific software it is four times worse.

This reframes G1 (§6). The gate is not globally over-admitting at score 3 — §5.1 and §15.4 saw
it on scientific repositories, and this run shows it does not happen on the ML ones. It is a
**domain-calibration** failure, which is a different problem with different fixes: the rubric and
the fine-scale map were both fitted on ML abstracts, and §9.4's two candidate repairs were
scored only on the scientific band. A repair aimed at the gate globally would be solving a
problem that only exists in half the benchmark.

Recorded as an observation. It needs its own pre-registration before anything is built on it.

### 16.6 What the project can now say

- **A current headline**: **+5.70 net@2 at 0.894 precision over 37 cases**, against **+0.86** for
  an Opus baseline. Every published figure before today described the pre-2026-08-19 profiler;
  this one describes what ships.
- **The profiler work cost the ML benchmark nothing measurable** (§16.1).
- **Scientific software is not measurably worse than ML/CS** (§16.2), with §16.3's caveat that
  the instrument could not have seen a moderate gap.
- Still unlicensed: the excluded compiled/manifest-less population (§14.2), a second draw, and
  any causal claim for §10–§13.

---
## 17. CORRECTION — §16.4's four cases are three different things, and one of my claims does not survive a judge swap (2026-08-20)

`rr why` (§6 D7) was built to answer §16.4, and the first thing it produced was a correction to
how §16.4 was written and to what I said about it.

### 17.1 The four cases are not one failure

§16.4 reported "four cases returned nothing and all four had good papers available" and treated
them as a single defect. They have three different causes, and only one is worth acting on.

| case | gate scores over its 15 ranked papers | judge ≥2 | what happened |
|---|---|---|---|
| linter | `{0: 1, 1: 14}` | 8/15 | **nothing reached the threshold** |
| webdev | `{0: 2, 1: 11, 2: 2}` | 3/15 | two reached the band; both failed the fine-scale bar |
| http | `{0: 13, 1: 2}` | 2/15 | gate and judge broadly agree; a thin pool |
| cli | `{0: 10, 1: 5}` | 1/15 | gate and judge broadly agree; a thin pool |

**http and cli are close to correct abstentions.** The gate scored most of their pools 0 and the
judge found 2 and 1 actionable papers respectively. Grouping them with linter overstated the
problem, and §16.4 should be read with this table beside it.

**webdev lost exactly one paper** to the fine-scale rescore: two papers reached the band, the
bar rejected both, and the judge scored one of them 2 (*Understanding Bugs in Template
Engine-Based Applications*) and the other 1. That is the stage working as designed on one paper
and costing a true positive on the other.

**linter is the one that matters**, and it is not a window or a rerank problem. Fourteen of its
fifteen ranked papers received the single gate score 1. Nothing was close to the threshold.

### 17.2 The claim I made, and why it does not hold

I wrote that the gate had "under-scored eight genuinely on-topic papers" for ruff, and listed
titles — *MetaLint*, *STYLE-ANALYZER*, *BitsAI-Fix*, *Automated Linter Configuration* — as if
their topicality settled it. **That was an assumption presented as a measurement.** What was
measured is that the gate and the judge disagree. Which of them is wrong is a separate question,
and this project has already measured enough to answer it — against me.

`evals/second_judge.py` re-judged 200 stratified labels with Sonnet under a byte-identical
rubric:

| statistic | value |
|---|---|
| Cohen's kappa (≥2 cut) | **0.507** — below its own pre-registered ≥0.60 bar |
| base rate actionable, GPT-5.5 / Sonnet | **40% / 22%** |
| GPT-scored 2 → Sonnet ≥2 | **8 of 48 (17%)** |
| GPT-scored 1 → Sonnet ≤1 | 58 of 61 (95%) |
| GPT-scored 0 → Sonnet 0 | **58 of 58 (100%)** |

Every one of my eight linter papers is a judge **2**. That is precisely the cell where the two
judges disagree most: a second judge would have scored roughly five in six of them below the
actionable cut — **agreeing with the gate**. The claim inverts under a judge swap, so it was
never mine to make.

### 17.3 The symmetric check, which the same evidence supports

The honest thing is to apply the test to the finding I liked as well. §16.5 reported that the
gate's score-3 band is non-actionable 31% of the time on scientific software against 7% on the
ML cases. All eleven of those non-actionable score-3 papers were scored **1** by the judge — and
GPT's 1s stay at or below 1 for Sonnet **95%** of the time.

**So that finding does survive a judge swap, and my linter claim does not.** Both sat on the
gate/judge boundary; only one sat on the *judges'* boundary. The asymmetry is the whole point,
and I would not have found it by arguing about which model is smarter.

What §16.5 licenses is therefore unchanged and worth restating precisely: on scientific software
the gate marks papers 3 that two independent judges both decline to call actionable. That is a
statement about the gate. What linter licenses is weaker: **the gate and one judge disagree, in
the region where judges disagree with each other.**

### 17.4 One thing that does not depend on the judge at all — **RETRACTED, see §18.4**

Fourteen of linter's fifteen ranked papers share the single gate score 1. A scorer that places
93% of a mixed pool in one bucket is degenerate on its own terms, whatever any judge thinks —
and the gate is not always like this: it spread http's pool across 0 and 1 (13/2) and CHGNet's
across 2 and 3. The concentration is evidence about the gate that needs no external label, and
it is the part of the linter observation that stands.

**It does not stand.** Swept across all 37 cases, the gate puts a median 73% of a ranked window
into one bucket, eight cases are at or above linter's 93%, and two are at 100% — both of them
good results. The two contrast cases above were picked, not sampled. §18.4.

### 17.5 What was built

`rr why` (#163) answers this for a real repository against `.reporadar/papers.db`. **The eval
harness writes no store**, so the command built for this question could not be pointed at the
data that raised it. `evals/why_case.py` is the adapter: same stages, read from a results
artifact, $0 and offline.

It reports gate/judge differences as **disagreement**, prints the second-judge transition rate
for the relevant cell beside each one, and refuses to say which side is wrong. It also flags a
degenerate gate distribution, which is the judge-independent signal of §17.4. What it cannot
show — `score_total`, `rrf_score`, `finescale_p` — it names rather than omits, because those are
not in the artifact and are the fields `rr why` exists to show.

**Two corrections.** `finescale_p` **is** in the artifact, for every score-2 band paper — the
sentence above was written from memory of the schema rather than from a file, and §18.1 is what
it cost. And the "degenerate gate distribution" flag rests on §17.4, which §18.4 retracts;
concentration is the gate's normal behaviour, so the flag marks something ordinary.

### 17.6 The general lesson, which is not new here

`net@2` is **defined** on judge labels, so for benchmark purposes the judge is the scoring
function by construction — "the digest lost points" is true whatever the judge's validity. That
is not the same as "the papers were actionable", and the slide from one to the other is easy
and was mine. RESULTS.md already says the levels are judge-specific and only the separations
survive; §17.2 is that sentence meeting a concrete claim and refuting it.

---
## 18. The two free checks — one instrument correction, one result, and §17.4 retracted (2026-08-20)

§8's plan item 3 and §17.4 were run together because both were free. Item 3 produced a result
and a correction to its own instrument; §17.4 produced a retraction of §17.4.

### 18.1 CORRECTION — the data was on disk the whole time

§8 item 3 named `calibrate_finescale.py --analyse` and estimated "$0 extra". **The cost was
right and the instrument was wrong.** `--analyse` re-reads that script's own per-case cache
under `.work/calibration/`, which holds the 22 legacy cases from the 2026-08-09 run and none of
the twelve scientific ones. Pointed at the cohort-3 artifact it prints `0/15 cached` twelve
times and analyses nothing. Filling that cache means cloning twelve repositories and paying for
a fresh gate and fine-scale pass — which is what "free" was hiding.

It is unnecessary. `run_judge_eval._apply_finescale` mutates the ranked window in place, so the
results artifact **already carries `finescale` and `finescale_p`**: 324 of 555 papers across the
37 cases, every one of them a score-2 band paper, none outside the band, no gaps.

§17.5 asserted the opposite — that `finescale_p` is "not in the artifact" — and `evals/why_case.py`
shipped that claim in its docstring and in a line it printed to the user. Both are corrected.
The error was asserting a schema from memory instead of opening the file, and its cost was that
the calibration check looked like it needed a paid pass for a day.

`evals/finescale_domains.py` is the $0 path: it reads the artifact rows and hands them to
`calibrate_finescale.analyse`, the same analysis the paid path runs.

A second defect surfaced while re-running `why_case.py` for this section: it crashed on a
default Windows console with `UnicodeEncodeError`, on a `≥` in its own output and then on a `λ`
in a paper title. It failed *after* printing the disagreement table, so the output looked
truncated rather than crashed. Titles carry whatever arXiv's metadata carries, so the fix is to
stop the console encoding from being able to fail rather than to sanitise the data.

### 18.2 RESULT — the map is worse on scientific software, and the reading that matters does not survive a judge swap

Reproduction first: the rebuilt policy reproduces **310 of 310** recorded Top Picks across both
populations, so the analysis is measuring the product.

**Judge-free**, a fact about the map:

| | band | withheld | mean P | median P |
|---|---|---|---|---|
| scientific-12 | 109 | **33 (30%)** | 0.692 | 0.760 |
| legacy-25 | 215 | **47 (22%)** | 0.740 | 0.847 |

**Judge-dependent**, against GPT-5.5:

| | ECE | AUC | net@2/case shipped | stage removed | the stage is worth |
|---|---|---|---|---|---|
| scientific-12 | **0.207** | 0.678 | +5.333 | +6.583 | **−1.250** |
| legacy-25 | 0.120 | 0.755 | +5.880 | +5.960 | −0.080 |

A leave-one-repo-out refit — the counterfactual `calibrate_finescale.py` sanctions, since it
never sees the repo it scores — gains **+1.250/case on scientific software** (95% CI
[+0.333, +2.083], sign test 8+/1−/3=, p = 0.039) and **+0.400/case on the legacy 25** (CI
[−0.080, +0.920], p = 0.227). On scientific software the refit's gain is *exactly* the gain from
deleting the stage: the map contributes nothing there that showing every band paper would not.

**And then the check that §17.2 taught.** The withheld papers are 82% actionable on scientific
software and 68% on the legacy 25, against a break-even of 2/3 — which reads as "the stage
withholds true positives". It is not usable. Of the withheld papers judged actionable, **25 of
27 (scientific) and 30 of 32 (legacy) are judge-2s** — the single cell where GPT and Sonnet
agree 8 times in 48. Projected through §17.2's transition table the withheld sets are **19% and
16%** actionable, both far below break-even, and the gap between the populations disappears.

**So exactly one form of the claim is licensed**, and it is the weaker one:

- **True by construction:** the fine-scale stage costs the twelve scientific cases 1.25 net@2
  each on this benchmark, and the legacy 25 nothing. `net@2` is *defined* on judge labels, so
  this holds whatever the judge's validity — it is a statement about the benchmark.
- **Not licensed:** that the stage withholds papers a maintainer would want. That claim needs
  the withheld papers to be actionable, and one judge's labels in their weakest cell cannot
  establish it.

The arm that would settle it is concrete and small: **second-judge the 80 withheld band papers**
with the instrument §17.2 already used on 200 labels. Until then §15.5's disposition stands.

### 18.3 CORRECTION — what §15.5's sweep arm actually is

§15.5 recorded `--rr-sweep min>=2` at +6.58 and set it aside as returning "more papers from a
20-candidate rerank pool", i.e. as a different configuration. **That description is wrong.**
`sweep_top_picks` filters the *same* ranked window on the *same* triage scores
(`run_judge_eval.py:375`), so `min>=2` is the shipped Top Picks **before `_apply_finescale`
runs**. Checked, not argued: rebuilding it reproduces the recorded sweep value on **37 of 37**
cases.

The arm is therefore far cleaner than §15.5 credited — it isolates exactly one shipped stage,
with the pool, window and gate held fixed. §15.5's *conclusion* is unaffected, because the real
reasons to withhold it were post-hoc selection and a single draw, and §18.2 adds a third that is
stronger than either.

### 18.4 RETRACTION — §17.4 does not survive its own control

§17.4 said fourteen of linter's fifteen ranked papers sharing one gate score was "degenerate on
its own terms, whatever any judge thinks", and offered it as the part of the linter observation
that stands after §17.2 took the rest. It cited two contrast cases and never asked what the
other thirty-four look like. `evals/gate_shape.py` asks:

- The gate puts a **median 73%** of a ranked window into one bucket.
- **Eight of thirty-seven** cases are at or above linter's 93%.
- **Two are at 100%** — `rl` (15/15 at score 2) and `thin-kv` (15/15 at score 2) — and both are
  among the run's better results (+13 and +10 net@2).

Concentration this high is the gate's ordinary behaviour, so "93% in one bucket" licenses
nothing. What is unusual about linter is not how sharp its distribution is but **where the mode
sits**: 14 of 15 at score 1, below the admit threshold, where `rl`'s sit at 2, above it. That is
a restatement of "linter returned nothing", not evidence for it.

**The lesson is the one this section exists for.** §17.2 removed the judge from a claim; §17.4
assumed that made it safe. A judge-free claim still needs a control, and the control for "this
distribution is degenerate" was the other thirty-six distributions — one `Counter` away, never
run. Removing a dependency changes which control a claim needs; it does not remove the need for
one. That is now the third claim about linter to fail a check, and the first two failed for
different reasons than this one.

### 18.5 What survives, and it needs no judge at all

The gate's *emission rates* are a fact about the gate:

| gate score | scientific-12 | legacy-25 | Fisher p |
|---|---|---|---|
| 0 | **0/180 (0.0%)** | 31/375 (8.3%) | 4.6e-06 |
| 1 | 35/180 (19.4%) | 99/375 (26.4%) | 0.090 |
| 2 | 109/180 (60.6%) | 215/375 (57.3%) | 0.520 |
| 3 | **36/180 (20.0%)** | 30/375 (8.0%) | **7.7e-05** |

The gate reaches for its top score **two and a half times more often** on scientific software and
**never once** reaches for its bottom score there. No labels are involved.

§16.5 measured the same asymmetry *through* the judge — score-3 papers non-actionable 31% vs 7%,
p = 0.027 — and §17.3 showed that finding survives a judge swap because its papers are judge-1s.
§18.5 now reaches it without a judge at all. **Two measurements of the same asymmetry that share
no dependency** is the strongest form evidence takes in this project, and it is the domain-
calibration thread's first judge-free support.

### 18.6 What this changes about the next step

§16.5 said the score-3 problem "needs its own pre-registration before anything is built on it".
That is still true, and §18 sharpens what it should ask:

1. The asymmetry is now visible in the gate's own output, so a pre-registration can state a
   judge-free primary endpoint alongside the judged one.
2. Two stages are implicated, not one — the gate at score 3 and the fine-scale map in the score-2
   band — and both are worse on the same population.
3. **The judge is the binding constraint on all of it.** Three claims in §17–§18 died on the same
   cell of the same transition table. Second-judging the 80 withheld band papers costs less than
   any repair and would convert the largest open question from unanswerable to measured.

Nothing here licenses shipping a change. What it licenses is knowing which measurement to buy.

---
## Appendix

**Scratchpad** (`C:\Users\raimo\AppData\Local\Temp\claude\C--Users-raimo-auto-features\56bd6727-3c61-4ec9-bf98-ad1b7916a373\scratchpad\`):
`wf/audit_rendered.md` (21 findings + verification + missed), `wf/biosource_rendered.md`,
`wf/arxivcov_rendered.md`, `wf/bio_scout_rendered.md`, `wf/mat_scout_rendered.md`,
`wf/runs_rendered.md`, `wf/judge_rendered.md`, `wf/critic_rendered.md` (gaps, audience
questions, risks, the §5.2 recall diagnostic), `judge_scisoft.py` / `judge_scisoft.json` /
`judge_scisoft.out` (per-paper verdicts with justification and `proposed_change`);
`repos/<name>` (19 shallow clones; LAMMPS is 673 MB and
can be deleted); `runs/<name>/{profile.yml,profile.txt,queries.txt,update.log,digest.md,digest.json}`;
`runs/biorxiv-probe/` and `runs/hyde_probe/` (probe scripts and raw API responses);
`<clone>/.reporadar.yml` and `<clone>/.reporadar/papers.db` for the six live runs.

**Commands** (from the repo root; `set -a; . evals/.env; set +a` first):
`uv run rr init --path <clone> --measured` → edit `arxiv.categories`, `hyde.index_dir:
C:/Users/raimo/auto-features/evals/.work/hyde_index`, `output.digest_path` → `uv run rr update
--config <clone>/.reporadar.yml -v` → `uv run rr digest --config … [--format json]`. Windows
note: `repo_path` and `index_dir` must be `C:/…` paths; a POSIX `/c/…` path is resolved as
`C:\c\…`.

**Not measured here, stated as such:** HyDE reach with *LLM-written* hypotheses on a bio repo
against a blind target list (§5.2 is the scouts' list, not gold); the Opus baseline on any of
the six; a second judge draw; any ablation arm (keyword-only, gate-off, rescore-off); any
multi-source run; Europe PMC rate limits; S2 keyless behaviour; the fine-scale calibration on
bio/matsci beyond the 33-verdict read in B5; end-to-end quality of any bioRxiv candidate (no
adapter delivers one today); journal-only literature; R/Julia/Rust/Nextflow/conda repos; the
recency path; fresh-machine `rr sync-index` time; `HF_HUB_OFFLINE=1`; HF Papers coverage of
`q-bio`/`cond-mat`. Costs are estimates from call counts, not invoices.
