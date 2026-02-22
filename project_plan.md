Here’s a **practical MVP plan** for a personal, repo-focused “arXiv discovery + digest + lightweight suggestions” tool, built almost entirely from **free, easy APIs/libraries**.

---

## MVP goal

For a given software repo, automatically:

1. build a simple “topic profile” of the repo,
2. query arXiv regularly for new relevant papers,
3. dedupe + rank results,
4. produce a **weekly Markdown digest** (and optionally open GitHub issues with suggested next steps).

arXiv is easy because it has a public API (`export.arxiv.org/api/query`) returning Atom feeds. ([arXiv][1])

---

## What the MVP should do (tight scope)

### Outputs you’ll actually use

* `digest.md` (or `digest.html`) with:

  * Top N papers this period
  * 3–5 bullet summary per paper (title/abstract-based)
  * “Why this matches your repo” (keywords/deps/components)
  * Optional “Action ideas” (small, concrete tasks)

### Inputs (minimal friction)

* Repo path (local)
* Optional: arXiv categories + a few seed keywords
* Optional: “ignore” keywords (to reduce noise)

---

## Architecture (simple, modular)

### 1) Repo Profiler

Extract a repo “topic profile” from:

* README + docs
* dependency manifests (`requirements.txt`, `pyproject.toml`, `package.json`, `pom.xml`, etc.)
* import statements from top-level modules (optional)

Output:

* keywords/phrases + weights (e.g., TF-IDF-ish)
* inferred domains (e.g., “retrieval”, “speech”, “time series”, “compilers”)
* “anchors” (library names, model names, datasets, acronyms)

### 2) arXiv Collector

Use either:

* **Direct API calls** to `export.arxiv.org/api/query?search_query=...&start=...&max_results=...` (Atom feed). ([arXiv][1])
* Or the **`arxiv.py`** Python wrapper (less boilerplate; has client/retry/rate limiting patterns). ([GitHub][2])

Key parameters:

* Sort newest first: `sortBy=submittedDate&sortOrder=descending` (documented in arXiv user manual). ([arXiv][3])
* Query formulation: start with `all:` or title `ti:` + AND/OR combinations.

### 3) Store + Dedupe

Local-first storage: `sqlite` (one file per repo), tables like:

* `papers(arxiv_id PRIMARY KEY, title, authors, abstract, categories, published, updated, url, pdf_url, raw_atom_hash, first_seen, last_seen)`
* `runs(run_id, run_time, query_set_hash, n_new, n_seen)`
* `paper_scores(arxiv_id, run_id, score_total, score_breakdown_json)`

Dedup rules:

* Primary key is arXiv id/version
* If you want, treat v2/v3 as “same paper, updated” but keep `updated` timestamp.

### 4) Ranker (MVP heuristics)

Start with cheap, explainable scoring:

* **Repo keyword overlap** with title+abstract (weighted)
* **Category match** (cs.LG vs cs.SE etc.)
* **Recency** (newer = slightly higher)
* **Penalty for generic terms** (e.g., “LLM”, “deep learning” alone)

Later upgrade: optional embeddings similarity (local `sentence-transformers`) if you want better relevance without paid APIs.

### 5) Digest Generator

Generate `digest.md` with sections:

* “New since last run”
* “Top picks”
* “Maybe relevant”
* “Muted/ignored” (to tune filters)

Each paper entry includes links + the exact query that surfaced it.

### 6) Suggestion Generator (keep it humble)

For each high-ranked paper, generate **1–3 repo-impact suggestions** using templates like:

* “Add evaluation on ___ (paper mentions benchmark/dataset)”
* “Compare your method to ___ baseline”
* “Try swapping component ___ (e.g., retriever, loss, optimizer)”
* “Add feature flag / module to support ___ approach”

Important: in MVP, suggestions should be clearly labeled **“ideas”** and always cite the paper abstract text snippets they’re derived from.

---

## “Free and easy” optional add-ons (still MVP-friendly)

### Papers With Code linking (nice for ML repos)

If you want “dataset / code implementation” links for an arXiv id, Papers with Code has a client library; read-only usage is straightforward (write mode needs a token). ([GitHub][4])

This can turn “paper found” → “here’s the dataset + reference implementation” (huge usefulness jump for ML repos).

### Citation / influence signals (optional)

If you need citation counts to help ranking:

* Semantic Scholar API exists but typically needs an API key and has rate limits (even with a key it starts low). ([Semantic Scholar][5])
* OpenAlex is free but uses an API-key + budget model. ([OpenAlex][6])

For MVP: skip citations unless you really need them.

---

## CLI UX (what you build)

Commands:

* `rr init` → creates `.reporadar.yml`, initializes sqlite
* `rr profile` → prints inferred keywords/categories; lets you pin/edit
* `rr update` → queries arXiv, stores new results, scores them
* `rr digest --since 7d` → writes digest markdown
* `rr open top` → opens top papers in browser
* `rr gh-issues --top 5` → (optional) opens GitHub issues with suggestions

Minimal config example:

```yaml
repo_path: .
arxiv:
  categories: [cs.LG, cs.CL]
  max_results_per_query: 50
  lookback_days: 14
  sort_by: submittedDate
  sort_order: descending
queries:
  seed:
    - "retrieval augmented generation"
    - "long context transformers"
  exclude:
    - "survey"
    - "benchmark"
ranking:
  w_keyword_overlap: 1.0
  w_category_match: 0.5
  w_recency: 0.3
output:
  digest_path: ./reporadar/digest.md
schedule:
  cron: "0 9 * * 1"   # weekly Monday 9am (optional)
```

---

## Milestones (no fluff, just what to build)

1. **End-to-end vertical slice**: `init → update → digest` using arXiv only (title/abstract ranking).
2. **Repo profiler v1**: extract keywords/deps; auto-generate 5–15 arXiv queries.
3. **Noise control**: ignore lists, query audit trail, dedupe, stable scoring.
4. **PapersWithCode link enrichment** (optional, high ROI for ML). ([GitHub][4])
5. **GitHub issues export** (optional).

---

## The hardest parts (even for an MVP)

* **Query generation**: if queries are bad, everything downstream is noise. Start with user-pinned seeds + a few auto keywords.
* **Relevance ranking**: “seems related” isn’t enough; you need low false positives or you’ll stop reading the digest.
* **Change management**: papers update versions; you want “what’s new since last digest” without repeating everything.
* **Action ideas without hallucination**: keep suggestions templated + grounded in abstract text until you add stronger grounding.

---

If you tell me what kind of repos you mean (ML research code, devtools, distributed systems, etc.), I can propose:

* a default set of arXiv categories,
* a better first-pass query generator,
* and a scoring scheme that fits that repo type.

[1]: https://info.arxiv.org/help/api/basics.html "arXiv API Basics - arXiv info"
[2]: https://github.com/lukasschwab/arxiv.py?utm_source=chatgpt.com "lukasschwab/arxiv.py: Python wrapper for the arXiv API"
[3]: https://info.arxiv.org/help/api/user-manual.html?utm_source=chatgpt.com "arXiv API User's Manual"
[4]: https://github.com/paperswithcode/paperswithcode-client "GitHub - paperswithcode/paperswithcode-client: API Client for paperswithcode.com"
[5]: https://www.semanticscholar.org/product/api?utm_source=chatgpt.com "Semantic Scholar Academic Graph API"
[6]: https://docs.openalex.org/how-to-use-the-api/api-overview?utm_source=chatgpt.com "API Overview"
