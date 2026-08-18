# RepoRadar

Monitor arXiv for papers relevant to your software repo and produce a ranked Markdown digest with actionable suggestions.

RepoRadar automatically profiles your repository (README, dependencies, docs), queries arXiv for matching papers, scores them by relevance, and generates a digest you can actually use.

## Features

- **Repo profiling** — extracts keywords via TF-IDF from README, docs, and dependency manifests (`requirements.txt`, `pyproject.toml`, `package.json`)
- **arXiv collection** — queries the arXiv API with auto-generated and user-defined seed queries
- **Multi-source** — opt into Semantic Scholar, OpenAlex, **bioRxiv** (biology), **DBLP** (systems/PL/DB), and **IACR ePrint** (cryptography) alongside arXiv via `sources:` — for repos whose literature isn't on arXiv. All are opt-in and off by default: none has been shown to improve results, and until 2026-08-12 they were all sent malformed queries (see `evals/RESULTS.md`, C-9), so treat them as unvalidated
- **SQLite storage** — deduplicates papers across runs, tracks collection history
- **Heuristic ranking** — scores papers by keyword overlap, category match, and recency with configurable weights
- **Markdown digest** — three-tier output (Top Picks / Maybe Relevant / Muted) with score breakdowns and arXiv links
- **HTML output** — optional `--format html` renders a real digest page (paper cards, score breakdowns, badges), not raw Markdown
- **GitHub Action + Pages** — a first-party action publishes a dated, ranked digest to GitHub Pages on a schedule; `rr archive` builds the browsable site
- **Local corpus search** — `rr search "<query>"` over every paper ever fetched: offline BM25, or `--semantic`/`--hybrid` embedding search backed by a cached, optionally sqlite-vec-accelerated index (also an MCP tool)
- **Action suggestions** — template-based ideas grounded in paper abstracts (benchmarks, baselines, datasets, modules)
- **Extends work you starred** — flags and boosts new papers that cite a paper you starred or rated highly (`ranking.w_citation_proximity`)
- **Learned recommendations** — your stars/ratings seed the free Semantic Scholar recommender; results are re-ranked locally before they reach the digest (`recommendations.enabled`)
- **SPECTER2 similarity** — citation-trained scientific embeddings (served free by Semantic Scholar, no local model) score how close each paper is to the work you starred/rated (`ranking.w_specter`)
- **Community attention** — optionally rank on Hugging Face Papers upvotes, log-scaled against the run's own maximum (`ranking.w_community`)
- **HyDE dense discovery** — queries built from a repository describe what it *has*; the paper that would improve it describes what it should *adopt*. Across nine benchmark repos, keyword search fetched 2,030 papers and reached **0** of 24 known-good ones. HyDE sidesteps that: an LLM writes the *abstract of the paper it wishes existed*, which is in the literature's register by construction, and that is matched against a local 3.1M-vector index of all arXiv. Measured end to end on the 22-repo benchmark it is worth **+1.36 mean net@2**, and it takes the system to **+4.55 vs the Opus 4.8 baseline's +1.82** — 15 wins to 3, p = 0.0075, the first result here that clears p < 0.05 against the baseline. Once synced, matching is fully offline (`hyde.enabled` + `rr sync-index`, opt-in, ~1.1 GB local)
- **Fine-scale rescore** — the 0-3 actionability gate is near-binary, and its score-2 band ran anywhere from 0% to 100% useful depending on the repo. A second pass rescores that band on a 0-9 scale, reads the *distribution* over the answer token rather than the sampled digit, and admits a paper only above a calibrated P ≥ 2/3. Worth **+1.86 → +3.18 mean net@2** on a live 22-repo benchmark run, with every net-negative repo eliminated (`triage.finescale`, opt-in, needs `OPENAI_API_KEY`)
- **Withdrawal detection** — flags papers their own authors withdrew and demotes them out of Top Picks, so you never act on retracted work (on by default)
- **Hacker News attention** — badges papers that were discussed, with points and a link to the thread (`signals.hackernews`)
- **Ranking eval** — `rr eval` scores the ranker against your own ratings, and `--compare a.yml b.yml` A/Bs two configs with a bootstrap interval, so "did that change help?" has an answer
- **Privacy audit** — `rr audit` prints every network destination and the exact query strings your profile would transmit, without sending any of them; `privacy.redact` strips internal codenames from queries and LLM prompts
- **Polite by design** — every arXiv request in the process passes one shared gate at arXiv's stated ceiling of 1 request / 3 s, identifies itself with a RepoRadar User-Agent, and backs off for 30 s (not 2 s) on a 429. One clock, not one per module
- **No API keys required** for the default arXiv pipeline — every core source is free and keyless. That default is also the configuration this project's own benchmark scores at mean net@2 **−8.12**, against **+5.72** with the LLM stages on, so "keyless" and "good" are not the same setting and we do not present them as one: see [The measured configuration](#the-measured-configuration). Three opt-in features need a key: OpenAlex (`openalex.api_key`, since 2026-02-13 it throttles keyless callers), LLM triage (`ANTHROPIC_API_KEY` or a local Ollama), and the fine-scale rescore (`OPENAI_API_KEY` — it is the only feature that needs a *second* vendor, because it reads token logprobs and Anthropic does not expose them)

## Installation

Requires Python 3.11+. Dependencies: `click`, `pyyaml`, `scikit-learn`, `jinja2`, `arxiv`.

```bash
# Clone and install with uv
git clone <repo-url>
cd auto-features
uv pip install -e .

# Or with dev dependencies (pytest, pytest-cov)
uv pip install -e ".[dev]"
```

## The measured configuration

**Read this before the quick start.** RepoRadar ships two configurations, and the default is
the weak one.

| | `rr init` (default) | `rr init --measured` |
|---|---|---|
| how papers are ordered | keyword overlap | keyword + BM25 fusion, LLM actionability gate, fine-scale rescore |
| where papers come from | arXiv keyword queries | the same, plus HyDE dense discovery over 3.1M arXiv abstracts |
| **mean net@2 on the 25-repo benchmark** | **−8.12** | **+5.72** |
| papers shown / of those actionable | 235 / 89 | **212 / 189** |
| precision | 0.379 | **0.892** |
| repositories where it scores negative | **19 of 25** | 0 of 25 *(this draw; see below)* |
| against the agentic Opus 4.8 baseline | — | +1.56 (paired **+4.16**, 95% CI [+2.44, +6.00], sign *p* = 0.0004) |
| API keys | none | Anthropic **and** OpenAI |
| disk | none | ~1.1 GB (one time) |
| cost per repo per run | $0 | **~$0.01–0.02** |

**"0 of 25" is this draw's value, not a property of the method.** The same configuration has
produced 1 and 2 net-negative repositories on other draws; a run that happens to have none
is a favourable draw, and reporting it as "never scores negative" is a mistake this project
has already made once and corrected (C-7 in [evals/RESULTS.md](evals/RESULTS.md)). What
replicates is the mean and the paired delta.

**One line in that configuration depends on your install.** `ranking.w_embedding: 1.5` is
worth about +1 net@2 per repository over the 0.0 it carried until 2026-08-16 — but only
with the `embeddings` extra present. Without it the weight is silently inert and you get
the lower-scoring configuration. `uv pip install -e ".[hyde]"` already supplies it, so
following the setup below is enough; setting the weight without the extra is not.

**The default is not a recommendation; it is what works without credentials.** A digest
ranked by keyword overlap alone scores *worse than emitting nothing*, because `net@2`
charges 2 for every unactionable paper shown and keyword rank does not predict
actionability (§"How this was measured" below, and [evals/RESULTS.md](evals/RESULTS.md)).
The measured configuration is not the default only because a tool that fails on first run
without an API key is worse than one that under-delivers — not because the difference is
small or a matter of taste.

```bash
rr init --measured
```

That writes a fully commented `.reporadar.yml` in which every value carries the measurement
that justifies it. Then, before the first run:

```bash
export ANTHROPIC_API_KEY=...     # actionability gate + HyDE hypotheses (Claude Haiku)
export OPENAI_API_KEY=...        # fine-scale rescore -- see "a second vendor" below
uv pip install -e ".[hyde]"      # sentence-transformers + pyarrow
rr sync-index                    # one time: 432 MB index + ~670 MB model weights
```

and **set `arxiv.categories` to your own fields** — the `cs.LG, cs.CL` default is a guess
that fits an ML repository and nothing else, and a wrong category list starves every stage
downstream.

**What each stage costs and buys**

| stage | needs | cost per repo per run | measured worth |
|---|---|---|---|
| actionability gate (`triage`) | `ANTHROPIC_API_KEY` or local Ollama | ~$0.01 (Haiku over 50 papers; measured ~$0.02/100) | most of the −8.12 → +5.72 gap; it is what declines to show a paper |
| fine-scale rescore (`triage.finescale`) | `OPENAI_API_KEY` | <$0.01 (one call per band paper) | +1.36 mean net@2; eliminates net-negative repos |
| HyDE discovery (`hyde`) | `.[hyde]` + `rr sync-index`, ~1.1 GB | <$0.01 (4 Haiku hypotheses/run) | +1.36 mean net@2; keyword search alone reached **0 of 24** targets |
| hybrid fusion (`ranking.hybrid`) | nothing — plain Python | $0 | better nDCG; keep it **with** the gate, see NR-11 |
| | | **~$0.01–0.02 total** | vs **~$0.80/repo** for the agentic baseline it beats |

**Why a second vendor.** The rescore reads the *token probability distribution* over the
score digit, not the sampled digit — that is the mechanism that works — and of the major
APIs only OpenAI exposes logprobs. It is structural, not a preference. Without an OpenAI
key, drop the `finescale` block: you keep the gate and HyDE and lose roughly the +1.36 that
stage is worth.

**Degrading honestly.** Each stage can be dropped independently and the ones below it still
work. What you cannot do is enable the gate halfway: `triage.enabled: true` with
`suggestions.provider` left at `template` gates *nothing*. Both fields are required, `rr
update` says so loudly if only one is set, and the measured config sets both.

**How this was measured.** 25 repositories, papers pooled across systems and judged blind
to source by GPT-5.5 under a fixed rubric, `net@2 = #actionable − 2·#non-actionable`. **Both
columns are runs of the actual configurations, at the same digest width, on the same 25
repositories** — the default arm on 2026-08-16 specifically to replace an earlier figure
(−11) that had been measured on *four* repositories in July, one of them a negative
control, and had no business being quoted against a 25-repository number. That row was
wrong when first published here and is corrected now.

Why the gate is worth 13 points: the ungated arm is not bad at *finding* papers — it
surfaces 89 actionable ones — but `net@2` pays `3p − 2` per paper shown, so at precision
0.379 **every paper displayed costs 0.86 on average**, and the keyword tiering has no way
to decline: it filled all 15 slots in 17 of the 25 repositories. The measured configuration
shows *fewer* papers (197 vs 235) and delivers nearly twice as many actionable ones. The
expensive stage is the one that says no.

Caveats: a single draw of this benchmark carries a ±0.6 spread, so the 13-point gap is far
outside the noise but the individual levels are not precise to the decimal; the two arms
come from different sessions and pool provenances, so this compares levels rather than a
paired delta; the system wins over the *agentic baseline* on volume at slightly lower
precision (0.888 vs 0.938); and the default arm was measured without the `embeddings`
extra, which the harness cannot reproduce. All of this is in
[evals/RESULTS.md](evals/RESULTS.md) and `paper/DRAFT.md` §8.7. The configuration written
by `rr init --measured` is asserted field-by-field against the benchmark's own
configuration by `evals/audit_product_divergence.py`, so this recommendation cannot
silently drift from the run it cites.

## Quick Start

```bash
# 1. Initialize RepoRadar in your repo
cd /path/to/your/repo
rr init                  # keyword-only; see "The measured configuration" above
# rr init --measured     # the configuration behind +5.72 (needs 2 API keys + ~1.1 GB)

# 2. (Optional) Edit .reporadar.yml to add seed queries and categories

# 3. See what RepoRadar infers about your repo
rr profile

# 4. Fetch and score papers from arXiv
rr update

# 5. Generate a digest
rr digest

# 6. Open top papers in your browser
rr open --top 5
```

## CLI Commands

### `rr init [--path DIR] [--measured]`

Creates `.reporadar.yml` config and `.reporadar/` storage directory. Safe to run multiple times — skips files that already exist.

`--measured` writes the configuration every published number in this project was measured under (mean net@2 **+5.72** against the agentic baseline's +1.56) instead of the keyword-only default (**−8.12**). It needs an Anthropic key, an OpenAI key and `rr sync-index`, and costs ~$0.01–0.02 per repository per run — see [The measured configuration](#the-measured-configuration). Without the flag, `rr init` prints what the default gives up.

### `rr profile [--config PATH]`

Prints the inferred topic profile: TF-IDF keywords with weights, detected packages (anchors), and inferred domains.

It also flags paper sources that match the repo but aren't in `sources:` — a repo built on `scanpy`/`anndata` gets pointed at bioRxiv, one built on `duckdb`/`rocksdb` at DBLP (`rr update` prints the same hint). Suggestions only: nothing is auto-enabled, since each source has costs worth opting into knowingly, and the hint says what they are.

### `rr update [--config PATH] [--explain] [--foundational] [--rebuild-embeddings] [-v]`

Runs the full pipeline: profile repo, build queries, fetch papers from arXiv, store in SQLite, score, and display top 5 results. Use `-v` for verbose logging.

`--explain` prints a per-component score breakdown for the top papers, which is the only way to see why a paper ranked where it did.

`--foundational` forces **all-time, relevance-first** discovery with the recency weight dropped. **This is now the default** (`lookback_days: 36500`, `sort_by: relevance`, `w_recency: 0.0`); the flag remains so a config that narrows the window can be overridden for a single run.

The default changed because the old one — a 14-day submitted-first window — was never the configuration the benchmark measured. Every headline Tier B number since 2026-07-06 was produced under all-time/relevance, and all 48 benchmark targets are ≥11 months old, so a fortnight's window cannot reach any of them. Repeat runs do not re-show the same papers: the store records `first_seen` and the digest marks what is new. For a strict recency digest, set `lookback_days: 14` and `sort_by: submitted` — that path is supported but has never been benchmarked.

### `rr digest [--config PATH] [--since 7d] [--run-id N] [-o PATH] [--format md|html|json|csv|rss] [--diff]`

Generates a digest from the latest (or specified) run. Options:

- `--since 7d` — time window (e.g. `7d`, `14d`)
- `--run-id N` — use scores from a specific run instead of the latest
- `-o PATH` — custom output file path
- `--format html` — a fully rendered HTML page (paper cards, score breakdowns, badges — not raw Markdown; auto-converts the `.md` extension to `.html`)
- `--format json|csv|rss` — machine-readable output for downstream tooling; the extension is rewritten to `.json` / `.csv` / `.xml` automatically
- `--diff` — mark papers `[NEW]` versus carried over from the previous run

### `rr open [--config PATH] [-n N | --top N]`

Opens the top N papers from the latest run in your default browser. Defaults to 5. Opening a paper also stars it, which counts as a weak positive signal.

### `rr rate ARXIV_ID RATING [--config PATH]`

Rates a paper from 1 (not useful) to 5 (very useful). This is the input everything
adaptive is built on: 4–5 seeds the [learned recommender](#configuration) and the
SPECTER2 query, 1–2 suppresses similar results, and both become the labels
[`rr eval`](#rr-eval---config-path--k-10---compare-ayml-byml---baseline---against-latest---history---format-textjson)
scores the ranker against. An explicit rating always outranks the implicit star from
`rr open`.

```bash
rr rate 2501.12948 5
```

### `rr archive [--config PATH] [--archive-dir DIR] [--date YYYY-MM-DD] [--since 7d]`

Publishes the latest run's digest as a rendered HTML page into a dated archive
(`<archive-dir>/<date>.html`, default dir `digests/`) and regenerates `index.html`
listing every edition newest-first. Re-running on the same date replaces that
edition. This is the content GitHub Pages serves for the [GitHub Action](#github-action-scheduled-digests--github-pages).

### `rr search QUERY [--config PATH] [-n LIMIT] [--since 7d] [--semantic] [--hybrid] [--format text|json]`

Free-text search across **every paper RepoRadar has ever fetched** — the store
accumulates into a personal corpus, and this queries all of it offline. The
default is Okapi BM25 (no network, no embeddings). `--semantic` ranks by embedding
similarity and `--hybrid` fuses both via Reciprocal Rank Fusion; both need the
`embeddings` extra and encode + **cache** each paper's vector once (reused across
runs and by `rr update`). Install the optional `vectors` extra to back semantic
KNN with a sqlite-vec index on large corpora — it works without it (numpy fallback).

```bash
rr search "low-rank adaptation quantization"
rr search "efficient attention" --semantic
rr search "retrieval augmented generation" --hybrid -n 5 --format json
```

### `rr notify --channel shell|slack|discord|email [--config PATH] [--run-id N]`

Pushes a one-line summary of a run to a configured channel (`hooks:` in
`.reporadar.yml`: `on_digest` for a shell command, `slack_webhook_url`,
`discord_webhook_url`, `email`):

```
RepoRadar digest #12: 24 new papers, 5 top picks (40 scored) — 2 papers extend work you starred
```

The trailing clause appears only when papers in the run cite something you starred
or rated 4–5, so a citation alert reaches you instead of waiting to be noticed in
the digest. Shell hooks additionally get the whole summary as environment
variables: `RR_DIGEST_PATH`, `RR_RUN_ID`, `RR_PAPERS_NEW`, `RR_PAPERS_SEEN`,
`RR_TOP_PICKS_COUNT`, `RR_TOTAL_SCORED`, `RR_EXTENDS_STARRED_COUNT`, `RR_FORMAT`.

### `rr eval [--config PATH] [-k 10] [--compare A.yml B.yml] [--baseline] [--against latest] [--history] [--format text|json]`

Scores the ranker against the ratings you have already given, so a ranking change is
falsifiable instead of taken on faith:

```
Judged papers: 34 (30 rated, 4 starred-only, 2 neutral 3-star ignored)
Relevant:      18

  precision@10: 1.000
  recall@10:    0.556
  nDCG@10:      1.000
  MRR:          1.000
```

4–5 stars (and stars) count as relevant, 1–2 as not, and 3 is ignored — the same
convention the feedback loop uses to tune weights, so the thing measured and the thing
tuned agree. An explicit rating always beats an implicit star, since `rr open` stars
papers as a side effect.

`--compare` re-scores the identical papers under two configs and bounds the difference,
because with a few dozen labels a small gap is noise:

```bash
rr eval --compare .reporadar.yml experiment.yml
```

```
metric                 A         B     delta
--------------------------------------------
nDCG@10            1.000     0.555    -0.445

ndcg@k difference (B - A): -0.445  90% interval [-0.729, -0.199]
  -> A is better on this data (interval excludes zero).
```

When the interval contains zero it says **NOT SHOWN** rather than inviting a decision
the data cannot support. If the two configs differ only in something the harness cannot
reproduce (`w_embedding` needs the optional embeddings extra; `w_citations` needs counts
fetched at digest time), it says that too — otherwise "cannot tell them apart" would be indistinguishable from a
genuine null result.

**As a CI gate.** `--baseline [--label NAME]` records a measurement together with the
weights that produced it; `--against latest` (or a snapshot id) compares a fresh run to
it and **exits 1 on a regression**:

```bash
rr eval --baseline --label before-change   # once, on a good config
rr eval --against latest                   # in CI, after any ranking change
```

Movement smaller than 0.02 is tolerated, since metrics shift whenever a new rating
lands — an exact-equality gate would fail constantly and get switched off. A baseline
taken at a different `-k` is refused rather than differenced, and a changed judged-set
size is flagged, because part of that movement is new ratings rather than a ranking
change. `--history` lists recorded baselines with the `k` each used.

**What it does and does not measure.** It measures whether the ranker *orders the
papers you judged* well. It cannot measure papers it never showed you, and your
ratings only exist for papers an earlier ranking surfaced — real selection bias that
no offline metric can correct. Treat it as a regression check on changes, not an
absolute quality score. Below ~20 judgments the output says so.

Papers unrated by you are **removed** from the ranking rather than counted as
irrelevant: with tens of labels against thousands of stored papers, counting the
unjudged as bad would drown every metric in papers you never looked at.

The optional components are measurable too. Citation proximity, SPECTER2, HF upvotes,
Hacker News points, withdrawal flags and hybrid RRF are rebuilt from what your own runs
already stored — **entirely offline, no network calls** — so `--compare` can tell you
whether `w_specter` or `w_attention` earns its weight on your corpus. A signal your runs
never fetched stays absent rather than becoming a zero, so the eval measures only the
components you actually populated. `w_embedding` and `w_citations` cannot be reproduced
offline; a comparison that turns on one of those says so rather than reporting a null.

SPECTER2 gets special handling: its query is the centroid of the papers you starred or
rated highly, which are *exactly* the relevant labels — so it is scored leave-one-out,
with each paper excluded from its own query. Without that, every relevant paper's score
is inflated by construction and the harness confidently recommends turning the weight
up even when the vectors are pure noise.

### `rr audit [--config PATH] [--json]`

Prints exactly what would leave this machine, and sends nothing to find out.

```
Reached by every `rr update` (3):
  [repo-derived] arXiv - export.arxiv.org/api/query
      sends: the search queries built from your repo profile (keywords, library names)
      enabled by: always (the core source)
  [repo-derived] DBLP - dblp.org/search/publ/api
      ...

Reached only by an explicit command (1):
  [interests] GitHub - the `gh` CLI, against your configured remote
      sends: paper titles, abstracts and suggestions, as issue bodies in your repo

Query strings that would be transmitted (10):
  1. (all:"learning to rank") AND (cat:cs.LG OR cat:cs.CL)
  ...

Redaction: 2 pattern(s) active - queries above are filtered.

What leaves regardless of redaction:
  Profile keywords, which encode your domain: arxiv, mcp, pytest, sqlite-vec, ...
```

Destinations are grouped by what they receive — `repo+paper text` (LLM prompts, the
most sensitive), `repo-derived` (search terms inferred from your code), `interests`
(arXiv ids of papers you track), `none` — and sorted worst-first. Anything reached only
by an explicit command (`rr gh-issues`, `rr notify`) gets its own section rather than
being omitted for not being part of `rr update`.

If LLM triage is on, the prompt also carries the first `profiler.prose_chars` characters
of your README (300 by default) so the model knows what the project is *for* — the
keyword profile only says what it *contains*. On a proprietary codebase that is a larger
disclosure than a term list, so the audit names it explicitly and
**`profiler.prose_chars: 0` withholds it**. 300 is the measured optimum rather than a
safe-looking number: 2,000 and 6,000 characters both score *worse* on the benchmark, so
the privacy-preserving setting is also the best-performing one.

Two things make the report trustworthy rather than decorative. The query strings come
from the same `build_queries` call `update` uses — profiled the same way, including
`profiler.scan_source` — so there is no second implementation to drift from the first.
And the destination list is **enforced**: a test walks the package for modules that make
outbound calls and fails CI if any is missing from the registry, so adding a source
without documenting it is a build error rather than a quietly stale privacy page. That
detector is static, and the test says so: it knows a list of request shapes and follows
private helpers imported from an already-outbound module (`specter` makes no request of
its own — it borrows one from `citations`). A module reaching the network some third way
would still slip past. It raises the cost of an undeclared destination; it does not make
one impossible.

`--json` emits the same data for scripting (e.g. failing a CI job if an unexpected
destination becomes active).

#### `privacy.redact` — stripping internal codenames

```yaml
privacy:
  redact:
    - projectatlas          # literal, case-insensitive
    - "re:acme-[0-9]{4}"    # regex, only with the `re:` prefix
```

Entries are **literal by default**; a `re:` prefix opts into a regex. This matters more
than it looks: `C++` is a *valid* Python regex — `++` parses as a possessive quantifier
— so treating entries as regexes by default would compile it happily and redact only
the letter `C`, leaving `++` in the query and the user believing otherwise.

Terms are removed at two choke points, both by construction rather than per-call-site:
`build_queries`, which every text-search source draws from, and `llm_client.complete`,
the last step before any prompt leaves the process. Redaction runs on the query *terms*
before they're assembled into arXiv syntax — filtering the finished string would leave
`(all: ) AND (cat:cs.IR)` behind, which is not a redacted search but a broken one. A
term that redacts to nothing is dropped rather than sent empty.

**What it does not do.** Removing a codename stops it reaching a search log. It does
not hide that you work on, say, distributed consensus: TF-IDF keywords still encode
the domain, and no denylist changes that. `rr audit` says so explicitly in its "what
leaves regardless of redaction" section, and warns when your patterns matched nothing
— a user who configures redaction and gets silence would otherwise assume it worked.
The audit view is the honest layer; the filter is a convenience on top of it.

### `rr mcp [--config PATH]`

Runs RepoRadar as an **MCP server** (stdio) so coding agents — Claude Code, Cursor, VS Code, Windsurf — can query your repo-aware paper store conversationally. Unlike generic arXiv MCP servers, its tools are grounded in *this repository's* profile and ranking. Tools exposed:

- `get_repo_profile` — the repo's keywords / libraries / domains
- `get_ranked_papers(limit)` — top papers from the latest `rr update`, best-first
- `explain_relevance(arxiv_id)` — score-component breakdown + LLM actionability reason
- `rate_paper(arxiv_id, rating)` — record a 1–5 rating (feeds the feedback loop)
- `search_papers(query, limit)` — free-text BM25 search over the whole stored corpus

Requires the optional extra: `uv pip install -e ".[mcp]"`. Register it with your agent, e.g. Claude Code:

```bash
claude mcp add reporadar -- rr mcp --config /abs/path/.reporadar.yml
```

### Other commands

| Command | What it does |
|---|---|
| `rr status` | Corpus size, last run, ratings/stars recorded |
| `rr history` | Past collection runs with counts |
| `rr queries` | The auto-generated queries `rr update` would run, without running them |
| `rr watch --interval 6h` | Continuous update+digest cycles with a desktop notification |
| `rr schedule` | Install/remove an OS-level scheduled run (crontab or schtasks) |
| `rr gh-issues` | Open a GitHub issue per top paper (needs the `gh` CLI) |
| `rr workspace` | Multi-repo workspaces: `init`, `add`, `list`, `remove`, `update`, `digest` — **reduced pipeline, see below** |

Run any of them with `--help` for the full flag list.

### `rr workspace update` does not run the measured pipeline

It collects across member repos and ranks, but stops before the rest: **no actionability gate,
no fine-scale rescore, no HyDE, no fusion, no embeddings, and no source beyond arXiv** —
whatever your config says. The gate is the stage that *declines to show* a paper, so a config
reading `triage.enabled: true` still produces an ungated digest there: closer to the **−8.12**
configuration than the **+5.72** one. It lists the stages it is skipping, per member, before
it starts.

**`rr watch` used to have the same gap and no longer does** — it runs the identical
`pipeline.run_pipeline` that `rr update` runs, so a watch cycle produces the configuration your
config file describes. It still reports `skipped_stages` every cycle, which is now empty and
stays checked: `tests/test_stages.py` walks the real import graph and fails if any entry point
and the registry disagree in either direction.

Workspace is not unified because it is a different shape rather than duplicated code: one
shared candidate pool across many member repos under a single run id. For the measured
pipeline on a schedule, use `rr update`:

```bash
rr schedule --cron "0 9 * * 1"   # registers `rr update && rr digest` -- the full pipeline
```

## GitHub Action (scheduled digests + GitHub Pages)

Run RepoRadar as scheduled team infrastructure with zero hosting: the
[`reporadar` action](action.yml) profiles the repo, fetches + ranks papers, and
publishes a dated, browsable **research radar** to GitHub Pages — the repo that
*is* the profile target runs its own radar.

**Setup**

1. Copy [`examples/reporadar.yml`](examples/reporadar.yml) to `.github/workflows/reporadar.yml`.
2. Enable Pages: **Settings → Pages → Source: GitHub Actions**.
3. (Optional) commit a `.reporadar.yml` to tune categories/queries; if it's absent the action creates a default with `rr init`.

Minimal workflow:

```yaml
permissions:
  contents: read
  pages: write
  id-token: write
jobs:
  radar:
    runs-on: ubuntu-latest
    environment: github-pages
    steps:
      - uses: actions/checkout@v4
      - uses: raimondasl/auto-features@v1   # moving major tag (or pin @v1.0.0 / a SHA)
        id: radar
        with:
          archive-dir: digests
      - uses: actions/upload-pages-artifact@v3
        with: { path: '${{ steps.radar.outputs.archive-dir }}' }
      - uses: actions/deploy-pages@v4
```

**Inputs** (all optional): `config` (default `.reporadar.yml`), `archive-dir`
(`digests`), `formats` (extra loose digest files — `md,json,…`; default none, HTML
archive only), `notify` (`slack,discord,email`), `github-issues` (`true` opens an
Issue per top pick — also needs `issues: write` + `GH_TOKEN`), `cache-db` (`true`),
`python-version` (`3.12`). See [`action.yml`](action.yml) for the full list and outputs.

**Secrets** never live in the committed config. Reference them by name —
`slack_webhook_url: ${SLACK_WEBHOOK}` in `.reporadar.yml` — and inject the value
in the workflow (`env: { SLACK_WEBHOOK: ${{ secrets.SLACK_WEBHOOK }} }`). Any
`${VAR}` in `.reporadar.yml` is expanded from the environment at load time (unset → empty).

**Run history** (the paper DB and ratings) is cached between runs via `actions/cache`.
Caches unused for 7 days are evicted; for durable history, commit the `.reporadar/`
directory to a branch instead of relying on the cache. If you set a non-default
`repo_path` in `.reporadar.yml`, point the action's `cache-path` input at
`<repo_path>/.reporadar` so the cache tracks where the DB actually lives.

## Configuration

`.reporadar.yml` in your repo root:

```yaml
repo_path: .                          # Path to the repo to profile (default: current dir)

arxiv:
  categories: [cs.LG, cs.CL]        # arXiv categories to search
  max_results_per_query: 50          # Max papers per query
  lookback_days: 36500               # All-time. Set to 14 for a strict recency digest
  sort_by: relevance                 # or "submitted" for newest-first

queries:
  seed:                               # Your own search terms (exact-match quoted)
    - "retrieval augmented generation"
    - "long context transformers"
  exclude:                            # Terms to penalize in ranking (0.5x per match)
    - "survey"
    - "benchmark"
  bigrams: verified                   # phrase queries: verified | adjacent | none
                                      #   verified — only phrases your repo actually contains
                                      #   adjacent — pair keywords by TF-IDF rank (pre-2026-08-12)
                                      #   none     — no phrase queries (measured worse)

sources: [arxiv]                      # add: semantic_scholar, openalex, biorxiv, dblp, iacr

ranking:
  w_keyword: 1.0                      # Weight for keyword overlap score
  w_category: 0.5                     # Weight for category match score
  w_recency: 0.3                      # Weight for recency score
  w_embedding: 0.0                    # `rr init` writes 1.5; needs the `embeddings` extra or it does nothing
  w_citations: 0.0                    # >0: rank on citation counts (fetched at digest time)
  w_citation_proximity: 0.0           # >0: fetch references + boost papers that cite work you starred/rated
  w_specter: 0.0                      # >0: SPECTER2 similarity to the papers you starred/rated highly
  w_community: 0.0                    # >0: rank on Hugging Face Papers upvotes (from cached enrichments)
  w_attention: 0.0                    # >0: rank on Hacker News points (needs signals.hackernews)
  withdrawn_penalty: 0.1              # multiplier for a withdrawn paper's score (1.0 disables)
  hybrid: false                       # true: fuse the heuristic order with BM25 via RRF
  category_weights: {}                # per-category multipliers, e.g. {cs.CL: 2.0}

hyde:                                 # dense discovery (see "HyDE discovery" below)
  enabled: false                      # true: needs `rr sync-index` first (~432 MB, one time)
  index_dir: ~/.cache/reporadar/hyde-index
  n_hypotheses: 4                     # 4 diverse guesses measured far better than 1
  top_k: 100                          # candidates per hypothesis; the union feeds the ranker

triage:
  enabled: false                      # true: LLM-score each top paper 0-3 for actionability
  min_actionable: 2                   # llm_score >= this qualifies as a Top Pick
  rerank: true                        # reorder by llm_score before the top-N cut
  finescale:                          # second-stage rescore of the score-2 band (see below)
    enabled: false                    # true: needs OPENAI_API_KEY (logprobs; OpenAI-only)
    openai_model: gpt-4o-mini
    threshold: 0.667                  # P(actionable) a band paper must clear

suggestions:
  provider: template                  # template | ollama | claude (triage needs ollama or claude)

feedback:
  enabled: false                      # true: learn ranking weights from your ratings
  min_ratings: 10                     # don't tune until this many ratings exist

profiler:
  scan_source: false                  # true: also scan source files for imports/ML patterns
  typed_anchors: false                # true: LLM-extract named entities from the README into
                                      #   anchors. Measured at -0.32 net@2/case (P11); the
                                      #   channel discriminates but does not reach the digest
  prose_chars: 300                    # README chars sent to LLM triage; 0 sends none

openalex:
  api_key: ""                         # required for real use since 2026-02-13; ${ENV} refs work

enrichment:
  provider: huggingface               # or `off` to skip the HF Papers lookup entirely

signals:
  integrity: true                     # check whether a paper was withdrawn by its authors
  hackernews: false                   # look up Hacker News discussion (badge; see below)

recommendations:
  enabled: false                      # true: seed the free S2 recommender with your stars/ratings
  limit: 20                           # how many recommendations to request per run
  max_seeds: 50                       # cap on example papers sent

privacy:
  redact: []                          # terms stripped from outbound queries and LLM prompts.
                                      # Literal and case-insensitive; prefix with `re:`
                                      # for a regex. See `rr audit`.

output:
  digest_path: ./reporadar_digest.md  # Default output path
  top_n: 15                           # Max papers in digest
```

`w_specter` also needs at least one **starred or 4–5-star** paper (it scores how close
each candidate is to that work, in citation-trained space). Vectors are fetched once
from Semantic Scholar and cached (~3 KB per paper); `rr update --rebuild-embeddings`
clears every cached vector. The component is skipped when a run's papers are too few
or too similar to rank meaningfully.

`w_community` scores each paper on its Hugging Face Papers upvote count, log-scaled
against the highest count in the same run (upvotes are heavy-tailed, so a linear
scale would flatten everything below the leader). It reuses the upvotes that
enrichment already stores — and because enrichment runs *after* ranking (it only
fetches for the papers that made the digest), the signal comes from **previous**
runs. A paper seen for the first time therefore has no community score yet and is
scored on the other components alone rather than penalized for it. Papers with zero
upvotes are likewise treated as "no signal", since a zero usually just means HF has
no page for that paper. Setting `enrichment.provider: off` therefore also stops the
community signal from ever refreshing (quoting it is not required — a bare `off` is
handled, even though YAML would otherwise read it as the boolean `false`).

### Withdrawal detection and Hacker News attention

`signals.integrity` (**on by default**) asks arXiv whether any of the run's papers
have been withdrawn by their authors. It is the one signal that defaults on, because
a withdrawn paper is the case where ranking can actively waste your time: it reads as
fresh and on-topic, and you can spend an hour on a result its own authors retracted.

A flagged paper's score is multiplied by `withdrawn_penalty` (0.1). That is a
multiplier, not another weighted component, so a withdrawn paper **cannot** reach Top
Picks by scoring well everywhere else — 1.0 × 0.1 = 0.1, below the Maybe Relevant
threshold. It is not dropped to zero, because a paper you may have seen elsewhere is
better shown flagged than silently missing; the digest gets a "Withdrawn by their
authors" section that appears regardless of tier.

arXiv has no withdrawal *field* — the notice is hand-written free text in the paper's
comment, so the matcher is the whole feature. It reads the comment liberally (the most
common real comment is the single word "Withdrawn") and prose conservatively, since an
abstract may legitimately discuss a drug withdrawn from the market, and it covers
"retracted" and "the authors withdrew this" as well as "withdrawn". Measured at
100% recall on notices phrased "withdrawn" and 83–85% on the other two verbs, with
no confirmed false positive over 600 ordinary papers. Papers are
re-checked at most weekly, capped per run, oldest first — withdrawal can happen days
after a paper is ingested, but chasing it every run would cost minutes of throttled
requests for a signal that fires for well under 1% of papers.

`signals.hackernews` (off by default) badges papers that were discussed on HN, with
points and a link. Be aware of what it is: across **340 random papers from the last
two weeks, zero had any HN story** — including 0/40 in cs.LG. HN surfaces a handful of
papers a week across all of science, so treat `w_attention` as a bonus that
occasionally fires, not a component you can rank on. Points are scaled against a fixed
reference rather than the run's maximum (the usual RepoRadar pattern), because with 0–1
discussed papers per run a relative scale would award 1.0 to a 12-point story.

Recommendations need at least one **starred or 4–5-star** paper (low-rated papers
become negative examples that suppress similar results; an explicit low rating
beats an implicit star from `rr open`). They're merged into the normal candidate
pool and **re-scored by RepoRadar**, and only those clearing the relevance bar are
shown — the API is repo-agnostic, so off-topic suggestions are dropped rather than
displayed, falling back to the local keyword recommender.

## How It Works

### Profiling

The profiler scans your repo for text to build a topic profile:

1. **README** (supports `.md`, `.rst`, `.txt` variants) and files in `docs/`
2. **Dependency manifests** — `requirements.txt`, `pyproject.toml`, `package.json`
3. **TF-IDF** — extracts up to 20 keywords (unigrams + bigrams) from the collected text
4. **Anchors** — package names from manifests, mapped to domain labels (e.g., `torch` → "deep learning")

### Query Building

Queries are built from three sources:

1. **Seed queries** from config — wrapped in exact-match quotes (e.g., `all:"retrieval augmented generation"`)
2. **Phrase queries** — pairs of adjacent profile keywords, quoted (e.g., `all:"speech recognition"`)
3. **Auto-generated** — top 5 profile keywords as individual queries (e.g., `all:transformers`)

All queries are scoped to your configured arXiv categories (e.g., `cat:cs.LG OR cat:cs.CL`).

`queries.bigrams` controls step 2, and defaults to **`verified`**: a pair is only asked for if
it occurs, literally, in your repo's own text. The original behaviour (`adjacent`) paired
keywords by TF-IDF rank alone, which produced phrases no repository contains — `"use page"` for
duckdb, `"data cd"` for redis. On arXiv this barely matters, because the category clause keeps
results in the right field regardless; on `sources:` like DBLP or bioRxiv, which have no
category to fall back on, it is the whole query. `none` disables phrase queries and measured
*worse* than either. Full three-arm measurement in `evals/RESULTS.md`.

### Scoring

Each paper gets a **weighted average** of whichever components have data:

```
score = (Σ wᵢ · componentᵢ / Σ wᵢ) · exclude_penalty · withdrawn_penalty
```

Always present:

- **Keyword score** (0–1) — fraction of profile keywords found in paper title + abstract, weighted by TF-IDF weight
- **Recency score** (0–1) — linear decay from 1.0 (today) to 0.0 at the lookback boundary
- **Category score** (0–1) — fraction of target categories that appear in the paper's categories

Optional, each contributing only when its weight is above zero *and* the signal exists:
`w_embedding`, `w_citations`, `w_citation_proximity`, `w_specter`, `w_community`, `w_attention`.

Two properties follow from that formula and are easy to get wrong:

- **A missing signal is not a zero.** A component with no data is left out of *both* the numerator and the
  denominator, so a paper SPECTER2 has never seen is not handicapped against one it has. The same rule drops
  the category term for non-arXiv sources (Semantic Scholar, DBLP, bioRxiv) that carry no categories at all.
- **The score is normalized, so scaling every weight changes nothing.** Only the *ratios* matter. This also
  means the digest tier thresholds (0.5 / 0.2) live on the normalized 0–1 scale, not on the raw weighted sum.

Multipliers apply after normalization: **exclude penalty** (each matched exclude term halves the score) and
**withdrawn penalty** (`ranking.withdrawn_penalty`, default 0.1 — see withdrawal detection above).

### Digest Tiers

Papers are categorized into three tiers based on their combined score:

- **Top Picks** (score >= 0.5) — full details with score breakdown, abstract snippet, and action suggestions
- **Maybe Relevant** (score >= 0.2) — condensed details
- **Muted** (score < 0.2) — title and link only

With `triage.enabled`, the LLM actionability score replaces the heuristic score for
tiering: `llm_score >= min_actionable` is a Top Pick, `>= 1` is Maybe, `0` is Muted.

### HyDE discovery (`hyde`)

The measurement that motivates this is the bluntest in the project: across nine benchmark
repositories RepoRadar's own queries fetched **2,030 papers and reached 0** of the 24 that
an independent agentic baseline recommended and a judge confirmed useful. Not a ranking
miss — a disjoint set, with 23 of the 24 inside the categories being searched, and arXiv
returning them at rank 1 given the right phrase.

The cause is a **register mismatch**: a codebase's vocabulary, and anything derived from
reading it, describes what the project *has*. The useful paper describes what it should
*adopt*. Asking an LLM to name what the repo lacks does not help either — it names a
plausible *different* research agenda, and ranks the true targets worse than random even
after the phrasing gap is closed entirely.

HyDE routes around the mismatch rather than trying to close it. The LLM writes the
**abstract of a paper that does not exist** but which, if it did, would most improve this
codebase. That text is in the literature's register by construction, so it can be matched
against real abstracts in a dense space — where "experience replay prioritization methods"
and "prioritized experience replay" are neighbours instead of disjoint strings.

```bash
rr sync-index          # one time: ~432 MB of column-pruned range reads, resumable
rr sync-index --verify # check our vectors reproduce the index bit-for-bit
```

Then set `hyde.enabled: true`.

**Retrieval reach** (blind — the generator never saw the targets): **27 of 48** known-good
papers in the top 1,000, median rank 837, against 10/48 for embedding the README and 3/48
for keyword queries *in the same index*. **15 of those targets are unreachable by the
citation hop**, including every repository with no arXiv bibliography.

**End to end**, with HyDE the only variable in a paired 22-repo run: mean net@2
**+3.18 → +4.55**, which takes the system past the Opus 4.8 baseline at **p = 0.0075**
(15 wins / 3 losses). It roughly doubles every candidate pool, and precision *rises*
(0.91 → 0.94) rather than falling — because the fine-scale rescore orders the extra
admits. Three of the six repos that had been losing to the baseline on pure recall are
fixed outright, including one that had been returning nothing at all.

Three things worth knowing:

- **Discovery becomes offline, which is a privacy improvement.** Once synced, matching
  sends nothing — versus the keyword path, which transmits repo-derived queries to arXiv on
  every run. `rr audit` reports it as such.
- **A wrong encoder is undetectable downstream**, so it is checked rather than assumed:
  before searching, RepoRadar reproduces published vectors bit-for-bit and **refuses to
  run** if it cannot. Leave `verify_encoder` on.
- **The index is a third-party mirror**, republished periodically. RepoRadar warns when the
  local copy passes `stale_after_days` (default 60), because a stale index silently stops
  seeing recent work.

Costs ~1.1 GB local (432 MB index + ~670 MB model weights), one LLM call per run for the
hypotheses, and a few seconds of CPU. Install with `uv pip install -e ".[hyde]"`.

### Fine-scale rescore (`triage.finescale`)

The 0-3 triage gate turns out to be near-binary: almost everything it admits scores 2,
score 3 is rare, and **within that score-2 band** the share of genuinely useful papers
ranged from 0% to 100% across a 22-repo benchmark. Two repos could each admit ten papers
with indistinguishable gate scores while one band was entirely useful and the other half
noise — so on band-heavy repos, Top Picks was a coin flip.

The fix is to ask the same question at a finer resolution and read the answer's
*distribution* instead of its sample: the model emits one digit 0-9, and the score is the
expectation over that digit's token probabilities, so "mostly 7, maybe 8" scores 7.3
instead of collapsing to a rung. That expectation goes through a frozen two-parameter
logistic to a probability, and a band paper is a Top Pick only if it clears `threshold`.

The default threshold of **2/3 is derived, not tuned**: the benchmark metric values a
shown paper at `3p - 2`, so showing pays exactly above p = 2/3.

Measured on a live end-to-end run of the 22-repo benchmark: it orders the band at
ROC-AUC 0.84 and lifts mean net@2 from **+1.86 to +3.18**, rescuing every net-negative
repo (six of them on that draw). Papers shown drop 131 → 97 while genuinely useful ones
only drop 101 → 88 — precision 0.77 → **0.91**. Against the Opus 4.8 baseline on the same
22 repos that is **+3.18 vs +1.82** (paired +1.36; 10 better, 6 worse, 6 tied — a sign
test at p = 0.45, so ahead on the mean but not established per repo). Details, plus the
four approaches that lost to it, are in [evals/RESULTS.md](evals/RESULTS.md) under
"Ranking the score-2 band".

Two things to know before enabling it:

- **It needs `OPENAI_API_KEY`, and only OpenAI.** Reading a token distribution requires
  logprobs, which the Anthropic API does not expose. Approximating them by sampling Haiku
  ten times was measured and is *much* worse (AUC 0.59 vs 0.84) — the model is nearly
  deterministic at default temperature, so the samples re-read the mode rather than
  revealing the distribution.
- **The probability map is calibrated to a specific prompt.** Editing the rubric or the
  repo-context block without refitting silently decalibrates the threshold. Both are
  pinned by tests for that reason.

Cost is roughly one gpt-4o-mini call per band paper (~15 per run, well under a cent). If
the stage fails for more than half the band — a bad key, an outage — the gate is skipped
for that run with a warning, rather than demoting everything and looking like a
deliberate abstention.

### Action Suggestions

Top-scoring papers get up to 3 template-based suggestions, derived from pattern matching against the abstract:

| Pattern detected | Example suggestion |
|---|---|
| Benchmark/evaluation mentioned | "Add evaluation on {benchmark}" |
| Outperforms a baseline | "Compare your approach against {baseline}" |
| Proposes a new method | "Explore integrating the proposed {method}" |
| Dataset/corpus referenced | "Consider using the {dataset} dataset" |
| SOTA claim | "Claims SOTA on {task} — worth checking" |
| Open-source code available | "Code/data may be publicly available" |
| Modular/plug-in component | "Consider adding as a feature flag" |
| New loss/optimizer | "Try swapping your optimizer/loss for {name}" |

Suggestions are clearly labeled as auto-generated starting points.

## Development

```bash
# Install with dev dependencies
uv pip install -e ".[dev]"

# Run tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=reporadar --cov-report=term-missing
```

## Project Structure

```
src/reporadar/
  cli.py              # Click CLI entry points (every `rr` command)
  config.py           # YAML config loading/validation
  store.py            # SQLite storage, dedup, and the versioned migration chain

  profiler.py         # Repo topic profiling (TF-IDF)
  source_analysis.py  # Optional import/ML-pattern scan of the repo's source
  collector.py        # arXiv API querying + query building
  sources/            # Adapters that FIND papers, opt-in via `sources:`
    semantic_scholar.py  openalex.py  biorxiv.py  dblp.py
    hf_papers.py         # HF Papers enrichment (code/model/dataset links, upvotes)
    s2_recommendations.py# learned recommendations seeded by your stars/ratings
    suggest.py           # suggests a domain source from the repo profile
  signals/            # Adapters that say something ABOUT a paper you already have
    integrity.py         # withdrawal detection (on by default, hard penalty)
    hn.py                # Hacker News attention (opt-in badge + w_attention)

  ranker.py           # Weighted-average scoring across all components
  retrieval.py        # BM25 + RRF hybrid fusion
  embeddings.py       # Sentence-transformer encoding (optional extra)
  embedding_cache.py  # Compute-once vector cache
  vec_index.py        # sqlite-vec KNN with a numpy fallback
  semantic.py         # Semantic/hybrid search over the stored corpus
  search.py           # Offline BM25 corpus search (`rr search`)
  specter.py          # SPECTER2 vectors + centroid query (served by S2)
  citations.py        # Citation counts and reference fetching
  citation_graph.py   # "extends work you starred" link finding
  triage.py           # LLM actionability scoring + listwise rerank
  llm_client.py       # Shared LLM transport (Ollama / Claude)
  llm_suggestions.py  # LLM-generated action ideas
  suggestions.py      # Template-based action suggestions
  feedback.py         # Learns ranking weights from your ratings
  trends.py           # Keyword-frequency trend detection

  evaluation.py       # `rr eval` — scores the ranker against your own ratings
  metrics.py          # Shared IR metrics (also re-exported by evals/)

  digest.py           # Digest generation: md / html / json / csv / rss
  archive.py          # Dated GitHub Pages archive + index
  notify.py           # Shell / Slack / Discord / email notifications
  gh_issues.py        # Opens GitHub issues for top papers
  watcher.py          # `rr watch` polling loop
  scheduler.py        # OS-level scheduling helpers
  workspace.py        # Multi-repo workspaces
  mcp_server.py       # MCP server (`rr mcp`) for coding agents
  output.py           # Console output helpers
  templates/          # Jinja2: digest.md, digest_page.html (the rendered page),
                      # digest.html (legacy wrapper), digest.rss.xml,
                      # archive_index.html, workspace_digest.md
tests/                # 46 test modules; fixtures/ holds sample repos + frozen arXiv data
evals/                # Standalone benchmark (Tier A/B/S) — see evals/README.md
```

## License

MIT
