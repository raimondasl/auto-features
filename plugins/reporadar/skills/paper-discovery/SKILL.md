---
name: paper-discovery
description: >
  Find research papers a repository should ACT ON — techniques it could adopt — rather than
  papers about its topic. Use when the user asks what recent work applies to this project, what
  they should read or implement, whether a paper is relevant here, or asks to review the
  research landscape for the codebase they are in. Not for general literature search unattached
  to a repository.
---

# Paper discovery for this repository

RepoRadar answers a narrower question than literature search: **given this repository, which
papers could genuinely improve it?** That is a utility judgment, not a topical-relevance one,
and the correct answer is often *none* — an empty result is a real answer, not a failure.

It profiles the working tree, retrieves candidates, gates them with an LLM actionability
judgment, and orders survivors with a calibrated rescore. On a 25-repository benchmark it scores
mean net@2 **+5.72** against **+1.84** for an agentic frontier-model baseline given the same
repository, at roughly one-fortieth the cost per repository.

## When to use this

Reach for these tools when the user asks:

- "what recent research applies to this project?"
- "is there work I should be building on here?"
- "should we implement anything from the literature?"
- "is this paper relevant to us?" (→ `explain_relevance`)
- "find me papers about X for this repo" (→ `search_papers`)

Do **not** use it for literature search unconnected to a repository — it conditions everything
on the working tree, and asked about a topic in the abstract it will answer about the wrong
thing.

## Tools

| tool | use |
|---|---|
| `get_repo_profile` | What RepoRadar thinks this project is. Check this first when results look wrong — a bad profile explains a bad digest. |
| `get_ranked_papers` | The digest: papers judged actionable for this repository, best first. |
| `explain_relevance` | Why one specific arXiv paper was or was not surfaced. |
| `search_papers` | Search the stored corpus by query, still conditioned on this repository. |
| `rate_paper` | Record the user's **1–5** usefulness rating. Anything outside 1–5 is rejected. A 3 is accepted and then *ignored* by the feedback loop, which learns only from 4–5 and 1–2 — so do not default to 3 when the user is vague, ask them. |

`get_ranked_papers` reads what has already been collected. If it returns nothing, the store is
probably empty rather than the literature — say so and suggest `rr update`, rather than
reporting "no relevant papers exist".

## Setup, and what to say when it is missing

The tools need RepoRadar initialised in the repository — and they need the `rr` command, which
**installing the plugin does not provide**. The plugin runs the MCP server in its own throwaway
environment, so if `rr` is not on the user's PATH that is the first thing to fix, not a sign that
anything is broken:

```bash
uv tool install reporadar-papers   # `rr` on PATH; add [hyde] if they also want `rr sync-index`
rr init --measured     # writes the configuration every published number was measured under
rr doctor              # says what is still missing and what each gap costs
```

**`rr doctor` is the thing to run when anything looks off**, and worth suggesting proactively
if results seem thin. It exits non-zero and names each gap with its measured cost. Three gaps
are common and every one of them fails *silently* at run time:

- **No API key** — the actionability gate is skipped entirely, and an ungated digest measured
  mean net@2 **−11**. One key suffices: set `suggestions.provider: openai` and the whole
  pipeline runs on OpenAI.
- **No dense index** — `rr sync-index` is a one-time ~1.1 GB download. Skipping it costs
  **−1.36 net@2**, and it is the *only* retrieval channel for 15 of 48 benchmark targets,
  including every repository with no arXiv bibliography. Large, optional, and worth it.
- **Default `arxiv.categories`** — `cs.LG, cs.CL` is a guess that fits an ML repository and no
  other. On the wrong field it is the difference between a digest and noise.

If the user has not run `rr init`, the MCP server will not start and will say so. Point them at
these two commands rather than guessing at the cause.

## Reporting results honestly

**An empty digest is a legitimate answer.** The system is built to abstain, three benchmark
repositories are negative controls where the correct output is nothing, and the metric gives
abstention a defined value. Do not pad a thin result to seem useful.

**Do not restate the paper's abstract as a recommendation.** The useful thing is what the
repository would *do* with it. `explain_relevance` gives the reasoning behind a pick; prefer it
over your own guess at why a paper was surfaced.

**Coverage is arXiv's.** For repositories whose literature lives elsewhere — cryptography,
databases, most of applied science — near-abstention is the best this can structurally do, and
that is a property of the source rather than of the project. Say so plainly rather than
presenting a weak digest as the state of the field.

**The judgments are one model's opinion under a rubric.** A second judge agrees on ranking and
differs on strictness; papers a repository actually went on to adopt score well above matched
controls, and above papers *other* repositories adopted, so the judgment tracks something real
and repository-specific. It is not ground truth.
