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
| `setup_repo` | Initialise RepoRadar here. Call it with no arguments first: it answers with the repository's profile and asks which arXiv categories to use. |
| `update_corpus` | Collect, rank and gate papers. Minutes, with progress — the only tool that fetches anything. |
| `rate_paper` | Record the user's **1–5** usefulness rating. Anything outside 1–5 is rejected. A 3 is accepted and then *ignored* by the feedback loop, which learns only from 4–5 and 1–2 — so do not default to 3 when the user is vague, ask them. |

`get_ranked_papers` reads what has already been collected and never fetches on its own. If it
returns nothing, the store is probably empty rather than the literature — call `update_corpus`,
rather than reporting "no relevant papers exist".

## Setup — you do this, the user does not

**Check the directory before you configure anything, and fix it if it is wrong.** Every tool
reports `repo_path` and `repo_source`. If `repo_source` says anything beginning `cwd`, the client
did not tell the server which project it is in, and the server is guessing from its own working
directory — which for a plugin install is the plugin's own folder, not the user's code.

Do not stop there, and do not initialise it either. **You know where the project is** — you are
working in it. Call `setup_repo` again with `repo_path` set to its absolute path. The server
remembers it for the rest of the session, so every later call uses it too.

```
setup_repo(repo_path="/absolute/path/to/the/project")
```

Confirm the path with the user if you are unsure. A digest built for the wrong repository is
worse than no digest, because it looks like an answer.

**Do not send the user to a terminal.** Setup is two tool calls:

1. `setup_repo` with no arguments. It comes back `needs_input`, asking for `categories` and
   handing you this repository's inferred profile — keywords, libraries, domains.
2. Read that profile, propose arXiv categories that fit it, **confirm them with the user**, then
   call `setup_repo` again with `categories`.

Propose rather than ask blind: "this looks like a cryptography library, so `cs.CR` — does that
match?" is a question someone can answer. "Which arXiv categories do you want?" is not. Getting
this wrong is expensive: on the wrong field it is the difference between a digest and noise, and
it is the one setting no benchmark number justifies.

Then call `update_corpus`. It takes minutes and reports progress; say what it is doing rather
than going quiet.

**Read its `warnings` before reporting the result.** They are separate from `progress` because
they change what the digest means: a stage that was configured and could not run — HyDE
unavailable, a source that failed — makes a thin result evidence about the *setup*, not about the
literature. Say which one it is. "Nothing came back, and dense discovery was not running" is a
useful sentence; "no relevant papers exist" in the same situation is wrong.

Any tool answering `{"status": "not_configured"}` means the repository has no config yet — start
at step 1. A tool answering `{"status": "needs_input"}` is asking you for something specific and
naming the call to retry; it is not an error, and reporting it as one strands the user.

## What still needs a key, and what it costs

- **No API key** — the actionability gate is skipped entirely, and an ungated digest measured
  mean net@2 **−11**. One key suffices: `suggestions.provider: openai` runs the whole pipeline
  on OpenAI.

  **This is the one thing you cannot do for the user, and you should not try.** Do not ask them
  to paste a key into the chat and do not accept one if they offer — anything said here is in
  the transcript. They run `rr auth` themselves, once; it prompts without echoing and stores the
  key where the server can read it. An exported `OPENAI_API_KEY` also works when the server can
  see it, which it often cannot, because an editor-launched server does not reliably inherit a
  shell environment.

  If `update_corpus` warns that the gate did not run, this is almost always why. Say what it
  costs — the digest you are looking at is the −11 configuration, not the +5.72 one.
- **No dense index** — the one step that is still a command, because 1.1 GB does not belong
  inside a tool call. Skipping it costs **−1.36 net@2**, and it is the *only* retrieval channel
  for 15 of 48 benchmark targets, including every repository with no arXiv bibliography. The
  `sync-index` skill has the command and the full cost; reach for it when `update_corpus`
  reports HyDE unavailable.

Two gaps that fail *silently* at run time:

- **No API key** — the actionability gate is skipped entirely, and an ungated digest measured
  mean net@2 **−11**. One key suffices: set `suggestions.provider: openai` and the whole
  pipeline runs on OpenAI.
- **No dense index** — `rr sync-index` is a one-time ~1.1 GB download. Skipping it costs
  **−1.36 net@2**, and it is the *only* retrieval channel for 15 of 48 benchmark targets,
  including every repository with no arXiv bibliography. Large, optional, and worth it.
- **Default `arxiv.categories`** — `cs.LG, cs.CL` is a guess that fits an ML repository and no
  other, which is why `setup_repo` refuses to write a config without being told.

`rr doctor` remains the fullest diagnosis and is worth suggesting when results look thin, but it
is a CLI command and the user may not have `rr` installed — the plugin does not provide it. Reach
for it only when the tools themselves have not explained the problem.

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
