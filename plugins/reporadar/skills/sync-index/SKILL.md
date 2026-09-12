---
name: sync-index
description: >
  Download RepoRadar's dense arXiv index, so paper discovery can search by meaning instead of
  keywords alone. Use when update_corpus warns that HyDE discovery is unavailable, when a digest
  comes back thin, or when the user asks how to get better results out of RepoRadar. One-time
  download of roughly 1.1 GB, shared by every repository on the machine.
---

# Turn on dense discovery

RepoRadar's default retrieval is keyword-based. Dense discovery (HyDE) writes the abstract of the
paper this repository *wishes existed* and searches for that — which finds work whose authors
never used the repository's vocabulary.

It is worth **+1.36 net@2**, and it is the **only** channel that reaches 15 of 48 benchmark
targets, including every repository with no arXiv bibliography. It is also the largest single
thing RepoRadar asks of anyone.

## This one is a command, not a tool

Everything else in this plugin is an MCP tool you call. This is not, for two reasons worth
telling the user if they ask:

- **1.1 GB does not belong inside a tool call.** It would exceed any sane timeout, and progress
  on a download that size is something the user should watch rather than infer.
- **The plugin's server is deliberately light.** Making this a tool would force the embedding
  model into every installation, including everyone who never syncs.

**This is the only part that is manual.** Once the index is on disk, `update_corpus` searches it
by itself — it runs the pipeline in a `uvx` environment that has the embedding model when this
server's own environment does not. So this is a one-time act, not a step to repeat before each
digest, and the user never types a `uvx` command for collection.

## What to run

**Confirm with the user first.** This is a large download and a several-minute wait; never start
it unprompted.

From the repository they want a digest for:

```bash
uvx --from "reporadar-papers[hyde]" rr sync-index
```

It is resumable — an interrupted sync re-fetches only the shards it had not finished, so a
failure part-way is not wasted work.

## What it costs, honestly

- **432 MB** of index plus roughly **670 MB** of embedding weights.
- Both land in user-global caches (`~/.cache/reporadar/hyde-index` and the Hugging Face cache),
  **not** in the repository and not in the plugin. **Sync once and every repository on the
  machine benefits** — if the user has already done this for another project, there is nothing
  to do here.
- The heavy Python environment `uvx` builds is disposable; only the downloaded artefacts persist.

## Afterwards

Nothing to configure: `setup_repo` already writes `hyde.enabled: true`, which is why
`update_corpus` was warning that the index was missing. Call `update_corpus` again and the
warning is gone and the channel is live.

**Tell the user the next collection will be slower** — several minutes longer, once. The
embedding model's dependencies are not in this server's environment and `uvx` builds one that
has them the first time it is needed, then caches it. The result's `collected_in` field names
the environment the pipeline actually ran in, so you can say whether dense discovery was really
in play rather than assuming it.

If the user declines, that is a legitimate choice — say what it costs them (**−1.36 net@2**, and
whole classes of repository unreachable) and carry on without it. RepoRadar works without dense
discovery; it just works less well, and they should know which one they are looking at.
