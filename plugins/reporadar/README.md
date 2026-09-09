# RepoRadar — GitHub Copilot plugin

Repository-conditioned paper discovery: papers this project should **act on**, not papers about
its topic.

```
/plugin marketplace add raimondasl/auto-features
/plugin install reporadar@reporadar
```

Then, in the repository you want a digest for:

```bash
rr init --measured   # the configuration every published number was measured under
rr doctor            # names what is missing and what each gap costs
rr update            # collect candidates
```

## What it needs

One API key. `suggestions.provider: openai` runs the whole pipeline on OpenAI; the fine-scale
rescore is OpenAI-only because it reads logprobs and no other vendor exposes them.

Optionally `rr sync-index` — a one-time ~1.1 GB download for dense discovery. It is worth
+1.36 net@2 and is the only retrieval channel for 15 of 48 benchmark targets, including every
repository with no arXiv bibliography. `rr doctor` will keep telling you it is missing.

## Notes

The MCP server is launched by `uvx` from a **pinned PyPI release**
(`reporadar-papers[mcp]==1.0.0`) — not from `main`, and no longer from a git tag — so what you run
does not change under you when this repository is pushed to. Upgrading is a version bump in
`.mcp.json`. It also installs a wheel instead of cloning and building the repository.

Moving off `git+` retired a trap worth recording. While the spec resolved against a tag, the
distribution name inside it had to match that tag's `pyproject.toml`, so renaming the distribution
and cutting the tag had to land in the same commit or `uv` failed the install with a metadata name
mismatch — a broken plugin rather than a broken build. A version pin has no such coupling. The
distribution is `reporadar-papers` because PyPI refuses `reporadar`; the command is still `rr`.

Every published number was measured with the gate on `claude-haiku-4-5`. An OpenAI gate is
measured to make no difference to the shipped digest — under both judges, across two independent
draws — but it is not the configuration the headline figures were produced under.
