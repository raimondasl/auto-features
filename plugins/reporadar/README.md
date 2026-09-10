# RepoRadar — GitHub Copilot plugin

Repository-conditioned paper discovery: papers this project should **act on**, not papers about
its topic.

```
/plugin marketplace add raimondasl/auto-features
/plugin install reporadar@reporadar
```

`reporadar@reporadar` is `PLUGIN@MARKETPLACE`; both are named `reporadar` here, so it reads like
a typo and is not.

## Then just ask

```
/paper-discovery
```

or ask in your own words — "what recent research applies to this project?". There is nothing to
initialise first. The agent will read the repository, propose arXiv categories for it, confirm
them with you, write the configuration and collect. Collection takes minutes and reports progress
as it goes.

You do not need the `rr` command for any of that. It is one exception, below, and it is for a
secret.

## One key, once

Without an API key the actionability gate is skipped, and an ungated digest measured mean net@2
**−11** — worse than showing nothing, because the metric charges 2 for every unactionable paper.
So this matters more than it looks.

```bash
uvx --from reporadar-papers rr auth      # prompts without echoing
```

This is the one thing the plugin cannot do for you, and it should not try: anything typed into a
chat is in the transcript. An exported `OPENAI_API_KEY` also works and takes precedence — `rr
auth` exists because a server your *editor* launched does not reliably inherit your shell
environment, and the other three places a key could live are all closed to it. `.reporadar.yml`
gets committed, `.mcp.json` is in a public repository, and a tool argument would reach the model.

One key is enough: `suggestions.provider: openai` runs the whole pipeline on OpenAI. The
fine-scale rescore is OpenAI-only regardless, because it reads logprobs and no other vendor
exposes them.

## Optional: dense discovery

```
/sync-index
```

A one-time ~1.1 GB download, worth **+1.36 net@2**, and the only retrieval channel that reaches
15 of 48 benchmark targets — including every repository with no arXiv bibliography. It is a
command rather than a tool because 1.1 GB does not belong inside a tool call, and because making
it a tool would force the embedding model into every installation including everyone who never
syncs. The download is shared by every repository on the machine, so it is once per machine, not
once per project.

## The CLI is still here

`rr` remains fully supported and is what the GitHub Action runs; `rr update`, `rr digest`,
`rr watch` and `rr schedule` all work as before. It is simply no longer a *prerequisite* for the
plugin. Both front doors call the same functions, so a repository set up either way is the same
repository.

```bash
uv tool install reporadar-papers            # if you want `rr` on PATH
uv tool install "reporadar-papers[hyde]"    # ...with dense discovery
```

`rr doctor` is still the fullest diagnosis of a configuration — it names every gap and what each
one costs — and is worth reaching for when results look thin.

## Where this works

Copilot CLI, the GitHub Copilot desktop app (Customize → Plugins → the gear by the marketplace
dropdown), VS Code (add `raimondasl/auto-features` to `chat.plugins.marketplaces`), the Copilot
cloud agent (declaratively, via `.github/copilot/settings.json`), and Claude Code. There is no
plugin support in Copilot Chat on github.com, on GitHub Mobile, or in JetBrains outside
enterprise-managed settings.

You need `uv` on PATH: the server is launched with `uvx`.

## What installing it runs

Plugins execute with your own permissions, and this one starts a process: `.mcp.json` launches
the server with `uvx`, which downloads `reporadar-papers` from PyPI on first use. That is ordinary
for the ecosystem, and the advice that goes with it is to read a plugin before installing it — the
whole of this one is the files in this directory.

## Notes

The MCP server is launched by `uvx` from a **pinned PyPI release**
(`reporadar-papers[mcp]==1.0.1`) — not from `main`, and no longer from a git tag — so what you run
does not change under you when this repository is pushed to. Upgrading is a version bump in
`.mcp.json`. It also installs a wheel instead of cloning and building the repository.

Moving off `git+` retired a trap worth recording. While the spec resolved against a tag, the
distribution name inside it had to match that tag's `pyproject.toml`, so renaming the distribution
and cutting the tag had to land in the same commit or `uv` failed the install with a metadata name
mismatch — a broken plugin rather than a broken build. A version pin has no such coupling. The
distribution is `reporadar-papers` because PyPI refuses `reporadar`; the command is still `rr`.

Every published number was measured with the gate on `claude-haiku-4-5`. An OpenAI gate has since
been measured too, and is a wash in the shipped digest: +0.08 net@2 paired over 37 cases, CI
[−0.86, +1.02], under both judges (NR-63).
