# RepoRadar — GitHub Copilot plugin

Repository-conditioned paper discovery: papers this project should **act on**, not papers about
its topic.

```
/plugin marketplace add raimondasl/auto-features
/plugin install reporadar@reporadar
```

`reporadar@reporadar` is `PLUGIN@MARKETPLACE`; both are named `reporadar` here, so it reads like
a typo and is not.

**Where this works.** Copilot CLI, the GitHub Copilot desktop app (Customize → Plugins → the gear
by the marketplace dropdown), VS Code (add `raimondasl/auto-features` to `chat.plugins.marketplaces`),
the Copilot cloud agent (declaratively, via `.github/copilot/settings.json`), and Claude Code. There
is no plugin support in Copilot Chat on github.com, on GitHub Mobile, or in JetBrains outside
enterprise-managed settings.

**What installing it runs.** Plugins execute with your own permissions, and this one starts a
process: `.mcp.json` launches the server with `uvx`, which downloads `reporadar-papers` from PyPI
on first use. That is ordinary for the ecosystem, and the advice that goes with it is to read a
plugin before installing it — the whole of this one is the four files in this directory.

Installing the plugin does **not** give you the `rr` command. The plugin launches the MCP
server in its own throwaway environment; the CLI is a separate install, and the setup below
needs it:

```bash
uv tool install reporadar-papers            # puts `rr` on PATH
uv tool install "reporadar-papers[hyde]"    # ...or this, if you also want `rr sync-index`
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

```bash
rr auth          # prompts without echoing, stores it readable only by you
rr auth --status # says what is stored and where it came from, never the key itself
```

An exported `OPENAI_API_KEY` still works and still wins over the stored file. `rr auth` exists
because the server here is launched by your editor rather than your shell, and does not reliably
inherit your environment — and the three other places a key could go are all closed to it:
`.reporadar.yml` gets committed, `.mcp.json` lives in a public repository, and anything passed
through a tool call ends up in the model's context.

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
