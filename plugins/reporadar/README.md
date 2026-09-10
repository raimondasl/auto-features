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

**The first launch is slow and this is expected.** `uvx` downloads the server and its
dependencies before it can answer, so your editor may sit on "starting" for 30–60 seconds, longer
on a cold cache. Every launch after that is around three seconds. You can get the wait over with
in advance:

```bash
uvx --from "reporadar-papers[mcp]==1.0.2" rr --version
```

If it is still "starting" well after that, it is not the download — check the server output (in
VS Code: **MCP: List Servers** → `reporadar` → **Show Output**), which is where its diagnostics
go.

You do not need the `rr` command for any of that. It is one exception, below, and it is for a
secret.

## One key, once

Without an API key the actionability gate is skipped, and an ungated digest measured mean net@2
**−11** — worse than showing nothing, because the metric charges 2 for every unactionable paper.
So this matters more than it looks.

```bash
uvx --from "reporadar-papers==1.0.2" rr auth                    # OpenAI, the default
uvx --from "reporadar-papers==1.0.2" rr auth --provider claude  # ...or Anthropic
uvx --from "reporadar-papers==1.0.2" rr auth --status           # what is stored, and from where
```

**Store the key before you ask for a digest.** `setup_repo` writes a configuration for whichever
key it can find — OpenAI if you have one, Anthropic if that is what you stored — so doing it in
this order gets you a config that matches your credentials instead of one demanding a vendor you
never signed up with.

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
once per project — and it needs no repository at all, so you can run it before you have set one
up:

```bash
uvx --from "reporadar-papers[hyde]==1.0.2" rr sync-index
```

**Until you do, expect a warning on every collection**: "HyDE discovery unavailable". That is
deliberate rather than a fault. `setup_repo` writes the measured configuration, which has dense
discovery *on*, and the alternative to warning is degrading to the keyword-only path in silence —
the path this project's own benchmark scores at 0 of 24. If you would rather not spend the 1.1 GB,
set `hyde.enabled: false` in `.reporadar.yml` and take the −1.36 knowingly; the warning stops
because the answer changed, not because it was hidden.

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

Copilot CLI, the GitHub Copilot desktop app, VS Code, the Copilot cloud agent, and Claude Code.
There is **no** plugin support in Copilot Chat on github.com, on GitHub Mobile, or in JetBrains
outside enterprise-managed settings.

No paid plan is needed. Agent Plugins are generally available on **all** Copilot plans, and
Copilot Free includes both agent mode and MCP. (The plans table's "third-party agents" row refers
to cloud coding agents — a different product — and does not apply here.) The one real gate is a
**Copilot Business or Enterprise** seat, where an admin must enable the "MCP servers in Copilot"
policy; personal Free/Pro accounts are not governed by it.

You need `uv` on PATH everywhere: the server is launched with `uvx`. Check it resolves in the
same shell you start your editor from.

### VS Code

Command Palette → **Preferences: Open User Settings (JSON)**, and add:

```json
"chat.plugins.enabled": true,
"chat.plugins.marketplaces": ["raimondasl/auto-features"]
```

Then Extensions view (`Ctrl+Shift+X`) → type `@agentPlugins` in the search box → find
**reporadar** → Install. The first install from a new marketplace shows a trust prompt.

Verify with **MCP: List Servers** (`reporadar` should be listed and started) and
**Chat: Configure Skills** (`paper-discovery` and `sync-index` should appear). Then use agent
mode and type `/` — the skills are slash commands.

### Copilot CLI

```bash
copilot plugin marketplace add raimondasl/auto-features
copilot plugin install reporadar@reporadar
```

Or the same two as `/plugin …` inside a session. Verify with `/mcp show reporadar`.

### The Copilot app

**Customize** in the sidebar → **Plugins** → the gear beside the marketplace dropdown → add
`raimondasl/auto-features` → filter to `reporadar` → **Install**.

### Copilot cloud agent

No UI; commit `.github/copilot/settings.json` in the repository the agent works in:

```json
{
  "extraKnownMarketplaces": {
    "reporadar": { "source": { "source": "github", "repo": "raimondasl/auto-features" } }
  },
  "enabledPlugins": { "reporadar@reporadar": true }
}
```

Add a `copilot-setup-steps.yml` step installing `uv` if the agent environment lacks it.

### Claude Code

```bash
claude plugin marketplace add raimondasl/auto-features
claude plugin install reporadar@reporadar
```

## What installing it runs

Plugins execute with your own permissions, and this one starts a process: `.mcp.json` launches
the server with `uvx`, which downloads `reporadar-papers` from PyPI on first use. That is ordinary
for the ecosystem, and the advice that goes with it is to read a plugin before installing it — the
whole of this one is the files in this directory.

## Notes

The MCP server is launched by `uvx` from a **pinned PyPI release**
(`reporadar-papers[mcp]==1.0.2`) — not from `main`, and no longer from a git tag — so what you run
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
