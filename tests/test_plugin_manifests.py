"""Structural checks on the Copilot plugin manifests.

These files had two type errors on required fields -- `owner` and `author` were strings
where the spec and GitHub's own manifests use objects -- and both survived into a public
repository because nothing validated them. They each declared a `$schema`, which looked
like validation but was not: both URLs 404, so no editor or CI ever checked anything.

So this is the validation those `$schema` lines were pretending to be. Shapes here were
checked against GitHub's canonical worked example, `github/awesome-copilot`:
`.github/plugin/marketplace.json` uses `owner: {name, email}` with `description`/`version`
nested under `metadata`, and `plugins/*/plugin.json` uses `author: {name, url}`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MARKETPLACE = ROOT / ".github" / "plugin" / "marketplace.json"
PLUGIN = ROOT / "plugins" / "reporadar" / "plugin.json"
MCP_JSON = ROOT / "plugins" / "reporadar" / ".mcp.json"
# Claude Code reads ONLY this path; Copilot reads MARKETPLACE above.
CLAUDE_MARKETPLACE = ROOT / ".claude-plugin" / "marketplace.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class TestTheManifestsAreShapedTheWayTheLoaderExpects:
    def test_marketplace_owner_is_an_object_not_a_string(self) -> None:
        """`owner` is a required top-level field, and a wrong type on a required field is
        the one error here that can fail `plugin marketplace add` outright rather than
        degrade quietly."""
        owner = _load(MARKETPLACE)["owner"]
        assert isinstance(owner, dict), f"owner must be an object, got {type(owner).__name__}"
        assert owner.get("name"), "owner.name is required"

    def test_plugin_author_is_an_object_not_a_string(self) -> None:
        author = _load(PLUGIN)["author"]
        assert isinstance(author, dict), f"author must be an object, got {type(author).__name__}"
        assert author.get("name"), "author.name is required"

    def test_marketplace_listing_fields_live_under_metadata(self) -> None:
        """`description` and `version` belong in `metadata`, not at the top level, which is
        where they were and where the reference documents no such fields."""
        m = _load(MARKETPLACE)
        assert "description" not in m and "version" not in m
        assert m["metadata"]["description"] and m["metadata"]["version"]

    @pytest.mark.parametrize("path", [MARKETPLACE, PLUGIN], ids=["marketplace", "plugin"])
    def test_no_schema_url_we_do_not_actually_validate_against(self, path: Path) -> None:
        """Both files pointed `$schema` at json.schemastore.org URLs that 404. A declared
        schema nobody checks is worse than none: it reads as validation and provides none,
        which is how two type errors reached a public repo. Add one back only alongside a
        check that actually runs it."""
        assert "$schema" not in _load(path)


class TestTheManifestsAgreeWithWhatTheyPointAt:
    def test_the_marketplace_entry_resolves_to_a_real_plugin(self) -> None:
        entry = _load(MARKETPLACE)["plugins"][0]
        target = (ROOT / entry["source"].removeprefix("./")).resolve()
        assert target.is_dir(), f"source {entry['source']} does not exist"
        assert (target / "plugin.json").is_file()

    def test_plugin_declares_paths_that_exist(self) -> None:
        p = _load(PLUGIN)
        base = PLUGIN.parent
        assert (base / p["mcpServers"].removeprefix("./")).is_file()
        skills = base / p["skills"].removeprefix("./")
        assert skills.is_dir()
        assert list(skills.glob("*/SKILL.md")), "no SKILL.md under the declared skills dir"

    def test_versions_move_together(self) -> None:
        """The marketplace entry, the plugin manifest and the version `.mcp.json` installs
        are kept in step deliberately. A marketplace entry's version pins what already
        installed users receive, so a manifest that stays put while `.mcp.json` moves ships
        a fix nobody gets -- the mirror of the metadata-name mismatch that preceded it.
        """
        entry_version = _load(MARKETPLACE)["plugins"][0]["version"]
        plugin_version = _load(PLUGIN)["version"]
        assert entry_version == plugin_version

        spec = " ".join(_load(MCP_JSON)["mcpServers"]["reporadar"]["args"])
        pinned = re.search(r"reporadar-papers\[[a-z]+\]==([0-9][^\s\"]*)", spec)
        assert pinned is not None, f"no pinned version found in .mcp.json args: {spec}"
        assert pinned.group(1) == plugin_version, (
            f".mcp.json installs {pinned.group(1)} but the plugin manifest says "
            f"{plugin_version}; keep them in step"
        )


class TestTheSkillDescribesTheServerItFronts:
    """SKILL.md is the agent's whole picture of these tools, and nothing checked it against
    the server. That is how it came to document `rate_paper` as a 0-3 scale for a tool that
    rejects anything outside 1-5: an agent following the skill would send values the tool
    refuses, and no gate anywhere would notice."""

    SKILL = ROOT / "plugins" / "reporadar" / "skills" / "paper-discovery" / "SKILL.md"
    SERVER = ROOT / "src" / "reporadar" / "mcp_server.py"

    def _registered(self) -> set[str]:
        # Parsed from source rather than imported: the `mcp` extra is not installed in CI
        # (`uv sync --extra dev --extra evals`), so `build_server` cannot be called here.
        source = self.SERVER.read_text(encoding="utf-8")
        return set(re.findall(r"@server\.tool\(\)\s*\n\s*def (\w+)\(", source))

    def _documented(self) -> set[str]:
        rows = self.SKILL.read_text(encoding="utf-8")
        return set(re.findall(r"^\| `(\w+)` \|", rows, re.MULTILINE))

    def test_the_skill_documents_exactly_the_tools_the_server_registers(self) -> None:
        registered, documented = self._registered(), self._documented()
        assert registered, "no @server.tool() functions found; the parse is out of date"
        assert documented == registered, (
            f"SKILL.md documents {sorted(documented)} but the server registers "
            f"{sorted(registered)}; undocumented tools go unused and documented-but-absent "
            f"ones get called and fail"
        )

    def test_the_documented_rating_range_is_the_one_the_tool_enforces(self) -> None:
        source = self.SERVER.read_text(encoding="utf-8")
        bounds = re.search(r"not (\d+) <= rating <= (\d+)", source)
        assert bounds is not None, "rate_paper's range check moved; update this guard"
        lo, hi = bounds.groups()

        row = next(
            line
            for line in self.SKILL.read_text(encoding="utf-8").splitlines()
            if line.startswith("| `rate_paper`")
        )
        # The FIRST range in the row, not merely some range in it. The row legitimately
        # mentions 4-5 and 1-2 further along (what the feedback loop learns from), so an
        # "appears anywhere" check passes even when the headline scale is wrong -- which is
        # precisely the drift this guard exists to catch.
        first = re.search(r"(\d)\s*[-–—]\s*(\d)", row)
        assert first is not None, f"SKILL.md's rate_paper row states no range: {row[:120]}"
        assert first.groups() == (lo, hi), (
            f"rate_paper enforces {lo}-{hi} but SKILL.md leads with "
            f"{first.group(1)}-{first.group(2)}: {row[:120]}"
        )


class TestBothEcosystemsGetTheSameMarketplace:
    """Two copies of one manifest, because the two loaders disagree on where it lives.

    Copilot searches `marketplace.json`, `.plugin/`, `.github/plugin/`, `.claude-plugin/`.
    Claude Code searches only `.claude-plugin/marketplace.json` -- verified on 2.1.198,
    where `claude plugin marketplace add` fails with "Marketplace file not found at
    .../.claude-plugin/marketplace.json", and the string ".github/plugin" does not
    appear in its binary at all. So supporting both means duplicating the file, and
    duplicated files drift. This is what stops them.
    """

    def test_claude_code_finds_a_marketplace_where_it_looks(self) -> None:
        assert CLAUDE_MARKETPLACE.is_file(), (
            "Claude Code reads only .claude-plugin/marketplace.json; without it "
            "`claude plugin marketplace add` fails outright"
        )

    def test_the_two_copies_have_not_drifted(self) -> None:
        """Compared parsed rather than byte-for-byte: line endings are normalised on
        checkout, and it is the content both loaders read that has to agree."""
        assert _load(CLAUDE_MARKETPLACE) == _load(MARKETPLACE), (
            ".claude-plugin/marketplace.json and .github/plugin/marketplace.json differ; "
            "they describe the same marketplace to two loaders and must stay identical"
        )

    def test_the_relative_source_resolves_from_the_repository_root(self) -> None:
        """Both loaders resolve a plugin `source` against the repo root, not against the
        manifest's own directory -- which is why one `./plugins/reporadar` works from two
        different places."""
        entry = _load(CLAUDE_MARKETPLACE)["plugins"][0]
        target = (ROOT / entry["source"].removeprefix("./")).resolve()
        assert (target / "plugin.json").is_file()
