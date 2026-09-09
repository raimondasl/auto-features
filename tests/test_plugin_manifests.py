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
