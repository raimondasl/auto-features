"""Prove the server follows the CLIENT's workspace, not its own working directory.

The plugin's server is launched by an editor, which chooses the working directory — and in
practice chose the plugin's own install directory. So a server that inferred the project
from its CWD profiled the plugin instead of the user's code and wrote a configuration
there, which is worse than failing outright: the digest still looks like an answer.

MCP roots is the protocol's answer, and this is the only thing that checks we actually use
it. It drives `rr mcp` as a real MCP client with the process CWD set to a decoy, declares a
root pointing somewhere else, and asserts the server followed the root. It also asserts the
CWD fallback still works when no roots are offered, because that is what keeps `rr mcp`
usable from a terminal.

Needs the `mcp` SDK, so run it with the interpreter that has the extra:

    <venv>/bin/python scripts/roots_probe.py <path-to-rr>
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import ListRootsResult, Root


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


async def _profile(rr: str, cwd: Path, root: Path | None) -> dict[str, Any]:
    """Start the server in *cwd*, optionally declaring *root*, and ask what it profiled."""

    async def _roots(_context: Any) -> ListRootsResult:
        assert root is not None
        return ListRootsResult(roots=[Root(uri=root.as_uri(), name="project")])

    params = StdioServerParameters(command=rr, args=["mcp"], cwd=str(cwd))
    async with (
        stdio_client(params) as (read, write),
        ClientSession(read, write, list_roots_callback=_roots if root else None) as session,
    ):
        await session.initialize()
        result = await session.call_tool("get_repo_profile", {})
        return dict(result.structuredContent or {})


async def main(rr: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        # Stands in for the plugin's install directory: somewhere real, and not the project.
        decoy = Path(tmp) / "decoy"
        project = Path(tmp) / "project"
        for path in (decoy, project):
            path.mkdir()
            (path / "README.md").write_text(f"# {path.name}", encoding="utf-8")
            # Both need a project marker: the server refuses to act on a directory it
            # cannot identify, and this probe is about WHICH directory it picks rather
            # than about that refusal.
            (path / "pyproject.toml").write_text("[project]", encoding="utf-8")

        without = await _profile(rr, cwd=decoy, root=None)
        got = Path(without.get("repo_path", ""))
        if got != decoy:
            fail(f"with no roots the server should fall back to its CWD {decoy}, used {got}")
        source = str(without.get("repo_source", ""))
        # A PREFIX, not an exact match: the fallback now explains itself -- "cwd (client
        # declares no roots capability)" -- and that detail is the point, because "cwd"
        # alone could not distinguish a client that offered nothing from one never asked.
        if not source.startswith("cwd"):
            fail(f"expected a cwd fallback, got {source!r}")
        print(
            f"ok: no roots offered — fell back to the working directory ({without['repo_source']})"
        )

        with_root = await _profile(rr, cwd=decoy, root=project)
        got = Path(with_root.get("repo_path", ""))
        if got != project:
            fail(
                f"the client declared {project} but the server profiled {got}. This is the "
                f"bug: the digest would describe the wrong repository and still look like "
                f"an answer."
            )
        print(f"ok: root declared — followed the client ({with_root['repo_source']})")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    anyio.run(main, sys.argv[1])
