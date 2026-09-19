"""The MCP handshake names RepoRadar's version, not the MCP SDK's.

`FastMCP` takes no version, and the low-level server it wraps falls back to
`importlib.metadata.version("mcp")` when none is set. So a 1.0.7 install introduced itself to
editors as `reporadar 1.30.0` -- the SDK release uvx happened to resolve that day -- which is
what a user reads in the server list when asking which RepoRadar they are running.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("mcp", reason="the mcp extra is not installed in this environment")

import anyio  # noqa: E402
from mcp.client.session import ClientSession  # noqa: E402
from mcp.shared.memory import create_client_server_memory_streams  # noqa: E402

from reporadar import __version__, mcp_server  # noqa: E402


def _repo(tmp_path: Path) -> Path:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n", encoding="utf-8")
    return tmp_path


def test_the_handshake_reports_reporadars_version(tmp_path: Path) -> None:
    server = mcp_server.build_server(repo_path=_repo(tmp_path))._mcp_server
    seen: dict[str, str] = {}

    async def main() -> None:
        async with (
            create_client_server_memory_streams() as (client, served),
            anyio.create_task_group() as tg,
        ):
            tg.start_soon(
                lambda: server.run(served[0], served[1], server.create_initialization_options())
            )
            async with ClientSession(client[0], client[1]) as session:
                result = await session.initialize()
                seen["name"] = result.serverInfo.name
                seen["version"] = result.serverInfo.version
            tg.cancel_scope.cancel()

    anyio.run(main)
    assert seen == {"name": "reporadar", "version": __version__}


def test_it_is_not_the_sdks_version() -> None:
    """Guards the test above against passing by coincidence, were the two ever equal."""
    from importlib.metadata import version

    assert __version__ != version("mcp")
