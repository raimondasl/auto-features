"""Prove the `mcp` extra still yields a server that answers, on a FRESH resolve.

Nothing in `tests/` can do this, and that is the point. CI installs with
``uv sync --locked``, and ``uv.lock`` pins a known-good mcp, so a locked environment can
never see what a user's own resolve gets. That gap shipped a Copilot plugin that could not
start for anyone: the extra was declared ``mcp>=1.0``, a fresh resolve took mcp 2.x, and
2.x dropped ``mcp.server.fastmcp`` -- the module ``build_server`` imports. All four gates
were green the whole time, because none of them ever resolved the extra.

So this runs outside the lock, against a real interpreter, and asserts the server completes
an MCP handshake and lists the tools the plugin advertises. Any extra with a runtime import
deserves the same treatment; this one had a published plugin riding on it.

Usage:  python scripts/mcp_smoke.py <path-to-rr-executable>
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile

# The tools plugins/reporadar/skills/paper-discovery/SKILL.md tells an agent to reach for.
# Listing them here rather than counting means a silently dropped tool fails the check.
EXPECTED_TOOLS = {
    "get_repo_profile",
    "get_ranked_papers",
    "explain_relevance",
    "rate_paper",
    "search_papers",
}

HANDSHAKE = [
    {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "ci-smoke", "version": "1"},
        },
    },
    {"jsonrpc": "2.0", "method": "notifications/initialized"},
    {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
]


def fail(msg: str, stdout: str = "", stderr: str = "") -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    if stdout:
        print(f"--- stdout ---\n{stdout[:2000]}", file=sys.stderr)
    if stderr:
        print(f"--- stderr ---\n{stderr[:2000]}", file=sys.stderr)
    raise SystemExit(1)


def main(rr: str) -> None:
    with tempfile.TemporaryDirectory() as work:
        # `rr mcp` refuses to start without a config, by design, so make one first -- the
        # same two steps the plugin's own README gives a user.
        init = subprocess.run([rr, "init"], cwd=work, capture_output=True, text=True, timeout=180)
        if init.returncode != 0:
            fail("`rr init` failed", init.stdout, init.stderr)

        payload = "".join(json.dumps(r) + "\n" for r in HANDSHAKE)
        try:
            proc = subprocess.run(
                [rr, "mcp"],
                cwd=work,
                input=payload,
                capture_output=True,
                text=True,
                timeout=300,
            )
        except subprocess.TimeoutExpired:
            fail("`rr mcp` never returned; the server hung on the handshake")

        if not proc.stdout.strip():
            # This is the exact shape of the shipped outage: exit non-zero, zero JSON-RPC
            # bytes, and a message on stderr that no MCP client ever shows a user.
            fail(
                f"`rr mcp` produced no JSON-RPC output (exit {proc.returncode})",
                proc.stdout,
                proc.stderr,
            )

        replies = {}
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(msg, dict) and msg.get("id") is not None:
                replies[msg["id"]] = msg

        if 1 not in replies or "result" not in replies[1]:
            fail("no initialize result", proc.stdout, proc.stderr)
        if 2 not in replies or "result" not in replies[2]:
            fail("no tools/list result", proc.stdout, proc.stderr)

        got = {t["name"] for t in replies[2]["result"].get("tools", [])}
        missing = EXPECTED_TOOLS - got
        if missing:
            fail(f"tools/list is missing {sorted(missing)}; got {sorted(got)}")

        server = replies[1]["result"].get("serverInfo", {})
        print(f"ok: handshake completed against {server.get('name')!r}")
        print(f"ok: tools/list returned all {len(EXPECTED_TOOLS)} tools: {sorted(got)}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
