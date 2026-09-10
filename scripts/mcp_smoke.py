"""Prove the `mcp` extra still yields a server that answers, on a FRESH resolve.

Nothing in `tests/` can do this, and that is the point. CI installs with
``uv sync --locked``, and ``uv.lock`` pins a known-good mcp, so a locked environment can
never see what a user's own resolve gets. That gap shipped a Copilot plugin that could not
start for anyone: the extra was declared ``mcp>=1.0``, a fresh resolve took mcp 2.x, and
2.x dropped ``mcp.server.fastmcp`` -- the module ``build_server`` imports. All four gates
were green the whole time, because none of them ever resolved the extra.

So this runs outside the lock, against a real interpreter, and asserts the server completes
an MCP handshake and lists the tools the plugin advertises.

Usage:  python scripts/mcp_smoke.py <path-to-rr-executable>
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

# The tools plugins/reporadar/skills/paper-discovery/SKILL.md tells an agent to reach for.
# Listing them here rather than counting means a silently dropped tool fails the check.
EXPECTED_TOOLS = {
    "get_repo_profile",
    "get_ranked_papers",
    "explain_relevance",
    "rate_paper",
    "search_papers",
    "setup_repo",
    "update_corpus",
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
    # Exercise a tool WRAPPER, not just the listing -- pytest can never reach these,
    # because CI installs `--extra dev --extra evals` and has no SDK. This call also
    # proves the server serves an UNINITIALISED repository: `setup_repo` with no
    # arguments must come back asking for categories, rather than the process having
    # died at startup with the explanation on stderr where no client shows it.
    {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {"name": "setup_repo", "arguments": {}},
    },
]

TOOLS_LIST_ID = 2
SETUP_CALL_ID = 3


def fail(msg: str, stdout: str = "", stderr: str = "") -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    if stdout:
        print(f"--- stdout ---\n{stdout[:2000]}", file=sys.stderr)
    if stderr:
        print(f"--- stderr ---\n{stderr[:2000]}", file=sys.stderr)
    raise SystemExit(1)


def _drain(stream, sink: list[str]) -> None:
    """Read a pipe to EOF into *sink*. Both pipes get one of these: an undrained stderr
    can fill its buffer and wedge the server mid-handshake."""
    try:
        for line in stream:
            sink.append(line)
    except (ValueError, OSError):  # pragma: no cover - pipe closed under us
        pass


def _replies(lines: list[str]) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for line in list(lines):
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(msg, dict) and isinstance(msg.get("id"), int):
            out[msg["id"]] = msg
    return out


def _await(proc: subprocess.Popen, lines: list[str], want: int, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        got = _replies(lines)
        if want in got:
            return got
        if proc.poll() is not None:
            # Server exited. Give the drain threads a moment to catch the tail.
            time.sleep(0.2)
            return _replies(lines)
        time.sleep(0.05)
    return _replies(lines)


def _shutdown(proc: subprocess.Popen) -> None:
    try:
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.close()
    except OSError:  # pragma: no cover
        pass
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:  # pragma: no cover
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def main(rr: str) -> None:
    with tempfile.TemporaryDirectory() as work:
        # Deliberately NOT initialised first. The server has to serve an unconfigured
        # repository -- that is the whole point of `setup_repo` being a tool -- so this
        # runs against a bare directory and lets the handshake prove it.
        (Path(work) / "README.md").write_text("# smoke", encoding="utf-8")

        proc = subprocess.Popen(
            [rr, "mcp"],
            cwd=work,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        out: list[str] = []
        err: list[str] = []
        for stream, sink in ((proc.stdout, out), (proc.stderr, err)):
            threading.Thread(target=_drain, args=(stream, sink), daemon=True).start()

        try:
            assert proc.stdin is not None
            for request in HANDSHAKE:
                proc.stdin.write(json.dumps(request) + "\n")
            proc.stdin.flush()
            # stdin stays OPEN while we wait. Closing it is EOF, and the server treats EOF
            # as "session over" and begins shutting down -- which raced the reply it was
            # already writing and lost on 3.11 while passing on 3.12/3.13. A real client
            # holds stdin open for the life of the session; so do we.
            replies = _await(proc, out, SETUP_CALL_ID, timeout=120)
        finally:
            _shutdown(proc)

        stdout, stderr = "".join(out), "".join(err)
        if not stdout.strip():
            # This is the exact shape of the shipped outage: exit non-zero, zero JSON-RPC
            # bytes, and a message on stderr that no MCP client ever shows a user.
            fail(
                f"`rr mcp` produced no JSON-RPC output (exit {proc.returncode})",
                stdout,
                stderr,
            )
        if 1 not in replies or "result" not in replies[1]:
            fail("no initialize result", stdout, stderr)
        if TOOLS_LIST_ID not in replies or "result" not in replies[TOOLS_LIST_ID]:
            fail("no tools/list result", stdout, stderr)

        got = {t["name"] for t in replies[TOOLS_LIST_ID]["result"].get("tools", [])}
        missing = EXPECTED_TOOLS - got
        if missing:
            fail(f"tools/list is missing {sorted(missing)}; got {sorted(got)}")

        setup = replies.get(SETUP_CALL_ID, {}).get("result")
        if setup is None:
            fail(
                "no setup_repo result - the server did not serve an unconfigured repo",
                stdout,
                stderr,
            )
        if "needs_input" not in json.dumps(setup):
            fail(f"setup_repo did not ask for categories; got {json.dumps(setup)[:400]}")

        server = replies[1]["result"].get("serverInfo", {})
        print(f"ok: handshake completed against {server.get('name')!r}")
        print(f"ok: tools/list returned all {len(EXPECTED_TOOLS)} tools: {sorted(got)}")
        print("ok: setup_repo answered on an uninitialised repository")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
