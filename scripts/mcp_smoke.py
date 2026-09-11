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
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
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


MCP_JSON = Path(__file__).resolve().parents[1] / "plugins" / "reporadar" / ".mcp.json"


def _handshake(cmd: list[str], label: str) -> tuple[set[str], dict[int, dict]]:
    """Drive one MCP server through initialize, tools/list and a setup_repo call.

    Runs against a bare directory on purpose: the server has to serve an UNINITIALISED
    repository, which is the whole point of `setup_repo` being a tool.
    """
    with tempfile.TemporaryDirectory() as work:
        (Path(work) / "README.md").write_text("# smoke", encoding="utf-8")
        # A project marker, deliberately. Without one the server now REFUSES to act on a
        # directory it cannot identify, and this check would still see "needs_input" -- but
        # for the refusal rather than for the categories question it means to exercise.
        (Path(work) / "pyproject.toml").write_text("[project]", encoding="utf-8")

        proc = subprocess.Popen(
            cmd,
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
            replies = _await(proc, out, SETUP_CALL_ID, timeout=300)
        finally:
            _shutdown(proc)

        stdout, stderr = "".join(out), "".join(err)
        if not stdout.strip():
            # The exact shape of the shipped outage: exit non-zero, zero JSON-RPC bytes,
            # and a message on stderr that no MCP client ever shows a user.
            fail(f"{label}: no JSON-RPC output (exit {proc.returncode})", stdout, stderr)
        if 1 not in replies or "result" not in replies[1]:
            fail(f"{label}: no initialize result", stdout, stderr)
        if TOOLS_LIST_ID not in replies or "result" not in replies[TOOLS_LIST_ID]:
            fail(f"{label}: no tools/list result", stdout, stderr)

        tools = {t["name"] for t in replies[TOOLS_LIST_ID]["result"].get("tools", [])}
        return tools, replies


def _pinned_spec() -> str:
    args = json.loads(MCP_JSON.read_text(encoding="utf-8"))["mcpServers"]["reporadar"]["args"]
    for i, arg in enumerate(args):
        if arg == "--from" and i + 1 < len(args):
            return str(args[i + 1])
    fail(f"no --from spec in {MCP_JSON}")
    raise AssertionError  # unreachable; fail() exits


def _is_published(spec: str) -> bool:
    match = re.search(r'==([0-9][^\s"]*)', spec)
    if match is None:
        return False
    url = f"https://pypi.org/pypi/reporadar-papers/{match.group(1)}/json"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return bool(response.status == 200)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return False
        raise


def check_pinned(local_tools: set[str]) -> None:
    """The version the plugin installs must serve what this repository's server serves.

    The offline guard keeps `.mcp.json`'s pin equal to pyproject's version. This is the
    other half of that invariant: that the version was actually PUBLISHED with this code.

    Between merging a server change and releasing it, the plugin installs a version that
    predates the change -- and that is not hypothetical. `setup_repo` and `update_corpus`
    landed across three PRs while the pin sat on a release with only the five read-only
    tools, so the skill instructed agents to call a tool the running server did not have.
    Nothing caught it, because every guard compared the repository against itself.
    """
    spec = _pinned_spec()
    if not _is_published(spec):
        # A real and temporary state: merged, not yet released. Loud rather than silent, so
        # it is visible in the CI log of the commit that created the window.
        print(f"PENDING RELEASE: {spec} is not on PyPI yet — publish it before anyone installs")
        return

    pinned_tools, _ = _handshake(["uvx", "--from", spec, "rr", "mcp"], "pinned release")
    missing = local_tools - pinned_tools
    if missing:
        fail(
            f"the pinned release is stale: {spec} serves {sorted(pinned_tools)} but this "
            f"repository's server has {sorted(missing)} as well. Anyone installing the "
            f"plugin gets skills that reference tools their server does not have."
        )
    print(f"ok: the pinned release serves every tool this repository does ({spec})")


def main(rr: str) -> None:
    tools, replies = _handshake([rr, "mcp"], "local build")

    missing = EXPECTED_TOOLS - tools
    if missing:
        fail(f"tools/list is missing {sorted(missing)}; got {sorted(tools)}")

    setup = replies.get(SETUP_CALL_ID, {}).get("result")
    if setup is None:
        fail("no setup_repo result - the server did not serve an unconfigured repo")
    missing = (setup.get("structuredContent") or setup).get("missing")
    if missing != ["categories"]:
        fail(
            f"setup_repo should ask for categories on a plausible project; it asked for "
            f"{missing!r}. Got: {json.dumps(setup)[:400]}"
        )

    server = replies[1]["result"].get("serverInfo", {})
    print(f"ok: handshake completed against {server.get('name')!r}")
    print(f"ok: tools/list returned all {len(EXPECTED_TOOLS)} tools: {sorted(tools)}")
    print("ok: setup_repo answered on an uninitialised repository")

    check_pinned(tools)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
