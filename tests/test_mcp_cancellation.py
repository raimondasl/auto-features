"""Stopping `update_corpus` must not take the MCP session down with it.

Found by an adversarial review of the delegation work and reproduced here independently.
When a client cancels a running `update_corpus` -- VS Code's Stop button sends
`notifications/cancelled`, and TypeScript-SDK clients send it when a request times out -- the
SDK answers "Request cancelled" at once and marks the request complete. But the collection is
waiting in `anyio.to_thread.run_sync`, which shields the handler, so the cancellation is
deferred and the work runs on. When it finishes, `update_corpus` returned without awaiting
anything, the deferred cancellation never landed, and the SDK tried to send a SECOND response:
`AssertionError: Request already responded to`, unhandled, inside the server's task group.

The session died silently. The process stayed up, so nothing looked wrong until the user's next
request -- `get_ranked_papers`, a retry, anything -- which was never answered. To the user it
looked like RepoRadar crashing on whatever they asked for next.

Drives the real server over an in-memory transport with raw JSON-RPC, so the request ids and
the cancellation are exactly what an editor sends. The collection is stubbed to be slow and
offline; what is under test is the tool's handling of the protocol, not the pipeline.
"""

from __future__ import annotations

import contextlib
import time
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("mcp", reason="the mcp extra is not installed in this environment")

import anyio  # noqa: E402
from mcp.shared.message import SessionMessage  # noqa: E402
from mcp.types import LATEST_PROTOCOL_VERSION, JSONRPCMessage  # noqa: E402

from reporadar import mcp_server  # noqa: E402

SLOW_SECONDS = 1.5


def _repo(tmp_path: Path) -> Path:
    # A project marker, or the server rightly refuses to guess which repository this is.
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# x\n", encoding="utf-8")
    (tmp_path / ".reporadar.yml").write_text("repo_path: .\n", encoding="utf-8")
    return tmp_path


def _slow_collect(cfg: Any, *, repo: Path, config_path: Path, db: Path, report: Any) -> dict:
    time.sleep(SLOW_SECONDS)  # runs in the worker thread, like the real pipeline
    return {
        "run_id": 1,
        "stopped": None,
        "queries": 0,
        "papers": 0,
        "scored": 0,
        "collected_in": "a stub",
    }


def _session(tmp_path: Path, *, cancel: bool) -> dict[str, Any]:
    """Initialize, call update_corpus, optionally cancel it, then check the session still
    answers a ping once the collection has finished."""
    server = mcp_server.build_server(repo_path=_repo(tmp_path))._mcp_server
    to_server_send, to_server_recv = anyio.create_memory_object_stream[SessionMessage | Exception](
        100
    )
    to_client_send, to_client_recv = anyio.create_memory_object_stream[SessionMessage](100)
    replies: dict[Any, dict[str, Any]] = {}
    outcome: dict[str, Any] = {"crash": None}

    async def serve() -> None:
        try:
            await server.run(to_server_recv, to_client_send, server.create_initialization_options())
        except BaseException as exc:  # an ExceptionGroup around the AssertionError
            outcome["crash"] = exc

    async def read() -> None:
        async for message in to_client_recv:
            body = message.message.model_dump(by_alias=True, exclude_none=True)
            if "id" in body:
                replies[body["id"]] = body

    async def send(**body: Any) -> None:
        payload = JSONRPCMessage.model_validate({"jsonrpc": "2.0", **body})
        # The server's end being gone means the session is dead; `reply` will say so.
        with contextlib.suppress(anyio.BrokenResourceError, anyio.ClosedResourceError):
            await to_server_send.send(SessionMessage(payload))

    async def reply(request_id: int, within: float) -> dict[str, Any] | None:
        with anyio.move_on_after(within):
            while request_id not in replies:
                await anyio.sleep(0.02)
        return replies.get(request_id)

    async def main() -> None:
        async with anyio.create_task_group() as tg:
            tg.start_soon(serve)
            tg.start_soon(read)

            await send(
                id=1,
                method="initialize",
                params={
                    "protocolVersion": LATEST_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "editor", "version": "0"},
                },
            )
            assert await reply(1, within=5), "the server never initialized"
            await send(method="notifications/initialized")

            await send(id=2, method="tools/call", params={"name": "update_corpus", "arguments": {}})
            if cancel:
                await anyio.sleep(0.3)  # the collection is under way in its worker thread
                await send(
                    method="notifications/cancelled",
                    params={"requestId": 2, "reason": "the user pressed Stop"},
                )
            outcome["call"] = await reply(2, within=SLOW_SECONDS + 5)

            # Let the collection's thread run out, which is when the crash used to happen,
            # then ask the session anything at all.
            await anyio.sleep(SLOW_SECONDS + 0.5)
            await send(id=3, method="ping")
            outcome["ping"] = await reply(3, within=5)
            outcome["crashed_by_now"] = outcome["crash"]
            tg.cancel_scope.cancel()

    anyio.run(main)
    return outcome


def _leaves(exc: BaseException | None) -> list[str]:
    """The real errors inside a (possibly nested) exception group, for a failure message."""
    if exc is None:
        return []
    if isinstance(exc, BaseExceptionGroup):
        return [leaf for sub in exc.exceptions for leaf in _leaves(sub)]
    return [f"{type(exc).__name__}: {exc}"]


def test_an_uncancelled_collection_returns_its_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The control. If this fails, the harness is broken and the cancellation test below
    proves nothing."""
    monkeypatch.setattr(mcp_server, "collect_payload", _slow_collect)
    outcome = _session(tmp_path, cancel=False)

    assert outcome["call"] is not None and "result" in outcome["call"]
    assert outcome["ping"] is not None, "the session stopped answering"
    assert outcome["crashed_by_now"] is None


def test_stopping_a_collection_does_not_kill_the_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mcp_server, "collect_payload", _slow_collect)
    outcome = _session(tmp_path, cancel=True)

    assert outcome["crashed_by_now"] is None, (
        "the server crashed after a cancelled update_corpus finished: "
        f"{_leaves(outcome['crashed_by_now'])}"
    )
    assert outcome["ping"] is not None, (
        "the session stopped answering after a cancelled update_corpus finished -- the user's "
        "next request would fail"
    )
