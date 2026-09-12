"""Run the collection pipeline in a heavier environment than this process has.

The plugin's MCP server is installed as ``reporadar-papers[mcp]``: click, pyyaml, arxiv and
the SDK. Deliberately light, because every installation pays for it and an editor launches
it on a cold ``uvx`` cache. Dense discovery needs sentence-transformers and, through it,
torch — gigabytes, landing on everyone including the majority who never turn HyDE on.

Those two requirements are in direct conflict, and this module is the join. When a
repository's configuration asks for HyDE and this interpreter cannot provide it, collection
runs as ``uvx --from "reporadar-papers[hyde]==<this version>" rr update`` instead of in
process. ``uvx`` builds that environment once and caches it, so only the people who opted
into dense discovery ever pay for it, and they pay once.

It is the same pipeline either way — one ``run_pipeline``, two invocations — which is the
shape the CLI and the MCP server already share. What is new here is only *where* it runs.

Two things make this less fragile than spawning a subprocess usually is:

* **The child is the same version as the parent**, pinned from this process's own installed
  metadata. The ``--progress-json`` protocol it speaks is therefore never older or newer
  than the parser in this file.
* **Delegation is never the only path.** Anything that prevents it — no ``uvx``, no
  published version to pin, a configuration that would send the child to a different
  repository, a child that dies — falls back to collecting in this process, with a warning
  saying what did not run. A keyword-only digest the caller knows is keyword-only is worth
  more than an error.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import queue
import shutil
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DISTRIBUTION = "reporadar-papers"
HEAVY_EXTRA = "hyde"

# The escape hatch for anyone who would rather this process never spawn another. Set it to
# 0/no/off/false and collection stays in-process, with the ordinary "HyDE unavailable"
# warning from the pipeline.
ENABLE_ENV = "RR_HYDE_SUBPROCESS"

# How long the child may go silent before we say something. Not cosmetic: Copilot CLI's
# 180-second per-request timeout is reset by every progress notification, and building the
# heavy environment for the first time is several silent minutes of downloading.
HEARTBEAT_SECONDS = 20.0

# A wall clock on the whole thing. Collection is minutes and a first environment build can
# be several more, so this is set where only a wedged child reaches it — a uv lock nobody
# releases, a stalled connection. VS Code applies no timeout to a tool call at all, so
# without this such a child would hold the call open until the editor restarted.
TIMEOUT_ENV = "RR_HYDE_SUBPROCESS_TIMEOUT"
DEFAULT_TIMEOUT_SECONDS = 3600.0

# Kept for the error message when the child fails. Enough to carry a traceback, bounded so a
# chatty failure cannot be the thing that fills the caller's context.
_TAIL_LINES = 40


class DelegationError(RuntimeError):
    """The delegated run could not be completed. The caller falls back; it does not fail."""


@dataclass(frozen=True)
class Plan:
    """Where collection should run, and why.

    ``reason`` is filled in on both branches deliberately. "Collection ran here rather than
    there" is the first question asked when a digest is thin or a run is slow, and it is
    not answerable after the fact from anything the run leaves behind.
    """

    command: list[str] | None
    cwd: Path | None = None
    spec: str | None = None
    reason: str = ""
    # Set only when HyDE was configured and will NOT run. The pipeline warns on its own when
    # it reaches the stage and cannot import the encoder; this warns about the reasons the
    # pipeline never sees, which are the ones a user can act on.
    warning: str | None = None

    @property
    def delegated(self) -> bool:
        return self.command is not None


def hyde_importable() -> bool:
    """Whether THIS interpreter could run dense discovery.

    Checked by import machinery rather than by importing: importing sentence-transformers
    pulls torch in and costs seconds, and the answer is needed on every collection.
    """
    return all(_have(name) for name in ("sentence_transformers", "pyarrow"))


def _have(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):  # pragma: no cover - a broken partial install
        return False


def installed_version() -> str | None:
    """The version of RepoRadar running in this process, or None if it is not installed.

    The pin for the child. Reading it from installed metadata rather than from a constant is
    what guarantees parent and child are the same code: there is no second number to keep in
    step, and a checkout that was never installed correctly answers None and is not
    delegated to a version that might not exist.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(DISTRIBUTION)
    except PackageNotFoundError:  # pragma: no cover - running from an uninstalled tree
        return None


def uvx_executable() -> str | None:
    """The full path to ``uvx``, or None. Resolved rather than passed as a bare name because
    that is what makes the spawn work the same way on Windows as everywhere else."""
    return shutil.which("uvx")


def _enabled() -> bool:
    return os.environ.get(ENABLE_ENV, "1").strip().lower() not in {"0", "no", "off", "false"}


def _timeout() -> float:
    raw = os.environ.get(TIMEOUT_ENV, "").strip()
    try:
        parsed = float(raw)
    except ValueError:
        return DEFAULT_TIMEOUT_SECONDS
    return parsed if parsed > 0 else DEFAULT_TIMEOUT_SECONDS


def manual_command(version: str | None = None) -> str:
    """What a user could run by hand to get the same collection. Quoted in every warning
    this module produces, because a warning that names no next step is just bad news."""
    pin = f"=={version}" if version else ""
    return f'uvx --from "{DISTRIBUTION}[{HEAVY_EXTRA}]{pin}" rr update'


def child_paths(cfg: Any, repo: Path) -> tuple[Path, Path]:
    """The repository and database a delegated ``rr update`` would actually use.

    `rr update` takes both from ``repo_path`` in the configuration, resolved against its
    working directory; the MCP server takes them from whichever repository the client named.
    Usually identical — ``repo_path: .`` with the child started in that repository — but
    when they are not, the child would collect into a different store than the one the
    server reads back, and the digest would silently be the previous run's. So this is
    computed and compared rather than assumed.
    """
    configured = Path(cfg.repo_path).expanduser()
    child_repo = configured if configured.is_absolute() else repo / configured
    child_repo = child_repo.resolve()
    return child_repo, (child_repo / ".reporadar" / "papers.db").resolve()


def plan(cfg: Any, *, repo: Path, config_path: Path, db: Path) -> Plan:
    """Decide where this collection runs.

    Returns a Plan with ``command`` set to delegate, or None to run in process. Every
    outcome carries a reason, and every outcome that loses dense discovery carries a
    warning: a plugin that quietly dropped the channel worth +1.36 net@2 — the only one
    reaching 15 of 48 benchmark targets — would be doing the exact thing this project
    refuses to do elsewhere.
    """
    if not getattr(cfg, "hyde", None) or not cfg.hyde.enabled:
        return Plan(command=None, reason="HyDE is not enabled for this repository")

    if hyde_importable():
        return Plan(command=None, reason="this environment can run dense discovery itself")

    if not _enabled():
        return Plan(
            command=None,
            reason=f"{ENABLE_ENV} disables the dense-discovery subprocess",
            warning=(
                f"Dense discovery cannot run here and {ENABLE_ENV} is set, so collection "
                f"used keyword retrieval only. Unset it, or run: {manual_command()}"
            ),
        )

    version = installed_version()
    if version is None:
        return Plan(
            command=None,
            reason="cannot pin a version for the dense-discovery environment",
            warning=(
                "Dense discovery cannot run here, and RepoRadar's own version could not be "
                "determined, so there is no version to install it at. Collection used "
                f"keyword retrieval only. Run by hand: {manual_command()}"
            ),
        )

    uvx = uvx_executable()
    if uvx is None:
        return Plan(
            command=None,
            reason="uvx is not on PATH",
            warning=(
                "Dense discovery needs sentence-transformers, which this server's "
                "environment does not have, and `uvx` is not on PATH to build one. "
                "Collection used keyword retrieval only. Install uv "
                "(https://docs.astral.sh/uv/), or run: " + manual_command(version)
            ),
        )

    child_repo, child_db = child_paths(cfg, repo)
    if child_repo != repo.resolve() or child_db != db.resolve():
        return Plan(
            command=None,
            reason="a delegated run would collect into a different store",
            warning=(
                f"Dense discovery cannot run in this server's environment, and collecting "
                f"it elsewhere would write to {child_db} while this server reads {db} — so "
                f"it was skipped rather than run somewhere you would not see it. Set "
                f"`repo_path: {repo}` in {config_path} to fix it."
            ),
        )

    spec = f"{DISTRIBUTION}[{HEAVY_EXTRA}]=={version}"
    return Plan(
        command=[
            uvx,
            "--from",
            spec,
            "rr",
            "update",
            "--config",
            str(config_path),
            "--progress-json",
        ],
        cwd=repo,
        spec=spec,
        reason=f"dense discovery needs an environment this server does not have ({spec})",
    )


def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    # The progress stream carries the pipeline's own prose, em-dashes included, and the
    # child's default stdio encoding is cp1252 on Windows. Without this the first such
    # message is a UnicodeEncodeError inside the child rather than a line of progress.
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    # uv's progress bars are carriage-return animations on the same stream the protocol
    # uses. They cannot be rendered through an MCP progress notification anyway, and in the
    # failure tail they push out the error that explains the failure.
    env["UV_NO_PROGRESS"] = "1"
    return env


def _drain(stream: Any, name: str, sink: queue.Queue[tuple[str, str]]) -> None:
    try:
        for line in stream:
            sink.put((name, line.rstrip("\r\n")))
    finally:
        sink.put(("eof", name))


def _event(line: str) -> dict[str, Any] | None:
    """One protocol record, or None for anything else on the stream.

    Everything uv and the child's own error path write lands here too, so a line that is not
    a protocol record is ordinary rather than exceptional.
    """
    text = line.strip()
    if not text.startswith("{"):
        return None
    try:
        parsed = json.loads(text)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) and "event" in parsed else None


def run(plan: Plan, *, report: Any, timeout: float | None = None) -> dict[str, Any]:
    """Run *plan*'s command, forwarding its progress to *report*.

    Returns the same counts the in-process pipeline reports, so the caller's payload does
    not depend on which branch produced it. Raises DelegationError on anything that stops
    the child from finishing — the caller falls back to collecting in process.
    """
    if plan.command is None or plan.cwd is None:  # pragma: no cover - guarded by callers
        raise DelegationError("this plan does not delegate")

    limit = _timeout() if timeout is None else timeout
    deadline = time.monotonic() + limit
    report.info(f"Collecting in a dense-discovery environment ({plan.spec}).")

    try:
        proc = subprocess.Popen(
            plan.command,
            cwd=str(plan.cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_child_env(),
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
    except OSError as exc:
        raise DelegationError(f"could not start {plan.command[0]}: {exc}") from exc

    events: queue.Queue[tuple[str, str]] = queue.Queue()
    for handle, name in ((proc.stderr, "protocol"), (proc.stdout, "plain")):
        threading.Thread(target=_drain, args=(handle, name, events), daemon=True).start()

    tail: deque[str] = deque(maxlen=_TAIL_LINES)
    result: dict[str, Any] | None = None
    open_streams = 2
    heard = False
    abandoned = False
    started = time.monotonic()

    try:
        while open_streams:
            # Checked every pass rather than only when the queue goes quiet: a child stuck
            # in a retry loop is talkative, and a deadline that only a SILENT child can
            # reach is not a deadline.
            if time.monotonic() > deadline:
                abandoned = True
                raise DelegationError(
                    f"the dense-discovery run did not finish within {limit:.0f}s and was stopped"
                )
            try:
                stream, line = events.get(timeout=HEARTBEAT_SECONDS)
            except queue.Empty:
                report.info(_waiting(heard, time.monotonic() - started))
                continue

            if stream == "eof":
                open_streams -= 1
                continue

            event = _event(line) if stream == "protocol" else None
            if event is None:
                if line.strip():
                    tail.append(line)
                continue
            # Only a PROTOCOL record means the pipeline itself has started. uv talks while
            # it is still building the environment, and treating that as "collecting" would
            # tell the user the long wait is nearly over when it has barely begun.
            heard = True
            kind = event.get("event")
            if kind == "result":
                result = event
            elif kind == "warn":
                report.warn(event.get("message", ""))
            else:
                report.info(event.get("message", ""))
    finally:
        _shutdown(proc, at_once=abandoned)

    code = proc.returncode
    if code != 0:
        raise DelegationError(_failure(code, tail))
    if result is None:
        # Exit 0 and no result record means the child never reached the end of the pipeline.
        # Reporting zero papers here would be indistinguishable from a genuinely empty
        # digest, which is an answer this project defends — so it must not be manufactured.
        lines = "\n".join(tail) if tail else "(no output)"
        raise DelegationError(
            f"the dense-discovery run exited cleanly without reporting a result:\n{lines}"
        )

    return {
        "run_id": result.get("run_id"),
        "stopped": result.get("stopped"),
        "queries": int(result.get("queries") or 0),
        "papers": int(result.get("papers") or 0),
        "scored": int(result.get("scored") or 0),
    }


def _shutdown(proc: subprocess.Popen[str], *, at_once: bool) -> None:
    """Make sure the child is gone before we return.

    Two cases with opposite right answers. Reaching the end of both pipes means the child is
    finishing its last writes, so it gets a moment. Passing the deadline means it is wedged
    — behind a uv lock nobody released, or a connection that never times out — and waiting
    on it again is waiting on the thing we just gave up on. An abandoned child would go on
    collecting into a store nobody is reading, outliving the editor session that started it.
    """
    if proc.poll() is not None:
        return
    if not at_once:
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=30)
        if proc.poll() is not None:
            return
    proc.kill()
    with contextlib.suppress(Exception):
        proc.wait(timeout=10)


def _waiting(heard: bool, elapsed: float) -> str:
    """The heartbeat. Says which of the two long waits this is, because they have different
    causes and only one of them is ever repeated."""
    if heard:
        return f"  Still collecting ({elapsed:.0f}s)..."
    return (
        f"  Building the dense-discovery environment ({elapsed:.0f}s). The first run "
        f"downloads the embedding model's dependencies; later runs reuse the cache."
    )


def _failure(code: int, tail: deque[str]) -> str:
    lines = "\n".join(tail) if tail else "(no output)"
    return f"the dense-discovery run exited {code}:\n{lines}"
