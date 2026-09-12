"""Tests for reporadar.delegate — running the pipeline in an environment this one is not.

Two things are load-bearing here and neither is the happy path.

The first is that **no branch may lose dense discovery silently**. It is worth +1.36 net@2
and is the only channel reaching 15 of 48 benchmark targets, so a plugin that quietly
collected without it would hand back a thin digest that looks exactly like a thin
literature. Every way this can decline to delegate is therefore checked for a warning.

The second is that **a delegated run that goes wrong must not take the collection with it**.
The subprocess is the most machinery in the plugin and the likeliest thing to break; the
fallback is what keeps a broken one costing a retrieval channel rather than an answer.
"""

from __future__ import annotations

import json
import shutil
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from reporadar import delegate


class _Hyde:
    def __init__(self, enabled: bool, index_dir: str) -> None:
        self.enabled = enabled
        self.index_dir = index_dir


class Cfg:
    """Stands in for a RepoRadarConfig: the fields this module reads."""

    def __init__(
        self, *, enabled: bool = True, repo_path: str = ".", index_dir: str = "no-index-here"
    ) -> None:
        self.hyde = _Hyde(enabled, index_dir)
        self.repo_path = repo_path


# Kept before any fixture replaces it, for the tests that exercise the real lookup.
_REAL_INDEX_SYNCED = delegate.index_synced


def _shard(directory: Path, year: int = 1991) -> Path:
    """One synced shard as `hyde.index_shards` recognises it: a vector file and its ids."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{year}.npy").write_bytes(b"")
    (directory / f"{year}.ids").write_text("", encoding="utf-8")
    return directory


@dataclass
class Recorder:
    """A Reporter that remembers, with `warn` kept apart the way McpReporter keeps it."""

    infos: list[str] = field(default_factory=list)
    warns: list[str] = field(default_factory=list)

    def info(self, message: str) -> None:
        self.infos.append(message)

    def warn(self, message: str) -> None:
        self.warns.append(message)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / ".reporadar").mkdir()
    return tmp_path


def _places(repo: Path) -> dict[str, Path]:
    return {
        "repo": repo,
        "config_path": repo / ".reporadar.yml",
        "db": repo / ".reporadar" / "papers.db",
    }


@pytest.fixture
def cannot_run_hyde(monkeypatch: pytest.MonkeyPatch) -> None:
    """The plugin's server on a machine that HAS synced the index: light on purpose, so it
    cannot embed anything, but with something worth delegating for."""
    monkeypatch.setattr(delegate, "hyde_importable", lambda: False)
    monkeypatch.setattr(delegate, "index_synced", lambda cfg, repo: True)
    monkeypatch.setattr(delegate, "installed_version", lambda: "9.9.9")
    monkeypatch.setattr(delegate, "uvx_executable", lambda: "/usr/bin/uvx")
    monkeypatch.delenv(delegate.ENABLE_ENV, raising=False)


class TestWhereCollectionRuns:
    def test_a_repository_that_never_asked_for_hyde_stays_here(self, repo: Path) -> None:
        plan = delegate.plan(Cfg(enabled=False), **_places(repo))
        assert not plan.delegated
        assert plan.warning is None  # nothing was lost, so there is nothing to say
        assert "not enabled" in plan.reason

    def test_an_environment_that_can_embed_does_not_spawn_anything(
        self, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`rr mcp` from a full install, and every existing CLI user. The subprocess exists
        for the plugin's stripped-down environment and must not appear anywhere else."""
        monkeypatch.setattr(delegate, "hyde_importable", lambda: True)
        plan = delegate.plan(Cfg(), **_places(repo))
        assert not plan.delegated
        assert plan.warning is None

    @pytest.mark.usefixtures("cannot_run_hyde")
    def test_it_delegates_at_this_processs_own_version(self, repo: Path) -> None:
        """The pin is what makes the two ends of the --progress-json protocol the same
        code. A floating spec would eventually put a parser and an emitter from different
        releases on opposite ends of the same pipe."""
        plan = delegate.plan(Cfg(), **_places(repo))
        assert plan.delegated
        assert plan.spec == "reporadar-papers[hyde]==9.9.9"
        assert plan.command is not None
        assert "--from" in plan.command and plan.spec in plan.command

    @pytest.mark.usefixtures("cannot_run_hyde")
    def test_the_child_is_told_the_configuration_and_started_in_the_repository(
        self, repo: Path
    ) -> None:
        """Both halves matter. `rr update` resolves `repo_path: .` against its working
        directory, so a child started anywhere else profiles somewhere else — the same
        class of bug as the server inferring a repository from its own CWD."""
        places = _places(repo)
        plan = delegate.plan(Cfg(), **places)
        assert plan.command is not None
        assert plan.command[-3:] == ["--config", str(places["config_path"]), "--progress-json"]
        assert plan.cwd == repo

    @pytest.mark.usefixtures("cannot_run_hyde")
    def test_the_escape_hatch_turns_it_off_and_says_so(
        self, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(delegate.ENABLE_ENV, "0")
        plan = delegate.plan(Cfg(), **_places(repo))
        assert not plan.delegated
        assert plan.warning is not None and delegate.ENABLE_ENV in plan.warning

    @pytest.mark.usefixtures("cannot_run_hyde")
    def test_without_uvx_it_names_the_thing_to_install(
        self, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(delegate, "uvx_executable", lambda: None)
        plan = delegate.plan(Cfg(), **_places(repo))
        assert not plan.delegated
        assert plan.warning is not None and "uv" in plan.warning

    @pytest.mark.usefixtures("cannot_run_hyde")
    def test_without_a_known_version_it_refuses_to_guess_one(
        self, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unpinned spec would install whatever PyPI has latest, which is not this code.
        Better to collect without dense discovery and say so than to run a different
        version of the pipeline than the one the user is talking to."""
        monkeypatch.setattr(delegate, "installed_version", lambda: None)
        plan = delegate.plan(Cfg(), **_places(repo))
        assert not plan.delegated
        assert plan.warning is not None

    @pytest.mark.usefixtures("cannot_run_hyde")
    def test_it_will_not_collect_into_a_store_the_server_does_not_read(
        self, repo: Path, tmp_path: Path
    ) -> None:
        """The failure this prevents is invisible from the outside: the child collects
        happily into another repository's database, the server reads its own, and the digest
        is silently the PREVIOUS run's. Nothing errors and nothing looks wrong."""
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        plan = delegate.plan(Cfg(repo_path=str(elsewhere)), **_places(repo))
        assert not plan.delegated
        assert plan.warning is not None and str(elsewhere) in plan.warning

    @pytest.mark.usefixtures("cannot_run_hyde")
    def test_a_configuration_pointing_back_at_this_repository_is_fine(self, repo: Path) -> None:
        """The absolute form `rr init` writes. It names the same directory, so the check
        must accept it: refusing here would disable delegation for everyone who ran init."""
        plan = delegate.plan(Cfg(repo_path=str(repo)), **_places(repo))
        assert plan.delegated

    @pytest.mark.parametrize(
        "break_it",
        [
            pytest.param({"uvx_executable": lambda: None}, id="no uvx"),
            pytest.param({"installed_version": lambda: None}, id="no version"),
        ],
    )
    @pytest.mark.usefixtures("cannot_run_hyde")
    def test_no_route_loses_dense_discovery_quietly(
        self, repo: Path, monkeypatch: pytest.MonkeyPatch, break_it: dict
    ) -> None:
        """The invariant, stated once. A repository configured for HyDE that does not get it
        must always hear why — the keyword-only path measured 0 of 24 on this project's own
        benchmark, so being handed it unannounced is the worst outcome available."""
        for name, value in break_it.items():
            monkeypatch.setattr(delegate, name, value)
        plan = delegate.plan(Cfg(), **_places(repo))
        assert not plan.delegated
        assert plan.warning, "a lost retrieval channel must be announced"
        assert "rr update" in plan.warning, "and the warning must name a way out"


class TestNoIndexMeansNoHeavyEnvironment:
    """Found by review: `plan` delegated without asking whether there was an index to search.

    `setup_repo` writes `hyde.enabled: true` for every user, so every new plugin user's first
    collection built the embedding model's environment -- several gigabytes of torch on Linux
    -- only for the child to report that no index was synced. That contradicted the promise
    #305's own docs make: only the people who opted in ever download it.
    """

    @pytest.mark.usefixtures("cannot_run_hyde")
    def test_a_machine_that_never_synced_does_not_build_the_environment(
        self, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(delegate, "index_synced", _REAL_INDEX_SYNCED)
        plan = delegate.plan(Cfg(index_dir=str(repo / "never-synced")), **_places(repo))
        assert not plan.delegated
        assert "index" in plan.reason
        # The pipeline raises its own "run `rr sync-index`" warning before any LLM call or
        # encoder load; a second one here would be the same news twice.
        assert plan.warning is None

    @pytest.mark.usefixtures("cannot_run_hyde")
    def test_once_synced_it_delegates(self, repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(delegate, "index_synced", _REAL_INDEX_SYNCED)
        index = _shard(repo / "synced")
        assert delegate.plan(Cfg(index_dir=str(index)), **_places(repo)).delegated

    @pytest.mark.usefixtures("cannot_run_hyde")
    def test_a_missing_index_is_reported_rather_than_the_escape_hatch(
        self, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With both true, the index is the cause worth naming: unsetting the variable would
        not help, and saying so would send the user to fix the wrong thing."""
        monkeypatch.setattr(delegate, "index_synced", _REAL_INDEX_SYNCED)
        monkeypatch.setenv(delegate.ENABLE_ENV, "0")
        plan = delegate.plan(Cfg(index_dir=str(repo / "never-synced")), **_places(repo))
        assert not plan.delegated
        assert "index" in plan.reason and plan.warning is None


class TestWhereTheIndexIsLookedFor:
    def test_an_absolute_index_dir_is_used_as_written(self, repo: Path, tmp_path: Path) -> None:
        index = _shard(tmp_path / "elsewhere" / "hyde-index")
        assert _REAL_INDEX_SYNCED(Cfg(index_dir=str(index)), repo)

    def test_an_empty_index_dir_is_not_synced(self, repo: Path, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        assert not _REAL_INDEX_SYNCED(Cfg(index_dir=str(empty)), repo)

    def test_a_relative_index_dir_resolves_against_the_repository(
        self, repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Against the repository, which is the CHILD's working directory -- not this
        process's. The server's own working directory is wherever the editor started it."""
        _shard(repo / "local-index")
        server_cwd = tmp_path / "where-the-editor-started-us"
        server_cwd.mkdir()
        monkeypatch.chdir(server_cwd)  # "local-index" does not exist relative to HERE
        assert _REAL_INDEX_SYNCED(Cfg(index_dir="local-index"), repo)


class TestWhichPathsAChildWouldUse:
    def test_a_relative_repo_path_resolves_against_the_repository(self, repo: Path) -> None:
        child_repo, child_db = delegate.child_paths(Cfg(), repo)
        assert child_repo == repo.resolve()
        assert child_db == (repo / ".reporadar" / "papers.db").resolve()

    def test_an_absolute_repo_path_is_taken_as_written(self, repo: Path, tmp_path: Path) -> None:
        other = tmp_path / "other"
        other.mkdir()
        child_repo, _ = delegate.child_paths(Cfg(repo_path=str(other)), repo)
        assert child_repo == other.resolve()


_EMIT = (
    "import sys, json, time\n"
    "def say(**kw):\n"
    "    sys.stderr.write(json.dumps(kw) + chr(10)); sys.stderr.flush()\n"
)


def _fake_child(body: str, tmp_path: Path) -> delegate.Plan:
    """A Plan that runs *body* instead of uvx. Real process, real pipes, no download.

    *body* is written against `say(event=..., ...)`, which puts one protocol record on
    stderr — the same stream and shape `rr update --progress-json` uses.
    """
    return delegate.Plan(
        command=[sys.executable, "-c", _EMIT + textwrap.dedent(body)],
        cwd=tmp_path,
        spec="reporadar-papers[hyde]==9.9.9",
        reason="test",
    )


class TestDrivingTheChild:
    def test_progress_warnings_and_counts_all_come_back(self, tmp_path: Path) -> None:
        plan = _fake_child(
            """
            say(event="info", message="Discovering papers by hypothesis (HyDE)...")
            say(event="warn", message="  HyDE index is 70 days old")
            say(event="result", run_id=7, stopped=None, queries=12, papers=114, scored=114)
            """,
            tmp_path,
        )
        report = Recorder()
        counts = delegate.run(plan, report=report)

        assert counts == {
            "run_id": 7,
            "stopped": None,
            "queries": 12,
            "papers": 114,
            "scored": 114,
        }
        assert any("HyDE" in m for m in report.infos)

    def test_a_warning_is_not_flattened_into_progress(self, tmp_path: Path) -> None:
        """The whole reason the protocol has two event kinds. A stage that was configured
        and could not run changes what a thin digest MEANS, and as one status line among
        sixty it is gone by the time anyone reads the result."""
        plan = _fake_child(
            """
            say(event="info", message="Collecting...")
            say(event="warn", message="  HyDE discovery unavailable: no index synced")
            say(event="result", run_id=1, stopped=None, queries=1, papers=0, scored=0)
            """,
            tmp_path,
        )
        report = Recorder()
        delegate.run(plan, report=report)

        assert report.warns == ["  HyDE discovery unavailable: no index synced"]
        assert not any("unavailable" in m for m in report.infos)

    def test_a_stopped_run_is_reported_rather_than_looking_like_a_crash(
        self, tmp_path: Path
    ) -> None:
        """`rr update` returns early when there is nothing to collect, and emits its result
        record before doing so. Without that a parent cannot tell a deliberate stop from a
        child that died, and those call for opposite responses."""
        plan = _fake_child(
            """
            say(event="result", run_id=3, stopped="no new papers", queries=4, papers=0,
                scored=0)
            """,
            tmp_path,
        )
        counts = delegate.run(plan, report=Recorder())
        assert counts["stopped"] == "no new papers"

    def test_uv_chatter_is_not_forwarded_as_progress(self, tmp_path: Path) -> None:
        """uv writes to the same stream. Its output is not the pipeline's, and forwarding it
        as progress would put install logs in front of the user as if they were findings."""
        plan = _fake_child(
            """
            sys.stderr.write("Installed 41 packages in 3.2s" + chr(10))
            sys.stdout.write("Top papers:" + chr(10))
            say(event="result", run_id=1, stopped=None, queries=1, papers=1, scored=1)
            """,
            tmp_path,
        )
        report = Recorder()
        delegate.run(plan, report=report)
        assert not any("Installed 41" in m for m in report.infos + report.warns)
        assert not any("Top papers" in m for m in report.infos + report.warns)

    def test_a_child_that_fails_raises_with_its_last_output(self, tmp_path: Path) -> None:
        """The tail is the only diagnosis available: the child's traceback goes nowhere a
        user of the plugin can see it."""
        plan = _fake_child(
            """
            sys.stderr.write("ModuleNotFoundError: No module named torch" + chr(10))
            sys.exit(2)
            """,
            tmp_path,
        )
        with pytest.raises(delegate.DelegationError) as exc:
            delegate.run(plan, report=Recorder())
        assert "No module named torch" in str(exc.value)
        assert "exited 2" in str(exc.value)

    def test_a_child_that_finishes_without_a_result_is_a_failure(self, tmp_path: Path) -> None:
        """Exit 0 with no result record means the run did not reach the end of the pipeline.
        Reporting zero papers would be indistinguishable from a real empty digest, which is
        a legitimate answer this project defends — so it must not be manufactured."""
        plan = _fake_child('sys.stderr.write("done" + chr(10))\n', tmp_path)
        with pytest.raises(delegate.DelegationError, match="without reporting a result"):
            delegate.run(plan, report=Recorder())

    def test_silence_is_narrated_rather_than_waited_out(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Building the heavy environment is minutes of nothing. Copilot CLI cancels a
        request after 180 silent seconds and resets that clock on every progress
        notification, so the heartbeat is what lets a first run finish at all."""
        monkeypatch.setattr(delegate, "HEARTBEAT_SECONDS", 0.1)
        plan = _fake_child(
            """
            time.sleep(0.45)
            say(event="result", run_id=1, stopped=None, queries=0, papers=0, scored=0)
            """,
            tmp_path,
        )
        report = Recorder()
        delegate.run(plan, report=report)
        assert any("Building the dense-discovery environment" in m for m in report.infos)

    def test_uv_talking_does_not_mean_the_pipeline_has_started(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """uv narrates the environment build on the same stream. Counting that as the
        pipeline starting would tell the user the long wait is nearly over when it has
        barely begun — and the environment build is the wait worth warning about."""
        monkeypatch.setattr(delegate, "HEARTBEAT_SECONDS", 0.1)
        plan = _fake_child(
            """
            sys.stderr.write("Resolved 41 packages in 1.2s" + chr(10))
            sys.stderr.flush()
            time.sleep(0.45)
            say(event="result", run_id=1, stopped=None, queries=0, papers=0, scored=0)
            """,
            tmp_path,
        )
        report = Recorder()
        delegate.run(plan, report=report)
        assert any("Building the dense-discovery environment" in m for m in report.infos)
        assert not any("Still collecting" in m for m in report.infos)

    def test_a_wedged_child_is_stopped_rather_than_held_open(self, tmp_path: Path) -> None:
        """VS Code applies no timeout to a tool call, so without this a child stuck behind a
        uv lock holds the call open until the editor is restarted."""
        plan = _fake_child("time.sleep(120)\n", tmp_path)
        with pytest.raises(delegate.DelegationError, match="did not finish within"):
            delegate.run(plan, report=Recorder(), timeout=0.5)

    def test_a_command_that_does_not_exist_is_an_error_the_caller_can_catch(
        self, tmp_path: Path
    ) -> None:
        plan = delegate.Plan(
            command=[str(tmp_path / "definitely-not-here"), "rr", "update"],
            cwd=tmp_path,
            spec="x",
            reason="test",
        )
        with pytest.raises(delegate.DelegationError, match="could not start"):
            delegate.run(plan, report=Recorder())


# Runs `delegate.run` the way the MCP server does: as a process whose stdin is a pipe that
# stays open with nothing arriving on it -- the JSON-RPC channel while a tool call is pending.
# Prints one JSON line: the counts on success, {"error": ...} when the delegated run failed.
_SERVER_STANDIN = """
import json, sys, threading, time
from pathlib import Path
from reporadar import delegate

mode = sys.argv[1]
if mode == "reader":
    # The MCP stdio reader: a synchronous read parked on stdin before anything is spawned.
    threading.Thread(target=lambda: sys.stdin.buffer.readline(), daemon=True).start()
    time.sleep(1.0)
    child = "print('never reads stdin')"
else:
    child = "import sys; sys.stdin.read()"
child += (
    "; import sys, json; sys.stderr.write(json.dumps({'event': 'result', 'run_id': 1,"
    " 'stopped': None, 'queries': 0, 'papers': 0, 'scored': 0}) + chr(10))"
)

class Quiet:
    def info(self, message): pass
    def warn(self, message): pass

plan = delegate.Plan(command=[sys.executable, "-c", child], cwd=Path.cwd(), spec="x", reason="t")
try:
    print(json.dumps(delegate.run(plan, report=Quiet(), timeout=float(sys.argv[2]))))
except delegate.DelegationError as exc:
    print(json.dumps({"error": str(exc)}))
sys.stdout.flush()
import os
os._exit(0)
"""


class TestTheChildNeverSharesTheServersStdin:
    """The 1.0.5 hang, found in a real VS Code session.

    `delegate.run` did not set `stdin`, so the child inherited the MCP server's — the JSON-RPC
    pipe from the editor. On Windows that deadlocks before a line of RepoRadar runs: the
    server has a synchronous read parked on that pipe, and the child's interpreter, setting
    up its own stdio, queries the same file object and waits for that read to finish. The
    read waits for the editor's next message; the editor waits for the tool result; the tool
    waits for the child. Observed directly: the child at 0.02 s of CPU for 22 minutes, its
    stack in `ZwQueryInformationFile` under `Py_InitializeFromConfig`, while the server sat
    in `NtReadFile` on stdin.

    Every earlier test launched from a shell or a Python-made pipe with no read pending,
    which is why none of them saw it. So these build the server's side on purpose.
    """

    def _standin(self, tmp_path: Path, mode: str, timeout: float) -> tuple[list[str], Path]:
        script = tmp_path / "server_standin.py"
        script.write_text(_SERVER_STANDIN, encoding="utf-8")
        return [sys.executable, str(script), mode, str(timeout)], script

    def test_the_child_cannot_see_the_servers_stdin(self, tmp_path: Path) -> None:
        """Portable, and the guard CI actually runs.

        The child here READS stdin. With the server's pipe inherited it would block for as
        long as the editor stays quiet — and on POSIX, a child that read it would steal
        JSON-RPC bytes from the session outright. Given its own empty stdin, it reads EOF
        and finishes at once.
        """
        import subprocess

        argv, _ = self._standin(tmp_path, "child-reads", timeout=8)
        server = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, cwd=tmp_path
        )
        try:
            # Never `communicate()`: it closes stdin, which hands the child EOF and would
            # pass this test with the bug still in place.
            server.wait(timeout=60)
            out = server.stdout.read() if server.stdout else ""
        finally:
            if server.poll() is None:
                server.kill()
            if server.stdin:
                server.stdin.close()

        result = json.loads(out.strip().splitlines()[-1])
        assert "error" not in result, (
            "the delegated child could see the server's stdin and blocked on it: "
            f"{result.get('error')}"
        )

    @pytest.mark.skipif(sys.platform != "win32", reason="the startup deadlock is Windows-only")
    @pytest.mark.skipif(shutil.which("node") is None, reason="needs Node, as VS Code uses")
    def test_launched_by_node_with_a_read_pending_the_child_still_starts(
        self, tmp_path: Path
    ) -> None:
        """The exact failure, reproduced: Node as the parent (VS Code's libuv pipes), a read
        parked on stdin, and a child that never touches stdin at all — it hangs anyway,
        inside interpreter startup, unless its stdin is its own."""
        import subprocess

        argv, _ = self._standin(tmp_path, "reader", timeout=20)
        launcher = tmp_path / "launch.js"
        launcher.write_text(
            "const { spawn } = require('child_process');\n"
            "const [cmd, ...args] = JSON.parse(process.argv[2]);\n"
            "const p = spawn(cmd, args, { stdio: ['pipe', 'pipe', 'inherit'] });\n"
            "let out = '';\n"
            "p.stdout.on('data', (d) => (out += d));\n"
            "p.on('exit', () => { process.stdout.write(out); process.exit(0); });\n",
            encoding="utf-8",
        )
        done = subprocess.run(
            ["node", str(launcher), json.dumps(argv)],
            capture_output=True,
            text=True,
            timeout=90,
            cwd=tmp_path,
        )

        result = json.loads(done.stdout.strip().splitlines()[-1])
        assert "error" not in result, (
            "under a Node parent with a pending stdin read, the delegated child hung at "
            f"startup: {result.get('error')}"
        )


class TestReadingTheProtocol:
    @pytest.mark.parametrize(
        "line",
        ["", "Installed 41 packages", "{not json", "[1, 2]", '{"message": "no event key"}'],
    )
    def test_anything_that_is_not_a_record_is_ignored(self, line: str) -> None:
        assert delegate._event(line) is None

    def test_a_record_survives_leading_whitespace(self) -> None:
        assert delegate._event('  {"event": "info", "message": "x"}  ') == {
            "event": "info",
            "message": "x",
        }


class TestTheTwoEndsOfTheProtocolAgree:
    """The drift guard.

    `rr update --progress-json` writes these records and this module reads them. They are
    pinned to the same version so they cannot be different releases, but they can still be
    edited apart — and the symptom would be a delegated collection that runs correctly,
    returns nothing, and is reported as a failure. So the emitter's own output is fed
    through the reader here rather than a hand-written approximation of it.
    """

    def _emitted(self, event: str, **fields: object) -> str:
        import click

        from reporadar.cli import JsonReporter

        written: list[str] = []
        reporter = JsonReporter(inner=Recorder())
        original = click.echo

        def capture(message: str = "", **kwargs: object) -> None:
            written.append(message)

        click.echo = capture  # type: ignore[assignment]
        try:
            if fields:
                reporter.emit(event, **fields)
            else:
                getattr(reporter, event)("a message")
        finally:
            click.echo = original  # type: ignore[assignment]
        assert len(written) == 1
        return written[0]

    def test_the_result_record_carries_every_field_the_reader_returns(self) -> None:
        line = self._emitted("result", run_id=7, stopped=None, queries=12, papers=114, scored=113)
        record = delegate._event(line)
        assert record is not None
        for key in ("run_id", "stopped", "queries", "papers", "scored"):
            assert key in record, f"the reader returns {key} and the emitter does not send it"
        assert json.loads(line)["event"] == "result"

    @pytest.mark.parametrize("kind", ["info", "warn"])
    def test_progress_records_are_the_kinds_the_reader_dispatches_on(self, kind: str) -> None:
        record = delegate._event(self._emitted(kind))
        assert record is not None
        assert record["event"] == kind
        assert record["message"] == "a message"


class TestTheManualCommand:
    def test_it_names_the_extra_that_carries_the_encoder(self) -> None:
        assert delegate.manual_command("1.2.3") == (
            'uvx --from "reporadar-papers[hyde]==1.2.3" rr update'
        )

    def test_without_a_version_it_is_still_runnable(self) -> None:
        assert "==" not in delegate.manual_command()


class TestThisEnvironment:
    def test_it_can_tell_whether_it_could_embed(self) -> None:
        """Whatever the answer here, it must be a decision rather than an exception: it is
        taken on every collection, including in environments with neither package."""
        assert isinstance(delegate.hyde_importable(), bool)

    def test_uvx_is_resolved_to_a_path_when_present(self) -> None:
        assert delegate.uvx_executable() == shutil.which("uvx")
