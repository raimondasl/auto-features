"""Every captured subprocess output, and what happens when a byte will not decode.

On 2026-08-25 an 11-case baseline batch died on its first case. `claude --output-format
json` emitted UTF-8; `subprocess.run(text=True)` decoded it with the LOCALE codec, cp1252 on
this project's Windows box; byte 0x81 at position 1556 is undefined there. The reader thread
raised `UnicodeDecodeError` -- and `subprocess.run` **prints that traceback and swallows it**,
handing back `stdout=None`. The failure then surfaced four frames away as
`TypeError: the JSON object must be str, bytes or bytearray, not NoneType`, uncaught, after
the answer had already been billed for.

Two properties are worth pinning, and they are not the same property:

* **Nothing that captures output may decode with an unpinned `errors` handler.** A decode
  that raises does not raise *at the call site* -- it leaves an attribute unset for
  something downstream to trip over, which is the hardest shape of failure to read.
* **The right handler differs by what the caller does with the bytes.** `replace` is correct
  where output is read or logged and thrown away. It is wrong where bytes are read, edited
  and written BACK: `add_cron_job` round-trips the user's whole crontab, so `replace` would
  overwrite entries we do not own with U+FFFD. That pair uses `surrogateescape`, which
  round-trips byte-exactly, and this file checks the two ends still agree.

The survey is the point. C-14b in this project is a guard that inherited the defect it was
written to prevent, by checking only the modules where the bug had been found -- so this
reads every `subprocess.run` in `src/` and `evals/` rather than the four that were fixed.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))

SEARCH_ROOTS = (ROOT / "src", ROOT / "evals")
# `evals/.work/` holds cloned benchmark repositories -- other people's code, not ours.
SKIP_PARTS = {".work", "cache", "results", "__pycache__"}

# Where the bytes are read, edited and written back, so a lossy handler would corrupt data
# belonging to someone else. Both ends must use the same handler or the round trip is not
# byte-exact.
ROUND_TRIP_PAIR = ("_get_current_crontab", "_set_crontab")


def _our_python_files() -> list[Path]:
    out = []
    for root in SEARCH_ROOTS:
        for path in root.rglob("*.py"):
            if not SKIP_PARTS.isdisjoint(path.parts):
                continue
            out.append(path)
    return sorted(out)


def _kw(call: ast.Call, name: str) -> ast.expr | None:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def _captures_output(call: ast.Call) -> bool:
    node = _kw(call, "capture_output")
    if isinstance(node, ast.Constant) and node.value is True:
        return True
    return any(_kw(call, k) is not None for k in ("stdout", "stderr"))


def _decodes_to_text(call: ast.Call) -> bool:
    for key in ("text", "universal_newlines"):
        node = _kw(call, key)
        if isinstance(node, ast.Constant) and node.value is True:
            return True
    return _kw(call, "encoding") is not None


def _subprocess_calls() -> list[tuple[Path, int, ast.Call]]:
    """Every `subprocess.run` / `check_output` / `Popen` in our own source."""
    found = []
    for path in _our_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            owner = node.func.value
            if not (isinstance(owner, ast.Name) and owner.id == "subprocess"):
                continue
            if node.func.attr in {"run", "check_output", "Popen"}:
                found.append((path, node.lineno, node))
    return found


class TestTheSurvey:
    def test_the_survey_actually_finds_the_calls(self):
        """A survey that matched nothing would pass every test below."""
        calls = _subprocess_calls()
        assert len(calls) >= 8, f"only found {len(calls)} subprocess calls; the walk is broken"
        files = {p.name for p, _, _ in calls}
        assert {"baseline.py", "scheduler.py", "gh_issues.py", "notify.py"} <= files

    def test_every_text_capture_pins_its_errors_handler(self):
        """The defect, stated as a property rather than as four fixed line numbers."""
        offenders = [
            f"{path.relative_to(ROOT).as_posix()}:{lineno}"
            for path, lineno, call in _subprocess_calls()
            if _captures_output(call) and _decodes_to_text(call) and _kw(call, "errors") is None
        ]
        assert not offenders, (
            "these decode captured output with an unpinned `errors` handler; a byte the "
            f"locale codec rejects leaves the attribute unset, not an exception: {offenders}"
        )

    def test_output_that_is_written_back_round_trips_losslessly(self):
        """`replace` is data loss when the caller edits and rewrites what it read."""
        handlers = {}
        for path, _, call in _subprocess_calls():
            if path.name != "scheduler.py":
                continue
            node = _kw(call, "errors")
            if isinstance(node, ast.Constant):
                handlers.setdefault(node.value, 0)
                handlers[node.value] += 1
        assert "surrogateescape" in handlers, (
            "the crontab read/write pair must round-trip bytes exactly; `replace` would "
            "rewrite the user's other cron entries as U+FFFD"
        )

    def test_the_crontab_pair_agrees_on_its_handler(self):
        """Different handlers at the two ends is a silent one-way corruption."""
        import reporadar.scheduler as scheduler

        source = ast.parse(Path(scheduler.__file__).read_text(encoding="utf-8"))
        handlers = {}
        for func in ast.walk(source):
            if not isinstance(func, ast.FunctionDef) or func.name not in ROUND_TRIP_PAIR:
                continue
            for node in ast.walk(func):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    errors = _kw(node, "errors")
                    if isinstance(errors, ast.Constant):
                        handlers[func.name] = errors.value
        assert set(handlers) == set(ROUND_TRIP_PAIR), f"missing a half: {handlers}"
        assert len(set(handlers.values())) == 1, f"the two ends disagree: {handlers}"


class TestTheCrontabRoundTripIsLossless:
    """The data-safety claim, exercised rather than asserted about the source.

    `add_cron_job` reads the user's whole crontab and writes back everything that is not
    ours. A byte the locale codec cannot decode must survive that trip untouched, or we
    corrupt cron entries belonging to other tools.
    """

    @staticmethod
    def _fake_crontab(monkeypatch, raw: bytes) -> list[str]:
        """Simulate `crontab -l` emitting *raw*; capture what `crontab -` receives."""
        import reporadar.scheduler as scheduler

        written: list[str] = []

        def fake_run(cmd, **kw):
            handler = kw.get("errors") or "strict"
            codec = kw.get("encoding") or "cp1252"
            if cmd == ["crontab", "-l"]:
                return subprocess.CompletedProcess(cmd, 0, raw.decode(codec, handler), "")
            written.append(kw["input"])
            return subprocess.CompletedProcess(cmd, 0, "", "")

        monkeypatch.setattr(scheduler.subprocess, "run", fake_run)
        assert scheduler.add_cron_job("0 9 * * *", "cfg.yaml") is True
        assert written, "nothing was written back"
        return written

    def test_a_foreign_entry_survives_byte_for_byte(self, monkeypatch):
        import reporadar.scheduler as scheduler

        # A cron line owned by someone else, holding a byte cp1252 cannot decode.
        foreign = b"0 3 * * * /usr/bin/backup --tag caf\x81\n"
        written = self._fake_crontab(monkeypatch, foreign)
        out = written[0].encode("cp1252", "surrogateescape")
        assert foreign.rstrip(b"\n") in out, "the foreign entry was altered on the way back"
        assert b"\xef\xbf\xbd" not in out and "�" not in written[0], (
            "a replacement character means we overwrote a byte we do not own"
        )
        assert scheduler.CRON_MARKER in written[0], "our own entry should still be added"

    def test_our_own_lines_are_still_replaced_not_duplicated(self, monkeypatch):
        import reporadar.scheduler as scheduler

        existing = f"0 1 * * * old-command {scheduler.CRON_MARKER}\n".encode("cp1252")
        written = self._fake_crontab(monkeypatch, existing)
        assert written[0].count(scheduler.CRON_MARKER) == 1
        assert "old-command" not in written[0]


class TestTheSwallowedDecodeError:
    """The behaviour that made this so hard to read, demonstrated rather than asserted."""

    @pytest.mark.skipif(sys.platform != "win32", reason="cp1252 default is Windows-specific")
    @pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
    def test_bare_text_true_yields_none_rather_than_raising(self):
        """This is why the traceback pointed at `json.loads`, four frames from the cause."""
        proc = subprocess.run(
            [sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'\\x81')"],
            capture_output=True,
            text=True,
            encoding="cp1252",
        )
        assert proc.stdout is None, "if this ever raises instead, the guards can relax"

    def test_pinning_errors_makes_the_same_byte_survive(self):
        proc = subprocess.run(
            [sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'\\x81')"],
            capture_output=True,
            text=True,
            encoding="cp1252",
            errors="replace",
        )
        assert proc.stdout == "�"


class TestTheBaselineParserSurvivesIt:
    """Belt and braces: even with the decode fixed, `None` must be a status, not a crash."""

    def test_none_stdout_is_a_reason_not_an_exception(self):
        import baseline as baseline_mod

        ok, reason = baseline_mod._parse_cli_payload(None)
        assert ok is None
        assert "no readable stdout" in reason

    def test_empty_stdout_is_a_reason_too(self):
        import baseline as baseline_mod

        ok, reason = baseline_mod._parse_cli_payload("")
        assert ok is None and reason

    def test_a_real_payload_still_parses(self):
        import baseline as baseline_mod

        payload = (
            '{"type":"result","subtype":"success","is_error":false,'
            '"total_cost_usd":0.5,"result":"see below\\n```json\\n'
            '[{\\"arxiv_id\\": \\"2401.12345\\", \\"title\\": \\"A Paper\\"}]\\n```"}'
        )
        ok, reason = baseline_mod._parse_cli_payload(payload)
        assert reason == ""
        assert ok is not None and ok["ids"] == ["2401.12345"] and ok["cost_usd"] == 0.5
