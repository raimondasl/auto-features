"""Which account pays for a `cli` baseline run, and whether you can tell afterwards.

The `claude` CLI authenticates two ways, and they are not interchangeable:

* **signed in** — the run bills the user's Claude subscription;
* **ANTHROPIC_API_KEY visible in the environment** — the run bills the API, and by the CLI's
  own warning (one of `_CLI_AUTH_CONFLICT_MARKERS`) a present key *disables connectors*.

So the choice is not only about money. Two runs of "the same" baseline under different auth
may not have offered the agent the same tools, which would make them different comparators —
the P13 distinction, one level down. Before 2026-08-26 nothing recorded which path a cached
run took, and the pre-existing behaviour only dropped the key *after* the CLI complained,
which it does not always do; a run that silently billed the API was indistinguishable
afterwards from one that did not.

What is pinned here:

* resolution order — an explicit `RR_EVAL_CLI_AUTH` always wins, `auto` asks the CLI, and an
  unanswerable question falls back to the key rather than to a guaranteed failure;
* the key is really absent from the child's environment under subscription auth, not merely
  intended to be;
* `cli_logged_in` asks with the key hidden, because with it visible the CLI reports the key
  as its auth method — the exact state we are trying to distinguish from a login;
* asking for subscription auth while signed out fails **before** spending, as a status
  rather than an exception.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))

import baseline as baseline_mod  # noqa: E402

AUTH_ENV = baseline_mod._CLI_AUTH_ENV


def _clear_probe_cache() -> None:
    """Drop `cli_logged_in`'s memo, tolerating tests that replaced it with a stub.

    It is lru_cached because auth does not change mid-run and a 37-case sweep should not
    spawn 37 identical probes — but that means the first stubbed answer would otherwise
    serve every test after it.
    """
    clear = getattr(baseline_mod.cli_logged_in, "cache_clear", None)
    if clear is not None:
        clear()


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    _clear_probe_cache()
    monkeypatch.delenv(AUTH_ENV, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    yield
    _clear_probe_cache()


def _status_payload(monkeypatch, payload: str | None, *, returncode: int = 0):
    """Stub `claude auth status`; return the env it was called with."""
    seen: dict[str, dict[str, str]] = {}

    def fake_run(cmd, **kw):
        seen["env"] = kw.get("env") or {}
        if payload is None:
            raise FileNotFoundError("claude")
        return subprocess.CompletedProcess(cmd, returncode, payload, "")

    monkeypatch.setattr(baseline_mod.subprocess, "run", fake_run)
    return seen


class TestLoggedInDetection:
    def test_reports_a_signed_in_cli(self, monkeypatch):
        _status_payload(monkeypatch, json.dumps({"loggedIn": True, "authMethod": "claudeai"}))
        assert baseline_mod.cli_logged_in() is True

    def test_reports_a_signed_out_cli(self, monkeypatch):
        _status_payload(monkeypatch, json.dumps({"loggedIn": False, "authMethod": "none"}))
        assert baseline_mod.cli_logged_in() is False

    def test_asks_with_the_api_key_hidden(self, monkeypatch):
        """With the key visible the CLI answers about the key, not about a login."""
        seen = _status_payload(monkeypatch, json.dumps({"loggedIn": False}))
        baseline_mod.cli_logged_in()
        assert "ANTHROPIC_API_KEY" not in seen["env"]

    @pytest.mark.parametrize("payload", ["", "not json", "null"])
    def test_unreadable_output_is_unknown_not_false(self, monkeypatch, payload):
        """None means "cannot tell"; returning False would assert something unmeasured."""
        _status_payload(monkeypatch, payload)
        assert baseline_mod.cli_logged_in() is None

    def test_a_missing_cli_is_unknown(self, monkeypatch):
        _status_payload(monkeypatch, None)
        assert baseline_mod.cli_logged_in() is None


class TestAuthModeResolution:
    @pytest.mark.parametrize("requested", ["api", "subscription"])
    def test_an_explicit_choice_wins_without_asking_the_cli(self, monkeypatch, requested):
        monkeypatch.setenv(AUTH_ENV, requested)

        def explode(*_a, **_kw):  # pragma: no cover - must not be reached
            raise AssertionError("an explicit choice must not consult the CLI")

        monkeypatch.setattr(baseline_mod, "cli_logged_in", explode)
        assert baseline_mod.cli_auth_mode() == requested

    def test_an_explicit_choice_is_case_and_space_insensitive(self, monkeypatch):
        monkeypatch.setenv(AUTH_ENV, "  SUBSCRIPTION ")
        assert baseline_mod.cli_auth_mode() == "subscription"

    def test_a_typo_is_refused_rather_than_silently_defaulted(self, monkeypatch):
        monkeypatch.setenv(AUTH_ENV, "subscribtion")
        with pytest.raises(ValueError, match="subscribtion"):
            baseline_mod.cli_auth_mode()

    def test_auto_prefers_the_subscription_when_signed_in(self, monkeypatch):
        monkeypatch.setattr(baseline_mod, "cli_logged_in", lambda *_a: True)
        assert baseline_mod.cli_auth_mode() == "subscription"

    @pytest.mark.parametrize("answer", [False, None])
    def test_auto_falls_back_to_the_key_when_not_signed_in(self, monkeypatch, answer):
        """Falling back to a guaranteed failure would be worse than billing the key."""
        monkeypatch.setattr(baseline_mod, "cli_logged_in", lambda *_a: answer)
        assert baseline_mod.cli_auth_mode() == "api"


class TestTheSubprocessEnvironment:
    """Intent is not enough — check the key is actually gone from the child."""

    @staticmethod
    def _capture(monkeypatch) -> list[dict[str, str]]:
        envs: list[dict[str, str]] = []

        def fake_run(cmd, **kw):
            envs.append(kw.get("env") or {})
            payload = json.dumps(
                {"subtype": "success", "is_error": False, "total_cost_usd": 0.0, "result": "x"}
            )
            return subprocess.CompletedProcess(cmd, 0, payload, "")

        monkeypatch.setattr(baseline_mod.subprocess, "run", fake_run)
        return envs

    def test_subscription_auth_hides_the_key_from_claude(self, monkeypatch, tmp_path):
        monkeypatch.setenv(AUTH_ENV, "subscription")
        monkeypatch.setattr(baseline_mod, "cli_logged_in", lambda *_a: True)
        envs = self._capture(monkeypatch)
        baseline_mod._run_cli(tmp_path, flags=[], timeout=5)
        assert envs and all("ANTHROPIC_API_KEY" not in e for e in envs)

    def test_api_auth_passes_the_key_through(self, monkeypatch, tmp_path):
        monkeypatch.setenv(AUTH_ENV, "api")
        envs = self._capture(monkeypatch)
        baseline_mod._run_cli(tmp_path, flags=[], timeout=5)
        assert envs and envs[0].get("ANTHROPIC_API_KEY") == "sk-ant-test"

    def test_the_judges_key_is_never_touched_in_this_process(self, monkeypatch, tmp_path):
        """Only the child's environment is edited; OPENAI_API_KEY must survive intact."""
        import os

        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
        monkeypatch.setenv(AUTH_ENV, "subscription")
        monkeypatch.setattr(baseline_mod, "cli_logged_in", lambda *_a: True)
        self._capture(monkeypatch)
        baseline_mod._run_cli(tmp_path, flags=[], timeout=5)
        assert os.environ["OPENAI_API_KEY"] == "sk-openai-test"
        assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-test"


class TestAskingForSubscriptionWhileSignedOut:
    def test_fails_before_spending_and_says_how_to_fix_it(self, monkeypatch, tmp_path):
        monkeypatch.setenv(AUTH_ENV, "subscription")
        monkeypatch.setattr(baseline_mod, "cli_logged_in", lambda *_a: False)

        def explode(*_a, **_kw):  # pragma: no cover - must not be reached
            raise AssertionError("no subprocess should run when auth cannot succeed")

        monkeypatch.setattr(baseline_mod.subprocess, "run", explode)
        out = baseline_mod._run_cli(tmp_path, flags=[], timeout=5)
        assert out["status"] == "no_cli_login"
        assert out["ids"] == [] and out["cost_usd"] == 0.0
        assert "auth login" in out["raw"] and AUTH_ENV in out["raw"]

    def test_a_failure_status_is_never_cached(self, monkeypatch, tmp_path):
        """The rule that keeps a broken run from becoming a permanent abstention."""
        monkeypatch.setattr(baseline_mod, "CACHE_DIR", tmp_path)
        monkeypatch.setenv(AUTH_ENV, "subscription")
        monkeypatch.setattr(baseline_mod, "cli_logged_in", lambda *_a: False)
        out = baseline_mod.run_baseline(tmp_path, repo_name="case", mode="cli")
        assert out["status"] == "no_cli_login"
        assert not list(tmp_path.rglob("case.json"))
