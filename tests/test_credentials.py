"""Tests for reporadar.credentials — where the API key lives and who is allowed to find it.

The precedence tests are the load-bearing ones. Key resolution used to be written out at six
call sites, and that is how `rr doctor` came to certify a gate as healthy while the pipeline
skipped it: two spellings of one rule, both of which looked right on their own.
"""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path
from unittest import mock

import pytest

from reporadar import credentials


@pytest.fixture(autouse=True)
def isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Never touch the developer's real credentials file."""
    monkeypatch.setenv("REPORADAR_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    return tmp_path


class Cfg:
    """Stands in for a SuggestionsConfig."""

    def __init__(self, openai: str = "", claude: str = "") -> None:
        self.openai_api_key = openai
        self.claude_api_key = claude


class TestWhereTheFileGoes:
    def test_the_override_wins(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REPORADAR_CONFIG_DIR", str(tmp_path / "elsewhere"))
        assert credentials.config_dir() == tmp_path / "elsewhere"
        assert credentials.auth_path().name == "auth.json"

    def test_the_secret_lives_in_its_own_file(self) -> None:
        """Separate from any other config, so "the one with the secret in it" is a single
        path you can name, back up deliberately, or delete in anger."""
        assert credentials.auth_path().name == "auth.json"
        assert credentials.auth_path() != credentials.config_dir() / "config.yml"


class TestStoringAndForgetting:
    def test_round_trip(self) -> None:
        credentials.store("openai", "sk-proj-abcdefghijklmnop")
        assert credentials.stored("openai") == "sk-proj-abcdefghijklmnop"

    def test_one_provider_does_not_clobber_the_other(self) -> None:
        credentials.store("openai", "sk-openai-aaaaaaaaaa")
        credentials.store("claude", "sk-ant-bbbbbbbbbb")
        assert credentials.stored("openai") == "sk-openai-aaaaaaaaaa"
        assert credentials.stored("claude") == "sk-ant-bbbbbbbbbb"

    def test_remove_reports_whether_there_was_anything(self) -> None:
        assert credentials.remove("openai") is False
        credentials.store("openai", "sk-openai-aaaaaaaaaa")
        assert credentials.remove("openai") is True
        assert credentials.stored("openai") == ""

    def test_an_unknown_provider_is_refused_rather_than_silently_stored(self) -> None:
        with pytest.raises(ValueError, match="unknown provider"):
            credentials.store("gemini", "x")

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
    def test_the_file_is_readable_only_by_its_owner(self) -> None:
        """Created with the mode rather than chmod'd afterwards: narrowing a file after
        creating it leaves a window where it is world-readable, and that window is exactly
        when the secret is being written into it."""
        path = credentials.store("openai", "sk-proj-abcdefghijklmnop")
        assert stat.S_IMODE(path.stat().st_mode) == 0o600

    def test_a_corrupt_file_does_not_take_down_a_run(self) -> None:
        """There may be a perfectly good environment variable sitting right there."""
        path = credentials.auth_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json at all", encoding="utf-8")
        assert credentials.stored("openai") == ""  # no exception

    def test_the_file_is_json_a_human_can_read_and_delete(self) -> None:
        path = credentials.store("openai", "sk-proj-abcdefghijklmnop")
        assert json.loads(path.read_text(encoding="utf-8"))["openai"]["api_key"]


class TestPrecedence:
    """Config, then environment, then file — chosen to change nothing that already works."""

    def test_config_wins_over_everything(self, monkeypatch: pytest.MonkeyPatch) -> None:
        credentials.store("openai", "sk-from-file-xxxxx")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env-xxxxxx")
        cfg = Cfg(openai="sk-from-config-xx")
        assert credentials.resolve_api_key("openai", cfg) == "sk-from-config-xx"
        assert credentials.source_of("openai", cfg) == "config"

    def test_environment_beats_the_file(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An exported key keeps working and costs nothing, which is what the GitHub Action
        and every existing shell rely on."""
        credentials.store("openai", "sk-from-file-xxxxx")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env-xxxxxx")
        assert credentials.resolve_api_key("openai", Cfg()) == "sk-from-env-xxxxxx"
        assert credentials.source_of("openai", Cfg()) == "environment"

    def test_the_file_is_the_fallback_neither_other_route_reaches(self) -> None:
        """The whole reason it exists: a server an editor started, with no shell env."""
        credentials.store("openai", "sk-from-file-xxxxx")
        assert credentials.resolve_api_key("openai", Cfg()) == "sk-from-file-xxxxx"
        assert credentials.source_of("openai", Cfg()) == "file"

    def test_nothing_anywhere_is_an_empty_string_not_an_error(self) -> None:
        assert credentials.resolve_api_key("openai", Cfg()) == ""
        assert credentials.source_of("openai", Cfg()) == "none"

    def test_source_of_never_disagrees_with_resolve(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """They are reported side by side, so a mismatch would be a lie in the output of
        the command written to stop exactly that kind of lie."""
        cases: list[tuple[Cfg, str | None, bool, str]] = [
            (Cfg(openai="sk-cfg-aaaaaaaaaa"), "sk-env-bbbbbbbbbb", True, "config"),
            (Cfg(), "sk-env-bbbbbbbbbb", True, "environment"),
            (Cfg(), None, True, "file"),
            (Cfg(), None, False, "none"),
        ]
        for cfg, env, with_file, expected in cases:
            credentials.remove("openai")
            monkeypatch.delenv("OPENAI_API_KEY", raising=False)
            if with_file:
                credentials.store("openai", "sk-file-cccccccccc")
            if env:
                monkeypatch.setenv("OPENAI_API_KEY", env)
            assert credentials.source_of("openai", cfg) == expected
            assert bool(credentials.resolve_api_key("openai", cfg)) == (expected != "none")

    def test_an_empty_environment_variable_is_not_a_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        credentials.store("openai", "sk-from-file-xxxxx")
        monkeypatch.setenv("OPENAI_API_KEY", "   ")
        assert credentials.resolve_api_key("openai", Cfg()) == "sk-from-file-xxxxx"


class TestTheStatusOutputCannotLeakTheKey:
    def test_the_fingerprint_hides_the_middle(self) -> None:
        key = "sk-proj-SECRETMIDDLE-1234"
        printed = credentials.fingerprint(key)
        assert "SECRETMIDDLE" not in printed
        assert printed.startswith("sk-") and printed.endswith("1234")

    def test_a_short_key_reveals_nothing_at_all(self) -> None:
        assert credentials.fingerprint("short") == "****"

    def test_absent_is_said_plainly(self) -> None:
        assert credentials.fingerprint("") == "(none)"


class TestNobodyResolvesKeysOnTheirOwnAnymore:
    """The guard for the bug that motivated this module.

    Key resolution was written out at six call sites. `rr doctor` read the config field and
    the environment variable; so did `llm_client`; and when `openai` was added to one and
    not the other, doctor reported a healthy gate for a pipeline that ran none. One rule
    with two implementations disagrees eventually, and the disagreement is silent.
    """

    def test_no_module_reads_a_vendor_key_out_of_the_environment_itself(self) -> None:
        root = Path(__file__).resolve().parents[1] / "src" / "reporadar"
        offenders = []
        for path in sorted(root.rglob("*.py")):
            if path.name == "credentials.py":
                continue  # the one place allowed to know
            body = path.read_text(encoding="utf-8")
            for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
                if f'os.environ.get("{var}"' in body or f'os.environ["{var}"]' in body:
                    offenders.append(f"{path.name} reads {var} directly")
        assert not offenders, (
            "resolve keys through reporadar.credentials.resolve_api_key instead: "
            + "; ".join(offenders)
        )

    def test_the_environment_variable_names_live_in_one_table(self) -> None:
        assert credentials.PROVIDERS["openai"][1] == "OPENAI_API_KEY"
        assert credentials.PROVIDERS["claude"][1] == "ANTHROPIC_API_KEY"
        assert set(credentials.PROVIDERS) == {"openai", "claude"}


def test_the_isolated_fixture_really_isolates(tmp_path: Path) -> None:
    """If this ever fails, the suite has been writing to the developer's real key file."""
    assert str(tmp_path) in str(credentials.auth_path())
    assert not os.environ.get("OPENAI_API_KEY")


class TestLookupNeverTakesTheRunWithIt:
    """`stored` is consulted on the way to every LLM call, so nothing about *looking* for a
    key may raise. Found by the existing suite: three tests clear the whole environment with
    `patch.dict(os.environ, {}, clear=True)`, which removes HOME/USERPROFILE and makes
    `Path.home()` raise RuntimeError outright. That is not only a test artifact — it is true
    inside containers, under `env -i`, and on some CI runners."""

    def test_a_missing_home_directory_reads_as_nothing_stored(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("REPORADAR_CONFIG_DIR", raising=False)
        with mock.patch.dict(os.environ, {}, clear=True):
            assert credentials.stored("openai") == ""
            assert credentials.resolve_api_key("openai") == ""
            assert credentials.source_of("openai") == "none"

    def test_an_unreadable_credentials_path_reads_as_nothing_stored(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            credentials, "_read", mock.Mock(side_effect=OSError("permission denied"))
        )
        assert credentials.stored("openai") == ""
